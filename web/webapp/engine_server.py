"""Persistent engine process: load models once, run every engine job in-process.

Run with ENGINE_PYTHON from /srv/audio-demo:  python -m webapp.engine_server
(systemd unit ``audio-fidelity-engine``). Only the standard library and the
website's stdlib-only modules are imported here; the engine is imported from
``<engine_root>/scripts`` and is NOT modified.

How the engine is kept unchanged:
  * The engine chains its stages with ``subprocess.run([sys.executable,
    scripts/X.py, ...])``. Those calls are intercepted and run in this
    process with ``runpy`` instead, isolated like a child process: os.environ,
    cwd, argv, sys.path and every engine module (scripts/*.py, which read
    paths from the environment at import time) are swapped in and restored
    afterwards. Anything else (espeak, ffmpeg …) still runs as a subprocess.
  * ``from_pretrained`` of the transformers classes the engine uses is
    memoised per process, so Wav2Vec2 / feature extractor / punctuation
    model are loaded once. emphases caches its model on ``emphases.core.infer``
    itself, which survives because site-packages modules are never reloaded.
  * 通义听悟 rerouting (asr_hook) is applied per job from the job's
    AFD_ASR_PROVIDER, both for the script and the import call sites.

Protocol: one JSON line per connection on a Unix socket.
  {"op": "ping"}                              → {"ready": true, "pid": …, …}
  {"op": "run", "spec": {...}, "env": {...}}  → runner-style result dict
Jobs are executed one at a time. On SIGTERM an idle server exits at once and a
busy one after the current job. The worker kills the process on timeout and
systemd restarts it (models are reloaded and warmed up again).
"""
from __future__ import annotations

import contextlib
import gc
import json
import os
import runpy
import signal
import socket
import sqlite3
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .asr import HOOK_DIR
from .config import get_settings
from .engine_adapter import engine_environment

SETTINGS = get_settings()
SCRIPTS = str(SETTINGS.engine_root / 'scripts')
MAX_REQUEST = 1 << 20
# Engine modules holding process-wide state (CPU affinity, torch thread pools —
# torch refuses to set interop threads twice). They are configured once at
# startup and never re-imported; every other engine module is per job.
PERSISTENT_ENGINE_MODULES = {'cpu_runtime'}
_ORIGINAL_RUN = subprocess.run
_JOURNAL = None
STATE = {'ready': False, 'busy': False, 'stop': False, 'jobs': 0, 'started_at': None,
         'warmup': None, 'model_loads': [], 'last_job_at': None}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')


def journal(message: str) -> None:
    """Server-level log line → systemd journal (the original stderr), never a job log."""
    print(f'[engine-server] {message}', file=_JOURNAL, flush=True)


def rss_mb() -> float:
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


# ---- isolation ----------------------------------------------------------------

def _is_engine_module(module) -> bool:
    path = getattr(module, '__file__', None)
    if not path or not os.path.abspath(path).startswith(SCRIPTS + os.sep):
        return False
    return os.path.splitext(os.path.basename(path))[0] not in PERSISTENT_ENGINE_MODULES


def _engine_modules() -> dict:
    return {name: mod for name, mod in list(sys.modules.items()) if _is_engine_module(mod)}


@contextlib.contextmanager
def isolated(env: dict | None = None, cwd: str | None = None, argv: list | None = None):
    """Make the block behave like a fresh child process for engine code."""
    saved_env, saved_cwd = dict(os.environ), os.getcwd()
    saved_argv, saved_path = sys.argv[:], sys.path[:]
    saved_exists = os.path.exists
    saved_modules = _engine_modules()
    for name in saved_modules:
        del sys.modules[name]
    try:
        if env is not None:
            os.environ.clear()
            os.environ.update({str(k): str(v) for k, v in env.items()})
        if cwd:
            os.chdir(cwd)
        if argv is not None:
            sys.argv = list(argv)
        yield
    finally:
        with contextlib.suppress(Exception):
            sys.stdout.flush()
            sys.stderr.flush()
        os.path.exists = saved_exists
        for name in _engine_modules():
            sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        os.environ.clear()
        os.environ.update(saved_env)
        os.chdir(saved_cwd)
        sys.argv, sys.path[:] = saved_argv, saved_path


def run_engine_script(script: str, args: list[str], env: dict | None, cwd: str | None) -> int:
    """`python scripts/<name>.py args` executed in this process; returns the exit code."""
    with isolated(env, cwd, [script, *args]):
        import afd_tingwu
        name = os.path.basename(script)
        provider = os.environ.get('AFD_ASR_PROVIDER', '').strip().lower()
        if name == 'xunfei_transcribe.py' and provider == 'disabled':
            print('[预热] 跳过转写', flush=True)
            return 1
        if name == 'xunfei_transcribe.py' and provider == 'tingwu':
            return afd_tingwu.cli_main(sys.argv)
        if name == 'analyze_reference.py' and provider == 'tingwu':
            afd_tingwu.allow_missing_xunfei_config(script)
        sys.path.insert(0, os.path.dirname(script))
        try:
            runpy.run_path(script, run_name='__main__')
            return 0
        except SystemExit as exit_:
            if exit_.code is None:
                return 0
            if isinstance(exit_.code, int):
                return exit_.code
            print(exit_.code, file=sys.stderr)
            return 1
        except Exception:  # noqa: BLE001 - same as an uncaught exception in a child
            traceback.print_exc()
            return 1


def _intercepting_run(args, *more, **kwargs):
    if (isinstance(args, (list, tuple)) and len(args) >= 2 and not more
            and os.path.realpath(str(args[0])) == os.path.realpath(sys.executable)
            and str(args[1]).endswith('.py')
            and os.path.abspath(str(args[1])).startswith(SCRIPTS + os.sep)
            and not kwargs.get('capture_output') and kwargs.get('stdout') is None
            and kwargs.get('input') is None):
        code = run_engine_script(os.path.abspath(str(args[1])), [str(a) for a in args[2:]],
                                 kwargs.get('env'), kwargs.get('cwd'))
        if kwargs.get('check') and code:
            raise subprocess.CalledProcessError(code, args)
        return subprocess.CompletedProcess(args, code)
    return _ORIGINAL_RUN(args, *more, **kwargs)


# ---- model cache ----------------------------------------------------------------

_MODEL_CACHE: dict = {}


def _memoise_from_pretrained(owner) -> None:
    original = owner.__dict__.get('from_pretrained')
    bound = owner.from_pretrained
    if getattr(bound, '_afd_cached', False):
        return
    function = original.__func__ if isinstance(original, classmethod) else bound.__func__

    def from_pretrained(cls, path, *args, **kwargs):
        key = (cls.__module__, cls.__qualname__, os.path.realpath(str(path)), repr(args),
               repr(sorted(kwargs.items())))
        if key not in _MODEL_CACHE:
            started = time.perf_counter()
            _MODEL_CACHE[key] = function(cls, path, *args, **kwargs)
            event = {'model': f'{cls.__qualname__}:{os.path.basename(str(path).rstrip("/"))}',
                     'seconds': round(time.perf_counter() - started, 2), 'at': utc()}
            STATE['model_loads'].append(event)
            journal(f"loaded {event['model']} in {event['seconds']} s")
        return _MODEL_CACHE[key]

    from_pretrained._afd_cached = True
    owner.from_pretrained = classmethod(from_pretrained)


def install_model_cache() -> None:
    import transformers
    for name in ('AutoModelForCTC', 'AutoFeatureExtractor', 'Wav2Vec2FeatureExtractor',
                 'AutoTokenizer', 'AutoModelForTokenClassification'):
        owner = getattr(transformers, name, None)
        if owner is not None:
            _memoise_from_pretrained(owner)


def preload_models() -> None:
    """Load the phoneme model eagerly (the punctuation model loads on first prepare)."""
    from transformers import AutoFeatureExtractor, AutoModelForCTC
    path = SETTINGS.engine_root / 'models' / 'wav2vec2-espeak'
    if (path / 'config.json').is_file():
        AutoModelForCTC.from_pretrained(str(path)).eval()
        with contextlib.suppress(Exception):
            AutoFeatureExtractor.from_pretrained(str(path))


# ---- jobs -----------------------------------------------------------------------

@contextlib.contextmanager
def output_to(log_path: Path):
    """Point fd 1/2 (Python prints and any child processes) at the job log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o640)
    try:
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        yield
    finally:
        with contextlib.suppress(Exception):
            sys.stdout.flush()
            sys.stderr.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for item in (fd, *saved):
            os.close(item)


def _emphases_loaded() -> bool:
    core = sys.modules.get('emphases.core')
    return bool(core is not None and hasattr(getattr(core, 'infer', None), 'model'))


def execute(spec: dict, env: dict) -> dict:
    result = {'action': spec['action'], 'pid': os.getpid(), 'server': True}
    loads_before = len(STATE['model_loads'])
    emphases_before = _emphases_loaded()
    with output_to(Path(spec['log_path'])), isolated(env, str(SETTINGS.engine_root)):
        sys.path.insert(0, SCRIPTS)
        try:
            from path_jobs import evaluate, prepare_reference
        except BaseException as error:  # noqa: BLE001
            result.update(ok=False, phase='import', error=f'{type(error).__name__}: {error}',
                          traceback=traceback.format_exc())
            return result
        result['started_at'] = utc()
        started = time.perf_counter()
        try:
            if spec['action'] == 'prepare':
                value = prepare_reference(spec['reference'], cache_dir=spec['cache_dir'],
                                          confirm_text=False, logs_dir=spec['logs_dir'])
            elif spec['action'] == 'evaluate':
                value = evaluate(reference=spec['reference'], imitation=spec['imitation'],
                                 cache_dir=spec['cache_dir'], output_dir=spec['output_dir'])
            else:
                raise ValueError('unknown action')
        except BaseException as error:  # noqa: BLE001
            result.update(ok=False, phase='engine', elapsed=time.perf_counter() - started,
                          finished_at=utc(), error=f'{type(error).__name__}: {error}',
                          traceback=traceback.format_exc())
        else:
            result.update(ok=True, elapsed=time.perf_counter() - started, finished_at=utc(),
                          value=value)
        finally:
            with contextlib.suppress(Exception):
                import matplotlib.pyplot as plt
                plt.close('all')
    if not emphases_before and _emphases_loaded():
        STATE['model_loads'].append({'model': 'emphases', 'seconds': None, 'at': utc()})
        journal('loaded emphases (inside job)')
    loads = STATE['model_loads'][loads_before:]
    result['model_loads'] = loads
    result['warm'] = not loads
    gc.collect()
    return result


def warm_up() -> None:
    """Short real evaluation (shortest ready reference against itself, no ASR call)."""
    started = time.perf_counter()
    temp = SETTINGS.temp_dir / 'engine'
    try:
        with sqlite3.connect(f'file:{SETTINGS.db_path}?mode=ro', uri=True, timeout=10) as conn:
            row = conn.execute(
                "SELECT a.id FROM refs r JOIN assets a ON a.id = r.asset_id WHERE r.status = 'ready' "
                'AND r.cache_version = ? ORDER BY a.duration_seconds LIMIT 1',
                (SETTINGS.cache_version,)).fetchone()
    except sqlite3.Error as error:
        row = None
        journal(f'warm-up: database unavailable ({error})')
    wav = SETTINGS.assets_dir / row[0] / 'audio.wav' if row else None
    if not wav or not wav.is_file():
        journal('warm-up: no ready reference, models preloaded only')
        STATE['warmup'] = {'mode': 'models-only', 'seconds': round(time.perf_counter() - started, 1)}
        return
    output = temp / f'warmup-{uuid.uuid4().hex[:8]}'
    env = engine_environment('disabled')
    spec = {'action': 'evaluate', 'reference': str(wav), 'imitation': str(wav),
            'cache_dir': str(SETTINGS.reference_cache_dir), 'output_dir': str(output),
            'logs_dir': None, 'log_path': str(temp / 'warmup.log')}
    with contextlib.suppress(OSError):
        (temp / 'warmup.log').unlink()
    result = execute(spec, env)
    import shutil
    shutil.rmtree(output, ignore_errors=True)
    STATE['warmup'] = {'mode': 'evaluate', 'ok': result.get('ok'), 'reference_asset': row[0],
                       'seconds': round(time.perf_counter() - started, 1),
                       'evaluate_seconds': round(result.get('elapsed') or 0, 1),
                       'error': result.get('error')}
    journal(f"warm-up: {STATE['warmup']}")


# ---- socket server ----------------------------------------------------------------

def _reply(conn: socket.socket, payload: dict) -> None:
    with contextlib.suppress(OSError):
        conn.sendall(json.dumps(payload, ensure_ascii=False, default=str).encode('utf-8') + b'\n')


def _read_request(conn: socket.socket) -> dict:
    conn.settimeout(30)
    data = b''
    while not data.endswith(b'\n'):
        chunk = conn.recv(65536)
        if not chunk:
            break
        data += chunk
        if len(data) > MAX_REQUEST:
            raise ValueError('request too large')
    return json.loads(data.decode('utf-8'))


def status() -> dict:
    return {'ready': STATE['ready'], 'busy': STATE['busy'], 'pid': os.getpid(), 'jobs': STATE['jobs'],
            'started_at': STATE['started_at'], 'last_job_at': STATE['last_job_at'],
            'warmup': STATE['warmup'], 'model_loads': STATE['model_loads'], 'rss_mb': round(rss_mb()),
            'engine_root': str(SETTINGS.engine_root)}


def handle(conn: socket.socket) -> None:
    try:
        request = _read_request(conn)
    except (OSError, ValueError) as error:
        _reply(conn, {'ok': False, 'error': f'bad request: {error}'})
        return
    if request.get('op') == 'ping':
        _reply(conn, status())
        return
    if request.get('op') != 'run':
        _reply(conn, {'ok': False, 'error': 'unknown op'})
        return
    spec = request['spec']
    journal(f"job start {spec.get('action')} log={spec.get('log_path')}")
    STATE['busy'] = True
    try:
        result = execute(spec, request.get('env') or {})
    finally:
        STATE['busy'] = False
    STATE['jobs'] += 1
    STATE['last_job_at'] = utc()
    result['rss_mb'] = round(rss_mb())
    journal(f"job done ok={result.get('ok')} elapsed={result.get('elapsed', 0):.1f}s "
            f"warm={result.get('warm')} rss={result['rss_mb']}MB")
    _reply(conn, result)


def _on_sigterm(signum, frame):
    STATE['stop'] = True
    if not STATE['busy']:
        raise SystemExit(0)
    journal('SIGTERM received, exiting after the current job')


def main() -> int:
    global _JOURNAL
    _JOURNAL = os.fdopen(os.dup(2), 'w', buffering=1, encoding='utf-8', errors='replace')
    STATE['started_at'] = utc()
    signal.signal(signal.SIGTERM, _on_sigterm)
    # stdin like the old child processes (the engine ends with input()).
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.close(devnull)
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    # Same environment as a one-shot engine process, set before torch/transformers
    # read it at import time (HF offline, thread caches, temp dirs).
    os.environ.update(engine_environment('xunfei'))
    os.environ.pop('PYTHONPATH', None)
    os.chdir(SETTINGS.engine_root)
    sys.path.insert(0, str(HOOK_DIR))
    import afd_tingwu
    afd_tingwu.install_import_hook()
    subprocess.run = _intercepting_run
    started = time.perf_counter()
    # Process-wide CPU affinity + thread limits, kept for the server's lifetime
    # (also exported to os.environ, like the engine does for its child processes).
    sys.path.insert(0, SCRIPTS)
    import cpu_runtime
    cpu_runtime.configure_cpu_runtime()
    journal(cpu_runtime.cpu_runtime_description())
    with isolated():
        import path_jobs  # noqa: F401 - import check + warms the engine's imports
    try:
        install_model_cache()
        preload_models()
    except ImportError as error:  # engine env without transformers (tests)
        journal(f'model cache disabled: {error}')
    journal(f'models ready in {time.perf_counter() - started:.1f} s, rss={rss_mb():.0f}MB')
    warm_up()

    path = SETTINGS.engine_socket
    path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(FileNotFoundError):
        path.unlink()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    os.chmod(path, 0o660)
    server.listen(8)
    STATE['ready'] = True
    journal(f'ready on {path} (startup {time.perf_counter() - started:.1f} s, rss={rss_mb():.0f}MB)')
    try:
        while not STATE['stop']:
            conn, _ = server.accept()
            with conn:
                handle(conn)
            if rss_mb() > SETTINGS.engine_server_max_rss_mb and not STATE['busy']:
                journal(f'rss {rss_mb():.0f}MB above limit, restarting to release memory')
                break
    except SystemExit:
        pass
    finally:
        server.close()
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
        journal('stopped')
    return 0


if __name__ == '__main__':
    sys.exit(main())
