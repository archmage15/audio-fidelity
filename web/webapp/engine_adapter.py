"""Bridge between the website and the existing engine.

* Runs ``engine_runner.py`` with ENGINE_PYTHON in a new session (own process
  group) with a hard time limit; on timeout, error or cancellation the whole
  group is terminated and reaped.
* Builds a clean environment for the engine (no website secrets).
* Maps engine output files to the website result summary and redacts
  server paths from reports.
"""
from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import time
import uuid
from pathlib import Path

from .config import get_settings

RUNNER = Path(__file__).resolve().with_name('engine_runner.py')
SCORE_FILE = '还原度综合评分.json'
REPORT_FILE = '还原度综合报告.txt'
REPORT_FILES = {'summary': REPORT_FILE, 'detail': '音素对比_报告.txt'}
CHART_FILES = {'rhythm': '节奏对比.png', 'pitch': '音高对比.png'}
PART_KEYS = {'发音': 'pronunciation', '音变': 'sound_changes', '节奏': 'rhythm', '重音位置': 'stress'}
_ENV_DROP = ('REDIS_URL', 'REDIS_PASSWORD', 'SESSION_SECRET')


class EngineError(Exception):
    def __init__(self, code: str, message: str, detail: str | None = None,
                 elapsed: float | None = None, started_at: str | None = None):
        super().__init__(message)
        self.code, self.message, self.detail = code, message, detail
        self.elapsed, self.started_at = elapsed, started_at


class EngineTimeout(EngineError):
    pass


def engine_environment(asr_provider: str = 'xunfei') -> dict[str, str]:
    settings = get_settings()
    temp = settings.temp_dir / 'engine'
    env = {k: v for k, v in os.environ.items() if k not in _ENV_DROP}
    nltk = settings.engine_root / 'tmp/nltk-data'
    env.update({
        'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8', 'PYTHONFAULTHANDLER': '1',
        'PYTHONDONTWRITEBYTECODE': '1', 'MPLBACKEND': 'Agg',
        'TMPDIR': str(temp), 'TEMP': str(temp), 'TMP': str(temp),
        'AUDIO_TEST_TEMP_DIR': str(temp), 'AUDIO_TEST_NO_PAUSE': '1',
        'HF_HOME': str(temp / 'huggingface'), 'HF_HUB_DISABLE_TELEMETRY': '1',
        'NUMBA_CACHE_DIR': str(temp / 'numba-cache'),
        'MPLCONFIGDIR': str(temp / 'matplotlib'),
        'NLTK_DATA': str(nltk if nltk.is_dir() else temp / 'nltk-data'),
    })
    from .asr import engine_env
    env.pop('PYTHONPATH', None)
    env.update(engine_env(asr_provider))
    if settings.hf_offline:
        env.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
    for key in ('HF_HOME', 'NUMBA_CACHE_DIR', 'MPLCONFIGDIR'):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    temp.mkdir(parents=True, exist_ok=True)
    return env


def _kill_group(proc: subprocess.Popen) -> None:
    for sig, grace in ((signal.SIGTERM, 5.0), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=grace)
            if sig == signal.SIGKILL or proc.returncode is not None:
                # Also remove any descendants that outlived the runner itself.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                return
        except subprocess.TimeoutExpired:
            continue


def run_engine(action: str, *, reference: Path, imitation: Path | None, output_dir: Path | None,
               logs_dir: Path | None, log_path: Path, timeout: float,
               asr_provider: str = 'xunfei') -> dict:
    """Run one engine call; returns the runner result dict (ok=True) or raises."""
    settings = get_settings()
    if settings.engine_mode == 'server':
        if timeout <= 1:
            raise EngineTimeout('timed_out', '任务处理超时。', 'no time left before engine start', 0.0)
        spec = {'action': action, 'engine_root': str(settings.engine_root),
                'reference': str(reference), 'imitation': str(imitation) if imitation else None,
                'cache_dir': str(settings.reference_cache_dir),
                'output_dir': str(output_dir) if output_dir else None,
                'logs_dir': str(logs_dir) if logs_dir else None, 'log_path': str(log_path)}
        return _run_via_server(spec, engine_environment(asr_provider), timeout)
    if not settings.engine_python.exists():
        raise EngineError('engine_unavailable', '推理环境尚未就绪，请联系管理员。',
                          f'ENGINE_PYTHON not found: {settings.engine_python}')
    if timeout <= 1:
        raise EngineTimeout('timed_out', '任务处理超时。', 'no time left before engine start', 0.0)
    scratch = settings.temp_dir / 'engine' / f'run-{uuid.uuid4().hex}'
    scratch.mkdir(parents=True)
    spec_path, result_path = scratch / 'spec.json', scratch / 'result.json'
    spec = {'action': action, 'engine_root': str(settings.engine_root),
            'reference': str(reference), 'imitation': str(imitation) if imitation else None,
            'cache_dir': str(settings.reference_cache_dir),
            'output_dir': str(output_dir) if output_dir else None,
            'logs_dir': str(logs_dir) if logs_dir else None, 'result_path': str(result_path)}
    spec_path.write_text(json.dumps(spec, ensure_ascii=False), encoding='utf-8')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        with log_path.open('ab') as log:
            proc = subprocess.Popen([str(settings.engine_python), str(RUNNER), str(spec_path)],
                                    cwd=str(settings.engine_root), env=engine_environment(asr_provider),
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            try:
                returncode = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                _kill_group(proc)
                raise EngineTimeout('timed_out', '推理超时，任务已终止。',
                                    f'engine exceeded {timeout:.0f}s', time.perf_counter() - started)
            finally:
                if proc.returncode is None:
                    _kill_group(proc)  # cancellation / RQ timeout / worker shutdown
            _kill_group(proc)  # reap stray descendants of a finished runner
        parent_elapsed = time.perf_counter() - started
        try:
            result = json.loads(result_path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            raise EngineError('engine_crashed', '推理进程异常退出。',
                              f'runner exit code {returncode}, no result file; see {log_path.name}',
                              parent_elapsed)
        if not result.get('ok'):
            code = 'engine_import_failed' if result.get('phase') == 'import' else 'engine_failed'
            raise EngineError(code, _friendly_engine_error(result.get('error', '')),
                              (result.get('error') or '') + '\n' + (result.get('traceback') or ''),
                              result.get('elapsed'), result.get('started_at'))
        if returncode != 0:
            raise EngineError('engine_failed', '推理进程退出码异常。', f'exit code {returncode}',
                              result.get('elapsed'), result.get('started_at'))
        result['parent_elapsed'] = parent_elapsed
        return result
    finally:
        for path in (spec_path, result_path, result_path.with_suffix('.json.tmp')):
            path.unlink(missing_ok=True)
        try:
            scratch.rmdir()
        except OSError:
            pass


# ---- persistent engine server (webapp/engine_server.py) -------------------------

def _server_request(payload: dict, timeout: float) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(timeout)
        conn.connect(str(get_settings().engine_socket))
        conn.sendall(json.dumps(payload, ensure_ascii=False).encode('utf-8') + b'\n')
        data = b''
        while not data.endswith(b'\n'):
            chunk = conn.recv(65536)
            if not chunk:
                break
            data += chunk
    if not data:
        raise ConnectionError('engine server closed the connection without a result')
    return json.loads(data.decode('utf-8'))


def server_status(timeout: float = 2.0) -> dict | None:
    """Ping the persistent engine; None when it is starting, busy past timeout or down."""
    try:
        return _server_request({'op': 'ping'}, timeout)
    except (OSError, ValueError):
        return None


def _kill_server(pid: int | None) -> None:
    # systemd (Restart=always) restarts it and reloads the models.
    if pid:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def _run_via_server(spec: dict, env: dict, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    status = server_status()
    while not (status and status.get('ready')):
        if time.monotonic() > deadline - 60:
            raise EngineError('engine_unavailable', '推理服务暂不可用，请稍后重试。',
                              f'engine server not ready at {get_settings().engine_socket}', 0.0)
        time.sleep(3)  # starting up (loading models / warm-up) or restarting
        status = server_status()
    pid = status.get('pid')
    started = time.perf_counter()
    try:
        result = _server_request({'op': 'run', 'spec': spec, 'env': env},
                                 max(1.0, deadline - time.monotonic()))
    except TimeoutError:
        _kill_server(pid)
        raise EngineTimeout('timed_out', '推理超时，任务已终止。', 'engine server exceeded the job deadline',
                            time.perf_counter() - started) from None
    except (OSError, ValueError) as error:
        raise EngineError('engine_crashed', '推理服务异常中断，请稍后重试。',
                          f'{type(error).__name__}: {error}', time.perf_counter() - started) from None
    except BaseException:
        _kill_server(pid)  # RQ timeout / cancellation: stop the in-flight engine work
        raise
    if not result.get('ok'):
        code = 'engine_import_failed' if result.get('phase') == 'import' else 'engine_failed'
        raise EngineError(code, _friendly_engine_error(result.get('error', '')),
                          (result.get('error') or '') + '\n' + (result.get('traceback') or ''),
                          result.get('elapsed'), result.get('started_at'))
    result['parent_elapsed'] = time.perf_counter() - started
    return result


def _friendly_engine_error(error: str) -> str:
    if '讯飞' in error or 'xunfei' in error.lower():
        return '原音频转写服务调用失败，请稍后重试或联系管理员。'
    if '尚未缓存' in error:
        return '原音频尚未准备好。'
    return '评测程序执行失败，请稍后重试；如反复失败请联系管理员并提供任务编号。'


# ---- result mapping ------------------------------------------------------

def read_transcript(cache_dir: Path) -> str | None:
    """Reference text from the cache: 转写文本.txt first, then 原音频缓存.json."""
    try:
        text = (cache_dir / '转写文本.txt').read_text(encoding='utf-8').strip()
        if text:
            return text
    except OSError:
        pass
    try:
        data = json.loads((cache_dir / '原音频/原音频缓存.json').read_text(encoding='utf-8'))
        return str(data.get('转写文本') or '').strip() or None
    except (OSError, ValueError):
        return None


def _number(value) -> float | None:
    try:
        return None if value is None else round(float(value), 1)
    except (TypeError, ValueError):
        return None


def parse_advice(report: str) -> list[str]:
    lines = report.splitlines()
    tips: list[str] = []
    for index, line in enumerate(lines):
        if '下一步怎么练' in line or '改进建议' in line:
            for item in lines[index + 1:]:
                text = item.strip()
                if not text or set(text) <= {'-', '='}:
                    if tips and not text:
                        break
                    continue
                if text.startswith('·'):
                    tips.append(text.lstrip('· ').strip())
                elif tips:
                    break
            break
    return tips


def parse_warnings(report: str) -> list[str]:
    return [line.strip().lstrip('⚠').strip() for line in report.splitlines()
            if line.strip().startswith('⚠')]


def build_evaluation_summary(output_dir: Path, replacements: dict[str, str]) -> dict:
    """Validate engine output and build the public result; raises EngineError."""
    try:
        score = json.loads((output_dir / SCORE_FILE).read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        raise EngineError('result_invalid', '评分结果缺失或无法读取。', str(error))
    report_path = output_dir / REPORT_FILE
    if not report_path.is_file() or report_path.stat().st_size == 0:
        raise EngineError('result_invalid', '未生成综合报告。', str(report_path))
    overall = _number(score.get('综合还原度'))
    if overall is None:
        raise EngineError('result_invalid', '评分结果缺少综合分。', json.dumps(score)[:500])
    raw_parts = score.get('分项分') or {}
    report = sanitize_text(report_path.read_text(encoding='utf-8', errors='replace'), replacements)
    return {
        'overall': overall,
        'grade': score.get('等级'),
        'parts': {english: _number(raw_parts.get(chinese)) for chinese, english in PART_KEYS.items()},
        'advice': parse_advice(report),
        'warnings': parse_warnings(report),
        'reports': [name for name, file in REPORT_FILES.items() if (output_dir / file).is_file()],
        'charts': [name for name, file in CHART_FILES.items() if (output_dir / file).is_file()],
    }


_PATH_RE = re.compile(r'(?<![\w.:/])(?:[A-Za-z]:)?[\\/](?:[^\s\\/\'"：，、（）()<>|]+[\\/])+'
                      r'[^\s\\/\'"：，、（）()<>|]*')


def sanitize_text(text: str, replacements: dict[str, str]) -> str:
    """Replace known paths with labels, then redact any other absolute path."""
    for path in sorted(replacements, key=len, reverse=True):
        if path:
            text = text.replace(path, replacements[path])
    return _PATH_RE.sub('〈服务器路径〉', text)
