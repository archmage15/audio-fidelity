"""File-based entry points with independent reference caches and run outputs."""

from cpu_runtime import configure_cpu_runtime
configure_cpu_runtime()
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parent.parent
FORMATS = {'.wav', '.mp3', '.flac', '.ogg'}
CACHE_SCHEMA = 1


def audio_file(value):
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f'音频文件不存在：{path}')
    if path.suffix.lower() not in FORMATS:
        raise ValueError(f'不支持 {path.suffix}：{path}；支持 wav/mp3/flac/ogg，m4a 请先转换。')
    import soundfile as sf
    info = sf.info(str(path))
    if info.frames <= 0:
        raise ValueError(f'音频为空：{path}')
    return path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def run_id():
    return datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]


def cache_home(value=None):
    return Path(value).expanduser().resolve() if value else ROOT / 'tmp/reference_cache'


def cache_work(reference, cache_dir=None):
    return cache_home(cache_dir) / file_sha256(reference)


def cached_reference_valid(work, sha256):
    work = Path(work)
    try:
        manifest = json.loads((work / 'cache_manifest.json').read_text(encoding='utf-8'))
        data = json.loads((work / '原音频/原音频缓存.json').read_text(encoding='utf-8'))
        logits = work / '原音频/原音频_logits.npy'
        return bool(
            manifest.get('schema') == CACHE_SCHEMA
            and manifest.get('reference_sha256') == sha256
            and data.get('原音频SHA256') == sha256
            and data.get('转写文本')
            and (data.get('参考音素') or {}).get('音素')
            and data.get('原音频_CTC音素')
            and file_sha256(logits) == manifest.get('logits_sha256'))
    except (OSError, ValueError, TypeError):
        return False


def runtime_env(reference, imitation, work, output):
    output, work = Path(output).resolve(), Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    (work / "原音频").mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    temp = Path(os.environ.get('AUDIO_TEST_TEMP_DIR') or ROOT / 'tmp/runtime')
    temp.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    # Always set every per-job input; inherited context must not leak between jobs.
    env.update({
        'AUDIO_TEST_REFERENCE': str(reference),
        'AUDIO_TEST_IMITATION': str(imitation) if imitation else '',
        'AUDIO_TEST_WORK_DIR': str(work),
        'AUDIO_TEST_OUTPUT_DIR': str(output),
        'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8',
        'PYTHONFAULTHANDLER': '1',
        'AUTO_CONFIRM': '1',
        'TEMP': str(temp), 'TMP': str(temp),
        'NUMBA_CACHE_DIR': os.environ.get('NUMBA_CACHE_DIR') or str(ROOT / 'tmp/numba-cache'),
        'HF_HOME': os.environ.get('HF_HOME') or str(ROOT / 'tmp/huggingface'),
        'AUDIO_TEST_NO_PAUSE': '1',
    })
    env.pop('LOG_OFF', None)
    env.pop('LOG_QUIET', None)
    return env


def run_stage(script, env, *args):
    result = subprocess.run([sys.executable, str(ROOT / 'scripts' / script), *args],
                            cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL)
    if result.returncode:
        raise RuntimeError(f'{script} 执行失败，退出码 {result.returncode}；请查看本次日志。')


def publish_manifest(work, reference, sha256):
    data_path = Path(work) / '原音频/原音频缓存.json'
    data = json.loads(data_path.read_text(encoding='utf-8'))
    data['原音频SHA256'] = sha256
    write_json(data_path, data)
    write_json(Path(work) / 'cache_manifest.json', {
        'schema': CACHE_SCHEMA, 'reference_sha256': sha256,
        'source_path': str(reference), 'created_at': datetime.now().isoformat(),
        'logits_sha256': file_sha256(Path(work) / '原音频/原音频_logits.npy'),
    })
    if not cached_reference_valid(work, sha256):
        raise RuntimeError('原音频缓存未通过完整性检查。')


def import_legacy_cache(work, reference, sha256):
    """Reuse the already verified legacy reference without retranscribing it."""
    legacy = ROOT / 'audio/原音频'
    try:
        data = json.loads((legacy / '原音频缓存.json').read_text(encoding='utf-8'))
        original = Path(data['原音频路径'])
        from audio_paths import reference_cache_stale
        # Content hashes remain meaningful when moving from Windows to Linux.
        if data.get('原音频SHA256') != sha256:
            if reference_cache_stale(data, str(original)) is not False:
                return False
            if file_sha256(original) != sha256:
                return False
        logits = legacy / '原音频_logits.npy'
        if not logits.is_file() or logits.stat().st_size == 0:
            return False
        destination = Path(work) / '原音频'
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(logits, destination / logits.name)
        data['原音频SHA256'] = sha256
        write_json(destination / '原音频缓存.json', data)
        write_json(Path(work) / '参考音素.json', data['参考音素'])
        (Path(work) / '转写文本.txt').write_text(data['转写文本'], encoding='utf-8')
        publish_manifest(work, reference, sha256)
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def prepare_reference(reference, *, cache_dir=None, confirm_text=False, logs_dir=None,
                      analyze=True):
    reference = audio_file(reference)
    sha256 = file_sha256(reference)
    home = cache_home(cache_dir)
    home.mkdir(parents=True, exist_ok=True)
    work = home / sha256
    from filelock import FileLock
    with FileLock(str(home / (sha256 + '.lock')), timeout=1800):
        if cached_reference_valid(work, sha256):
            return {'reference': str(reference), 'sha256': sha256, 'cache': str(work), 'status': 'cached'}
        work.mkdir(parents=True, exist_ok=True)
        if import_legacy_cache(work, reference, sha256):
            return {'reference': str(reference), 'sha256': sha256, 'cache': str(work), 'status': 'imported'}
        if not analyze:
            raise ValueError(f'此原音频尚未缓存：{reference}\n请先运行 cache_audio.py "{reference}" 或批量缓存其文件夹。')
        output = Path(logs_dir) if logs_dir else ROOT / 'output/cache_batches' / run_id()
        env = runtime_env(reference, None, work, output)
        env['AUTO_CONFIRM'] = '0' if confirm_text else '1'
        env['AUDIO_TEST_FORCE_REFERENCE_ANALYSIS'] = '1'
        run_stage('analyze_reference.py', env, *([] if confirm_text else ['--auto']))
        if file_sha256(reference) != sha256:
            raise RuntimeError('缓存期间原音频发生变化，请停止修改后重新运行。')
        publish_manifest(work, reference, sha256)
        return {'reference': str(reference), 'sha256': sha256, 'cache': str(work), 'status': 'created'}


def evaluate(reference, imitation, *, output_dir=None, cache_dir=None):
    reference, imitation = audio_file(reference), audio_file(imitation)
    # Evaluation deliberately preserves the prepare-reference-first workflow.
    cache = prepare_reference(reference, cache_dir=cache_dir, analyze=False)
    output = Path(output_dir).expanduser().resolve() if output_dir else ROOT / 'output/runs' / run_id()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f'输出目录非空，请使用一个新的目录，避免混入旧报告：{output}')
    # Claim an empty output directory atomically for this invocation.
    claim = output / '.evaluation.lock'
    with claim.open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    imitation_hash = file_sha256(imitation)
    manifest = {'reference': str(reference), 'imitation': str(imitation),
                'reference_sha256': cache['sha256'], 'imitation_sha256': imitation_hash,
                'reference_cache': cache['cache'], 'output': str(output), 'status': 'running'}
    try:
        write_json(output / 'run.json', manifest)
        env = runtime_env(reference, imitation, cache['cache'], output)
        run_stage('run_all.py', env)
        if file_sha256(reference) != cache['sha256'] or file_sha256(imitation) != imitation_hash:
            raise RuntimeError('评测期间输入音频发生变化，本次结果无效，请重跑。')
        score = json.loads((output / '还原度综合评分.json').read_text(encoding='utf-8'))
        report = output / '还原度综合报告.txt'
        if not report.is_file() or not report.stat().st_size:
            raise RuntimeError('未生成综合报告。')
        inputs = score.get('输入音频') or {}
        if inputs.get('原音频') != str(reference) or inputs.get('模仿音频') != str(imitation):
            raise RuntimeError('报告输入路径与本次请求不一致，拒绝返回结果。')
        manifest.update(status='complete', score=score['综合还原度'], report=str(report))
        write_json(output / 'run.json', manifest)
        return manifest
    except Exception as error:
        manifest.update(status='failed', error=str(error))
        write_json(output / 'run.json', manifest)
        raise
    finally:
        claim.unlink(missing_ok=True)
