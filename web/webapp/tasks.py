"""RQ task entry ``webapp.tasks.execute`` and business state synchronization.

The queue only carries (job_id, attempt_no). The worker claims the attempt in
SQLite, normalizes inputs, runs the engine through the adapter, records one
inference_runs row per engine call and writes the final state only while it
still owns the attempt.
"""
from __future__ import annotations

import fcntl
import logging
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

from rq.timeouts import JobTimeoutException

from . import asr, storage
from .config import get_settings
from .engine_adapter import (EngineError, EngineTimeout, build_evaluation_summary,
                             read_transcript, run_engine)
from .media import MediaError, check_duration, normalize

log = logging.getLogger('audio.tasks')


class TaskError(Exception):
    def __init__(self, code: str, message: str, detail: str | None = None):
        super().__init__(message)
        self.code, self.message, self.detail = code, message, detail


@contextmanager
def asset_lock(asset_id: str):
    path = storage.asset_dir(asset_id) / '.lock'
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def ensure_normalized(asset_id: str, max_seconds: float) -> dict:
    """Make sure assets/<id>/audio.wav exists and is valid; returns the asset row."""
    settings = get_settings()
    with storage.db() as conn:
        asset = storage.get_asset(conn, asset_id)
    if not asset or asset['status'] == 'deleted':
        raise TaskError('asset_missing', '音频文件已被删除，请重新上传。')
    if asset['status'] == 'invalid':
        raise TaskError(asset.get('error') or 'decode_failed', '音频无法解码，请重新上传。')
    wav = storage.asset_wav_path(asset_id)
    with asset_lock(asset_id):
        if (asset['status'] == 'ready' and wav.is_file()
                and asset['transcode_version'] == settings.transcode_version):
            check_duration(asset['duration_seconds'], settings.min_audio_seconds, max_seconds)
            return asset
        try:
            info = normalize(storage.asset_source_path(asset_id), wav)
            check_duration(info['duration'], settings.min_audio_seconds, max_seconds)
        except MediaError as error:
            with storage.db() as conn:
                conn.execute("UPDATE assets SET status = 'invalid', error = ?, updated_at = ? WHERE id = ?",
                             (error.code, storage.iso(), asset_id))
            wav.unlink(missing_ok=True)
            raise TaskError(error.code, error.message)
        with storage.db() as conn:
            conn.execute(
                "UPDATE assets SET status = 'ready', duration_seconds = ?, sha256 = ?, error = NULL, "
                'transcode_version = ?, updated_at = ? WHERE id = ?',
                (info['duration'], info['sha256'], settings.transcode_version, storage.iso(), asset_id))
            return storage.get_asset(conn, asset_id)


def _remaining(deadline: float) -> float:
    return deadline - time.monotonic()


# ---- reference preparation -------------------------------------------------

def _prepare_reference(job: dict, deadline: float) -> dict:
    settings = get_settings()
    with storage.db() as conn:
        ref = storage.get_ref(conn, job['reference_id'])
    if not ref:
        raise TaskError('reference_missing', '原音频记录不存在。')
    asset = ensure_normalized(ref['asset_id'], settings.max_reference_seconds)
    storage.set_stage(job, 'preparing_reference')
    metrics = {'reference_duration_seconds': asset['duration_seconds'], 'imitation_duration_seconds': None}
    provider = asr.current_provider()
    storage.record_attempt_started(job, 'prepare_reference', metrics, reference_asset_id=asset['id'],
                                   imitation_asset_id=None, asr_provider=provider)
    logs_dir = settings.cache_logs_dir / job['id'] / job['attempt_id']
    outcome = _run_with_metrics(job, 'prepare', reference=storage.asset_wav_path(asset['id']),
                                imitation=None, output_dir=None, logs_dir=logs_dir,
                                log_path=settings.cache_logs_dir / job['id'] / f"{job['attempt_id']}.log",
                                asr_provider=provider, deadline=deadline)
    value = outcome.get('value') or {}
    storage.set_stage(job, 'saving')
    cache_path = Path(value.get('cache') or '')
    transcript = read_transcript(cache_path) if value.get('cache') else None
    with storage.db() as conn:
        conn.execute("UPDATE refs SET status = 'ready', cache_sha256 = ?, transcript = ?, error = NULL, "
                     'cache_version = ?, updated_at = ? WHERE id = ?',
                     (value.get('sha256'), transcript, settings.cache_version, storage.iso(), ref['id']))
    return {'reference_id': ref['id'], 'cache_status': value.get('status'),
            'transcript': transcript, 'metrics': metrics}


# ---- evaluation ------------------------------------------------------------

def _evaluate(job: dict, deadline: float) -> tuple[dict, str]:
    settings = get_settings()
    with storage.db() as conn:
        ref = storage.get_ref(conn, job['reference_id'])
    if not ref or ref['status'] != 'ready':
        raise TaskError('reference_not_ready', '原音频尚未准备好。')
    if ref['cache_version'] != settings.cache_version:
        raise TaskError('reference_not_ready', '原音频缓存版本已更新，请联系管理员重新准备。')
    reference = ensure_normalized(ref['asset_id'], settings.max_reference_seconds)
    imitation = ensure_normalized(job['imitation_asset_id'], settings.max_audio_seconds)
    storage.set_stage(job, 'evaluating')
    metrics = {'reference_duration_seconds': reference['duration_seconds'],
               'imitation_duration_seconds': imitation['duration_seconds']}
    provider = asr.current_provider()
    storage.record_attempt_started(job, 'evaluate', metrics, reference_asset_id=reference['id'],
                                   imitation_asset_id=imitation['id'], asr_provider=provider)
    output_dir = storage.run_output_dir(job['id'], job['attempt_id'])
    reference_wav = storage.asset_wav_path(reference['id'])
    imitation_wav = storage.asset_wav_path(imitation['id'])
    _run_with_metrics(job, 'evaluate', reference=reference_wav, imitation=imitation_wav,
                      output_dir=output_dir, logs_dir=None,
                      log_path=output_dir.parent / f"{job['attempt_id']}.log", asr_provider=provider,
                      deadline=deadline)
    storage.set_stage(job, 'saving')
    replacements = {str(reference_wav): '原音频.wav', str(imitation_wav): '模仿音频.wav',
                    str(output_dir): '〈本次报告目录〉', str(settings.reference_cache_dir): '〈缓存库〉',
                    str(settings.engine_root): '〈评测程序〉', str(settings.data_root): '〈数据目录〉'}
    summary = build_evaluation_summary(output_dir, replacements)
    summary['metrics'] = metrics
    return summary, str(output_dir)


def _run_with_metrics(job: dict, action: str, *, deadline: float, **kwargs) -> dict:
    try:
        outcome = run_engine(action, timeout=_remaining(deadline), **kwargs)
    except EngineTimeout as error:
        storage.finish_attempt(job, 'timed_out', error.elapsed, error_code=error.code)
        raise
    except EngineError as error:
        storage.finish_attempt(job, 'failed', error.elapsed, started_at=error.started_at,
                               error_code=error.code)
        raise
    except BaseException:
        # RQ timeout / shutdown while the engine ran: elapsed is unknown here.
        storage.finish_attempt(job, 'timed_out' if _is_rq_timeout() else 'failed', None,
                               error_code='interrupted')
        raise
    value = outcome.get('value') or {}
    cache_hit = value.get('status') in ('cached', 'imported') if action == 'prepare' else None
    storage.finish_attempt(job, 'succeeded', outcome.get('elapsed'), started_at=outcome.get('started_at'),
                           finished_at=outcome.get('finished_at'), cache_hit=cache_hit,
                           engine_warm=outcome.get('warm'))
    return outcome


def _is_rq_timeout() -> bool:
    import sys
    return isinstance(sys.exc_info()[1], JobTimeoutException)


# ---- entry point -----------------------------------------------------------

def execute(job_id: str, attempt_no: int) -> dict:
    job = storage.claim_job(job_id, attempt_no)
    if job is None:
        with storage.db() as conn:
            current = storage.get_job(conn, job_id)
        log.info('job %s attempt %s not claimable (status=%s)', job_id, attempt_no,
                 current and current['status'])
        return {'job_id': job_id, 'skipped': True, 'status': current and current['status']}
    settings = get_settings()
    deadline = time.monotonic() + settings.engine_timeout_seconds
    log.info('job %s %s attempt %s started', job_id, job['type'], job['attempt_id'])
    try:
        if job['type'] == 'prepare_reference':
            result, output_dir = _prepare_reference(job, deadline), None
        else:
            result, output_dir = _evaluate(job, deadline)
    except (EngineTimeout, JobTimeoutException) as error:
        detail = getattr(error, 'detail', None) or traceback.format_exc()
        _fail(job, 'timed_out', 'timed_out', '任务处理超时，已终止。可稍后点击重试。', detail)
        return {'job_id': job_id, 'status': 'timed_out'}
    except (TaskError, EngineError) as error:
        _fail(job, 'failed', error.code, error.message, error.detail or traceback.format_exc())
        return {'job_id': job_id, 'status': 'failed'}
    except Exception as error:  # noqa: BLE001
        log.exception('job %s crashed', job_id)
        _fail(job, 'failed', 'internal_error', '服务器内部错误，请稍后重试。',
              f'{type(error).__name__}: {error}\n{traceback.format_exc()}')
        return {'job_id': job_id, 'status': 'failed'}
    if not storage.finish_job(job, 'succeeded', stage='done', result=result, output_dir=output_dir):
        log.warning('job %s lost ownership before saving', job_id)
        return {'job_id': job_id, 'status': 'lost'}
    log.info('job %s succeeded', job_id)
    return {'job_id': job_id, 'status': 'succeeded'}


def _fail(job: dict, status: str, code: str, message: str, detail: str | None) -> None:
    log.warning('job %s %s: %s', job['id'], status, code)
    storage.finish_job(job, status, error_code=code, error_message=message, error_detail=detail)
    if job['type'] == 'prepare_reference' and job.get('reference_id'):
        with storage.db() as conn:
            conn.execute("UPDATE refs SET status = 'failed', error = ?, updated_at = ? "
                         "WHERE id = ? AND status = 'preparing'", (message, storage.iso(), job['reference_id']))
