"""Periodic reconciliation (queue ⇄ SQLite) and retention cleanup."""
from __future__ import annotations

import logging
import shutil
import time
from datetime import timedelta

from . import queueing, storage
from .config import get_settings

log = logging.getLogger('audio.maintenance')


def reconcile_queue() -> None:
    """Re-deliver attempts stuck before execution; close attempts whose worker vanished."""
    settings = get_settings()
    now = storage.utcnow()
    with storage.db() as conn:
        pending = [dict(r) for r in conn.execute(
            "SELECT * FROM jobs WHERE (status = 'pending_enqueue' AND updated_at < ?) "
            "OR (status = 'queued' AND updated_at < ?)",
            (storage.iso(now - timedelta(seconds=15)), storage.iso(now - timedelta(minutes=3))))]
        running = [dict(r) for r in conn.execute(
            "SELECT * FROM jobs WHERE status = 'running' AND started_at < ?",
            (storage.iso(now - timedelta(seconds=60)),))]
    if not queueing.redis_ok():
        return
    for job in pending:
        queueing.compensate(job)
    for job in running:
        status = queueing.rq_status(job['id'], job['attempt_no'])
        started = storage.parse_iso(job['started_at'])
        overdue = started and (now - started).total_seconds() > settings.rq_job_timeout + 120
        if status in ('failed', 'stopped', 'canceled', None) or overdue:
            close_lost_attempt(job, 'timed_out' if overdue else 'failed',
                               f'reconcile: rq status={status}, overdue={bool(overdue)}')


def close_lost_attempt(job: dict, status: str, detail: str) -> None:
    message = ('任务处理超时，已终止。可稍后点击重试。' if status == 'timed_out'
               else '处理进程意外中断，请点击重试。')
    if storage.finish_job(job, status, error_code='worker_lost', error_message=message,
                          error_detail=detail):
        log.warning('closed lost attempt %s/%s as %s', job['id'], job['attempt_id'], status)
        with storage.db() as conn:
            # Forced termination: no reliable timing, keep elapsed NULL.
            conn.execute("UPDATE inference_runs SET status = 'abandoned', error_code = 'worker_lost', "
                         "inference_finished_at = NULL WHERE job_id = ? AND attempt_id = ? "
                         "AND status = 'running'", (job['id'], job['attempt_id']))
            if job['type'] == 'prepare_reference':
                conn.execute("UPDATE refs SET status = 'failed', error = ?, updated_at = ? "
                             "WHERE id = ? AND status = 'preparing'",
                             (message, storage.iso(), job['reference_id']))


def reconcile_after_worker_start() -> None:
    """Called by the (single) worker on startup: nothing can be running yet."""
    with storage.db() as conn:
        running = [dict(r) for r in conn.execute("SELECT * FROM jobs WHERE status = 'running'")]
    for job in running:
        close_lost_attempt(job, 'failed', 'worker restarted while attempt was running')


def _remove_asset_files(asset_id: str) -> None:
    shutil.rmtree(storage.asset_dir(asset_id), ignore_errors=True)


def cleanup() -> None:
    from .asr import cleanup_media
    cleanup_media()
    settings = get_settings()
    now = storage.utcnow()
    retention = storage.iso(now - timedelta(days=settings.retention_days))
    idle = storage.iso(now - timedelta(hours=settings.idle_upload_hours))
    active = "('pending_enqueue','queued','running')"
    with storage.db() as conn:
        conn.execute('DELETE FROM sessions WHERE expires_at < ?', (storage.iso(now),))
        conn.execute('DELETE FROM idempotency WHERE created_at < ?', (retention,))
        # Run outputs of ordinary users past retention.
        old_runs = conn.execute(
            f'SELECT j.id FROM jobs j JOIN users u ON u.id = j.owner_id WHERE u.is_admin = 0 '
            f'AND j.files_purged = 0 AND j.status NOT IN {active} AND j.created_at < ?',
            (retention,)).fetchall()
        for (job_id,) in old_runs:
            shutil.rmtree(settings.runs_dir / job_id, ignore_errors=True)
            conn.execute('UPDATE jobs SET files_purged = 1 WHERE id = ?', (job_id,))
        # Ordinary user uploads: idle (never submitted) after 24h, all after retention.
        # Assets behind any reference record are materials and are never auto-deleted.
        candidates = conn.execute(
            f"SELECT a.id, a.created_at, EXISTS (SELECT 1 FROM jobs j WHERE j.imitation_asset_id = a.id) "
            f"FROM assets a JOIN users u ON u.id = a.owner_id WHERE u.is_admin = 0 "
            f"AND a.status != 'deleted' AND a.created_at < ? "
            f"AND NOT EXISTS (SELECT 1 FROM refs r WHERE r.asset_id = a.id) "
            f"AND NOT EXISTS (SELECT 1 FROM jobs j WHERE j.imitation_asset_id = a.id AND j.status IN {active})",
            (idle,)).fetchall()
        for asset_id, created_at, used in candidates:
            if used and created_at >= retention:
                continue
            _remove_asset_files(asset_id)
            conn.execute("UPDATE assets SET status = 'deleted', updated_at = ? WHERE id = ?",
                         (storage.iso(), asset_id))
    # Stale temporary files (interrupted uploads, engine scratch).
    cutoff = time.time() - 86400
    for path in (settings.temp_dir / 'uploads').glob('*'):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            pass


def run_once() -> None:
    for step in (reconcile_queue, cleanup):
        try:
            step()
        except Exception:  # noqa: BLE001 - keep the loop alive, log for operators
            log.exception('maintenance step %s failed', step.__name__)
