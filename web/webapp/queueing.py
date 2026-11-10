"""Redis/RQ queue access and the enqueue + compensation logic."""
from __future__ import annotations

import logging

from redis import Redis
from redis.exceptions import RedisError
from rq import Queue
from rq.exceptions import NoSuchJobError
from rq.job import Job

from . import storage
from .config import get_settings

log = logging.getLogger('audio.queue')
_redis: Redis | None = None


def redis_conn() -> Redis:
    global _redis
    if _redis is None:
        _redis = Redis.from_url(get_settings().redis_url, socket_connect_timeout=3, socket_timeout=10)
    return _redis


def audio_queue() -> Queue:
    return Queue(get_settings().queue_name, connection=redis_conn())


def rq_job_id(job_id: str, attempt_no: int) -> str:
    return f'{job_id}-a{attempt_no}'


def rq_status(job_id: str, attempt_no: int) -> str | None:
    try:
        job = Job.fetch(rq_job_id(job_id, attempt_no), connection=redis_conn())
    except NoSuchJobError:
        return None
    status = job.get_status(refresh=False)
    return getattr(status, 'value', status)


def enqueue(job_id: str, attempt_no: int) -> bool:
    """Send one attempt to RQ and mark it queued. False when Redis is unavailable."""
    settings = get_settings()
    try:
        audio_queue().enqueue_call(
            'webapp.tasks.execute', args=(job_id, attempt_no), job_id=rq_job_id(job_id, attempt_no),
            timeout=settings.rq_job_timeout, result_ttl=86400, failure_ttl=604800,
            description=f'{job_id} attempt {attempt_no}')
    except RedisError as error:
        log.warning('enqueue %s failed: %s', job_id, error)
        return False
    with storage.db() as conn:
        storage.mark_queued(conn, job_id, attempt_no)
    return True


def compensate(job: dict) -> None:
    """Re-deliver a pending/lost attempt with its original RQ id."""
    status = rq_status(job['id'], job['attempt_no'])
    if status in ('queued', 'started', 'deferred', 'scheduled'):
        with storage.db() as conn:
            storage.mark_queued(conn, job['id'], job['attempt_no'])
        return
    if status in ('finished',):
        return  # worker already handled it; its DB write is authoritative
    enqueue(job['id'], job['attempt_no'])


def redis_ok() -> bool:
    try:
        return bool(redis_conn().ping())
    except RedisError:
        return False
