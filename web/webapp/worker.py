"""Single RQ worker entry: ``python -m webapp.worker``.

Before taking jobs it applies the engine's CPU affinity to this process, so
the forked work-horse, FFmpeg and every engine subprocess share one CPU set
(at most 3 logical CPUs, leaving one free when possible). It also closes
attempts left 'running' by a previous worker process.
"""
from __future__ import annotations

import logging
import sys

from rq import Worker

from . import maintenance, queueing, storage
from .config import get_settings


def apply_cpu_affinity() -> str:
    settings = get_settings()
    sys.path.insert(0, str(settings.engine_root / 'scripts'))
    try:
        from cpu_runtime import configure_cpu_runtime, cpu_runtime_description
    finally:
        sys.path.pop(0)
    configure_cpu_runtime()
    return cpu_runtime_description()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    log = logging.getLogger('audio.worker')
    storage.init_db()
    log.info(apply_cpu_affinity())
    maintenance.reconcile_after_worker_start()
    settings = get_settings()
    worker = Worker([settings.queue_name], connection=queueing.redis_conn(),
                    name=f'audio-worker-{storage.new_id("w")[2:10]}')
    worker.work(with_scheduler=False, logging_level='INFO')
    return 0


if __name__ == '__main__':
    sys.exit(main())
