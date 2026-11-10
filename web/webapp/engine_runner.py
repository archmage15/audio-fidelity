"""Engine entry executed by ENGINE_PYTHON in its own process group.

Usage: ENGINE_PYTHON engine_runner.py <spec.json>

Only the standard library is imported here; the existing engine is imported
from ``<engine_root>/scripts``. The spec names an atomic result file that
receives the timing (perf_counter around the engine call), the engine return
value or the error. The worker kills the whole process group on timeout.
"""
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone


def utc():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')


def write_result(path, data):
    temporary = path + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def main():
    with open(sys.argv[1], encoding='utf-8') as stream:
        spec = json.load(stream)
    sys.path.insert(0, os.path.join(spec['engine_root'], 'scripts'))
    result = {'action': spec['action'], 'pid': os.getpid()}
    try:
        # path_jobs applies the engine CPU affinity on import (inherited set).
        from path_jobs import evaluate, prepare_reference
    except BaseException as error:  # noqa: BLE001 - report import problems verbatim
        result.update(ok=False, phase='import', error=f'{type(error).__name__}: {error}',
                      traceback=traceback.format_exc())
        write_result(spec['result_path'], result)
        return 3
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
        write_result(spec['result_path'], result)
        return 1
    result.update(ok=True, elapsed=time.perf_counter() - started, finished_at=utc(), value=value)
    write_result(spec['result_path'], result)
    return 0


if __name__ == '__main__':
    sys.exit(main())
