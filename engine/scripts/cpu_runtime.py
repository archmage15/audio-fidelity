"""Bound the audio pipeline to at most three logical CPUs on Windows/Linux.

Call configure_cpu_runtime before importing numerical libraries. Child processes
inherit the chosen CPU IDs, so each stage uses the same CPUs without repeatedly
subtracting a reserved CPU from its parent's already limited affinity.
"""
import os
import sys

MAX_LOGICAL_CPUS = 3
_AFFINITY_ENV = '_AUDIO_TEST_CPU_AFFINITY'
_cpus = None
_threads = None
_configured = False


def _windows_affinity():
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GetProcessAffinityMask.argtypes = [wintypes.HANDLE,
                                             ctypes.POINTER(ctypes.c_size_t),
                                             ctypes.POINTER(ctypes.c_size_t)]
    kernel.GetProcessAffinityMask.restype = wintypes.BOOL
    kernel.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    kernel.SetProcessAffinityMask.restype = wintypes.BOOL
    handle = kernel.GetCurrentProcess()
    process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(handle, ctypes.byref(process_mask),
                                         ctypes.byref(system_mask)):
        raise ctypes.WinError(ctypes.get_last_error())
    allowed = {i for i in range(ctypes.sizeof(ctypes.c_size_t) * 8)
               if process_mask.value & (1 << i)}

    def apply(selected):
        if not kernel.SetProcessAffinityMask(handle, sum(1 << i for i in selected)):
            raise ctypes.WinError(ctypes.get_last_error())

    return allowed, apply


def _linux_affinity():
    from pathlib import Path

    allowed = set(os.sched_getaffinity(0))

    def apply(selected):
        # Linux affinity is per-thread; include any threads already present.
        for task in Path('/proc/self/task').iterdir():
            try:
                os.sched_setaffinity(int(task.name), selected)
            except ProcessLookupError:
                pass  # A helper thread exited while enumerating the tasks.
        os.sched_setaffinity(0, selected)

    return allowed, apply


def configure_cpu_runtime():
    """Apply affinity and worker limits without importing torch or numpy."""
    global _cpus, _threads
    if _cpus is not None:
        return _cpus
    if os.name == 'nt':
        allowed, apply = _windows_affinity()
    elif sys.platform.startswith('linux'):
        allowed, apply = _linux_affinity()
    else:
        raise RuntimeError('CPU affinity requires Windows or Linux.')
    if not allowed:
        raise RuntimeError('No logical CPU is available to this process.')

    inherited = os.environ.get(_AFFINITY_ENV, '')
    if inherited:
        try:
            selected = sorted(allowed & {int(cpu) for cpu in inherited.split(',')})
        except ValueError as error:
            raise ValueError('Invalid inherited audio CPU affinity.') from error
        if not selected:
            raise RuntimeError('Inherited audio CPU affinity is unavailable.')
        selected = selected[:MAX_LOGICAL_CPUS]
    else:
        budget = min(MAX_LOGICAL_CPUS, max(1, len(allowed) - 1))
        selected = sorted(allowed)[:budget]
        if len(allowed) == 1:
            print('[CPU] Only one logical CPU is available; no spare CPU can be left.',
                  file=sys.stderr, flush=True)

    requested = int(os.environ.get('AUDIO_TEST_CPU_THREADS', str(len(selected))))
    if requested < 1:
        raise ValueError('AUDIO_TEST_CPU_THREADS must be positive')
    threads = min(requested, len(selected))
    # Fail visibly if affinity cannot be applied; do not silently use every CPU.
    apply(set(selected))
    os.environ[_AFFINITY_ENV] = ','.join(map(str, selected))
    os.environ['AUDIO_TEST_CPU_THREADS'] = str(threads)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'GOTO_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'NUMEXPR_MAX_THREADS', 'NUMBA_NUM_THREADS',
                'RAYON_NUM_THREADS'):
        os.environ[key] = str(threads)
    os.environ['OMP_THREAD_LIMIT'] = str(threads)
    os.environ['OMP_MAX_ACTIVE_LEVELS'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    _cpus, _threads = tuple(selected), threads
    return _cpus


def configure_torch_cpu():
    global _configured
    configure_cpu_runtime()
    if _configured:
        return
    import torch
    torch.set_num_threads(_threads)
    torch.set_num_interop_threads(1)
    _configured = True


def cpu_runtime_description():
    cpus = configure_cpu_runtime()
    return (f'CPU：最多使用 {len(cpus)} 个逻辑核心 '
            f'（编号 {", ".join(map(str, cpus))}），计算线程上限 {_threads}')
