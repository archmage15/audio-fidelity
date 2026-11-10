"""Prevent legacy Windows console selection from pausing a running job."""
import atexit
import os

_checked = False
_enabled = False

def prevent_console_pause():
    global _checked, _enabled
    if _checked:
        return _enabled
    _checked = True
    if os.name != "nt":
        return False
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD,
        wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
        wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetConsoleMode.restype = wintypes.BOOL
    kernel.SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.SetConsoleMode.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    # stdin is redirected to NUL by the batch launcher; open the console directly.
    handle = kernel.CreateFileW("CONIN$", 0xC0000000, 3, None, 3, 0, None)
    if handle in (None, ctypes.c_void_p(-1).value):
        return False
    original = wintypes.DWORD()
    if not kernel.GetConsoleMode(handle, ctypes.byref(original)):
        kernel.CloseHandle(handle)
        return False
    updated = (original.value | 0x0080) & ~0x0040
    if not kernel.SetConsoleMode(handle, updated):
        kernel.CloseHandle(handle)
        return False
    _enabled = True
    def restore():
        kernel.SetConsoleMode(handle, original.value)
        kernel.CloseHandle(handle)
    atexit.register(restore)
    return True
