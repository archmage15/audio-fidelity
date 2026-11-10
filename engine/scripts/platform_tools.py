"""Platform-specific helpers kept separate from the audio scoring logic."""
import os
import shlex
import shutil
import subprocess


def espeak_candidates(platform_name=None):
    platform_name = platform_name or os.name
    candidates = []
    override = os.environ.get('ESPEAK_NG_BINARY')
    if override:
        candidates.append(override)
    if platform_name == 'nt':
        folder = os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs', 'eSpeak NG')
        executable = os.path.join(folder, 'espeak-ng.exe')
        if os.path.isfile(executable):
            os.environ['ESPEAK_DATA_PATH'] = folder
            candidates.append(executable)
    candidates.append(shutil.which('espeak-ng') or 'espeak-ng')
    if platform_name == 'nt':
        candidates.extend([r'C:\Program Files\eSpeak NG\espeak-ng.exe',
                           r'C:\Program Files (x86)\eSpeak NG\espeak-ng.exe'])
    return list(dict.fromkeys(candidates))


def open_text_editor(path):
    if os.name == 'nt':
        try:
            os.startfile(path)
            return True
        except OSError:
            try:
                subprocess.Popen(['notepad', path])
                return True
            except OSError:
                return False
    editor = os.environ.get('VISUAL') or os.environ.get('EDITOR')
    if editor:
        try:
            with open('/dev/tty', 'r+') as terminal:
                result = subprocess.run(shlex.split(editor) + [path], stdin=terminal,
                                        stdout=terminal, stderr=terminal)
            return result.returncode == 0
        except (OSError, ValueError):
            pass
    if (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')) and shutil.which('xdg-open'):
        try:
            subprocess.Popen(['xdg-open', path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except OSError:
            pass
    return False


def plot_font_families():
    from matplotlib import font_manager
    installed = {entry.name for entry in font_manager.fontManager.ttflist}
    preferred = ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'WenQuanYi Micro Hei',
                 'WenQuanYi Zen Hei', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    return [name for name in preferred if name in installed] or ['DejaVu Sans']
