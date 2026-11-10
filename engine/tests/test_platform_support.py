import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import platform_tools
import path_jobs


class PlatformSupport(unittest.TestCase):
    def test_linux_espeak_has_no_windows_fallback(self):
        with patch.dict(os.environ, {'ESPEAK_NG_BINARY': '/custom tools/espeak-ng'}), patch('platform_tools.shutil.which', return_value='/usr/bin/espeak-ng'):
            candidates = platform_tools.espeak_candidates('posix')
            self.assertEqual(candidates, ['/custom tools/espeak-ng', '/usr/bin/espeak-ng'])
            self.assertFalse(any(path.endswith('.exe') for path in candidates))

    @unittest.skipIf(os.name == 'nt', 'POSIX editor branch')
    def test_headless_editor_does_not_try_notepad(self):
        with patch.dict(os.environ, {}, clear=True), patch('platform_tools.subprocess.Popen') as spawn:
            self.assertFalse(platform_tools.open_text_editor('/tmp/reference text.txt'))
            spawn.assert_not_called()

    def test_cache_with_hash_can_move_from_windows_to_linux(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / 'audio/原音频'
            legacy.mkdir(parents=True)
            reference = root / 'renamed reference.wav'
            reference.write_bytes(b'audio content used as a cache identity')
            digest = path_jobs.file_sha256(reference)
            data = {'原音频路径': r'Z:\old Windows machine\missing.wav',
                    '原音频SHA256': digest, '转写文本': 'a different reference',
                    '参考音素': {'音素': ['a']}, '原音频_CTC音素': ['a']}
            (legacy / '原音频缓存.json').write_text(json.dumps(data), encoding='utf-8')
            np.save(legacy / '原音频_logits.npy', np.zeros((2,392), dtype=np.float32))
            work = root / 'new cache'
            work.mkdir()
            with patch.object(path_jobs, 'ROOT', root):
                self.assertTrue(path_jobs.import_legacy_cache(work, reference, digest))
                self.assertTrue(path_jobs.cached_reference_valid(work, digest))

    @unittest.skipIf(os.name == 'nt', 'requires Linux Bash')
    def test_linux_dispatch_preserves_arguments_cwd_and_exit_status(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            env_path = temporary / 'fake environment'
            (env_path / 'bin').mkdir(parents=True)
            python = env_path / 'bin/python'
            python.write_text('''#!/usr/bin/env bash
if [[ "${1:-}" == -c ]]; then exit 0; fi
exec "$REAL_TEST_PYTHON" -c 'import json,os,sys;open(os.environ["ARGUMENT_LOG"],"w").write(json.dumps({"args":sys.argv[1:],"cwd":os.getcwd()}));sys.exit(int(os.environ.get("TEST_EXIT","0")))' "$@"
''', encoding='utf-8')
            python.chmod(0o755)
            log = temporary / 'arguments.json'
            env = os.environ.copy()
            env.update(AUDIO_TEST_VENV=str(env_path), REAL_TEST_PYTHON=sys.executable, ARGUMENT_LOG=str(log))
            arguments = ['原音频 & spaces.wav', '跟读 (draft).wav', '--output', 'relative output']
            result = subprocess.run(['bash', str(ROOT / 'linux.sh'), 'evaluate', *arguments],
                                    cwd=temporary, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            actual = json.loads(log.read_text())
            self.assertEqual(actual['args'][0], str(ROOT / 'evaluate.py'))
            self.assertEqual(actual['args'][1:], arguments)
            self.assertEqual(actual['cwd'], str(temporary))
            env['TEST_EXIT'] = '23'
            result = subprocess.run(['bash', str(ROOT / 'linux.sh'), 'cache', 'folder & spaces'],
                                    cwd=temporary, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 23)

    def test_shell_entry_has_linux_line_endings(self):
        source = (ROOT / 'linux.sh').read_bytes()
        self.assertTrue(source.startswith(b'#!/usr/bin/env bash\n'))
        self.assertNotIn(b'\r', source)


if __name__ == '__main__':
    unittest.main(verbosity=2)
