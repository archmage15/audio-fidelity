import ast
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.stdout.reconfigure(encoding='utf-8')
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'scripts'))
import path_jobs as jobs
import audio_paths


class DynamicPaths(unittest.TestCase):
    def test_explicit_paths_override_legacy_names_and_empty_never_falls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / '随意范本 & A.wav'
            imitation = Path(directory) / '我的练习.WAV'
            reference.write_bytes(b'reference')
            imitation.write_bytes(b'imitation')
            with patch.dict(os.environ, {'AUDIO_TEST_REFERENCE': str(reference), 'AUDIO_TEST_IMITATION': str(imitation)}):
                self.assertEqual(audio_paths.find_audio_pair('not-the-input-directory'), (str(reference), str(imitation)))
                os.environ['AUDIO_TEST_IMITATION'] = ''
                self.assertIsNone(audio_paths.find_audio('模仿'))

    def test_same_name_size_and_timestamp_different_content_is_stale(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / 'same.wav'
            audio.write_bytes(b'AAAA')
            fingerprint = audio_paths._file_fingerprint(str(audio))
            cache = {'原音频指纹': fingerprint, '原音频SHA256': jobs.file_sha256(audio)}
            timestamp = audio.stat().st_mtime
            audio.write_bytes(b'BBBB')
            os.utime(audio, (timestamp, timestamp))
            self.assertTrue(audio_paths.reference_cache_stale(cache, str(audio)))

    def test_matching_content_can_reuse_after_rename(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / 'a.wav', Path(directory) / 'b.wav'
            a.write_bytes(b'same content')
            b.write_bytes(a.read_bytes())
            cache = {'原音频SHA256': jobs.file_sha256(a)}
            self.assertFalse(audio_paths.reference_cache_stale(cache, str(b)))
            self.assertEqual(jobs.cache_work(a), jobs.cache_work(b))

    def test_bad_inputs_are_rejected_before_running_models(self):
        with self.assertRaises(ValueError):
            jobs.audio_file('missing_input.wav')
        with tempfile.TemporaryDirectory() as directory:
            unsupported = Path(directory) / 'unsupported.m4a'
            unsupported.write_bytes(b'anything')
            with self.assertRaisesRegex(ValueError, '不支持'):
                jobs.audio_file(unsupported)

    def test_each_environment_replaces_inherited_job_state(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            with patch.dict(os.environ, {'AUDIO_TEST_REFERENCE': 'old reference', 'AUDIO_TEST_IMITATION': 'old imitation', 'AUDIO_TEST_OUTPUT_DIR': 'old output'}):
                a = jobs.runtime_env(base / 'A.wav', base / 'B.wav', base / 'work A', base / 'out A')
                b = jobs.runtime_env(base / 'C.wav', None, base / 'work B', base / 'out B')
                self.assertNotEqual(a['AUDIO_TEST_REFERENCE'], b['AUDIO_TEST_REFERENCE'])
                self.assertEqual(b['AUDIO_TEST_IMITATION'], '')
                self.assertNotEqual(a['AUDIO_TEST_OUTPUT_DIR'], b['AUDIO_TEST_OUTPUT_DIR'])
                self.assertEqual(os.environ['AUDIO_TEST_REFERENCE'], 'old reference')
                self.assertTrue((base / 'work A/原音频').is_dir())

    def test_missing_cache_never_triggers_analysis_during_evaluation(self):
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            reference = base / 'uncached.wav'
            sf.write(reference, np.zeros(1600, dtype=np.float32), 16000)
            with patch.object(jobs, 'import_legacy_cache', return_value=False), patch.object(jobs, 'run_stage') as stage:
                with self.assertRaisesRegex(ValueError, '尚未缓存'):
                    jobs.prepare_reference(reference, cache_dir=base / 'cache', analyze=False)
                stage.assert_not_called()

    def test_transcription_temporaries_do_not_use_input_folder(self):
        import numpy as np
        import soundfile as sf
        import xunfei_transcribe
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / 'original.wav'
            sf.write(source, np.zeros((4410, 2), dtype=np.float32), 44100)
            digest = jobs.file_sha256(source)
            first = Path(xunfei_transcribe.prep_audio(str(source)))
            second = Path(xunfei_transcribe.prep_audio(str(source)))
            try:
                self.assertNotEqual(first, second)
                self.assertNotEqual(first.parent, source.parent)
                self.assertEqual(jobs.file_sha256(source), digest)
                self.assertEqual([p.name for p in base.iterdir()], ['original.wav'])
            finally:
                first.unlink(missing_ok=True)
                second.unlink(missing_ok=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
