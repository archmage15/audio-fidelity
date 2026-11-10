"""Test helper: like engine modules, captures paths from the environment at import time."""
import os
OUTPUT = os.environ.get('AUDIO_TEST_OUTPUT_DIR')
COUNTER = 0
