"""Test helper: a stage that run_all starts with subprocess.run([sys.executable, ...])."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import probe_mod
probe_mod.COUNTER += 1
with open(os.path.join(os.environ['AUDIO_TEST_OUTPUT_DIR'], 'nested.json'), 'w', encoding='utf-8') as f:
    json.dump({'pid': os.getpid(), 'cwd': os.getcwd(), 'argv': sys.argv[1:], 'probe_output': probe_mod.OUTPUT,
               'probe_counter': probe_mod.COUNTER, 'asr_provider': os.environ.get('AFD_ASR_PROVIDER')}, f)
print('[stub] nested stage done')
sys.exit(int(os.environ.get('FAKE_NESTED_EXIT', '0')))
