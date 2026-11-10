"""Test stub replacing scripts/analyze_reference.py (no models, no Xunfei)."""
import json, os, sys, time
work = os.environ['AUDIO_TEST_WORK_DIR']
if os.environ.get('FAKE_ENGINE_FAIL') == 'prepare':
    print('[stub] simulated 讯飞 failure', file=sys.stderr)
    sys.exit(2)
time.sleep(float(os.environ.get('FAKE_ENGINE_SLEEP', '1')))
os.makedirs(os.path.join(work, '原音频'), exist_ok=True)
data = {'转写文本': 'This is a stub transcript for testing.',
        '参考音素': {'音素': ['ð', 'ɪ', 's']}, '原音频_CTC音素': ['ð', 'ɪ', 's']}
with open(os.path.join(work, '原音频/原音频缓存.json'), 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False)
with open(os.path.join(work, '原音频/原音频_logits.npy'), 'wb') as f:
    f.write(b'stub-logits')
print('[stub] reference analyzed')
