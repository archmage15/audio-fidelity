"""Test stub replacing scripts/run_all.py: writes realistic output files."""
import base64, json, os, sys, time
out = os.environ['AUDIO_TEST_OUTPUT_DIR']
ref, mine = os.path.abspath(os.environ['AUDIO_TEST_REFERENCE']), os.path.abspath(os.environ['AUDIO_TEST_IMITATION'])
if os.environ.get('FAKE_ENGINE_FAIL') == 'evaluate':
    sys.exit(3)
time.sleep(float(os.environ.get('FAKE_ENGINE_SLEEP', '2')))
# Exercise the engine-server mechanics: nested stage via subprocess + env-at-import module.
import subprocess
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
import probe_mod
probe_mod.COUNTER += 1
assert probe_mod.OUTPUT == out, f'stale engine module: {probe_mod.OUTPUT} != {out}'
_r = subprocess.run([sys.executable, os.path.join(_here, 'nested_stage.py'), 'arg1'],
                    cwd=os.path.dirname(_here), stdin=subprocess.DEVNULL)
if _r.returncode != 0:
    sys.exit(_r.returncode)
assert subprocess.run(['echo', 'external ok'], capture_output=True, text=True).stdout.strip() == 'external ok'
score = {'输入音频': {'原音频': ref, '模仿音频': mine}, '综合还原度': 74.1, '等级': '良好',
         '分项分': {'发音': 79.1, '音变': 87.2, '节奏': 61.8, '重音位置': None}}
with open(os.path.join(out, '还原度综合评分.json'), 'w', encoding='utf-8') as f:
    json.dump(score, f, ensure_ascii=False)
report = '\n'.join(['=' * 60, '   还原度综合报告', '=' * 60,
    f'原音频：{ref}', f'模仿：{mine}', f'缓存：{os.environ["AUDIO_TEST_WORK_DIR"]}/原音频/x.json',
    '⚠ 检测到「半截模仿」：你只跟读到 2 / 3 句（67%）。',
    '综合还原度：74.1 / 100     等级：良好', '-' * 60, '六、下一步怎么练', '-' * 60,
    '    · 节奏偏慢，注意句内连读。', '    · 停顿数量与原音频不一致（原音频 2 处 / 你 3 处），注意不要多停或少停。', ''])
with open(os.path.join(out, '还原度综合报告.txt'), 'w', encoding='utf-8') as f:
    f.write(report)
with open(os.path.join(out, '音素对比_报告.txt'), 'w', encoding='utf-8') as f:
    f.write('逐句明细（stub）\n路径 /root/secret/thing.wav\n')
structured_report = {
    'schema_version': 1,
    'focus_sounds': ['θ', 's'],
    'top_issues': [{'sentence': 2, 'word': 0, 'issue': 0}],
    'sentences': [{
        'index': 2, 'text': 'Thanks for the lesson.', 'missed': False,
        'reference_ms': [3100, 5200], 'student_ms': [2900, 5600],
        'words': [{
            'text': 'Thanks', 'ipa': 'θ æ ŋ k s', 'student_ipa': 's æ ŋ k s',
            'student_ms': [2900, 3350],
            'issues': [{'category': 'sound_change', 'expected': 'θ', 'heard': 's',
                        'what': 'θ 读成了 s', 'how': '路径 /root/secret/thing.wav'}],
        }],
        'hints': [{'category': 'prosody', 'text': '原音频在 lesson 后停了一下'}],
    }],
}
with open(os.path.join(out, 'report.json'), 'w', encoding='utf-8') as f:
    json.dump(structured_report, f, ensure_ascii=False, indent=2)
png = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==')
for name in ('节奏对比.png', '音高对比.png'):
    open(os.path.join(out, name), 'wb').write(png)
print('[stub] evaluation done')
