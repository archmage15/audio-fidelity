# -*- coding: utf-8 -*-
r"""还原度对比 · 第一块：节奏 / 重音

数据来源分两条线，各管各的：

  节奏（词块时间来自 音素对比.json.词块 —— 09-28 块 10 起去讯飞）
    - 词块配对：按【词块序号】一一对应（两侧按"任一侧连读即对称合并"，块数恒相等）
    - 词块时长：CTC 帧给的 span（音素起点到起点+20ms）；用【能量实证】复核，差得离谱时以实证为准
    - 停顿    ：块与块之间的 gap；【换人处】的 gap 标"预期停顿"，不计入处数（§5.5）
    - 整句净时长：首块 start -> 末块 end
    - ⚠️ 必须先跑 compare_phoneme.py（产出 音素对比.json），本脚本才有输入

  重音（音频信号，各侧自比；09-28 块 10 起按【角色】分组）
    - 每块算 响度峰值 / 音高均值，与【本侧同角色组】的均值比（不跨侧、不跨角色比绝对值，
      因为两段录音的电平本来就不一样，两人对话时全句统计量也不成立）
    - 超过 组内基准 × STRESS_X 即判为突出；响度或音高任一过线就算
    - 两侧的"突出块序号集合"对比 -> 漏 / 多 / 命中
    - 只报差异，不下"你错了"的结论（对错取决于说话意图，机器判不了）

用法：
  1. 把两个音频放进 audio\ 文件夹，命名成：原音频.* 和 模仿.*
     （扩展名随意：mp3 / wav / flac / ogg 都可以，m4a / aac 不行）
  2. 双击 scripts\run_rhythm.bat（或根目录的 一键跑还原度.bat）
  3. 屏幕出报告，结果写到 output\ 文件夹

目录约定（本脚本在 scripts\ 里，用 ROOT 指回上一层）：
  audio\   放录音（你手动放，只读）
  output\  放本次结果（自动生成，可随时清空）

依赖：soundfile / numpy / matplotlib（09-28 起不再依赖 requests）

注：语调（F0 音高走向）本次暂不出分，等判分标准重做后再接回来。
"""

from cpu_runtime import configure_cpu_runtime
configure_cpu_runtime()
import json
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---- 目录约定 ----
# 本脚本位于 <项目>\scripts\ 下，ROOT = 项目根（上一层）
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC_DIR = os.environ.get("AUDIO_TEST_WORK_DIR") or os.path.join(ROOT, "audio")
OUT_DIR = os.environ.get("AUDIO_TEST_OUTPUT_DIR") or os.path.join(ROOT, "output")
for _d in (SRC_DIR, OUT_DIR):
    os.makedirs(_d, exist_ok=True)

REF_FILE = None      # 运行时按格式探测（见 audio_paths），可能是 .mp3 / .wav / .flac
MY_FILE = None
OUT_PNG = os.path.join(OUT_DIR, "节奏对比.png")
OUT_PNG_PITCH = os.path.join(OUT_DIR, "音高对比.png")
OUT_JSON = os.path.join(OUT_DIR, "节奏对比.json")   # 给机器/综合评分器用
# 本脚本的唯一词块时间来源：compare_phoneme.py 的产出（09-28 块 10 起，去讯飞）
PHONEME_JSON = os.path.join(OUT_DIR, "音素对比.json")

SR = 16000
FRAME_MS = 25
HOP_MS = 10
VOICED_RATIO = 0.12      # 能量低于 全句 max*该比例 算静音（判"实测有声时长"用，供报告参考）

# ---- 节奏参数 ----
BOUNDARY_GAP_S = 0.12    # 词间空隙 >= 这个 且 落在句子边界 -> 算停顿
REL_FAST = 0.30          # 块时长相对差 >= +30% -> 要读快一些
REL_SLOW = -0.30         # 块时长相对差 <= -30% -> 要读慢一些


# ---- 日志（统一模块，带时间戳；详细模式见 logkit.py 顶部说明）----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logkit import log, out, log_setup, log_done, log_step, log_error, VERBOSE
from audio_paths import find_audio_pair, missing_hint, short_name

REF_FILE, MY_FILE = find_audio_pair(SRC_DIR)


# ---------------- 读音频 + 变单声道 + 重采样 ----------------
def load_mono(path):
    if not os.path.exists(path):
        return None
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SR:
        n_out = int(round(len(audio) * SR / sr))
        x_in = np.linspace(0, 1, len(audio), endpoint=False)
        x_out = np.linspace(0, 1, n_out, endpoint=False)
        audio = np.interp(x_out, x_in, audio)
    return audio.astype(np.float32)


# ---------------- 能量包络 ----------------
def frame_rms(y):
    fl = int(SR * FRAME_MS / 1000)
    hl = int(SR * HOP_MS / 1000)
    if len(y) < fl:
        return np.zeros(1)
    n = 1 + (len(y) - fl) // hl
    idx = np.arange(fl)[None, :] + hl * np.arange(n)[:, None]
    return np.sqrt((y[idx] ** 2).mean(axis=1))


# ---------------- 音高 F0（pyin，概率 YIN；librosa 不可用时回退自相关） ----------------
def pitch_contour(y, fmin=70, fmax=400):
    """F0 序列（hop=HOP_MS，与 frame_rms 帧数严格对齐）；unvoiced=0.0。

    2026-11-02 从自相关法换成 librosa.pyin（概率 YIN，现代标准）：
      - 自相关对噪声/清浊音分界/低能量段都不稳，F0 测不准会传导到重音判定。
      - pyin 用贝叶斯平滑（Viterbi 转移概率），对清浊切换更鲁棒。
      - 参数刻意对齐 frame_rms：hop_length=160(=SR*HOP_MS/1000)、center=False、
        frame_length=512（够 fmin=70Hz 两个周期，避免低频警告）。帧数 = frame_rms 帧数。
    librosa 未安装（如托管 Python 跑 selftest 之外的场景）→ 回退旧自相关法。
    """
    try:
        import librosa
        f0 = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=SR,
                          frame_length=512, hop_length=int(SR * HOP_MS / 1000),
                          center=False)[0]
        return np.nan_to_num(f0, nan=0.0).astype(np.float32)
    except Exception:
        pass  # librosa 不可用 → 回退自相关
    fl = int(SR * 0.04)
    hl = int(SR * HOP_MS / 1000)
    lag_min = int(SR / fmax)
    lag_max = int(SR / fmin)
    f0 = []
    for i in range(0, max(1, len(y) - fl), hl):
        fr = y[i:i + fl]
        if len(fr) < fl:
            break
        fr = fr - fr.mean()
        corr = np.correlate(fr, fr, mode="full")[len(fr) - 1:]
        if corr[0] <= 0 or lag_max >= len(corr):
            f0.append(0.0)
            continue
        seg = corr[lag_min:lag_max]
        if len(seg) == 0:
            f0.append(0.0)
            continue
        peak = int(np.argmax(seg)) + lag_min
        if corr[peak] / corr[0] < 0.3:
            f0.append(0.0)
        else:
            f0.append(SR / peak)
    return np.array(f0)


# ================= 词块读入（来自 音素对比.json.词块，09-28 块 10 起） =================
def load_blocks():
    """读 compare_phoneme.py 产出的 词块，两侧对称合并连读区，返回
    (blocks_r, blocks_m, words_r, words_m)。读不到 → 抛异常（main 里降级退出）。

    ⚠️ 合并规则（排期 §5.5 / 计划书阶段 4）：
      音素对比.json.词块 是**逐词**条目（带 连读 标记 + 角色）。节奏配对要的是块。
      - 某词在**任一侧**被标 连读=True → 这个词段在**两侧**都并入连读块。
        为什么取并集：两侧块数必须相等（序号对齐是配对的前提）；
        而且连读块里的短词（弱读功能词，如 oh/im/are，CTC 帧下往往只占 1~3 帧、
        几十毫秒）单独成块时是纯测量噪声，并进大块才能把噪声摊掉。
      - 实测本例：原音频有 Where+are / oh+im / at+the，我的有 still+at
        → 并集后 13 词 → 9 块/侧（still+at+the 并成一块）。
    """
    if not os.path.exists(PHONEME_JSON):
        raise RuntimeError("缺少 " + PHONEME_JSON + " —— 请先跑 compare_phoneme.py"
                           "（或直接双击 一键跑还原度.bat）")
    with open(PHONEME_JSON, encoding="utf-8") as f:
        data = json.load(f)
    word_lists = (data.get("词块") or {}).get("原音频"), (data.get("词块") or {}).get("我的")
    if not word_lists[0] or not word_lists[1]:
        raise RuntimeError("音素对比.json 里没有 词块（或为空）—— 请重跑 compare_phoneme.py")

    # 词表输出用：逐词条目原样保留（带时间）
    words_r = [dict(w) for w in word_lists[0]]
    words_m = [dict(w) for w in word_lists[1]]
    n = min(len(words_r), len(words_m))

    # 连读区合并：两侧各自的 zone（连续 连读=True 段）取**区间并集**，
    # **只重叠才并、相邻不并**（跨侧共享词下标才算同一块）。
    #   例：原音频有 at+the[10,11]、我的有 still+at[9,10] —— 共享词 10 → 并成 [9,11]；
    #   但 oh+im[7,8] 和 still+at+the[9,11] 相邻（8 和 9）不重叠 → 保持两块。
    #   ⚠️ 旧实现用"逐词标记取并集"会把相邻的也粘成一块（曾把 oh+im+still+at+the
    #   错误合成 5 词一块），已修。
    def collect_zones(words):
        zs, i = [], 0
        while i < n:
            if words[i].get("连读"):
                j = i
                while j + 1 < n and words[j + 1].get("连读"):
                    j += 1
                if j > i:
                    zs.append((i, j))
                i = j + 1
            else:
                i += 1
        return zs

    raw_zones = sorted(collect_zones(words_r) + collect_zones(words_m))
    zones = []          # 合并后的 [(a, b)]，闭区间
    for a, b in raw_zones:
        if zones and a <= zones[-1][1]:    # 与上一区间重叠（共享词下标）
            zones[-1] = (zones[-1][0], max(zones[-1][1], b))
        else:
            zones.append((a, b))

    zone_start = {a: (a, b) for a, b in zones}

    def to_blocks(words):
        blocks = []
        i = 0
        while i < n:
            if i in zone_start:
                a, b = zone_start[i]
                grp = words[a:b + 1]
                blocks.append({"词": [w["词"] for w in grp],
                               "start_ms": grp[0]["start_ms"],
                               "end_ms": grp[-1]["end_ms"],
                               "连读": True,
                               "角色": grp[0].get("角色")})
                i = b + 1
            else:
                w = words[i]
                blocks.append({"词": [w["词"]],
                               "start_ms": w["start_ms"],
                               "end_ms": w["end_ms"],
                               "连读": False,
                               "角色": w.get("角色")})
                i += 1
        return blocks

    return to_blocks(words_r), to_blocks(words_m), words_r, words_m


def block_span(block):
    return (block["end_ms"] - block["start_ms"]) / 1000.0


# ================= 停顿（块间 gap；换人处单独标） =================
def find_boundary_pauses(blocks):
    """块与块之间的空隙 >= 阈值处记为停顿候选。

    09-28 块 10 两处改动（计划书阶段 4 + 排期 §5.5 连带坑）：
    1. 输入从「讯飞逐词」换成「合并后的词块」—— gap 是块 span 之间的真实空隙。
    2. **换人处**（相邻块角色不同，如 Terry(A) -> oh+im(B)）的 gap 标
       「预期停顿」并**不计入处数** —— 两人对话轮流说，那里本来就该停，
       拿它扣"多余停顿"是误判（§5.5 连带坑，本例实测原音频换人 gap 880ms、
       我的 680ms，旧口径下会全部被记成停顿错误）。
    """
    result = []
    for i in range(len(blocks) - 1):
        gap = (blocks[i + 1]["start_ms"] - blocks[i]["end_ms"]) / 1000.0
        if gap < BOUNDARY_GAP_S:
            continue
        role_now = blocks[i].get("角色")
        role_next = blocks[i + 1].get("角色")
        is_turn = bool(role_now and role_next and role_now != role_next)
        result.append({"after_word": "+".join(blocks[i]["词"]),
                       "after_index": i + 1,
                       "gap": round(gap, 3),
                       "换人处": is_turn})
    return result, [p for p in result if not p["换人处"]]


# ================= 时长复核（能量实证） =================
def measure_voiced(env, block):
    """块区间内"实际有声"的时长（能量 > 全句 max*比例 的帧数 × hop）。"""
    hop_s = HOP_MS / 1000.0
    if len(env) == 0 or env.max() <= 0:
        return 0.0
    thr = VOICED_RATIO * env.max()
    i0 = max(0, int(block["start_ms"] / 1000.0 / hop_s))
    i1 = min(len(env), max(i0 + 1, int(block["end_ms"] / 1000.0 / hop_s) + 1))
    return float(np.sum(env[i0:i1] > thr)) * hop_s


def resolve_duration(block, env):
    """块时长：直接取 CTC span（音素起点拼的，偏紧但可信）。

    09-28 去讯飞后词块时间来自 CTC 帧。旧版这里有一道"能量实证复核"——
    讯飞时代讯飞词边界会"虚高"（把静音算进词时长），需用实测有声时长修正。
    但 CTC span 本身偏紧、不含 stop-closure 静音，可信；若再拿"实测有声"去
    替代，反而会把爆破音成阻段（t/p/k 的闭嘴段）当静音剥掉、低估词时长。
    所以这道复核已废弃，直接信 span（voiced 仍算出来供报告/调试参考）。"""
    span = block_span(block)
    voiced = measure_voiced(env, block)
    return span, "CTC帧", span, voiced


# ================= 重音（邻域法，计划书 §2.1；09-28 块 11 起替掉旧均值法） =================
def block_signal_features(env, f0, blocks):
    """每块算 响度峰值 / 音高均值 / 时长（音高检测不到浊音时为 None）。"""
    hop_s = HOP_MS / 1000.0
    feats = []
    for b in blocks:
        i0 = max(0, int(b["start_ms"] / 1000.0 / hop_s))
        i1 = min(len(env), max(i0 + 1, int(b["end_ms"] / 1000.0 / hop_s) + 1))
        seg_env = env[i0:i1]
        loud = float(seg_env.max()) if len(seg_env) else 0.0

        nf0 = min(len(f0), len(env))
        seg_f0 = f0[i0:min(i1, nf0)]
        voiced = seg_f0[seg_f0 > 0] if len(seg_f0) else np.array([])
        pitch = float(voiced.mean()) if len(voiced) >= 2 else None

        feats.append({"响度": loud, "音高": pitch,
                      "时长": (b["end_ms"] - b["start_ms"]) / 1000.0})
    return feats


# ---------------- 重音明细渲染（先摊开依据，再下纠错结论） ----------------
# ⚠️ 顺序是刻意的：**先列全两侧每一块的判定依据**（含没突出的），
#    读者才能自己看出「为什么这个突出、那个没突出」，再看漏/多结论。
#    以前只打印「哪几块突出」，依据全在 json 里，等于要求读者自己去翻机器数据。
def _pad(text, width):
    """按显示宽度左对齐补空格（中文算 2 格）。"""
    w = sum(2 if ord(c) > 0x2E80 else 1 for c in text)
    return text + " " * max(0, width - w)


def render_stress_detail(detail, label, miss=(), extra=()):
    """把一侧的全部重音判定明细渲染成文本行（返回 list[str]）。

    每个块一行，格式：
        #序号词  角色  三路「归一值(落差σ)」  判定 + 命中路
      ★ = 该侧判定为突出，○ = 没突出，M = 原音频有我漏了，X = 我多出来的

    miss / extra 传入的是**对侧**的差集序号，只影响本侧标记，不改判定。
    """
    if not detail:
        return ["   （该侧没有词块）"]
    miss, extra = set(miss), set(extra)

    def cue_cell(d, c):
        """一格：归一值 + (落差σ)。缺该路数据 → '-'。"""
        v = (d.get("归一值") or {}).get(c)
        z = (d.get("落差σ") or {}).get(c)
        if v is None:
            return "-"
        return "%.2f(%+.2f)" % (v, z if z is not None else 0.0)

    def word_of(d):
        """detail 里的「词」可能是 '+'.join(...) 字符串，也可能是原始 list（json 回读/测试）。"""
        w = d.get("词")
        if isinstance(w, (list, tuple)):
            w = "+".join(str(x) for x in w)
        return w or "?"

    def dwidth(text):
        return sum(2 if ord(c) > 0x2E80 else 1 for c in text)

    w_no = max([len(str(d["序号"])) for d in detail] + [2])
    w_word = max(dwidth(word_of(d)) for d in detail)
    w_role = max(dwidth(d.get("角色") or "-") for d in detail)

    lines = ["   【%s】共 %d 块，其中判定突出 %d 块"
             % (label, len(detail), sum(1 for d in detail if d.get("突出")))]
    head = "   %s %s  %s  %s  %s  %s" % (
        _pad("#", w_no + 1), _pad("词", w_word), _pad("角色", w_role),
        _pad("音高 半音(z)", 15), _pad("响度 倍(z)", 15), _pad("时长 倍(z)", 15))
    lines.append(head)
    lines.append("   " + "-" * (len(head) - 3))
    for d in detail:
        no = d["序号"]
        if no in miss:
            flag = "M"
        elif no in extra:
            flag = "X"
        else:
            flag = "★" if d.get("突出") else "○"
        hits = "/".join(d.get("命中路") or []) or "-"
        lines.append("   %s %s  %s  %s  %s  %s  %s %s"
                     % (_pad(str(no), w_no + 1), _pad(word_of(d), w_word),
                        _pad(d.get("角色") or "-", w_role),
                        _pad(cue_cell(d, "音高"), 15),
                        _pad(cue_cell(d, "响度"), 15),
                        _pad(cue_cell(d, "时长"), 15),
                        flag, hits))
    lines.append("   图例：★ 该侧突出  ○ 不突出  M 原音频有、你漏了  X 你多出来的"
                 "   （z = 相对同角色邻居中位数的落差 ÷ σ）")
    # 「-」是「该路没数据」，不是 0 —— 音高需浊音才测得到。统计一下免得被当成 bug。
    miss_pitch = sum(1 for d in detail
                     if (d.get("归一值") or {}).get("音高") is None)
    if miss_pitch:
        lines.append("   注：%d/%d 块音高显示为 -（该处测不到浊音，非 0 值），"
                     "这几块只能靠响度/时长两路判定。"
                     % (miss_pitch, len(detail)))
    return lines


def render_stress_verdict(nr, nm, marked_r, marked_m):
    """纠错结论（依赖上面的明细才看得懂）。返回 list[str]。"""
    miss = sorted(set(marked_r) - set(marked_m))
    extra = sorted(set(marked_m) - set(marked_r))
    hit = sorted(set(marked_r) & set(marked_m))

    def name(seq, table):
        return ("、".join("#%d %s" % (i, table.get(i, "?")) for i in seq) or "无")

    return [
        "   → 漏（原音频有、我没有）：%s" % name(miss, nr),
        "   → 多（我有、原音频没有）：%s" % name(extra, nm),
        "   → 命中：%s" % name(hit, nr),
        "   （重音位置分只罚漏不罚多：多读不算错，但漏读要扣）",
    ]


# ---------------- 重点词清单（算法直接给结论，不让用户自己从表里找） ----------------
# 大法师 10-21：光给「哪几块漏了」不够 —— 用户要的是**要重读哪个词、怎么改**。
# 下面这个函数把两侧的归一值/落差σ换算成可执行的说法。
#
# ⚠️⚠️ **不给假精确的数字**（10-21 返工踩过）：
#   第一版直接输出「再拖长 298%」「往下压 11.3 个半音」—— 这些是相对**同角色邻居中位**
#   的比值差，邻居组随素材变，298% 没有可执行意义（用户没法"拖长 3 倍"），
#   11.3 个半音也超出音域常识。**精确的假数字比不给更糟**。
#   改成：方向（往上/往下/更响/更轻/拖长/缩短）+ 幅度三档（轻微/明显/很大）。
#   依据仍是已算好的量（归一值差 + 落差σ），只是表达成人能执行的粒度。
#
# 幅度分档（按归一值差，音高用半音、响度/时长用百分比）：
#   轻微 < 1 半音 / <15%   明显 1~3 半音 / 15~50%   很大 > 3 半音 / > 50%
def _magnitude(cue, gap):
    """把归一值差换算成三档幅度标签。"""
    if cue == "音高":
        a = abs(gap)
        if a < 1.0:
            return "轻微"
        return "明显" if a < 3.0 else "很大"
    a = abs(gap) * 100.0
    if a < 15:
        return "轻微"
    return "明显" if a < 50 else "很大"


def _too_extreme(cue, gap):
    """差值大到不像「读法差异」，更像测量异常（块边界/静音/吞音）→ 不给方向建议。

    实测踩过：某块时长比值差 298% —— 那是「原音频这块 1.2 倍、我的 0.3 倍」，
    也就是我这块几乎没读出时长（吞音或边界切错），不是「要拖长 3 倍」。
    对着这种数给建议是误导。
    """
    if cue == "音高":
        return abs(gap) > 12.0        # 12 个半音 = 一个八度，正常语流到不了
    return abs(gap) >= 3.0           # 3 倍差 = 差一个数量级，多半是异常不是读法


def _cue_advice(cue, ref_val, my_val):
    """一路信号的改法建议（方向 + 幅度档位）。gap = 原侧领先量。"""
    if ref_val is None or my_val is None:
        return "（该路测不到，无法量化）"
    gap = ref_val - my_val
    if cue == "音高":
        if abs(gap) < 0.5:
            return "音高已接近"
        # ⚠️ 音高路是**双向**的（升调/降调都算 prominence，因为英语重音可由降调实现）。
        #    所以 gap<0 有两种含义：① 原声靠降调实现重读、你要学它降下来；
        #    ② 原声抬高了、你没抬。分不清是哪种（n=1 未标定），
        #    **不能**武断说「压低」—— 措辞改成中性并提示两种可能。
        if gap > 0:
            if _too_extreme(cue, gap):
                return "音高差得反常（多半是这块没读出来/切边错误，不是读法问题）"
            return "音高往上挑（%s）" % _magnitude(cue, gap)
        if _too_extreme(cue, gap):
            return "音高差得反常（多半是这块没读出来/切边错误，不是读法问题）"
        return "音高和原声反了（%s）· 原声可能是降调重读，你听着原声往哪边走就往哪边走" \
            % _magnitude(cue, gap)
    if cue == "响度":
        if abs(gap) * 100 < 5:
            return "响度已接近"
        if _too_extreme(cue, gap):
            return "响度差得反常（多半是这块没读出来/切边错误，不是读法问题）"
        return ("再读响些（%s）" % _magnitude(cue, gap)) if gap > 0 \
            else ("读轻些（%s）" % _magnitude(cue, gap))
    if cue == "时长":
        if abs(gap) * 100 < 8:
            return "时长已接近"
        if _too_extreme(cue, gap):
            return "时长差得反常（多半是这块没读出来/切边错误，不是读法问题）"
        return ("拖长些（%s）" % _magnitude(cue, gap)) if gap > 0 \
            else ("缩短些（%s）" % _magnitude(cue, gap))
    return cue


def render_focus_words(detail_r, detail_m, nr, nm):
    """输出「重点词清单」：原音频突出 + 你漏掉的词，逐个说清怎么改。

    返回 list[str]。这是给人**照着念**的清单，不是给机器的。
    """
    by_no_r = {d["序号"]: d for d in detail_r}
    by_no_m = {d["序号"]: d for d in detail_m}
    marked_r = {d["序号"] for d in detail_r if d.get("突出")}
    marked_m = {d["序号"] for d in detail_m if d.get("突出")}
    miss = sorted(marked_r - marked_m)

    lines = []
    if not miss:
        lines.append("   ✅ 原音频的每个重点词你都重读了 —— 没有漏读，不用改。")
        return lines

    lines.append("   重点词清单（原音频重读、你没重读 → 照这个单子练）：")
    for no in miss:
        dr = by_no_r.get(no)
        dm = by_no_m.get(no)
        word = (dr or {}).get("词") or nm.get(no) or nr.get(no) or "?"
        cues = (dr or {}).get("命中路") or []
        tips = []
        for cue in ("音高", "响度", "时长"):
            if cue not in cues:
                continue
            rv = ((dr or {}).get("归一值") or {}).get(cue)
            mv = ((dm or {}).get("归一值") or {}).get(cue)
            tips.append(_cue_advice(cue, rv, mv))
        lines.append("   · %-16s → %s" % (word, "；".join(tips) if tips else "（无量化建议）"))
    lines.append("   怎么用：先只练上面这几个词。重音 = 把这个字读得更突出"
                 "（挑高 / 读响 / 拖长），别整句一起使劲。")
    lines.append("   ⚠ 幅度是相对同句其他词估的，不是精确目标 —— "
                 "照着「更突出」的方向练，不必追求具体数值。")
    return lines


def render_extra_words(detail_r, detail_m, nr, nm):
    """输出「你多出来的重音」：原音频没重读、你重读了 —— 不扣分，但可以听听是不是读过头。"""
    marked_r = {d["序号"] for d in detail_r if d.get("突出")}
    marked_m = {d["序号"] for d in detail_m if d.get("突出")}
    extra = sorted(marked_m - marked_r)
    if not extra:
        return []
    words = "、".join(str((d.get("词") or nm.get(d["序号"]) or "?"))
                     for d in detail_m if d["序号"] in extra)
    return ["",
            "   （附）你多出来的重音（不扣分）：%s" % words,
            "        原音频这里是弱读的，别读得太重 —— 听起来像在抢话。"]


# ---------------- 节奏分 ----------------
# ---- 打分参数（09-28 新增）----
# 逐块相对差 → 节奏分的汇总方式。
#   power = 1.0 → 算术平均（旧行为）—— 对「某一处崩了」**不敏感**：
#                 10 块里坏 2 块，会被其余 8 块摊平。
#   power = 3.0 → 幂平均，对大的偏差加权更重，比 max 抗噪、比 mean 敏感。
# ⚠️ 这个 3.0 **尚未标定**（目前只有 1 句 / 13 词的样本）。换样本后需重标。
AGG_POWER = 3.0


def aggregate_penalty(diffs, power=AGG_POWER):
    """把逐块偏差汇总成一个惩罚量（0~1）。幂平均：p=1 退化为算术平均。"""
    x = np.clip(np.abs(np.asarray(diffs, dtype=float)), 0.0, 1.0)
    if len(x) == 0:
        return 0.0
    return float((np.mean(x ** power)) ** (1.0 / power))


def score_blocks(pairs):
    """按块算相对差，给每块一个判定；再汇总成节奏分。

    节奏分 = (1 − 汇总惩罚) × 100，最低 0 分。汇总用幂平均（见 AGG_POWER）。

    ⚠️ 09-28 两处改动（原因见计划书 §4.5.4 / §4.5.6）：

    1. **打分先归一化**。每块时长先除以「本侧块时长总和」，再算相对差。
       为什么必须这样：旧版拿**绝对**块时长比 `(我−原)/原`，若对方只是**整体**
       说慢了 17.6%，则**每一块**都是 +17.6% → 被判成"每块都读错"。
       计划书风险 20 曾断言「分母同步变化 → 对均匀变速天然免疫」——
       **该推理有误**：分母是"该块自己的基准时长"，不随变体变化，整体缩放消不掉。
       （实测：S1 只是慢 15%，旧版却给 88.8；设计意图应为 ≈100。）

    2. **汇总改幂平均**（旧为算术平均）。算术平均结构上表达不出「某一处崩了」：
       S3 的 10 块里只有 2 块坏（−38% / −47%），被其余 8 块 0% 一平摊，
       扣分只剩 11% → 88.9 分，比"只是慢了"的 S1 还高。

    ⚠️ 逐块提示（items 里的 `相对差` / `判定`）仍按**绝对**块时长算 ——
       它回答的是"这块比基准快/慢多少"，与"整体语速"是两件事，
       由 `整句净时长` 字段单独报告。**只有打分走归一化。**
    """
    items = []
    norms = []
    total_r = sum(p["原音频时长"] for p in pairs if p["原音频时长"] > 0)
    total_m = sum(p["我的时长"] for p in pairs if p["我的时长"] > 0)
    for p in pairs:
        dr, dm = p["原音频时长"], p["我的时长"]
        if dr <= 0:
            continue
        rel = (dm - dr) / dr
        if rel >= REL_FAST:
            verdict = "要读快一些"
        elif rel <= REL_SLOW:
            verdict = "要读慢一些"
        else:
            verdict = "差不多"
        items.append({"序号": p["序号"], "块": p["块"], "连读": p["连读"],
                      "原音频时长": round(dr, 3), "我的时长": round(dm, 3),
                      "相对差": round(rel, 4), "判定": verdict})
        # 归一化相对差：占本侧块时长总和的比例，再比
        if total_r > 0 and total_m > 0:
            norms.append((dm / total_m - dr / total_r) / (dr / total_r))
        else:
            norms.append(rel)
    if not items:
        return items, 0.0
    penalty = aggregate_penalty(norms)
    return items, max(0.0, (1 - penalty) * 100)


# ---------------- 画图 ----------------
def plot_blocks(items, title="节奏对比（按词块序号）"):
    """逐块时长对比柱状图。横轴标签自适应：
      - 块数 <= LABEL_ALL_MAX（30）：每块都标「序号.词」，转 30°（原行为，单句模式）
      - 块数更多（长音频）：图幅拉宽、**每块都标「序号.词」**但转 90°、字调小。
        90° 旋转把横向拥挤转成纵向堆叠（每标签横向只占一字宽 ~8px，
        75 块在 18 英寸图上每人 28px，不会碰）；并对「相对差最大的 5 个块」
        挂判定提示（原来给所有 ≥30% 的块都挂，横排必叠）
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from platform_tools import plot_font_families
    plt.rcParams["font.sans-serif"] = plot_font_families()
    plt.rcParams["axes.unicode_minus"] = False

    n = len(items)
    x = np.arange(n)
    dr = [it["原音频时长"] for it in items]
    dm = [it["我的时长"] for it in items]
    width = 0.38

    LABEL_ALL_MAX = 30
    dense = n > LABEL_ALL_MAX
    if dense:
        fig, ax = plt.subplots(figsize=(18, 7))
        rot, fs = 90, 7
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        rot, fs = 30, 9

    ax.bar(x - width / 2, dr, width, label="原音频", color="#c0392b")
    ax.bar(x + width / 2, dm, width, label="我的模仿", color="#2471a3")
    ax.set_xticks([int(i) for i in x])
    ax.set_xticklabels(["%d.%s" % (it["序号"], it["块"]) for it in items],
                       rotation=rot, ha="right", fontsize=fs)
    ax.set_ylabel("块时长 (秒)")
    ax.set_title(title + "   ·   横轴是词块序号，不是绝对时间"
                 + ("（块多，标签竖排；提示只标差异最大的5块）" if dense else ""))
    ax.grid(alpha=0.25, axis="y")
    ax.legend()
    # 标出差异大的块
    if not dense:
        for i, it in enumerate(items):
            if abs(it["相对差"]) >= REL_FAST:
                ax.annotate(it["判定"], (i, max(dr[i], dm[i])),
                            textcoords="offset points", xytext=(0, 6),
                            ha="center", fontsize=8, color="#7d3c98")
    else:
        # 长音频：只标「相对差最大」的 5 个块，词+判定都带上，避免叠字
        worst = sorted(range(n), key=lambda i: -abs(items[i]["相对差"]))[:5]
        for i in worst:
            ax.annotate("%s %s" % (items[i]["块"], items[i]["判定"]),
                        (i, max(dr[i], dm[i])),
                        textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=7, color="#7d3c98", rotation=90)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    out("图已保存：" + OUT_PNG)


def plot_pitch(semi_r, semi_m):
    """画音高走向对比图（原音频 vs 我的，半音曲线叠加，看语调升降）。
    半音相对本侧浊音中位 F0；unvoiced 帧为 None（跳过不画，曲线自然断开）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from platform_tools import plot_font_families
    plt.rcParams["font.sans-serif"] = plot_font_families()
    plt.rcParams["axes.unicode_minus"] = False

    hop_s = HOP_MS / 1000.0
    t_r = [i * hop_s for i, v in enumerate(semi_r) if v is not None]
    v_r = [v for v in semi_r if v is not None]
    t_m = [i * hop_s for i, v in enumerate(semi_m) if v is not None]
    v_m = [v for v in semi_m if v is not None]

    fig, ax = plt.subplots(figsize=(12, 4))
    if t_r:
        ax.plot(t_r, v_r, label="原音频", color="#c0392b", linewidth=1.2)
    if t_m:
        ax.plot(t_m, v_m, label="我的模仿", color="#2471a3", linewidth=1.2)
    ax.set_xlabel("时间 (秒)")
    ax.set_ylabel("音高（半音，相对本侧中位）")
    ax.set_title("音高走向对比（语调可视化参考，不出分）")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_PNG_PITCH, dpi=120)
    out("图已保存：" + OUT_PNG_PITCH)


def pitch_to_semitone(f0):
    """F0(Hz) 序列 → 半音序列（相对该侧浊音中位 F0，消性别/个体基频）；unvoiced→None。
    计划书阶段 5：语调只作可视化，不出分。"""
    voiced = f0[f0 > 0]
    if len(voiced) < 1:
        return [None] * len(f0)
    med = float(np.median(voiced))
    if med <= 0:
        return [None] * len(f0)
    return [None if v <= 0 else round(12.0 * float(np.log2(v / med)), 2) for v in f0]


def rms_series(env):
    """RMS 能量序列（保留绝对量级，不归一化）—— 两侧电平本不同，仅作录音质量参考。"""
    return [round(float(v), 5) for v in env]


def write_json(net_r, net_m, pauses_r, pauses_m, items, rhythm_score,
               stress_r, stress_m, blocks_r, blocks_m, words_r, words_m,
               detail_r, detail_m, semi_r, semi_m, env_r, env_m):
    """写 节奏对比.json —— 给机器/综合评分器读。
    注：字段「节奏分」保持不变（overall_score.py 按这个名字读）；
       「语调分」本次不产出（语调整理中），overall_score.py 会自动跳过缺项。
    （09-28 去讯飞后，顶层键改为「节奏」，不再带「讯飞」字样；内部「数据源」字段说明真实来源。）

    ⚠️ 本函数只被 compare_rhythm.py 的 main() 调用，而 main 已被 run_all 弃用
    （10-28 起统一走 compare_long）。重音判定已改 emphases，故「重音_信号」字段
    不再产出（stress_r/stress_m/detail_r/detail_m 参数保留仅为兼容旧调用）。
    """
    nr = {i + 1: "+".join(b["词"]) for i, b in enumerate(blocks_r)}
    nm = {i + 1: "+".join(b["词"]) for i, b in enumerate(blocks_m)}

    # 停顿：换人处的 gap 不算错误（§5.5 预期停顿），处数只数非换人的
    def pause_part(pauses):
        turns = [p for p in pauses if p.get("换人处")]
        normal = [p for p in pauses if not p.get("换人处")]
        return normal, turns

    normal_r, turns_r = pause_part(pauses_r)
    normal_m, turns_m = pause_part(pauses_m)

    data = {
        "节奏": {
            "数据源": "音素对比.json.词块（CTC 帧，09-28 块 10 起去讯飞）",
            "整句净时长": {"原音频": round(net_r, 3), "我的": round(net_m, 3),
                           "相对差": round((net_m - net_r) / net_r, 4) if net_r > 0 else 0},
            "词块数": {"原音频": len(blocks_r), "我的": len(blocks_m)},
            "序号对齐风险": (len(blocks_r) != len(blocks_m)),
            "句子边界停顿": {
                "原音频处数": len(normal_r), "我的处数": len(normal_m),
                "原音频": normal_r, "我的": normal_m,
                "换人处_预期停顿": {"原音频": turns_r, "我的": turns_m},
                "说明": "换人处（角色切换）的停顿是预期的，不计入处数（排期 §5.5）",
            },
            "词块时长": items,
        },
        "词表": {"原音频": words_r, "我的": words_m},
        "节奏分": round(rhythm_score, 1),
        "语调分": None,          # 语调整理中，暂不出分
        # ---- 阶段 5 可视化字段（不参与任何分，仅供 UI/报告参考）----
        "音高曲线": {
            "单位": "半音（相对本侧浊音中位 F0，0=中位）",
            "原音频": semi_r,
            "我的": semi_m,
            "说明": "unvoiced 帧为 null；两侧基频不同，直接比绝对半音无意义，看走向",
        },
        "绝对响度": {
            "单位": "RMS 振幅（0~1，未归一化）",
            "原音频": rms_series(env_r),
            "我的": rms_series(env_m),
            "原音频均值": round(float(np.mean(env_r)), 5) if len(env_r) else None,
            "我的均值": round(float(np.mean(env_m)), 5) if len(env_m) else None,
            "说明": "两侧录音电平本不同（设备/距离），仅作录音质量参考，不参与任何分",
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    log_setup("节奏/重音")
    out("=" * 62)
    out("   还原度对比 · 节奏 / 重音")
    out("=" * 62)

    ref = load_mono(REF_FILE)
    mine = load_mono(MY_FILE)
    if ref is None or mine is None:
        out("")
        out("X 没找到音频文件：")
        out(missing_hint([("原音频", REF_FILE), ("模仿", MY_FILE)], SRC_DIR))
        log_done("（音频缺失，提前退出）")
        return

    hop_s = HOP_MS / 1000.0
    log_step("读入音频")
    log("   原音频 %.2fs ｜ 我的模仿 %.2fs" % (len(ref) / SR, len(mine) / SR),
        verbose=True)

    log_step("算能量包络 / 音高曲线")
    env_r, env_m = frame_rms(ref), frame_rms(mine)
    f0_r, f0_m = pitch_contour(ref), pitch_contour(mine)

    # ---- 读词块（来自 音素对比.json，09-28 块 10 起去讯飞）----
    log_step("读词块（音素对比.json.词块）")
    try:
        blocks_r, blocks_m, words_r, words_m = load_blocks()
        log("   原音频 %d 块 ｜ 我的 %d 块（合并连读后）"
            % (len(blocks_r), len(blocks_m)), verbose=True)
    except Exception as e:
        out("")
        out("X 读词块失败，本次节奏对比无法进行：")
        out("    " + str(e))
        out("  （先双击 一键跑还原度.bat 让 compare_phoneme.py 跑出 音素对比.json）")
        log_error(str(e))
        try:
            write_json(0, 0, [], [], [], 0.0, [], [], [], [], [], [], [], [])
        except Exception:
            pass
        log_done("（缺词块输入，提前退出）")
        return

    # ---- 词块配对（按序号，两侧块数恒相等）----
    log_step("词块配对（按序号）")
    pairs = []
    n = min(len(blocks_r), len(blocks_m))
    for i in range(n):
        br, bm = blocks_r[i], blocks_m[i]
        dr, src_r, _span_r, _ = resolve_duration(br, env_r)
        dm, src_m, _span_m, _ = resolve_duration(bm, env_m)
        pairs.append({"序号": i + 1,
                      "块": "+".join(br["词"]),
                      "连读": br["连读"],
                      "角色": br.get("角色"),
                      "原音频时长": dr, "我的时长": dm,
                      "原音频时长来源": src_r, "我的时长来源": src_m})

    # ---- 停顿（块间 gap；换人处单独标）----
    pauses_r_all, pauses_r = find_boundary_pauses(blocks_r)
    pauses_m_all, pauses_m = find_boundary_pauses(blocks_m)
    turns_r = [p for p in pauses_r_all if p["换人处"]]
    turns_m = [p for p in pauses_m_all if p["换人处"]]

    # ---- 节奏分（score_blocks 内部已归一化 + 幂平均 p=3）----
    items, rhythm_score = score_blocks(pairs)
    for i, it in enumerate(items):
        if i < len(blocks_r):
            it["角色"] = blocks_r[i].get("角色")

    # ---- 终端简报 ----
    net_r = (blocks_r[-1]["end_ms"] - blocks_r[0]["start_ms"]) / 1000.0
    net_m = (blocks_m[-1]["end_ms"] - blocks_m[0]["start_ms"]) / 1000.0

    out("")
    out("整句净时长：原音频 %.2fs   |   我的 %.2fs   |   相对差 %+.1f%%"
        % (net_r, net_m, (net_m - net_r) / net_r * 100 if net_r > 0 else 0))
    out("句子边界停顿：原音频 %d 处   |   我的 %d 处   （换人处已排除，单独列出）"
        % (len(pauses_r), len(pauses_m)))
    for label, ps in (("原音频", pauses_r), ("我的", pauses_m)):
        for p in ps:
            out("   [%s] \"%s\" 后停了 %.2fs" % (label, p["after_word"], p["gap"]))
    if turns_r or turns_m:
        out("   换人处预期停顿（不计入处数）：")
        for p in turns_r:
            out("     [原音频] \"%s\" 后 %.2fs（换人）" % (p["after_word"], p["gap"]))
        for p in turns_m:
            out("     [我的]   \"%s\" 后 %.2fs（换人）" % (p["after_word"], p["gap"]))
    out("")
    if len(blocks_r) != len(blocks_m):
        out("⚠ 两侧词块数不一致（%d vs %d），序号可能错位，结果仅供参考"
            % (len(blocks_r), len(blocks_m)))
    out("词块时长对比（横轴=序号，不卡点）：")
    for it in items:
        tag = " [连读]" if it["连读"] else ""
        role = " [%s]" % it.get("角色") if it.get("角色") else ""
        out("   #%-2d %-16s 原音频 %.2fs | 我 %.2fs | %+.1f%%  %s%s%s"
            % (it["序号"], it["块"], it["原音频时长"], it["我的时长"],
               it["相对差"] * 100, it["判定"], tag, role))
    out("")
    out("节奏分：%.1f / 100  （归一化+幂平均 p=3，见 score_blocks）" % rhythm_score)
    out("语调分：暂不出（判分标准重做中）")
    out("")

    # ---- 画图 ----
    try:
        plot_blocks(items)
    except Exception as e:
        log("（画图失败，不影响上面数据：%s）" % e, show=True)
    try:
        semi_r = pitch_to_semitone(f0_r)
        semi_m = pitch_to_semitone(f0_m)
        plot_pitch(semi_r, semi_m)
    except Exception as e:
        log("（画音高图失败，不影响上面数据：%s）" % e, show=True)

    # ---- 写 json ----
    try:
        semi_r = pitch_to_semitone(f0_r)
        semi_m = pitch_to_semitone(f0_m)
        write_json(net_r, net_m, pauses_r_all, pauses_m_all, items, rhythm_score,
                   [], [], blocks_r, blocks_m, words_r, words_m,
                   [], [], semi_r, semi_m, env_r, env_m)
        out("数据（机器）：节奏对比.json")
    except Exception as e:
        log("（写 json 失败：%s）" % e, show=True)

    log_done("节奏 / 重音")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log_error(traceback.format_exc())
    try:
        input("按回车退出 ...")
    except Exception:
        pass
