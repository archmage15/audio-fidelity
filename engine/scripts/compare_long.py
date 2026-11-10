# -*- coding: utf-8 -*-
r"""还原度对比 · 长音频模式：按内容对齐，逐句评测

为什么单独一个脚本（而不是改 compare_phoneme.py）：
  单句链路已经 12 块排期打磨验证过（综合分 72.7 稳定），不能动。
  长音频是**新能力**，单独入口，零回归风险；单句音频走原链路不变。

核心思路（避开 §7.3「两段录音时间轴不匹配」的坑）：
  1. 参考音素序列（文本生成）按标点切成 N 句，每句有确定音素区间 [a,b)
  2. 模仿的识别音素 vs 完整参考序列做一次 difflib 对齐 —— **按内容对齐**，
     自动处理：读快读慢、跳过某句、多读、半截（只读前半）
  3. 全局对齐结果把「模仿音素」分配进各句区间；每句内部再各自对齐
     （对齐范围回到单句级别 → 词块不再崩坏）
  4. 区间内一个模仿音素都没对上的句 = **漏句**（不扣分，单独列出）
  5. 发音/音变按句汇总（求和天然按音素数加权）

产出：
  output\音素对比.json     与单句版同 schema（overall_score.py 直接可用），
                           外加「长音频」段（逐句明细）
  output\音素对比_报告.txt  人读：逐句明细 + 漏句清单 + 半截提示

依赖：与 compare_phoneme.py 相同（torch/transformers/soundfile/numpy）
用法：python scripts\compare_long.py
"""

from cpu_runtime import configure_cpu_runtime
configure_cpu_runtime()
import datetime
import difflib
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC_DIR = os.environ.get("AUDIO_TEST_WORK_DIR") or os.path.join(ROOT, "audio")
OUT_DIR = os.environ.get("AUDIO_TEST_OUTPUT_DIR") or os.path.join(ROOT, "output")
os.makedirs(OUT_DIR, exist_ok=True)

REF_FILE = None      # 运行时按格式探测（见 audio_paths），可能是 .mp3 / .wav / .flac
MY_FILE = None
REF_PHONEME_JSON = os.path.join(SRC_DIR, "参考音素.json")
REF_DIR = os.path.join(SRC_DIR, "原音频")
REF_CACHE_JSON = os.path.join(REF_DIR, "原音频缓存.json")   # 原音频分析缓存（10-28 加）
REF_LOGITS_NPY = os.path.join(REF_DIR, "原音频_logits.npy")  # 原音频帧级 logits（强制对齐用）
OUT_JSON = os.path.join(OUT_DIR, "音素对比.json")
OUT_REPORT = os.path.join(OUT_DIR, "音素对比_报告.txt")
RHYTHM_JSON = os.path.join(OUT_DIR, "节奏对比.json")   # 同单句版 schema，overall_score 直接读

sys.path.insert(0, HERE)
from logkit import log, out, log_setup, log_done, log_step, log_error
from audio_paths import find_audio, missing_hint, short_name, reference_cache_stale

# 原音频可能放在 audio\原音频\ 子目录（10-28 拆缓存后），模仿在 audio\ 根目录
REF_FILE = find_audio("原音频", REF_DIR) or find_audio("原音频", SRC_DIR)
MY_FILE = find_audio("模仿", SRC_DIR)

# 复用单句链路的函数（import 不会触发 main，有 __main__ 保护）
import numpy as np
import compare_phoneme as cp
import compare_rhythm as cr

# 少于这么多音素的句不判漏句（讯飞转写会把一句话拆成碎句，如 "There. Are"，
# 2 音的 "There" 若判漏句会虚增漏句数 —— 实测踩过）
MIN_SENT_PH = 3


def assign_mine_to_sentences(opcodes, sentences, n_ref, n_mine):
    """把全局对齐的「模仿音素下标」分配到各句。

    opcodes: difflib.get_opcodes() 的结果（a=text_ph 参考序列, b=mine 模仿序列）
    返回：dict 句号 -> [mine 音素下标,...]

    规则：
      - equal/replace：参考区间 [i1,i2) 落在哪句，对应的模仿 [j1,j2) 就归哪句；
        跨句时按重叠比例拆分（极少发生）。
      - insert（模仿多读、参考没有）：归到前一个参考位置所在句（i1-1）。
    """
    # 参考位置 -> 句号
    pos2sent = {}
    for s in sentences:
        for i in range(s["a"], min(s["b"], n_ref)):
            pos2sent[i] = s["句号"]

    assign = {s["句号"]: [] for s in sentences}
    for op, i1, i2, j1, j2 in opcodes:
        if op == "insert":
            # 多读的音：归到前一个参考位置所在句
            prev = i1 - 1
            sent = pos2sent.get(prev)
            if sent is not None:
                assign[sent].extend(range(j1, j2))
            continue
        # equal / replace / delete：按参考区间归属
        if i2 <= i1:
            continue
        # 统计该区间内每个句号的重叠量
        overlap = {}
        for i in range(i1, i2):
            s = pos2sent.get(i)
            if s is not None:
                overlap[s] = overlap.get(s, 0) + 1
        if not overlap:
            continue
        total = sum(overlap.values())
        mine_ids = list(range(j1, j2))
        if op == "delete" or not mine_ids:
            continue        # delete 没有模仿音素可分配
        # 按重叠比例把模仿音素分给各句
        cursor = 0
        for s in sorted(overlap):
            k = round(len(mine_ids) * overlap[s] / total)
            if s == sorted(overlap)[-1]:
                k = len(mine_ids) - cursor      # 最后的句拿剩下的，避免舍入丢失
            assign[s].extend(mine_ids[cursor:cursor + k])
            cursor += k
    return assign


def tokenize_words(text):
    """把句子文本切成词（小写、去标点），返回 list[str]。

    单词级对比用：只保留字母/撇号构成的词，统一小写，忽略标点/数字。
    """
    if not text:
        return []
    words = []
    for w in text.split():
        # 去首尾标点，保留词内撇号（I'm / it's）
        w = w.strip(".,!?;:()[]{}\"'")
        # 只保留含字母的词
        if w and any(ch.isalpha() for ch in w):
            words.append(w.lower())
    return words


def align_words(ref_words, my_words):
    """两个词序列做词级对齐，返回逐词对比条目 list[dict]。

    每条：
      {"类型": "读对"|"读错"|"漏读"|"多读", "原词": str, "实读": str}

    词级 difflib 对齐没有音素级那种错位假象（词是自然单元），
    这里只用于「单词级纠音」展示，不参与评分（评分仍走强制对齐）。

    ⚠️ 比较时忽略大小写（讯飞转写统一小写，但原文本词边界保留首字母大写，
    直接比会把「Hey vs hey」误判成读错）—— 只影响判「读对/读错」，
    展示仍用原始词。
    """
    out = []
    if not ref_words and not my_words:
        return out
    # 对齐用 lower 后的词（讯飞统一小写，原词保留大写，直接比会把「Hey vs hey」
    # 误判成 replace/错位），展示仍用原始词。
    ref_key = [w.lower() for w in ref_words]
    my_key = [w.lower() for w in my_words]
    sm = difflib.SequenceMatcher(a=ref_key, b=my_key, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for k in range(i1, i2):
                out.append({"类型": "读对", "原词": ref_words[k],
                            "实读": my_words[j1 + k - i1]})
        elif op == "replace":
            # 两边长度不等时逐词配对，多出的归「漏读/多读」
            n = max(i2 - i1, j2 - j1)
            for k in range(n):
                rw = ref_words[i1 + k] if i1 + k < i2 else None
                mw = my_words[j1 + k] if j1 + k < j2 else None
                if rw is not None and mw is not None:
                    out.append({"类型": "读错", "原词": rw, "实读": mw})
                elif rw is not None:
                    out.append({"类型": "漏读", "原词": rw, "实读": ""})
                else:
                    out.append({"类型": "多读", "原词": "", "实读": mw})
        elif op == "delete":
            for k in range(i1, i2):
                out.append({"类型": "漏读", "原词": ref_words[k], "实读": ""})
        else:  # insert
            for k in range(j1, j2):
                out.append({"类型": "多读", "原词": "", "实读": my_words[k]})
    return out


def split_sentences_text(text):
    """把整段转写文本按句末标点切成句子（返回 list[str]，已 trim）。

    与 gen_ref_phonemes 的断句口径保持一致：以句号/问号/感叹号结尾切分。
    """
    if not text:
        return []
    import re
    # 按 . ! ? 后跟空格/结尾 切分
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def classify_diffs(scored):
    """把逐句差异按「性质」分类，返回 {类别: [(句号, 差异条目), ...]}。

    分类（10-30 起，差异条目来自强制对齐的可疑音）：
      口音型（L2 音变）→ 练发音：θ→f、ð→d、ɹ→l 这类，母语者几乎不犯
      未读出           → 读太轻/漏读/被吞了，练清晰度
      读太轻           → 读对了但太轻，读清晰一点
      替换             → 其他读错

    ⚠️ 差异条目的「类别」字段在 evaluate_sentence 里已统一成上述四类
    （未读出/读太轻/口音型/替换），这里**只按类别分桶**，不再做名称映射
    —— 若再映射一次，「未读出」会因不在映射表里被误归「替换」（10-30 踩过）。
    """
    cats = {"口音型": [], "未读出": [], "读太轻": [], "替换": []}
    for r in scored or []:
        for d in r.get("差异") or []:
            c = d.get("类别") or "替换"
            cats.setdefault(c, []).append((r["句号"], d))
    return cats


def compute_baseline(text_ph, ref_verdict):
    """算「母语者基准」—— 工具自己的误差底线（强制对齐口径）。

    目的：让用户知道「这个工具自己会错多少」。
          否则 60 分看着像失败，其实母语者读同一段也只有这个水平，
          用户会以为差 40 分，实际只差 28 分。

    口径必须**与本次完全一致**（同一套强制对齐、同一份参考音素），
    才可比 —— 拿别处的数字来对比是自欺欺人。

    10-30 改：从 difflib ratio 口径切换到强制对齐口径（发音/音变分都已
    改用强制对齐，母语者基准必须跟它们同口径才能减出「真实差距」）。
    怎么算：把**原音频自己的强制对齐可疑音**当成"模仿"去跟参考音素比。
          原音频 = 母语者 + 干净录音 = 工具的误差底线。
          它也被判可疑的那些音 → 工具的固有误差（不是任何人的问题）。

    参数
      text_ph     : 参考音素序列
      ref_verdict : 原音频的强制对齐可疑音清单（align_phone_verdict 输出，
                    每个含 类别/音素/下标）
    返回 dict；输入为空时返回 None。
    """
    if not text_ph or ref_verdict is None:
        return None
    n_ref = len(text_ph)
    # 母语者基准的可疑率 = 原音频被判可疑的音数 / 参考音素数
    n_susp = len(ref_verdict)
    l2 = sum(1 for v in ref_verdict if v.get("类别") == "口音型")
    miss = sum(1 for v in ref_verdict if v.get("类别") == "没读出来")
    return {
        "口径": "原音频强制对齐可疑率（母语者+干净录音 = 工具误差底线）",
        "可疑数": n_susp,
        "可疑率": round(100.0 * n_susp / max(n_ref, 1), 1),
        "口音型": l2,
        "没读出来": miss,
        "参考音素数": n_ref,
    }


def evaluate_sentence(sent, text_ph, mine, mine_frames, word_bounds, sent_susp):
    """单句评测：漏句/碎句判断 + 归句的强制对齐可疑音（差异明细）。

    10-30 改：删掉句内 difflib 评分对齐（那是「替换/缺失/多余」错位假象的根源）。
    差异明细改由**强制对齐的可疑音**按参考音素下标归句提供 —— 强制对齐是
    「已知要读什么，去音频里找它在哪」，逐音给后验，读快读慢不影响，
    不会像 difflib 那样把整词错位。

    参数
      sent        : 句子 dict（含 a/b 音素区间、句号、文本、_mine 分配的模仿音素下标）
      text_ph     : 完整参考音素序列
      mine        : 模仿识别音素序列
      mine_frames : 模仿音素帧号
      word_bounds : 词边界（用于差异条目挂词标签）
      sent_susp   : 归到本句的强制对齐可疑音清单（已按下标过滤）

    返回 dict：句号/文本/音素区间/参考音素/模仿音素/参考音素数/模仿音素数/
              漏句/忽略/差异/问题词
    """
    sa, sb = sent["a"], sent["b"]
    sent_ref = text_ph[sa:sb]
    # 句内词边界（下标平移到句内）
    sent_words = [dict(wb, a=wb["a"] - sa, b=wb["b"] - sa)
                  for wb in word_bounds if wb["a"] >= sa and wb["b"] <= sb]

    mine_ids = sent["_mine"]
    sent_mine = [mine[j] for j in mine_ids]
    sent_frames = [mine_frames[j] for j in mine_ids]

    # 句内参考位置 -> 词标签（差异条目挂词用，口径同 compare_phoneme.format_words）
    def word_at(pos):
        for wb in sent_words:
            if wb["a"] <= pos < wb["b"]:
                return wb["word"]
        return None

    # 差异明细：本句的强制对齐可疑音（sent_susp 已按下标归好，含「下标」是全局下标）
    diff_items = []
    for v in sent_susp or []:
        pos = v["下标"] - sa              # 转成句内下标
        word = word_at(pos)
        label = ("【单词 \"%s\"】" % word) if word else ""
        ref_ph = v.get("音素", "")
        got = v.get("实读", "")
        cat = v.get("类别", "其他")
        if cat == "没读出来":
            tip = "【要领】这个音没读出来（可能读太轻、漏读或被吞掉了）"
            _g = cp.phoneme_tips(ref_ph)
            if _g:
                tip += "　" + _g
            diff_items.append({"类型": "缺失", "词": label, "应读": ref_ph,
                               "实读": "", "类别": "未读出", "要领": tip,
                               "后验": v.get("后验", 0)})
        elif cat == "读太轻":
            tip = "【要领】读对了但太轻，读清晰一点"
            diff_items.append({"类型": "替换", "词": label, "应读": ref_ph,
                               "实读": got, "类别": "读太轻", "要领": tip,
                               "后验": v.get("后验", 0)})
        elif cat == "口音型":
            tip = cp.L2_CONFUSIONS.get((ref_ph, got), "") or cp.phoneme_tips(ref_ph)
            diff_items.append({"类型": "替换", "词": label, "应读": ref_ph,
                               "实读": got, "类别": "口音型",
                               "要领": ("【口音型】%s → %s：%s" % (ref_ph, got, tip)) if tip else "",
                               "后验": v.get("后验", 0)})
        else:  # 其他替换
            tip = cp.phoneme_tips(ref_ph)
            diff_items.append({"类型": "替换", "词": label, "应读": ref_ph,
                               "实读": got, "类别": "替换", "要领": tip,
                               "后验": v.get("后验", 0)})

    # 碎句（转写把一句拆成 "There."+"Are..."）：不判漏句、不计分
    ignored = len(sent_ref) < MIN_SENT_PH
    # 漏句：全局对齐时这句一个模仿音素都没分到（读快读慢/跳句都不影响这句判漏）
    is_missed = (not ignored) and (len(sent_mine) == 0)

    # 词级错误：供"原句框出"用。从可疑音按词归集（错音≥2 或占词比例≥40%）。
    word_problems = []
    from collections import Counter
    wrong_by_word = Counter()
    for v in sent_susp or []:
        pos = v["下标"] - sa
        word = word_at(pos)
        if word:
            wrong_by_word[word] += 1
    for wb in sent_words:
        word = wb["word"]
        wrong = wrong_by_word.get(word, 0)
        if wrong == 0:
            continue
        ratio = wrong / float(max(wb["b"] - wb["a"], 1))
        if wrong >= 2 or ratio >= 0.40:
            word_problems.append({"词": word, "错音数": wrong,
                                  "错误比例": round(ratio, 3)})

    return {
        "句号": sent["句号"],
        "文本": sent["文本"],
        "角色": sent.get("角色"),
        "音素区间": [sa, sb],
        "参考音素": " ".join(sent_ref),
        "模仿音素": " ".join(sent_mine),
        "参考音素数": len(sent_ref),
        "模仿音素数": len(sent_mine),
        "漏句": is_missed,
        "忽略": ignored,
        "差异": diff_items,
        "问题词": word_problems,
    }


def build_sentence_blocks(sent, text_ph, ref, ref_frames, mine, mine_frames, word_bounds):
    """给单句构造两侧对称的词块（用于节奏/重音）。

    原音频侧：识别音素按 参考序列↔原音频识别序列 的全局对齐分配进本句区间；
    模仿侧：同上（用 参考序列↔模仿识别序列 的对齐）。
    两侧词块数与词序天然对称（同一批 sent_words）→ 满足 compare_rhythm 的序号对齐前提。

    返回 (blocks_r, blocks_m)，元素 {"词","start_ms","end_ms","连读","角色"}。
    """
    sa, sb = sent["a"], sent["b"]
    sent_ref = text_ph[sa:sb]
    sent_words = [dict(wb, a=wb["a"] - sa, b=wb["b"] - sa)
                  for wb in word_bounds if wb["a"] >= sa and wb["b"] <= sb]
    if not sent_words:
        return [], []

    def side(key, audio_ph, frames):
        ids = sent["_%s" % key]
        ph = [audio_ph[j] for j in ids]
        fr = [frames[j] for j in ids]
        if not ph:
            return []
        owner = cp.map_words_to_audio(sent_ref, ph, sent_words)
        spans = cp.build_text_spans(sent_words, owner, [f * cp.FRAME_MS for f in fr])
        # ⚠️ 词 必须是**列表**（与单句版词块契约一致）：compare_rhythm 里
        # "+".join(b["词"]) 按列表拼接；传字符串会按字符拆成 "W+h+e+r+e"（实测踩过）
        return [{"词": [sp["word"]], "start_ms": sp["start_ms"], "end_ms": sp["end_ms"],
                 "连读": False, "角色": sent.get("角色")} for sp in spans]

    return side("ref", ref, ref_frames), side("mine", mine, mine_frames)


def build_per_sentence_stress(block_sent_map, marked_r, marked_m, nr, nm, sent_results):
    """把整段的重音判定按句聚合，供报告逐句展示「应读哪些重音 / 我读了哪些」。

    输入：
      block_sent_map  {块序号 -> 句号}
      marked_r/marked_m  原音频/模仿判定为突出的块序号集合
      nr/nm  {块序号 -> 词}（如 {5: "fun", ...}）
      sent_results  compare_long 的逐句结果（含 句号/文本/漏句/忽略）

    返回：list[dict]，每句一个元素，只含「有重音可说的」句（应读或实读有突出块的）：
      {"句号", "文本", "应读重音": [词...], "实读重音": [词...],
       "漏读重音": [词...], "多读重音": [词...]}

    「应读重音」= 该句内原音频突出的块；「实读重音」= 该句内我突出的块。
    「漏读重音」= 应读但我没突出（漏）；「多读重音」= 我突出但原音频没突出（多）。
    """
    marked_r, marked_m = set(marked_r), set(marked_m)
    # 句号 -> 该句的块序号（按全局序号升序，保证词序）
    blocks_by_sent = {}
    for no, sid in block_sent_map.items():
        blocks_by_sent.setdefault(sid, []).append(no)
    for v in blocks_by_sent.values():
        v.sort()

    # 句号 -> 文本（供报告直接引用，不用再回 overall_score 找）
    text_by_sent = {r["句号"]: r.get("文本", "") for r in sent_results}

    result = []
    for sid in sorted(blocks_by_sent, key=lambda x: (x is None, x)):
        nos = blocks_by_sent[sid]
        should = [nr.get(n, "?") for n in nos if n in marked_r]
        got = [nm.get(n, "?") for n in nos if n in marked_m]
        missed = [nr.get(n, "?") for n in nos if n in marked_r and n not in marked_m]
        extra = [nm.get(n, "?") for n in nos if n in marked_m and n not in marked_r]
        # 每句都返回（含无重音的句），让报告能逐句完整标注「该读/实读」。
        result.append({"句号": sid, "文本": text_by_sent.get(sid, ""),
                       "应读重音": should, "实读重音": got,
                       "漏读重音": missed, "多读重音": extra})
    return result


def build_align_stat(scored, n_ref):
    """构造对齐统计 dict（10-30 改：difflib 四类计数已删，只留基础计数）。

    发音分/音变分都已改强制对齐口径（读「强制对齐统计」），这里只保留
    「原生音数（参考音素数）+ 模仿音数」两个基础计数，供 overall_score
    展示用，不再参与任何评分。
    """
    return {"原生音数": n_ref,
            "模仿音数": sum(r["模仿音素数"] for r in scored)}


def main():
    log_setup("长音频评测")
    out("=" * 62)
    out("   还原度对比 · 长音频（按内容对齐·逐句评测）")
    out("=" * 62)

    if not (REF_FILE and MY_FILE):
        out("X 没找到音频文件：")
        out(missing_hint([("原音频", REF_FILE), ("模仿", MY_FILE)], SRC_DIR))
        log_done("（缺音频，提前退出）")
        return 1
    # ---- 读原音频缓存（10-28 加）----
    #   原音频是固定范本，它的转写文本/参考音素表/CTC 音素/RMS/音高都不变。
    #   有缓存就直接用，跳过原音频的模型推理 + 特征计算，只分析模仿音频。
    ref_cache = None
    if os.path.exists(REF_CACHE_JSON):
        try:
            with open(REF_CACHE_JSON, encoding="utf-8") as f:
                ref_cache = json.load(f)
        except Exception:
            ref_cache = None
    # 失效检测（10-30 加）：换过原音频则缓存过期，拒绝用旧缓存。
    #   否则会用旧参考音素/logits 评测新音频、结果错乱且不报错。
    if ref_cache:
        stale = reference_cache_stale(ref_cache, REF_FILE)
        if stale is True:
            out("")
            out("X 原音频缓存已过期：原音频文件换过了，旧缓存对应的是旧原音频。")
            out("  直接拿来评测会得出错误结论。请重新跑 audio\\原音频\\分析原音频.bat。")
            log_done("（原音频缓存过期，提前退出）")
            return 1
        if stale is None:
            log("   ⚠ 原音频缓存缺少指纹字段（旧版缓存），无法判断是否过期；"
                "若已换过原音频请先重跑 分析原音频.bat", show=True)
    if ref_cache:
        log("   用原音频缓存（跳过原音频的模型推理与特征计算）", show=True)
    else:
        log("   没找到原音频缓存 —— 请先跑 audio\\原音频\\分析原音频.bat", show=True)

    log_step("加载音素识别模型")
    model_path = cp.ensure_model()
    fe = cp.load_feature_extractor(model_path)
    model = cp.AutoModelForCTC.from_pretrained(model_path) if hasattr(cp, "AutoModelForCTC") else None
    if model is None:
        from transformers import AutoModelForCTC
        model = AutoModelForCTC.from_pretrained(model_path)
    model.eval()
    id2tok = cp.load_vocab(model_path)

    import torch
    results, frames_map = {}, {}
    # 强制对齐需要帧级 logits（10-29 加）—— 之前 argmax 后就把 logits 丢了。
    # logits_map: tag -> numpy [T, V]；argmax_map: tag -> 每帧贪心 id（判"实读"用）
    logits_map, argmax_map = {}, {}
    if ref_cache:
        # 原音频 CTC 结果直接用缓存，只对模仿音频跑模型
        results["原音频"] = ref_cache.get("原音频_CTC音素") or []
        frames_map["原音频"] = ref_cache.get("原音频_帧") or []
        log("   原音频（缓存）：%d 音素" % len(results["原音频"]))
        # 原音频 logits 从 .npy 缓存读（没有就跳过，强制对齐只用模仿侧也能出结果）
        if os.path.exists(REF_LOGITS_NPY):
            _ref_logits = np.load(REF_LOGITS_NPY)
            logits_map["原音频"] = _ref_logits
            argmax_map["原音频"] = _ref_logits.argmax(axis=1).tolist()
            log("   原音频 logits（缓存）：%d 帧 × %d 符号" % _ref_logits.shape)
        else:
            log("   （无原音频 logits 缓存，跳过原音频强制对齐基准；跑一次 分析原音频.bat 补上）",
                show=True)
        log_step("识别模仿音频")
        audio = cp.load_mono(MY_FILE)
        inputs = fe(audio, sampling_rate=cp.SR, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        logits_map["我的模仿"] = logits[0].detach().cpu().numpy()
        argmax_map["我的模仿"] = torch.argmax(logits, dim=-1)[0].tolist()
        ids = argmax_map["我的模仿"]
        toks, frames = cp.ctc_decode(ids, id2tok)
        results["我的模仿"] = toks
        frames_map["我的模仿"] = frames
        log("   我的模仿：%d 音素（%.1f 秒）" % (len(toks), len(audio) / cp.SR))
    else:
        log_step("识别两段音频")
        for tag, path in (("原音频", REF_FILE), ("我的模仿", MY_FILE)):
            audio = cp.load_mono(path)
            inputs = fe(audio, sampling_rate=cp.SR, return_tensors="pt")
            with torch.no_grad():
                logits = model(inputs.input_values).logits
            logits_map[tag] = logits[0].detach().cpu().numpy()
            argmax_map[tag] = torch.argmax(logits, dim=-1)[0].tolist()
            ids = argmax_map[tag]
            toks, frames = cp.ctc_decode(ids, id2tok)
            results[tag] = toks
            frames_map[tag] = frames
            log("   %s：%d 音素（%.1f 秒）" % (tag, len(toks), len(audio) / cp.SR))

    ref = results["原音频"]
    mine = results["我的模仿"]

    log_step("读参考音素表（含句子边界）")
    # 有缓存 → 参考音素表也从缓存读（自包含，不依赖 参考音素.json 文件）
    if ref_cache and ref_cache.get("参考音素"):
        ref_data = ref_cache["参考音素"]
    elif os.path.exists(REF_PHONEME_JSON):
        with open(REF_PHONEME_JSON, encoding="utf-8") as f:
            ref_data = json.load(f)
    else:
        out("X 缺少 参考音素.json —— 请先跑 分析原音频.bat 或 gen_ref_phonemes.py")
        log_done("（缺参考音素表，提前退出）")
        return 1
    text_ph = ref_data.get("音素") or []
    word_bounds = ref_data.get("词边界") or []
    sentences = ref_data.get("句子边界") or []
    if not sentences:
        out("X 参考音素表里没有 句子边界 —— 请用新版 gen_ref_phonemes.py 重新生成")
        log_done("（缺句子边界，提前退出）")
        return 1
    log("   参考 %d 音素 / %d 词 / %d 句" % (len(text_ph), len(word_bounds), len(sentences)))

    # ---- 单词级纠音（10-30 加）：转写模仿音频成文字，跟原文本逐词对齐 ----
    #   音素级对比只能告诉你「θ 读成 f」，单词级能告诉你「Thanks 读成 Sanks」「漏了 for」。
    #   这里转写「模仿」音频（复用讯飞），做 transcript_fix 修复，再整段词级对齐。
    #   ⚠️ 关键：讯飞断句和原文断句常不一致（如「lesson, no problem」合并成一句），
    #     所以**不能按句号下标一一对应**，而是把原文词序列和模仿词序列做整段对齐，
    #     再按「原词所属句号」把对齐结果归组（10-30 踩过：按句切会整体错位）。
    #   注意：这是**展示辅助**，不参与评分（评分仍走强制对齐），转写失败只降级不阻断。
    word_align = {}        # 句号 -> list[逐词条目]
    my_transcript = None   # 模仿转写整段文本（修复后）
    try:
        import xunfei_transcribe as xt
        import transcript_fix as tf
        log_step("转写模仿音频（单词级纠音）")
        my_raw, _my_sents, _ = xt.transcribe_audio(MY_FILE)
        my_transcript, _fixes = tf.fix_transcript(my_raw)
        # 原词序列：按词边界顺序（每个词带句号），拼出完整原词 + 每个词归属的句号
        ref_all_words = [wb["word"] for wb in word_bounds]
        ref_word_sent = [wb.get("句号", 0) for wb in word_bounds]
        my_all_words = tokenize_words(my_transcript)
        # 整段词级对齐（词是自然单元，无音素级那种错位假象）
        all_align = align_words(ref_all_words, my_all_words)
        # 按原词所属句号归组：读对/读错/漏读消耗一个原词下标（与 ref_all_words 对应），
        # 多读不消耗（归到前一个原词的句号）
        last_sent = 1
        ri = 0
        for it in all_align:
            if it["类型"] == "多读":
                sid = ref_word_sent[min(ri, len(ref_word_sent) - 1)] if ref_word_sent else last_sent
            else:
                sid = ref_word_sent[ri] if ri < len(ref_word_sent) else last_sent
                ri += 1
            word_align.setdefault(sid, []).append(it)
            last_sent = sid
        log("   模仿转写：%s" % (my_transcript[:80] + ("..." if len(my_transcript) > 80 else "")))
    except Exception as e:
        log("   ⚠ 单词级纠音跳过（转写模仿音频失败：%s）" % e, show=True)
        my_transcript = None


    #   把参考音素序列对齐到两侧音频的帧级 logits，得到逐音后验。
    #   这是「已知要读什么，去音频里找它在哪」，不依赖词边界，也不怕读快读慢。
    #   读快读慢不影响，逐音给后验，不会像 difflib 那样整词错位。
    log_step("强制对齐（逐音后验，发音/音变/差异明细唯一口径）")
    fa_result = {"模仿": None, "原音频": None}
    for tag, key in (("我的模仿", "模仿"), ("原音频", "原音频")):
        if tag not in logits_map:
            continue
        aligned = cp.forced_align(logits_map[tag], text_ph, id2tok)
        verdict = cp.align_phone_verdict(aligned, text_ph, argmax_map.get(tag, []),
                                         id2tok, threshold=0.5)
        fa_result[key] = {"逐音后验": aligned, "可疑": verdict}
        if key == "模仿":
            _n_susp = len(verdict)
            _n_l2 = sum(1 for v in verdict if v["类别"] == "口音型")
            _n_miss = sum(1 for v in verdict if v["类别"] == "没读出来")
            log("   模仿：%d 音素对齐，可疑 %d（口音型 %d / 没读出 %d / 其他+读太轻 %d）"
                % (len(aligned), _n_susp, _n_l2, _n_miss, _n_susp - _n_l2 - _n_miss))
        else:
            log("   原音频（母语者基准）：可疑 %d / %d 音素"
                % (len(verdict), len(aligned)))

    # 强制对齐统计（10-29 起，发音分改用这个口径）：
    #   发音分 = 1 - (口音型 + 其他 + 读太轻) / 参考音素数。
    #   「没读出来」归音变（吞音漏读），不扣发音分 —— 那是清晰度问题。
    fa_stat = None
    if fa_result.get("模仿") and fa_result["模仿"].get("可疑") is not None:
        _v = fa_result["模仿"]["可疑"]
        fa_stat = {
            "参考音素数": len(text_ph),
            "可疑数": len(_v),
            "口音型": sum(1 for x in _v if x["类别"] == "口音型"),
            "没读出来": sum(1 for x in _v if x["类别"] == "没读出来"),
            "读太轻": sum(1 for x in _v if x["类别"] == "读太轻"),
            "其他": sum(1 for x in _v if x["类别"] == "其他"),
        }

    # ---- 全局对齐（按内容，不用时间戳）----
    log_step("全局对齐（按内容）")
    sm = difflib.SequenceMatcher(a=text_ph, b=mine, autojunk=False)
    opcodes = sm.get_opcodes()
    cov = sum(i2 - i1 for op, i1, i2, j1, j2 in opcodes if op == "equal") / max(1, len(text_ph))
    log("   音素级覆盖率 %.1f%%" % (cov * 100))

    assign = assign_mine_to_sentences(opcodes, sentences, len(text_ph), len(mine))
    for s in sentences:
        s["_mine"] = assign.get(s["句号"], [])
    # 原音频侧识别音素也要分配进各句（节奏/重音需要两侧对称词块）
    sm_ref = difflib.SequenceMatcher(a=text_ph, b=ref, autojunk=False)
    assign_ref = assign_mine_to_sentences(sm_ref.get_opcodes(), sentences,
                                          len(text_ph), len(ref))
    for s in sentences:
        s["_ref"] = assign_ref.get(s["句号"], [])

    # ---- 逐句评测 ----
    #   10-30 改：差异明细来自强制对齐的可疑音（按下标归句），不再做句内 difflib。
    #   先把「模仿侧」可疑音按参考音素下标归到各句区间 [a,b)。
    log_step("逐句评测")
    mine_susp = (fa_result.get("模仿") or {}).get("可疑") or []
    sent_susp = {s["句号"]: [] for s in sentences}
    for v in mine_susp:
        idx = v.get("下标", -1)
        for s in sentences:
            if s["a"] <= idx < s["b"]:
                sent_susp[s["句号"]].append(v)
                break
    sent_results = [evaluate_sentence(s, text_ph, mine, frames_map["我的模仿"],
                                      word_bounds, sent_susp.get(s["句号"], []))
                    for s in sentences]
    scored = [r for r in sent_results if not r["漏句"] and not r["忽略"]]
    missed = [r for r in sent_results if r["漏句"]]
    ignored = [r for r in sent_results if r["忽略"]]
    log("   %d 句中：已模仿 %d 句，漏读 %d 句，碎句忽略 %d 句"
        % (len(sentences), len(scored), len(missed), len(ignored)))

    # ---- 半截检测（句子级）----
    tail_missed = 0
    for r in reversed(sent_results):
        if r["漏句"]:
            tail_missed += 1
        else:
            break
    coverage_ratio = (len(sent_results) - tail_missed) / float(len(sent_results))
    is_partial = tail_missed >= 2 and coverage_ratio < 0.95
    if is_partial:
        log("   ⚠ 半截模仿：只读到第 %d/%d 句，后 %d 句整句未读"
            % (len(sent_results) - tail_missed, len(sent_results), tail_missed), show=True)

    # ---- 汇总（10-30 起统一强制对齐口径，删 difflib 评分对齐）----
    #   发音分 = 1 - (口音型+其他+读太轻) / 参考音素数
    #   音变分 = 1 - (没读出来+读太轻) / 参考音素数
    #   （「多余音」difflib 是假象、强制对齐单向定位又算不了，直接丢掉不扣分）
    #   相似度 = 100 - 模仿可疑率（供展示/真实差距，统一到强制对齐口径）
    n_ref = len(text_ph)
    if fa_stat:
        pron = max(0.0, (1 - (fa_stat["口音型"] + fa_stat["其他"] + fa_stat["读太轻"])
                         / max(fa_stat["参考音素数"], 1))) * 100
        change = max(0.0, (1 - (fa_stat["没读出来"] + fa_stat["读太轻"])
                            / max(fa_stat["参考音素数"], 1))) * 100
        # 相似度：用「可疑率」的补数（100 - 可疑率）。读太轻在发音/音变都扣，
        # 但不属于「读错」，故相似度只扣「口音型+其他+没读出来」三类硬错。
        sim = max(0.0, (1 - (fa_stat["口音型"] + fa_stat["其他"] + fa_stat["没读出来"])
                         / max(n_ref, 1))) * 100
    else:
        # 没跑出强制对齐（如缺 logits）→ 无法出分，记 None 由 overall_score 显示「—」
        pron = None
        change = None
        sim = None

    # ---- 母语者基准（10-30 改强制对齐口径）----
    #   基准 = 原音频自己的强制对齐可疑率（工具误差底线）。
    #   真实差距 = 模仿可疑率 - 原音频可疑率（百分点），不再是 difflib ratio 之差。
    ref_verdict = (fa_result.get("原音频") or {}).get("可疑")
    baseline = compute_baseline(text_ph, ref_verdict)
    if baseline:
        log("   母语者基准：可疑率 %.1f%%（可疑 %d 音 / %d 参考音素，工具固有误差）"
            % (baseline["可疑率"], baseline["可疑数"], baseline["参考音素数"]), verbose=True)

    # ---- 节奏 / 重音（逐句两侧对称词块，整段汇总）----
    log_step("节奏 / 重音（逐句对称词块）")
    blocks_r_all, blocks_m_all, pairs = [], [], []
    # 原音频侧 RMS/音高若有缓存直接用（跳过 load_mono + 特征计算）
    # ⚠️ 缓存里存的是 JSON 可序列化的 list，但 block_signal_features 需要
    #    numpy 数组（env.max() / f0[bool] 布尔索引）—— 必须转回 np.array。
    if ref_cache:
        env_r = np.array(ref_cache.get("原音频_RMS") or [], dtype=float)
        f0_r = np.array(ref_cache.get("原音频_音高") or [], dtype=float)
    else:
        _ref_wave = cp.load_mono(REF_FILE)  # 只加载一次，RMS/音高共用
        env_r = cr.frame_rms(_ref_wave)
        f0_r = cr.pitch_contour(_ref_wave)
    _my_wave = cp.load_mono(MY_FILE)  # 只加载一次，RMS/音高共用
    env_m = cr.frame_rms(_my_wave)
    f0_m = cr.pitch_contour(_my_wave)
    # 块序号 -> 句号 映射：重音判定是整段算的（序号全局 1..N），
    # 但报告要按句展示「应读哪些重音 / 我读了哪些」，
    # 所以 extend 词块时同步记下每个块属于哪一句。
    block_sent_map = {}
    for s in sentences:
        r = next(x for x in sent_results if x["句号"] == s["句号"])
        if r["漏句"] or r["忽略"]:
            continue
        br, bm = build_sentence_blocks(s, text_ph, ref, frames_map["原音频"],
                                       mine, frames_map["我的模仿"], word_bounds)
        if not br or not bm:
            continue
        # 本句块在本句内的下标从 0 开始；全局序号 = 已 extend 的块数 + 下标 + 1
        base = len(blocks_r_all)
        blocks_r_all.extend(br)
        blocks_m_all.extend(bm)
        for k in range(len(br)):
            block_sent_map[base + k + 1] = s["句号"]
        for i in range(min(len(br), len(bm))):
            pairs.append({"序号": len(pairs) + 1,
                          "块": "+".join(br[i]["词"]),
                          "连读": br[i].get("连读", False),
                          "角色": br[i].get("角色"),
                          "原音频时长": cr.block_span(br[i]),
                          "我的时长": cr.block_span(bm[i])})
    # 节奏分：把所有已读句的词块对合成一次 score_blocks（归一化+幂平均，同单句口径）
    items, rhythm_score = cr.score_blocks(pairs) if pairs else ([], 0.0)
    # 重音：emphases 找重读词（2026-11-02 大法师拍板，替代邻域法）
    import emphases_stress as es
    _ref_wave = cp.load_mono(REF_FILE)  # emphases 需要原音频波形（缓存里没存波形）
    marked_r, scores_r = (es.stress_marks(_ref_wave, cp.SR, blocks_r_all)
                          if blocks_r_all else ([], np.zeros(0)))
    marked_m, scores_m = (es.stress_marks(_my_wave, cp.SR, blocks_m_all)
                          if blocks_m_all else ([], np.zeros(0)))
    miss_s = sorted(set(marked_r) - set(marked_m))
    extra_s = sorted(set(marked_m) - set(marked_r))
    hit_s = sorted(set(marked_r) & set(marked_m))
    ref_prom = len(marked_r)
    stress_pos = (round(max(0.0, (1.0 - len(miss_s) / float(max(ref_prom, 1))) * 100.0), 1)
                  if ref_prom else None)
    log("   节奏分 %.1f ｜ 重音位置 %s（原重读 %d / 我重读 %d / 漏 %d / 多 %d）"
        % (rhythm_score, stress_pos, ref_prom, len(marked_m), len(miss_s), len(extra_s)))

    # ---- 画图（复用 compare_rhythm 的绘图函数；不画就会留下上一次的旧图）----
    log_step("画图")
    try:
        cr.plot_blocks(items, title="节奏对比（长音频·按词块序号，仅已读句）")
    except Exception as e:
        log("（画节奏图失败，不影响数据：%s）" % e, show=True)
    try:
        cr.plot_pitch(cr.pitch_to_semitone(f0_r), cr.pitch_to_semitone(f0_m))
    except Exception as e:
        log("（画音高图失败，不影响数据：%s）" % e, show=True)

    out("")
    out("=" * 62)
    out("   长音频评测结果")
    out("=" * 62)
    out("整段音素相似度：%s / 100" % ("%.1f" % sim if sim is not None else "—"))
    if baseline:
        out("母语者基准：可疑率 %.1f%%（原音频自己跟参考音素比 = 工具误差底线）"
            % baseline["可疑率"])
        if sim is not None:
            out("  → 你的可疑率比母语者高 %.1f 个百分点（这就是真实差距）"
                % max(0.0, (100 - sim) - baseline["可疑率"]))
    out("已模仿 %d / %d 句（漏读 %d 句，碎句忽略 %d 句）"
        % (len(scored), len(sent_results), len(missed), len(ignored)))
    if is_partial:
        out("⚠ 半截模仿：只读到 %.0f%%，后 %d 句未读（不计分）" % (coverage_ratio * 100, tail_missed))
    out("发音分 %s ｜ 音变分 %s ｜ 节奏分 %.1f ｜ 重音位置 %s（仅按已模仿句计算）"
        % (("%.1f" % pron) if pron is not None else "—",
           ("%.1f" % change) if change is not None else "—",
           rhythm_score,
           ("%.1f" % stress_pos) if stress_pos is not None else "—"))
    out("")
    for r in sent_results:
        tag = " [漏句-未读]" if r["漏句"] else (" [碎句-忽略]" if r["忽略"] else "")
        n_diff = len(r.get("差异") or [])
        out("  句%-2d 参考%3d音/模仿%3d音 可疑%2d音  %s%s"
            % (r["句号"], r["参考音素数"], r["模仿音素数"], n_diff,
               r["文本"][:36], tag))

    out("")
    out("=" * 62)
    out("   重音（emphases）：每个词的重读分数 + 纠错结论")
    out("=" * 62)
    _nr = {i + 1: "+".join(b["词"]) for i, b in enumerate(blocks_r_all)}
    _nm = {i + 1: "+".join(b["词"]) for i, b in enumerate(blocks_m_all)}
    if blocks_r_all or blocks_m_all:
        out("   【原音频】共 %d 词，重读 %d 词 ｜【我的模仿】共 %d 词，重读 %d 词"
            % (len(blocks_r_all), len(marked_r), len(blocks_m_all), len(marked_m)))
        out("")
        out("   序号  词               原音频分   模仿分   标记")
        for i in range(max(len(blocks_r_all), len(blocks_m_all))):
            no = i + 1
            word = _nr.get(no, _nm.get(no, "?"))
            sr_v = scores_r[i] if i < len(scores_r) else None
            sm_v = scores_m[i] if i < len(scores_m) else None
            flag = "M" if no in miss_s else ("X" if no in extra_s
                                             else ("★" if no in marked_r else "○"))
            out("   %-4d %-14s %8s %8s   %s" % (no, word,
                  ("%.3f" % sr_v) if sr_v is not None else "-",
                  ("%.3f" % sm_v) if sm_v is not None else "-", flag))
        out("")
        out("   标记：★ 两侧都重读  ○ 都不重读  M 原重读你漏了  X 你多读的")
        out("")
        out("纠错结论：")
        for line in cr.render_stress_verdict(_nr, _nm, marked_r, marked_m):
            out(line)
        out("")
        out("重点词清单（照这个单子练）：")
        if miss_s:
            out("   漏读重音（该重读没重读）：%s" % "、".join(_nr[i] for i in miss_s))
        if extra_s:
            out("   多读重音（不该重读却重读）：%s" % "、".join(_nm[i] for i in extra_s))
        if not miss_s and not extra_s:
            out("   （重音位置完全命中，无漏无多）")
    else:
        out("   （无已读句，无重音判定）")

    # ---- 写 JSON（兼容单句版 schema，overall_score.py 可直接用）----
    log_step("写结果")
    align_stat = build_align_stat(scored, n_ref)
    diff_lines = []
    for r in scored:
        for d in r["差异"]:
            # 类型标记带分类（10-30 改：类别来自强制对齐的四类，已无「多读出」）
            _cat = d.get("类别") or "替换"
            _mark = {"口音型": "口音", "未读出": "没读出", "读太轻": "读太轻",
                     "替换": "替换"}.get(_cat, "替换")
            if _cat == "未读出":
                line = "  [%s] 句%d %s%s：没读出来" % (_mark, r["句号"], d.get("词") or "",
                                                    ("应读 %s" % d["应读"]) if d["应读"] else "")
            elif _cat == "读太轻":
                line = "  [%s] 句%d %s%s：读对了但太轻" % (_mark, r["句号"], d.get("词") or "",
                                                     ("应读 %s" % d["应读"]) if d["应读"] else "")
            else:
                line = "  [%s] 句%d %s%s：%s" % (_mark, r["句号"], d.get("词") or "",
                                             ("应读 %s" % d["应读"]) if d["应读"] else "",
                                             ("实读 %s" % d["实读"]) if d["实读"] else "")
            if d.get("要领"):
                line += "   " + d["要领"]
            diff_lines.append(line)
    word_list = [{"word": wb["word"], "角色": wb.get("角色"), "句号": wb.get("句号")}
                 for wb in word_bounds]
    # 问题严重的词（去重，保序）——overall_score 优先用它框原句
    problem_words = []
    seen = set()
    for r in scored:
        for wp in r.get("问题词") or []:
            if wp["词"] not in seen:
                seen.add(wp["词"])
                problem_words.append(wp["词"])

    # 问题分类（10-30 改：类别来自强制对齐四类，已无 difflib 的「多读出」假象）。
    #   口音型（L2 音变）→ 练发音：θ→f、ð→d、ɹ→l 这类，母语者几乎不犯
    #   未读出           → 读太轻/漏读/被吞了，练清晰度
    #   读太轻           → 读对了但太轻，读清晰一点
    #   替换             → 其他读错
    _cats = classify_diffs(scored)

    payload = {
        "原音频": ref, "我的模仿": mine,
        "相似度": (round(sim, 1) if sim is not None else None), "对齐统计": align_stat,
        "差异": diff_lines,
        "词表": word_list,
        "问题词": problem_words,
        "词块": {"原音频": blocks_r_all, "我的": blocks_m_all},
        "半截模仿": {"半截": is_partial,
                     "覆盖比例": round(coverage_ratio, 4),
                     "已读句数": len(scored), "总句数": len(sent_results),
                     "尾部漏读句数": tail_missed,
                     "漏句": [{"句号": r["句号"], "文本": r["文本"]} for r in missed],
                     "说明": "句子级：区间内无模仿音素对齐上的句判为漏句，不计分；"
                             "少于 %d 音的碎句忽略（转写瑕疵）" % MIN_SENT_PH},
        "长音频": {"模式": "按内容对齐·逐句评测",
                   "句子": sent_results,
                   "发音分": (round(pron, 1) if pron is not None else None),
                   "音变分": (round(change, 1) if change is not None else None),
                   "节奏分": round(rhythm_score, 1),
                   "重音位置分": stress_pos,
                   "全局音素覆盖率": round(cov, 4),
                   # 问题分类计数（10-28）：三类问题要采取的行动不同，
                   # 混在一起报，用户不知道该练发音还是该练清晰度。
                   "问题分类": {"口音型": len(_cats.get("口音型", [])),
                                "未读出": len(_cats.get("未读出", [])),
                                "读太轻": len(_cats.get("读太轻", [])),
                                "其他替换": len(_cats.get("替换", []))},
                   # 母语者基准：同口径下「工具自己会错多少」。
                   # 有了它，报告里的分数才有参照 —— 否则用户只能跟满分比，
                   # 以为差 40 分，实际母语者读同一段也差这个数。
                   "母语者基准": baseline},
        # 单词级纠音（10-30 加）：转写模仿音频成文字，跟原文本逐词对齐。
        #   逐词条目 {类型: 读对/读错/漏读/多读, 原词, 实读}，按句号归组。
        #   展示辅助，不参与评分。
        "模仿转写": my_transcript,
        "单词级纠音": word_align,
        # 强制对齐（10-30 起：发音分/音变分/差异明细的唯一口径）：
        #   逐音后验 + 可疑音清单。fa_result["模仿"] / ["原音频"] 各含 逐音后验 / 可疑。
        "强制对齐": fa_result,
        "强制对齐统计": fa_stat,
        "参考音素可用": True,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # ---- 写 节奏对比.json（schema 同单句版，overall_score.py 直接可用）----
    net_r = ((blocks_r_all[-1]["end_ms"] - blocks_r_all[0]["start_ms"]) / 1000.0
             if blocks_r_all else 0.0)
    net_m = ((blocks_m_all[-1]["end_ms"] - blocks_m_all[0]["start_ms"]) / 1000.0
             if blocks_m_all else 0.0)
    # 停顿：只在句内统计（句间停顿两侧边界不一致，跨句比较无意义）
    pauses_r_all = cr.find_boundary_pauses(blocks_r_all)[0] if blocks_r_all else []
    pauses_m_all = cr.find_boundary_pauses(blocks_m_all)[0] if blocks_m_all else []
    nr = {i + 1: "+".join(b["词"]) for i, b in enumerate(blocks_r_all)}
    nm = {i + 1: "+".join(b["词"]) for i, b in enumerate(blocks_m_all)}
    rhythm_payload = {
        "节奏": {
            "数据源": "compare_long.py 逐句对称词块（长音频模式）",
            "整句净时长": {"原音频": round(net_r, 3), "我的": round(net_m, 3),
                           "相对差": round((net_m - net_r) / net_r, 4) if net_r > 0 else 0},
            "词块数": {"原音频": len(blocks_r_all), "我的": len(blocks_m_all)},
            "序号对齐风险": (len(blocks_r_all) != len(blocks_m_all)),
            "句子边界停顿": {
                "原音频处数": len(pauses_r_all), "我的处数": len(pauses_m_all),
                "原音频": pauses_r_all, "我的": pauses_m_all,
                "换人处_预期停顿": {"原音频": [], "我的": []},
                "说明": "长音频模式：只统计句内停顿（句间停顿两侧边界不一致，不跨句比）",
            },
            "词块时长": items,
        },
        "重音_信号": {
            "方法": "emphases（神经网络 prominence 模型，真人母语者标注训练）",
            "阈值": es.THRESHOLD,
            "原音频重读词": [{"序号": i, "词": nr.get(i, "?")} for i in marked_r],
            "我的重读词": [{"序号": i, "词": nm.get(i, "?")} for i in marked_m],
            "漏（原音频有）": [{"序号": i, "词": nr.get(i, "?")} for i in miss_s],
            "多（我的）": [{"序号": i, "词": nm.get(i, "?")} for i in extra_s],
            "命中": [{"序号": i, "词": nr.get(i, "?")} for i in hit_s],
            "逐词分数": {"原音频": [round(float(s), 4) for s in scores_r],
                         "我的": [round(float(s), 4) for s in scores_m]},
            "块序号到句号": block_sent_map,
            "按句重音": build_per_sentence_stress(
                block_sent_map, marked_r, marked_m, nr, nm, sent_results),
            "说明": "长音频模式：emphases 逐词 prominence 分数，> 阈值判重读",
        },
        "词表": {"原音频": [{"词": b["词"], "start_ms": b["start_ms"], "end_ms": b["end_ms"],
                             "连读": b.get("连读", False), "角色": b.get("角色")}
                            for b in blocks_r_all],
                 "我的": [{"词": b["词"], "start_ms": b["start_ms"], "end_ms": b["end_ms"],
                           "连读": b.get("连读", False), "角色": b.get("角色")}
                          for b in blocks_m_all]},
        "节奏分": round(rhythm_score, 1),
        "语调分": None,
    }
    with open(RHYTHM_JSON, "w", encoding="utf-8") as f:
        json.dump(rhythm_payload, f, ensure_ascii=False, indent=2)

    # ---- 人读报告 ----
    L = []
    L.append("=" * 60)
    L.append("   音素对比报告（长音频·按内容对齐·逐句评测）")
    L.append("=" * 60)
    L.append("")
    L.append("%s  vs  %s" % (short_name(REF_FILE), short_name(MY_FILE)))
    L.append("整段音素相似度：%.1f / 100" % sim)
    L.append("已模仿 %d / %d 句（漏读 %d，碎句忽略 %d）"
             % (len(scored), len(sent_results), len(missed), len(ignored)))
    if is_partial:
        L.append("⚠ 半截模仿：只读到 %.0f%%（后 %d 句未读，不计分）"
                 % (coverage_ratio * 100, tail_missed))
    L.append("")
    L.append("-" * 60)
    L.append("一、问题分类总览（先看哪类最多，再看明细）")
    L.append("-" * 60)
    # _cats 在上面（写 payload 之前）已算好，这里直接用
    L.append("   口音型错误 %d 处（学习者最典型的音变，练发音）"
             % len(_cats.get("口音型", [])))
    L.append("   未读出的音 %d 处（读太轻/漏读，练清晰度）"
             % len(_cats.get("未读出", [])))
    L.append("   读太轻的音 %d 处（读对了但太轻）"
             % len(_cats.get("读太轻", [])))
    L.append("   其他替换   %d 处" % len(_cats.get("替换", [])))
    if baseline:
        L.append("")
        L.append("   母语者基准：同一口径下，原音频自己跟参考音素比，可疑率 %.1f%%"
                 % baseline["可疑率"])
        if sim is not None:
            L.append("   → 你的可疑率比母语者高 %.1f 个百分点（这是真实差距）"
                     % max(0.0, (100 - sim) - baseline["可疑率"]))
    L.append("")
    L.append("-" * 60)
    L.append("二、逐句明细")
    L.append("-" * 60)
    for r in sent_results:
        L.append("  句%-2d %s" % (r["句号"], r["文本"]))
        if r["漏句"]:
            L.append("      [漏句] 这句整句未读，不计分")
        elif r["忽略"]:
            L.append("      [碎句] 参考音素少于 %d 个（转写瑕疵），忽略" % MIN_SENT_PH)
        else:
            L.append("      参考 %d 音 / 模仿 %d 音 ｜ 可疑 %d 音"
                     % (r["参考音素数"], r["模仿音素数"], len(r.get("差异") or [])))
            for d in r["差异"][:6]:
                label = d.get("词") or ""
                tip = ("   " + d["要领"]) if d.get("要领") else ""
                cat = d.get("类别") or ""
                if cat == "未读出":
                    # 明确说「没读出来」——原来写"应读 X"会让人误以为要读 X
                    L.append("      [没读出] %s这个音没出来（%s）%s"
                             % (label, d["应读"], tip))
                elif cat == "读太轻":
                    L.append("      [读太轻] %s应读 %s（读对了但太轻）%s"
                             % (label, d["应读"], tip))
                elif cat == "口音型":
                    L.append("      [口音] %s应读 %s → 实读 %s%s"
                             % (label, d["应读"], d["实读"], tip))
                else:
                    L.append("      [替换] %s应读 %s → 实读 %s%s"
                             % (label, d["应读"], d["实读"], tip))
    # 单词级纠音（10-30 加）：转写模仿音频，跟原文本逐词对齐全量展示。
    #   比音素级更直观 —— 直接告诉你「哪个词读错了/漏了/多读了」。
    if word_align and my_transcript:
        L.append("")
        L.append("-" * 60)
        L.append("三、单词级纠音（你实际读成了哪些词）")
        L.append("-" * 60)
        L.append("   逐词对照：[✓]读对 [✗]读错 [−]漏读 [+]多读")
        L.append("")
        for s in sentences:
            sid = s["句号"]
            items = word_align.get(sid) or []
            if not items:
                continue
            # 原句行 + 你读的行，逐词并排；漏读/多读用占位符对齐
            ref_line, my_line = [], []
            for it in items:
                if it["类型"] == "多读":
                    ref_line.append("·")
                    my_line.append(it["实读"])
                elif it["类型"] == "漏读":
                    ref_line.append(it["原词"])
                    my_line.append("·")
                else:
                    ref_line.append(it["原词"])
                    my_line.append(it["实读"])
            L.append("  句%-2d %s" % (sid, s.get("文本", "")))
            L.append("      原：%s" % " ".join(ref_line))
            L.append("      我：%s" % " ".join(my_line))
            # 只把有问题的词标出来
            bad = [it for it in items if it["类型"] != "读对"]
            if bad:
                marks = []
                for it in bad:
                    if it["类型"] == "漏读":
                        marks.append("漏读「%s」" % it["原词"])
                    elif it["类型"] == "多读":
                        marks.append("多读「%s」" % it["实读"])
                    else:
                        marks.append("「%s」读成「%s」" % (it["原词"], it["实读"]))
                L.append("      ⚠ %s" % "；".join(marks))
            L.append("")
    # 口音型专段：把该练的音变集中列出来，一眼能抄下来练
    if _cats.get("口音型"):
        L.append("")
        L.append("-" * 60)
        L.append("四、口音型错误专段（这些是中国学习者最典型的音变，优先练）")
        L.append("-" * 60)
        for sid, d in _cats["口音型"][:30]:
            L.append("   句%-2d %s应读 %s → 实读 %s"
                     % (sid, d.get("词") or "", d["应读"], d["实读"]))
            if d.get("要领"):
                L.append("        %s" % d["要领"])
        if len(_cats["口音型"]) > 30:
            L.append("   （还有 %d 处，见上方逐句明细）"
                     % (len(_cats["口音型"]) - 30))
    if missed:
        L.append("")
        L.append("-" * 60)
        L.append("五、漏句清单（未读到的句子，不扣分）")
        L.append("-" * 60)
        for r in missed:
            L.append("   句%d：%s" % (r["句号"], r["文本"]))
    L.append("")
    L.append("-" * 60)
    L.append("六、重音（emphases）：每个词的重读分数 + 纠错结论")
    L.append("-" * 60)
    L.append("")
    _nr = {i + 1: "+".join(b["词"]) for i, b in enumerate(blocks_r_all)}
    _nm = {i + 1: "+".join(b["词"]) for i, b in enumerate(blocks_m_all)}
    L.append("判定口径：emphases 神经网络逐词 prominence 分数（> %.2f 判重读）。"
             % es.THRESHOLD)
    L.append("")
    if blocks_r_all or blocks_m_all:
        L.append("  【原音频】%d 词，重读 %d 词 ｜【我的模仿】%d 词，重读 %d 词"
                 % (len(blocks_r_all), len(marked_r), len(blocks_m_all), len(marked_m)))
        L.append("")
        for i in range(max(len(blocks_r_all), len(blocks_m_all))):
            no = i + 1
            word = _nr.get(no, _nm.get(no, "?"))
            sr_v = scores_r[i] if i < len(scores_r) else None
            sm_v = scores_m[i] if i < len(scores_m) else None
            flag = "M" if no in miss_s else ("X" if no in extra_s
                                             else ("★" if no in marked_r else "○"))
            L.append("    %-4d %-14s %8s %8s   %s" % (no, word,
                      ("%.3f" % sr_v) if sr_v is not None else "-",
                      ("%.3f" % sm_v) if sm_v is not None else "-", flag))
        L.append("    标记：★ 两侧都重读  ○ 都不重读  M 原重读你漏了  X 你多读的")
        L.append("")
        L.append("纠错结论：")
        L.extend(cr.render_stress_verdict(_nr, _nm, marked_r, marked_m))
        L.append("")
        L.append("重点词清单（照这个单子练）：")
        if miss_s:
            L.append("   漏读重音（该重读没重读）：%s" % "、".join(_nr[i] for i in miss_s))
        if extra_s:
            L.append("   多读重音（不该重读却重读）：%s" % "、".join(_nm[i] for i in extra_s))
        if not miss_s and not extra_s:
            L.append("   （重音位置完全命中，无漏无多）")
    else:
        L.append("   （无已读句，无重音判定）")
    L.append("")
    L.append("-" * 60)
    L.append("七、汇总（仅按已模仿句）")
    L.append("-" * 60)
    L.append("   发音分 %s / 100 ｜ 音变分 %s / 100"
             % (("%.1f" % pron) if pron is not None else "—",
                ("%.1f" % change) if change is not None else "—"))
    L.append("   节奏分 %.1f / 100 ｜ 重音位置 %s / 100"
             % (rhythm_score, ("%.1f" % stress_pos) if stress_pos is not None else "—"))
    L.append("   净时长：原音频 %.2fs ｜ 你 %.2fs ｜ 停顿 原 %d 处 / 你 %d 处"
             % (net_r, net_m, len(pauses_r_all), len(pauses_m_all)))
    L.append("   重音：原音频突出 %d 块 ｜ 你 %d 块 ｜ 命中 %d ｜ 漏 %d ｜ 多 %d"
             % (len(marked_r), len(marked_m), len(hit_s), len(miss_s), len(extra_s)))
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    out("")
    out("报告（人看）：音素对比_报告.txt")
    out("数据（机器）：音素对比.json")
    log_done("长音频评测")
    return 0


if __name__ == "__main__":
    try:
        result = main()
    except Exception:
        import traceback
        log_error(traceback.format_exc())
        result = 1
    try:
        input("按回车退出 ...")
    except Exception:
        pass
    sys.exit(result)
