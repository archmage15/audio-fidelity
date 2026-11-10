# -*- coding: utf-8 -*-
"""还原度对比 · 总装：综合评分 + 纠音报告

把前两步的产物合起来算一个"综合还原度"：
  ① 音素对比.json  → 发音分 + 音变分（都按强制对齐口径）
  ② 节奏对比.json  → 节奏分 + 重音位置分

权重（用户定调：发音最重要，其余持平）：
  发音 40% ｜ 音变 25% ｜ 节奏 20% ｜ 重音位置 15%

产出（两份，职责分开，都写在 output\\ 里）：
  - 还原度综合评分.json   给机器/AI：各项分值 + 权重 + 明细
  - 还原度综合报告.txt    给人看：综合分 + 分项分 + 问题清单 + 改进建议

目录约定（本脚本在 scripts\\ 里，用 ROOT 指回上一层）：
  audio\\   放录音     output\\  放结果（前两步写这里，本脚本从这里读）

依赖：只用标准库（读 json、写 txt），不需要 torch —— 秒出结果。
用法：先跑 音素对比 / 节奏对比（或直接双击 一键跑还原度.bat），再跑本脚本。
"""
import datetime
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---- 目录约定 ----
# 本脚本位于 <项目>\scripts\ 下，ROOT = 项目根（上一层）
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.environ.get("AUDIO_TEST_OUTPUT_DIR") or os.path.join(ROOT, "output")
os.makedirs(OUT_DIR, exist_ok=True)

PHONEME_JSON = os.path.join(OUT_DIR, "音素对比.json")
RHYTHM_JSON = os.path.join(OUT_DIR, "节奏对比.json")
OUT_JSON = os.path.join(OUT_DIR, "还原度综合评分.json")
OUT_REPORT = os.path.join(OUT_DIR, "还原度综合报告.txt")
HISTORY_JSON = os.path.join(OUT_DIR, "history.json")

# ---- 维度定义：全局唯一来源 ----
# 顺序 = 展示顺序；权重必须和为 1。
# 本文件不再出现第二处维度列表 —— 终端简报、报告、改进建议、payload 全部迭代它。
# 加/删维度：改这里 + collect_parts()，两处即可。
DIMENSIONS = [
    ("发音", 0.40),
    ("音变", 0.25),
    ("节奏", 0.20),
    ("重音位置", 0.15),
]
# 注：语调暂不出分（计划书 §4）—— 不进 DIMENSIONS、不参与加权；
#    语调分仍从 rhythm["语调分"] 读，仅供报告显示。

# ---- 评测模式（history.json 必须记，否则跨模式比分会误导）----
# 10-28 并轨后：单句 = 长文本切 1 句，所有音频统一走 compare_long，
# 不再有独立的单句链路。所以新记录统一记 MODE_LONG。
# 旧历史里「单句」模式的记录（compare_phoneme 旧口径）自动与新记录隔离
# —— 模式字段不同，find_prev_same_mode 不会拿它们比。
MODE_LONG = "长音频"
MODE_SINGLE = "单句"

# ---- 口径版本（history.json 必须记，否则跨版本比分会误导）----
# 为什么需要它（10-28 补的欠账）：
#   同一个「模式」下，**评分口径本身也会变**。10-26 修了「espeak 连读合并
#   导致词边界错位」的 P0 bug —— 修之前，报告里「单词 X 实读 Y」的指控
#   指错了词（"the" 头上装的是 lesson 的音）。修之后归因口径变了。
#   但 10-21 那条 60.6 分是**错位口径**算出来的，跟修复后的 60.3 分
#   直接相减，就会得出"我进步了"或"我退步了"的错误结论 ——
#   其实只是口径变了，跟练得好不好没关系。
#
# 纪律同「模式」隔离：**跨口径版本一律不对比**。
#   旧记录没有「口径版本」字段 → 视为未知版本 → 一律跳过。
#   宁可不给对比，也不给错的对比。
#
# 什么时候要 +1：
#   改了任何影响「分数怎么算」或「差异怎么归因」的东西时。
#   纯 UI 改动、纯文案改动 → 不用 +1。
SCORE_VERSION = 7
SCORE_VERSION_NOTE = (
    "7=10-30 删 difflib 序列比对，音变分改强制对齐口径（没读出来+读太轻/参考音素数），"
    "母语者基准改可疑率口径，多余音删除（difflib 错位假象）；"
    "6=10-30 修长音频「原生音数」多加了「多余」的音变分口径 bug（分母 301→280，"
    "音变分 86→85，与 compare_long 单句模式 len(ref) 对齐）；"
    "5=10-30 标点恢复断句（讯飞句号→逗号导致三句并一句，句子边界 13→16，漏句判断口径变）；"
    "4=10-30 讯飞错字修复生效（wellsort→well sort，词边界 87→88，节奏/重音/音变归因变）；"
    "3=10-29 发音分改用强制对齐口径（口音型+其他+读太轻 / 参考音素数）；"
    "2=10-26 修词边界错位（espeak 连读合并 → DP 对齐重建词边界）；"
    "1=修之前（词边界错位，词级指控不可信）"
)


def detect_mode(phoneme):
    """从音素对比.json 判定本次是长音频还是单句模式。

    判据用 `is not None` 而非真值判断：compare_long.py 一定会写「长音频」段
    （内含 模式/句子/四个分项等 7 个键），但若哪天它写了空段，
    `phoneme.get("长音频") or None` 会把"存在但为空"误判成单句。

    10-28 并轨后：所有音频都走 compare_long，所以实际上总是返回 MODE_LONG。
    MODE_SINGLE 分支保留作兜底（万一 compare_long 没写「长音频」段）。
    """
    return MODE_LONG if phoneme.get("长音频") is not None else MODE_SINGLE


# ---- 日志（统一模块，带时间戳；详细模式见 logkit.py 顶部说明）----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logkit import log, out, log_setup, log_done, log_step, log_error, VERBOSE


def read_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def read_history():
    """读 history.json（历史分数列表）；不存在/损坏返回空列表。"""
    if not os.path.exists(HISTORY_JSON):
        return []
    try:
        with open(HISTORY_JSON, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def find_prev_same_mode(hist, mode, version=None):
    """在历史里**往回找**最近一条同模式、**同口径版本**的记录。

    为什么不能直接用 hist[-2]（旧版做法）：
      单句（4秒）与长音频（72秒）的综合分口径不同，跨模式比会得出
      "进步了 +4.0" 的假结论 —— 分数变化其实来自素材变化，不是练得更好。
      宁可不给对比，也不给错的对比。
    旧记录没有「模式」字段 → 一律跳过（无法判断模式 = 不敢用来比）。

    口径版本隔离（10-28 加）：
      同一模式下**评分/归因口径也会变**。10-26 修了词边界错位 bug，
      修之前的记录是「错位口径」算的分数，跟修之后的直接相减会得出
      "我进步了"的假结论 —— 其实只是口径变了。
      所以：版本不同 → 跳过；旧记录没有「口径版本」字段 → 也跳过
      （不知道是哪个版本 = 不敢用来比）。

    参数 version 传 None → 只按模式找（用于"能不能比"的诊断，不用于出对比）。
    """
    for entry in reversed(hist):
        if not isinstance(entry, dict):
            continue
        if entry.get("模式") != mode:
            continue
        if version is not None and entry.get("口径版本") != version:
            continue
        return entry
    return None


def count_comparable(hist, mode, version=None):
    """统计历史里有几条能用于对比（同模式 + 同口径版本）。"""
    n = 0
    for entry in hist:
        if not isinstance(entry, dict):
            continue
        if entry.get("模式") != mode:
            continue
        if version is not None and entry.get("口径版本") != version:
            continue
        n += 1
    return n


def append_history(overall, parts, mode):
    """把本次综合分追加进 history.json，供报告显示「上次 vs 这次」的进步。

    mode 必须记（单句/长音频）—— 否则报告会把两种模式混着比。
    口径版本也必须记 —— 否则改过评分/归因口径后，新旧分数会被直接相减，
    得出"我进步了"的假结论（10-28 踩过：修词边界 bug 前后 60.6 vs 60.3）。
    """
    entry = {
        "时间": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "综合分": round(overall, 1),
        "模式": mode,
        "口径版本": SCORE_VERSION,
        "分项": {k: (round(parts[k], 1) if parts.get(k) is not None else None)
                  for k, _w in DIMENSIONS},
    }
    hist = read_history()
    hist.append(entry)
    try:
        with open(HISTORY_JSON, "w", encoding="utf-8") as f:
            json.dump(hist, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log("（写 history.json 失败，不影响评分：%s）" % e, show=True)


def score_pronunciation(fa_stat):
    """发音分（10-29 起改用强制对齐口径）：可疑音越少越高。
    可疑 = 口音型 + 其他替换 + 读太轻（「没读出来」归音变，不扣这里）。
    分母 = 参考音素数（强制对齐逐音定位，天然按该读的音总数计）。"""
    if not fa_stat:
        return None
    total = fa_stat.get("参考音素数") or 0
    wrong = ((fa_stat.get("口音型") or 0)
             + (fa_stat.get("其他") or 0)
             + (fa_stat.get("读太轻") or 0))
    base = max(total, 1)
    return max(0.0, (1 - wrong / base) * 100)


def score_phoneme_change(fa_stat):
    """音变分（10-30 起改用强制对齐口径）：吞音/漏读/读太轻越少越高。

    公式 = 1 − (没读出来 + 读太轻) / 参考音素数。
    「没读出来」= 强制对齐后验低且该帧是静音/特殊符（吞音漏读）；
    「读太轻」= 后验低但贪心仍是它自己（读对了但太轻，属弱化）。
    与发音分共享同一份「强制对齐统计」，分母都是参考音素数。
    ⚠️ 「多余音」已删除：difflib 的多余是整词错位假象，强制对齐单向定位
    又算不了多余音，故不再有这一项（已知局限，见报告标注）。
    """
    if not fa_stat:
        return None
    total = fa_stat.get("参考音素数") or 0
    wrong = (fa_stat.get("没读出来") or 0) + (fa_stat.get("读太轻") or 0)
    base = max(total, 1)
    return max(0.0, (1 - wrong / base) * 100)


def grade(score):
    if score is None:
        return "—"
    if score >= 90:
        return "优"
    if score >= 80:
        return "良"
    if score >= 70:
        return "中"
    if score >= 60:
        return "及格"
    return "待加强"


# 各维度低于门槛时的针对性建议（键须与 DIMENSIONS 对齐；新增维度时在这里补文案）
ADVICE = {
    "发音": "发音是当前短板：对照报告里【强制对齐逐音纠音】的可疑音，逐个纠音（注意长短音、θ/ð、r/l）。",
    "音变": "音变（连读/弱化/吞音）偏弱：留意原音频里被“吃掉”或黏连的音，先慢读再连读。",
    "语调": "语调（音高起伏）偏弱：模仿原音频的升降调，疑问句尾音上扬、陈述句尾音下压。",
    "节奏": "节奏偏弱：跟着原音频打拍子，注意停顿位置和每句的重音落点。",
    "重音位置": "重音落点偏弱：原音频强调的词你没强调，留意原音频哪几个词读得重、跟着加重。",
}


# ---- 读 节奏对比.json 里「真实存在」的键 ----
# 旧字段 总时长秒 / 对齐 / 停顿 / 子项分 已失效（现网 JSON 里不存在）。
# 若继续用 .get(..., 0) 兜底，会静默变成 0 —— 所以这里取不到就告警。
def rhythm_section(rhythm, key):
    """取 节奏.<key>；取不到则记日志告警并返回 None。"""
    if not rhythm:
        return None
    sec = (rhythm.get("节奏") or {}).get(key)
    if sec is None:
        log("⚠ 节奏对比.json 里没有 节奏.%s（字段失效或脚本未更新），该项显示为「—」" % key,
            show=True)
    return sec


def collect_parts(phoneme, rhythm):
    """按维度取分：键须与 DIMENSIONS 对齐。加维度时改这里（全局唯一取数处）。"""
    stat = phoneme.get("对齐统计")
    fa_stat = phoneme.get("强制对齐统计")
    # 长音频模式：只信 compare_long.py 写的 节奏对比.json（数据源带标记），
    # 防止把上一次单句运行的陈旧数据混进长音频报告（实测踩过）。
    long_mode = bool(phoneme.get("长音频"))
    src = ((rhythm or {}).get("节奏") or {}).get("数据源") or ""
    rhythm_ok = bool(rhythm) and (not long_mode or "compare_long" in src)
    return {
        "发音": score_pronunciation(fa_stat),
        "音变": score_phoneme_change(fa_stat),
        "语调": rhythm.get("语调分") if rhythm_ok else None,
        "节奏": rhythm.get("节奏分") if rhythm_ok else None,
        "重音位置": score_stress_position(rhythm) if rhythm_ok else None,
    }


def score_stress_position(rhythm):
    """重音位置分 = (1 − 漏数 / max(原音频突出块数,1)) × 100（计划书 §4 公式）。
    漏 = 原音频突出、模仿没突出的块。⚠️ 只罚「漏」，不罚「多」——
    多突出可能是感情色彩，漏原音频强调的词才是没还原。"""
    if not rhythm:
        return None
    sig = rhythm.get("重音_信号") or {}
    miss = sig.get("漏（原音频有）") or []
    ref = sig.get("原音频重读词") or []
    if not ref:
        return None       # 原音频一个重读词都没判出来 → 没基准，不出分
    return round(max(0.0, (1.0 - len(miss) / float(max(len(ref), 1))) * 100.0), 1)


def build_advice(parts, phoneme, rhythm):
    """根据分项分与明细生成"下一步怎么练"的建议。维度顺序来自 DIMENSIONS。"""
    tips = []
    lows = [k for k, _w in DIMENSIONS if parts.get(k) is not None and parts[k] < 70]
    if not lows:
        tips.append("各维度都在 70 分以上，整体还原度不错，可以练更长、更快的句子。")
    for k, _w in DIMENSIONS:
        v = parts.get(k)
        if v is not None and v < 75 and k in ADVICE:
            tips.append(ADVICE[k])
    # 从停顿明细再补一条针对性提示（真实键：节奏.句子边界停顿）
    ps = rhythm_section(rhythm, "句子边界停顿")
    if ps and ps.get("原音频处数") != ps.get("我的处数"):
        tips.append("停顿数量与原音频不一致（原音频 %s 处 / 你 %s 处），注意不要多停或少停。"
                    % (ps.get("原音频处数"), ps.get("我的处数")))
    return tips


def main():
    log_setup("综合评分")
    out("=" * 62)
    out("   还原度综合评分")
    out("=" * 62)
    out("")

    log_step("读取前两步的产出")
    phoneme = read_json(PHONEME_JSON)
    rhythm = read_json(RHYTHM_JSON)
    if phoneme is None:
        out("X 缺少 音素对比.json —— 请先跑 compare_phoneme.py（或一键 bat）。")
        log_done("（缺前置数据，提前退出）")
        return 1
    long_mode = bool(phoneme.get("长音频"))
    if rhythm is None:
        if long_mode:
            # 长音频模式但没跑出 节奏对比.json：用空 dict 占位
            # （collect_parts 里 rhythm_ok 会把它判为不可信，相关项不参与加权）
            rhythm = {}
        else:
            out("X 缺少 节奏对比.json —— 请先跑 compare_rhythm.py（或一键 bat）。")
            log_done("（缺前置数据，提前退出）")
            return 1
    if long_mode:
        log("   长音频模式：发音/音变/节奏/重音 均由 compare_long.py 逐句评测汇总", verbose=True)
    log("   音素对比.json：相似度 %s，差异 %d 条"
        % (phoneme.get("相似度"), len(phoneme.get("差异") or [])), verbose=True)
    log("   节奏对比.json：节奏 %s ｜ 语调 %s ｜ 重音位置 %s"
        % (rhythm.get("节奏分"), rhythm.get("语调分"),
           ((rhythm.get("重音_信号") or {}).get("原音频重读词") and "有" or "—")), verbose=True)

    log_step("计算分项分与综合分")
    stat = phoneme.get("对齐统计")
    fa_stat = phoneme.get("强制对齐统计")
    parts = collect_parts(phoneme, rhythm)

    # 综合分：只有拿到的项按权重归一（缺项不参与，避免误判为 0）
    avail = {k: parts[k] for k, _w in DIMENSIONS if parts.get(k) is not None}
    wsum = sum(w for k, w in DIMENSIONS if k in avail)
    overall = sum(w * parts[k] for k, w in DIMENSIONS if k in avail) / wsum if wsum else 0.0
    # 有效权重 = 额定权重在「缺项」后归一的结果（与 payload.权重 的额定值口径不同）
    eff_weights = ({k: round(w / wsum, 4) for k, w in DIMENSIONS if k in avail}
                   if wsum else {})

    # ---- 终端简报 ----
    out("综合还原度：%.1f / 100   （等级：%s）" % (overall, grade(overall)))
    out("")
    out("分项：")
    for k, w in DIMENSIONS:
        v = parts.get(k)
        out("   %s  %s  / 100   (权重 %d%%)"
            % (k, ("%.1f" % v) if v is not None else "  —", int(w * 100)))
    out("")
    out("明细参考：")
    if fa_stat:
        out("   发音/音变（强制对齐口径）：可疑 %d 音（口音型 %d ｜ 没读出来 %d ｜ "
            "读太轻 %d ｜ 其他 %d）／ 参考音素 %d"
            % (fa_stat.get("可疑数", 0), fa_stat.get("口音型", 0),
               fa_stat.get("没读出来", 0), fa_stat.get("读太轻", 0),
               fa_stat.get("其他", 0), fa_stat.get("参考音素数", 0)))
    elif stat:
        out("   音素（对齐统计）：原生 %d 音 ｜ 模仿 %d 音"
            % (stat.get("原生音数", 0), stat.get("模仿音数", 0)))
    if rhythm and not phoneme.get("长音频"):
        net = rhythm_section(rhythm, "整句净时长") or {}
        ps = rhythm_section(rhythm, "句子边界停顿") or {}
        st = rhythm.get("重音_信号") or {}
        if net:
            out("   节奏：净时长 原 %.2fs / 我 %.2fs（相对差 %+.1f%%）"
                % (net.get("原音频", 0), net.get("我的", 0), (net.get("相对差") or 0) * 100))
        if ps:
            out("   停顿：原音频 %s 处 / 我 %s 处"
                % (ps.get("原音频处数", "—"), ps.get("我的处数", "—")))
        if st:
            out("   重音：重读词 原 %d / 我 %d（命中 %d ｜ 漏 %d ｜ 多 %d）"
                % (len(st.get("原音频重读词") or []), len(st.get("我的重读词") or []),
                   len(st.get("命中") or []), len(st.get("漏（原音频有）") or []),
                   len(st.get("多（我的）") or [])))

    # ---- 写 JSON（给机器）----
    log_step("写综合评分与报告")
    # 注意：这里叫 payload 不叫 out —— out 是 logkit 的结果行函数，
    # 若用同名局部变量会遮蔽它，导致 main() 前面调用 out() 报 UnboundLocalError。
    from audio_paths import find_audio_pair
    source_reference, source_imitation = find_audio_pair()
    payload = {
        "输入音频": {"原音频": source_reference, "模仿音频": source_imitation},
        "综合还原度": round(overall, 1),
        "等级": grade(overall),
        "权重": dict(DIMENSIONS),
        "权重(有效)": eff_weights,
        "分项分": {k: (round(parts[k], 1) if parts.get(k) is not None else None)
                   for k, _w in DIMENSIONS},
        "音素明细": stat,
        "节奏明细": rhythm,
        "音素相似度(原始)": phoneme.get("相似度"),
        "母语者基准": (phoneme.get("长音频") or {}).get("母语者基准"),
    }
    try:
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log("（写 json 失败：%s）" % e, show=True)

    # ---- 记历史（供报告显示进步）----

    # ---- 写报告（给人）----
    try:
        write_report(overall, parts, phoneme, rhythm)
    except Exception as e:
        log("（写报告失败：%s）" % e, show=True)

    out("")
    out("数据（机器）：还原度综合评分.json")
    out("报告（人看）：还原度综合报告.txt")
    append_history(overall, parts, detect_mode(phoneme))
    log_done("综合评分")
    return 0


def _fmt_sent_diff(d):
    """把一句的结构化差异 dict 格式化成一行（与 compare_long 差异行口径一致）。"""
    who = (d.get("词") or "").strip()
    cat = d.get("类别") or "替换"
    tip = (d.get("要领") or "").strip()
    ref = (d.get("应读") or "").strip()
    got = (d.get("实读") or "").strip()
    if cat == "口音型":
        line = "[口音] %s应读 %s：实读 %s" % (who, ref, got)
        if tip:
            line += "   " + tip
        return line
    if cat == "未读出":
        line = "[没读出] %s应读 %s：没读出来" % (who, ref)
        if tip:
            line += "   " + tip
        return line
    if cat == "读太轻":
        line = "[读太轻] %s应读 %s：读对了但太轻" % (who, ref)
        if tip:
            line += "   " + tip
        return line
    if cat == "多读出":
        return "[多读出] %s：实读 %s" % (who, got)
    # 其他替换
    line = "[替换] %s应读 %s：实读 %s" % (who, ref, got)
    if tip:
        line += "   " + tip
    return line


def _mark_stressed_words(text, stressed, marker="【】"):
    """在句子里把「重读的词」用标记框出来。

    text 是原句文本，stressed 是应框的词列表（与重音块的「词」对应）。
    逐个词查找（不区分大小写、按词边界），命中则替换为该词的【】框住版本。
    找不到的词原样保留在 stressed 里不影响其它词。
    """
    if not stressed:
        return text
    words = text.split()
    hit = set(s.lower() for s in stressed)
    marked = []
    for w in words:
        key = w.strip(".,!?;:'\"")
        if key.lower() in hit:
            marked.append(marker[0] + key + marker[1])
        else:
            marked.append(w)
    return " ".join(marked)


def _render_stress_by_sentence(rhythm):
    """把按句重音渲染成「每句三行」结构，返回 list[str]。

    每句：
      ① 应读：原句，把「应该读出的重音词」用【】框出
      ② 实读：原句，把「我实际读出的重音词」用【】框出
      ③ 纠音：一句话点出漏了哪些、多了哪些
    无按句数据（单句模式或旧 JSON）→ 返回空列表。
    """
    st = (rhythm or {}).get("重音_信号") or {}
    per_sent = st.get("按句重音") or []
    if not per_sent:
        return []
    L = []
    for s in per_sent:
        text = s.get("文本") or ""
        should = s.get("应读重音") or []
        got = s.get("实读重音") or []
        missed = s.get("漏读重音") or []
        extra = s.get("多读重音") or []
        L.append("    句%d  %s" % (s.get("句号"), text))
        if should:
            L.append("        应读重音：%s  →  %s"
                     % ("、".join("【%s】" % w for w in should),
                        _mark_stressed_words(text, should)))
        else:
            L.append("        应读重音：（这句没有需要重读的词，按原样读即可）")
        if got:
            L.append("        你读的重音：%s  →  %s"
                     % ("、".join("【%s】" % w for w in got),
                        _mark_stressed_words(text, got)))
        elif not should:
            L.append("        你读的重音：（你也没乱加重，正确）")
        else:
            L.append("        你读的重音：（一个都没读出来）")
        # 纠音：漏了 + 多了 分开说
        if missed:
            L.append("        ➜ 漏了：该重读的「%s」你没加重 —— 把这几个词读得比周围更响/更长。"
                     % "、".join(missed))
        if extra:
            L.append("        ➜ 多了：不该重读的「%s」你加重了 —— 这几个词读轻一点、带过去。"
                     % "、".join(extra))
        if not missed and not extra:
            L.append("        ➜ 重音位置完全一致，很棒。")
    return L


def write_report(overall, parts, phoneme, rhythm):
    L = []
    L.append("=" * 60)
    L.append("   还原度综合报告")
    L.append("=" * 60)
    L.append("")
    from audio_paths import find_audio_pair, short_name
    _ref, _my = find_audio_pair()
    L.append("原音频路径：%s" % _ref)
    L.append("模仿音频路径：%s" % _my)
    L.append("%s  vs  %s（跟读还原度）" % (short_name(_ref), short_name(_my)))
    L.append("")
    # 半截模仿提示：模仿只覆盖原音频一部分时，明确告诉用户本次只按已模仿部分评分
    coverage = (phoneme.get("半截模仿") or {})
    if coverage.get("半截"):
        long_info = phoneme.get("长音频") or {}
        if long_info:
            # 长音频格式：按句报
            L.append("⚠ 检测到「半截模仿」：你只跟读到 %d / %d 句（%.0f%%）。"
                     % (coverage.get("已读句数", 0), coverage.get("总句数", 0),
                        coverage.get("覆盖比例", 0) * 100))
            missed = coverage.get("漏句") or []
            if missed:
                L.append("   未读到的句子（不计分）：")
                for m in missed:
                    L.append("     · 句%s：%s" % (m.get("句号"), m.get("文本", "")))
        else:
            head = coverage.get("头部漏读词") or []
            tail = coverage.get("尾巴漏读词") or []
            L.append("⚠ 检测到「半截模仿」：你只跟读到原音频的 %.1f%%。" % (coverage.get("覆盖比例", 0) * 100))
            if head:
                L.append("   开头没读到的部分：%s" % " ".join(head))
            if tail:
                L.append("   结尾没读到的部分：%s" % " ".join(tail))
        L.append("   本次只按已模仿到的部分评分，未读到的部分不算错。")
        L.append("")
    L.append("综合还原度：%.1f / 100     等级：%s" % (overall, grade(overall)))
    # 母语者基准（10-28 加）：让分数有参照。
    #   没有它，用户只能跟满分比 —— 60 分看着像失败，
    #   但母语者读同一段在同一口径下也只有 88.4，真实差距是 28 分不是 40 分。
    _bl = (phoneme.get("长音频") or {}).get("母语者基准")
    if _bl and _bl.get("可疑率") is not None:
        _sim = phoneme.get("相似度")
        if _sim is not None:
            L.append("（母语者基准：可疑率 %.1f%% —— 原音频自己跟参考音素比，"
                     "是工具的误差底线；你的可疑率比母语者高 %.1f 个百分点）"
                     % (_bl["可疑率"], max(0.0, (100 - _sim) - _bl["可疑率"])))
        else:
            L.append("（母语者基准：可疑率 %.1f%% —— 原音频自己跟参考音素比，"
                     "是工具的误差底线）" % _bl["可疑率"])
    # 进步对比：**只跟同模式 + 同口径版本的历史记录比**。
    #   单句(≤20s) 与长音频(逐句评测) 的素材长度、评分口径都不同，
    #   跨模式比会写出"进步 +4.0"这种假结论（分数变化来自素材变化）。
    #   同理，改过归因口径后（10-26 修词边界错位）新旧分数也不能直接比。
    #   找不到可比的记录就不显示，并说明原因 —— 宁可不给对比，也不给错的对比。
    mode = detect_mode(phoneme)
    L.append("（本次：%s模式 · 口径版本 %d）" % (mode, SCORE_VERSION))
    hist = read_history()
    prev = find_prev_same_mode(hist, mode, SCORE_VERSION)
    if prev and prev.get("综合分") is not None:
        prev_score = prev["综合分"]
        diff = round(overall, 1) - prev_score
        if abs(diff) < 0.05:
            delta = "持平"
        else:
            delta = ("↑ %+.1f" if diff > 0 else "↓ %+.1f") % diff
        L.append("   上次%s模式 %.1f 分（%s）→ 这次 %.1f 分，%s"
                 % (mode, prev_score, prev.get("时间", "?"), overall, delta))
    else:
        same_mode = count_comparable(hist, mode)          # 同模式（不限版本）
        same_ver = count_comparable(hist, mode, SCORE_VERSION)
        if same_mode > 0 and same_ver == 0:
            L.append("   暂无可比的历史记录 —— 已跑过 %d 次%s模式，"
                     "但都是**旧口径版本**（评分/归因方式已改），"
                     "直接比会得出错的结论，所以不比。"
                     % (same_mode, mode))
            L.append("   （下次跑就有基准了）")
        else:
            L.append("   暂无同模式（%s）的历史记录可对比 —— 已跑过 %d 次，"
                     "其中能用于对比的是 %d 次。"
                     % (mode, len(hist), same_ver))
    L.append("")
    L.append("-" * 60)
    L.append("一、分项得分（发音最重，其余持平）")
    L.append("-" * 60)
    for k, w in DIMENSIONS:
        v = parts.get(k)
        bar = "—" if v is None else "%.1f" % v
        L.append("   %s  %6s / 100    权重 %d%%    %s"
                 % (k, bar, int(w * 100), grade(v) if v is not None else ""))
    L.append("")

    # 二、发音 + 音变纠音（强制对齐口径，发音分/音变分都按这里算）
    L.append("-" * 60)
    L.append("二、发音 + 音变纠音（强制对齐口径 · 发音分/音变分按这个算）")
    L.append("-" * 60)
    L.append("   口径：拿参考音素表去录音里逐音定位，每个音给「像不像」分（后验 < 0.5 判可疑）。")
    L.append("   发音分 = 1 −（口音型 + 其他 + 读太轻）/ 参考音素数；")
    L.append("   音变分 = 1 −（没读出来 + 读太轻）/ 参考音素数。")
    L.append("   ※ 不再有「多读出」这一项：difflib 的“多余”是整词错位假象，")
    L.append("     强制对齐是单向定位、算不了多余音，故直接去掉（已知局限）。")
    L.append("")
    diffs = phoneme.get("差异") or []
    # 分类计数：四类问题行动不同，先给总览
    _cats = (phoneme.get("长音频") or {}).get("问题分类")
    if _cats:
        L.append("   口音型错误 %d 处 —— 学习者最典型的音变，优先练发音"
                 % _cats.get("口音型", 0))
        L.append("   未读出的音 %d 处 —— 读太轻/漏读，练清晰度"
                 % _cats.get("未读出", 0))
        if _cats.get("读太轻"):
            L.append("   读太轻的音 %d 处 —— 读对了但太轻" % _cats.get("读太轻", 0))
        if _cats.get("其他替换"):
            L.append("   其他替换 %d 处" % _cats.get("其他替换", 0))
        L.append("")
    if mode == MODE_LONG:
        # 长音频：按句分组，每句「原句 → 原音频音素 → 我的音素 → 该句纠音（分四类）」
        _sentences = (phoneme.get("长音频") or {}).get("句子") or []
        for _r in _sentences:
            if _r.get("漏句"):
                L.append("  · 句%d【%s】—— 漏读（不计分）"
                         % (_r.get("句号"), _r.get("文本", "")))
                L.append("")
                continue
            L.append("  句%d  %s" % (_r.get("句号"), _r.get("文本", "")))
            if _r.get("参考音素"):
                L.append("      原音频：%s" % _r["参考音素"])
            if _r.get("模仿音素"):
                L.append("      你读的：%s" % _r["模仿音素"])
            _sdiffs = _r.get("差异") or []
            if _sdiffs:
                # 该句差异按四类分组
                _sgroups = {"口音型": [], "未读出": [], "读太轻": [], "替换": []}
                for _d in _sdiffs:
                    _c = _d.get("类别") or "替换"
                    _sgroups.setdefault(_c, []).append(_d)
                for _c, _title in (("口音型", "口音型"),
                                   ("未读出", "没读出"),
                                   ("读太轻", "读太轻"),
                                   ("替换", "其他替换")):
                    _items = _sgroups.get(_c) or []
                    if not _items:
                        continue
                    L.append("      【%s】" % _title)
                    for _d in _items:
                        L.append("         " + _fmt_sent_diff(_d))
            else:
                L.append("      （本句完全一致）")
            L.append("")
        L.append("    ※ [口音型] 是学习者典型音变（θ→f、ð→d 之类），"
                 "母语者几乎不犯；[没读出] 是读太轻或漏读；")
        L.append("      [读太轻] 是读对了但太轻；[其他替换] 是普通读错。")
    elif diffs:
        # 单句：按类别分组呈现：先给口音型（该练发音），再给其他
        _groups = {"口音": [], "没读出": [], "读太轻": [], "替换": []}
        for d in diffs:
            _s = d.strip()
            for _k in _groups:
                if _s.startswith("[%s]" % _k):
                    _groups[_k].append(_s)
                    break
            else:
                _groups["替换"].append(_s)
        for _k, _title in (("口音", "口音型（优先练这些）"),
                           ("没读出", "没读出来的音（练清晰度）"),
                           ("读太轻", "读太轻的音"),
                           ("替换", "其他替换")):
            if not _groups[_k]:
                continue
            L.append("   【%s】%d 处" % (_title, len(_groups[_k])))
            for d in _groups[_k]:
                L.append("    " + d)
            L.append("")
        L.append("    ※ [口音] 是学习者典型音变（θ→f、ð→d 之类），"
                 "母语者几乎不犯；[没读出] 是读太轻或漏读；")
        L.append("      [读太轻] 是读对了但太轻；[替换] 是其他读错。")
    else:
        L.append("    （音素序列完全一致，很棒）")
    L.append("")
    # 原句
    words = phoneme.get("词表") or []
    problem_words = set()
    long_problem = phoneme.get("问题词")
    if long_problem:
        # 长音频模式：用 compare_long 挑好的"问题严重词"（错音≥2 或比例≥40%）。
        # 不能改成解析差异行的标签——那样满句都是框，等于没框（实测踩过）。
        problem_words |= set(long_problem)
    else:
        for d in diffs:
            for seg in d.split("【"):
                if "】" not in seg:
                    continue
                inner = seg.split("】")[0]
                # inner 形如：单词 "so"  或  单词 "Where" → "are" 之间
                for q in inner.split('"')[1::2]:        # 取所有被双引号括起的词
                    w = q.strip()
                    if w:
                        problem_words.add(w)
    if words:
        sent = " ".join(("【%s】" % w["word"]) if w["word"] in problem_words else w["word"]
                       for w in words)
        L.append("-" * 60)
        L.append("四、原句（有问题的词已用【】框出）")
        L.append("-" * 60)
        L.append("    " + sent)
        L.append("")

    # 五、节奏 / 语调
    L.append("-" * 60)
    L.append("五、节奏 / 语调")
    L.append("-" * 60)
    long_src = "compare_long" in (((rhythm or {}).get("节奏") or {}).get("数据源") or "")
    if phoneme.get("长音频") and not long_src:
        L.append("    长音频模式但缺 节奏对比.json —— 请先跑 compare_long.py。")
    elif rhythm:
        net = rhythm_section(rhythm, "整句净时长")
        ps = rhythm_section(rhythm, "句子边界停顿")
        st = rhythm.get("重音_信号") or {}
        if net:
            L.append("    净时长：原音频 %.2fs   |   你 %.2fs   |   相对差 %+.1f%%"
                     % (net.get("原音频", 0), net.get("我的", 0), (net.get("相对差") or 0) * 100))
        else:
            L.append("    净时长：（数据缺失）")
        if (rhythm.get("节奏") or {}).get("序号对齐风险"):
            L.append("    ⚠ 两侧词块序号可能未对齐，以下节奏明细仅供参考")
        if ps:
            L.append("    停顿：原音频 %s 处  |  你 %s 处"
                     % (ps.get("原音频处数", "—"), ps.get("我的处数", "—")))
            mine = ps.get("我的") or []
            if mine:
                L.append("          你多停的位置：" + "、".join(
                    "%s 后（%+.2fs）" % (p.get("after_word"), p.get("gap", 0)) for p in mine))
        else:
            L.append("    停顿：（数据缺失）")
        if st:
            L.append("    重音：原音频重读 %d 词  |  你 %d 词  |  命中 %d  |  漏 %d  |  多 %d"
                     % (len(st.get("原音频重读词") or []), len(st.get("我的重读词") or []),
                        len(st.get("命中") or []), len(st.get("漏（原音频有）") or []),
                        len(st.get("多（我的）") or [])))
        # 语调：阶段 5 起降级为"仅可视化"，不出分
        L.append("    语调：仅作可视化参考（音高曲线 / 绝对响度见 节奏对比.json），不出分")
        # 按句重音详表（10-28）：用户要的是「每句标出该读哪些重音、我读了哪些」，
        # 不是只报一个「漏掉重音」的清单。逐句三行：应读 / 实读 / 纠音。
        _stress_lines = _render_stress_by_sentence(rhythm)
        if _stress_lines:
            L.append("")
            L.append("    重音位置逐句对照（【】框出的词 = 该重读的词）：")
            L.extend(_stress_lines)
    L.append("")

    # 六、改进建议
    L.append("-" * 60)
    L.append("六、下一步怎么练")
    L.append("-" * 60)
    for tip in build_advice(parts, phoneme, rhythm):
        L.append("    · " + tip)
    L.append("")

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


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
