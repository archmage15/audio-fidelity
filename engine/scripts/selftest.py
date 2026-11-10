# -*- coding: utf-8 -*-
r"""还原度测试 · 自检（不跑模型，只验算法）

为什么要它
----------
项目零测试，但算法改过12 轮。历史上多次"静默错误"——数字照出、报告照出，
但结论是错的：
  · 长音频不画图，PNG 里是上一次的旧图（10-17）
  · 陈旧的 节奏对比.json 混进长音频报告，显示"净时长 3.70s"（10-17）
  · history 把单句 63.2 和长音频 67.2 混着比，写出"↑+4.0"假进步（10-18/P1）
  · 半截检测用 delete 段判漏头 → 抓不到（10-16）
  · 幂平均 p=1 时"10 块坏 2 块"被摊平，比"只是慢了"分还高（09-28）
这些都不是"代码崩了"，是"代码好好地算出了一个没意义的数"。单元测试挡不住
崩溃，但**能挡住口径被改坏**——所以每个用例都锚定一次真实踩过的坑。

怎么用
------
    python scripts\selftest.py

约1 秒跑完（纯计算，不加载 torch / 模型 / 音频）。
退出码 0 = 全过；1 = 有失败。可直接挂 git 钩子或改完代码先跑一遍。

设计约束
--------
1. **只用纯函数**。不碰模型、不碰音频、不写 output/ —— 自检不该产生副作用，
   否则它自己就成了干扰源（这正是 10-17 那几个 bug 的形态）。
2. **每个用例写明锚定哪个历史坑**，见各用例上方注释。将来有人想删某个用例，
   能看出它护的是什么。
3. 断言用精确值或窄容差；只在"数学上本该有浮点误差"处用 toFixed 风格比较。
"""

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import compare_long as cl
import compare_phoneme as cp
import compare_rhythm as cr
import overall_score as os_
import run_all as ra          # 顶层只有路径常量，导入无副作用（main 有 __main__ 保护）

# ---- 小工具 --------------------------------------------------------------

_PASS = []
_FAIL = []
_group = ""


def group(name):
    global _group
    _group = name
    print("\n%s" % name)


def check(desc, got, want):
    """断言 got == want；不等就记失败并打印差异。"""
    ok = got == want
    mark = "OK  " if ok else "FAIL"
    line = "  [%s] %s" % (mark, desc)
    if not ok:
        line += "\n         期望 %r\n         实际 %r" % (want, got)
    print(line)
    (_PASS if ok else _FAIL).append((_group, desc))
    return ok


def check_near(desc, got, want, tol):
    """数值断言（带容差）。用于浮点计算结果。"""
    ok = got is not None and abs(got - want) <= tol
    mark = "OK  " if ok else "FAIL"
    line = "  [%s] %s" % (mark, desc)
    if not ok:
        line += "\n         期望 %.4f ±%.4f\n         实际 %r" % (want, tol, got)
    print(line)
    (_PASS if ok else _FAIL).append((_group, desc))
    return ok


def check_true(desc, got):
    return check(desc, bool(got), True)


# ==========================================================================
# 1. 半截检测 detect_partial_coverage
#    锚定：10-16 踩过的坑 —— 用 opcodes 的 delete 段判漏头**抓不到**
#    （跳开头时 difflib 判成 replace 不是 delete）。现在靠匹配块跨度。
# ==========================================================================
group("1. 半截检测（锚定 10-16 漏头抓不到的坑）")

# 参考序列：**非周期**的音素串。
# ⚠️ 别用 ["s","oʊ","w"]*N 这种循环节律——截取 REF[20:40] 得到的片段
# 与 REF[0:20] 完全相同，difflib 会把它对到 index 0，"漏头"就测不出来了
# （真实代码没错，是测试数据自欺）。非周期串保证任意两段都不会撞。
REF40 = (["s", "oʊ", "w", "ɛ", "r", "t", "n", "i", "æ", "b"]
         + ["k", "d", "m", "ɛ", "l", "p", "h", "oʊ", "g", "v"]
         + ["θ", "z", "f", "ʃ", "j", "u", "aɪ", "d", "b", "l"]
         + ["f", "eɪ", "t", "ʃ", "ɔ", "r", "m", "h", "k", "w"])


def slice_ref(start, length):
    """取参考序列的连续片段（模拟"只跟读了这一段"）。

    ⚠️ 片段必须**就是参考序列本身**，才能产生真实匹配块。
    若换成全新的假音素（与参考无交集）→ 匹配块为空 → 直接 return empty，
    测的就不是漏尾逻辑而是"完全没读"的另一个分支了。
    """
    return REF40[start:start + length]


# 完整模仿：长度接近参考，不该判半截
r = cp.detect_partial_coverage(REF40, slice_ref(0, 38))
check("完整模仿不判半截", r["半截"], False)
check("完整模仿覆盖比例=1.0", r["覆盖比例"], 1.0)

# 漏尾：只读前半（20/40 = 50%，明显短一截且尾漏 20 音）
r = cp.detect_partial_coverage(REF40, slice_ref(0, 20))
check("漏尾判半截", r["半截"], True)
check("漏尾头漏=0", r["头部漏读音数"], 0)
check_near("漏尾覆盖比例≈0.5", r["覆盖比例"], 0.5, 0.05)

# 漏头：只读后半（跳开头）—— 这是 10-16 用 delete 段抓不到的那个
r = cp.detect_partial_coverage(REF40, slice_ref(20, 20))
check("漏头判半截", r["半截"], True)
check("漏头覆盖起始=20", r["覆盖起始位置"], 20)
check("漏头尾部漏读=0", r["尾巴漏读音数"], 0)

# 头尾都漏（只读中间一小段）→ 也该判半截
r = cp.detect_partial_coverage(REF40, slice_ref(18, 12))
check("掐头去尾只读中间 → 判半截", r["半截"], True)

# 只漏 1 个音（零星漏读不算整段没读）
r = cp.detect_partial_coverage(REF40, slice_ref(0, 39))
check("零星漏 1 音不判半截", r["半截"], False)

# ⚠️ 三个门槛里**只有 LEN_RATIO 真正生效**（变异测试 + 边界扫描实测得出）：
#   LEN_RATIO=0.75 要求 mine <= 30/40；MIN_TAIL_RATIO=0.10 只要漏 ≥4 音就够；
#   MIN_TAIL=3 更宽松。逐点扫描 40 音参考的真实边界：
#     mine=30（漏10 音/25%）→ 判半截   mine=31（漏9 音/22.5%）→ 不判
#   边界恰好等于 LEN_RATIO，与 10% 那条线无关 —— MIN_TAIL 与 MIN_TAIL_RATIO
#   在短句上都是死门槛（真要生效得漏 ≥25%，而那时 LEN_RATIO 早就拦下了）。
#   所以只锁 LEN_RATIO 这条真实边界，别为 MIN_TAIL 写用例（改了测不出差异）。
#
# ⚠️ 参考数据必须**全局互异**。变异测试踩过两次坑：
#   ① ["s","oʊ","w"]*13 循环 → slice(20,20) 片段与开头完全相同，difflib 对到
#      index 0，"漏头"根本测不出来；
#   ② REF40+REF40 拼 100 音 → 尾部与前段重复，尾漏被算成 0。
#   两次都是**用例自己骗自己**，不是代码 bug。
REF_UNIQ = ["ph%02d" % i for i in range(40)]
r = cp.detect_partial_coverage(REF_UNIQ, REF_UNIQ[:30])
check("mine=30(正好75%) → 判半截（LEN_RATIO 边界内）", r["半截"], True)
r = cp.detect_partial_coverage(REF_UNIQ, REF_UNIQ[:31])
check("mine=31(超过75%) → 不判半截（未短一截）", r["半截"], False)
r = cp.detect_partial_coverage(REF_UNIQ, REF_UNIQ[:35])
check("mine=35(87.5%，只漏5音) → 不判半截", r["半截"], False)

# LEN_RATIO 前提：模仿未短到 75% → 根本不该进半截判断（哪怕尾部漏很多）
r = cp.detect_partial_coverage(REF_UNIQ, REF_UNIQ[:36])
check("漏4音但长度仍90% → 不判半截（未短一截）", r["半截"], False)

# 模仿比参考长（读多了）→ 不可能是半截
r = cp.detect_partial_coverage(REF40, REF40 + ["n", "e", "w"])
check("模仿更长不判半截", r["半截"], False)

# 中间读错几个音但整体长度够 → 不该判半截
mixed = list(slice_ref(0, 38))
for i in (5, 20, 33):
    mixed[i] = "WRONG"
r = cp.detect_partial_coverage(REF40, mixed)
check("中间错音不判半截", r["半截"], False)

# 模仿为空 → 不崩，返回"未覆盖任何"
r = cp.detect_partial_coverage(REF40, [])
check("空模仿 → 不判半截且不崩", r["半截"], False)


# ==========================================================================
# 2. 节奏分score_blocks
#    锚定：09-28 的两个坑 —— ①不归一化则"整体慢 15%"被判成"每块都错"
#    ②算术平均把"坏 2 块"摊平。幂平均 p=3 是为此引入的。
# ==========================================================================
group("2. 节奏分（锚定 09-28 归一化 + 幂平均的坑）")


def pairs_from(durations_r, durations_m):
    """按两侧时长列表造score_blocks 的输入。"""
    return [{"序号": i + 1, "块": "w%d" % (i + 1), "连读": False, "角色": "A",
             "原音频时长": dr, "我的时长": dm}
            for i, (dr, dm) in enumerate(zip(durations_r, durations_m))]


# ① 整体等比例变慢：归一化后应接近 100 分（不惩罚"就是语速慢"）
items, sc = cr.score_blocks(pairs_from([0.40] * 10, [0.46] * 10))
check_near("整体慢 15% → 节奏分≈100（归一化生效）", sc, 100.0, 0.5)
check("整体慢 15% → 无块被判'要读慢一些'", len([i for i in items if i["判定"] != "差不多"]), 0)

# ② 10 块里坏 2 块：幂平均必须比算术平均扣得多
base = [0.40] * 10
slow_blocks = [0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.18, 0.22, 0.40, 0.40]
items, sc_bad = cr.score_blocks(pairs_from(base, slow_blocks))
check_true("10 块坏 2 块 → 分明显低于 100", sc_bad < 85)
# 时长被读短 → 相对差为负 → 判「要读慢一些」（不是"读快"，方向别搞反）
check("读短的2 块判'要读慢一些'",
      sorted(i["序号"] for i in items if i["判定"] == "要读慢一些"), [7, 8])

# 反向：时长被读长 → 判「要读快一些」
fast_blocks = [0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.62, 0.58, 0.40, 0.40]
items, _ = cr.score_blocks(pairs_from(base, fast_blocks))
check("读长的 2 块判'要读快一些'",
      sorted(i["序号"] for i in items if i["判定"] == "要读快一些"), [7, 8])

# 小幅偏差（±10%）不该被判为错误
mild = [0.40 * 1.1] * 10
items, _ = cr.score_blocks(pairs_from(base, mild))
check("小幅偏差 10% → 全部判'差不多'",
      len([i for i in items if i["判定"] != "差不多"]), 0)

# 幂平均确实比算术平均更敏感（p=3 vs p=1）
# ⚠️ 必须直接调 aggregate_penalty 并显式传 power：aggregate_penalty 的默认参数
# `power=AGG_POWER` 是**函数定义时**绑定的，改AGG_POWER 变量它不读。
# 换句话说：用 score_blocks 测不出 AGG_POWER 被改坏（变异测试实测踩过）。
diffs = [0.0] * 8 + [0.55, 0.62]
check_true("幂平均 p=3 比算术平均更敏感",
           cr.aggregate_penalty(diffs, power=3.0) > cr.aggregate_penalty(diffs, power=1.0))

# 锁定 AGG_POWER 必须是 3.0 —— 直接读模块常量（score_blocks 读不到，见上）
check("AGG_POWER 仍为 3.0（改它会退回09-28 的'坏块被摊平'）",
      cr.AGG_POWER, 3.0)

# p=3 在"8好2坏"上应显著低于 p=1（这是选p=3 的唯一理由，必须锁住）
check_true("p=3 对坏块比 p=1 扣得更狠",
           cr.aggregate_penalty(diffs, power=3.0) > cr.aggregate_penalty(diffs, power=1.0) + 0.1)

# 空输入不崩
items, sc_empty = cr.score_blocks([])
check("空输入返回空列表+0分", (items, sc_empty), ([], 0.0))


# ==========================================================================
# 3. 停顿判定 find_boundary_pauses
#    锚定：§5.5 连带坑 —— 换人处的停顿是**预期停顿**，不计入处数。
#    实测原音频换人 gap 880ms、模仿 680ms，旧口径会全记成停顿错误。
# ==========================================================================
group("3. 停顿判定（锚定 §5.5 换人处不应算停顿错误）")

blocks_same_role = [
    {"词": ["where"], "start_ms": 0, "end_ms": 400, "角色": "A"},
    {"词": ["are"], "start_ms": 900, "end_ms": 1200, "角色": "A"},   # gap 500ms 同人
]
result, counted = cr.find_boundary_pauses(blocks_same_role)
check("同人 gap 500ms → 记为停顿", len(counted), 1)
check("同人停顿 非换人处", counted[0]["换人处"], False)

blocks_turn = [
    {"词": ["terry"], "start_ms": 0, "end_ms": 400, "角色": "A"},
    {"词": ["oh", "im"], "start_ms": 1280, "end_ms": 1600, "角色": "B"},  # gap 880ms 换人
]
result, counted = cr.find_boundary_pauses(blocks_turn)
check("换人 gap 880ms → 出现但不计入处数", len(counted), 0)
check("换人处被标记出来", result[0]["换人处"], True)

# gap 不足阈值 → 不算停顿
blocks_tight = [
    {"词": ["a"], "start_ms": 0, "end_ms": 400, "角色": "A"},
    {"词": ["b"], "start_ms": 450, "end_ms": 800, "角色": "A"},   # gap 50ms
]
_, counted = cr.find_boundary_pauses(blocks_tight)
check("gap 50ms → 不算停顿", len(counted), 0)


# ==========================================================================
# 4. 词级错误 compute_word_errors
#    锚定：09-23 定的基准 —— 以**原音频实际音素**为准，不是参考文本。
#    连读吞音若原音频也吞、学生也吞 → 不算错（反例：正确连读成 wanna 会被误报）。
# ==========================================================================
group("4. 词级错误（锚定 09-23 基准=原音频实际音素）")

wb = [{"word": "want", "角色": "A"}, {"word": "to", "角色": "A"}]

# 原音频吞音（wanna 只有一个音），学生也吞成一样 → 不算错
ref_owner = [0, 1]
my_owner = [0, 1]
errs = cp.compute_word_errors(wb, ["w", "n"], ref_owner, ["w", "n"], my_owner)
check("原音频也吞音、学生同样吞 → 不报错", len(errs), 0)

# 学生确实读错→ 要报
# ⚠️ owner 数组长度必须等于音素数（compare_word_errors 按 len(音素) 索引 owner）
errs = cp.compute_word_errors(wb, ["w", "n"], [0, 1], ["w", "t", "o"], [0, 1, 1])
check_true("学生读错 → 报出问题词", len(errs) >= 1)

# 只错1 个音、且占该词比例低（4 音里错 1 = 25% < 40%，且错音数 1 < 2）
# → 低于双阈值，不报（避免把吹毛求疵的差异全标成错）
errs = cp.compute_word_errors(
    [{"word": "bank", "角色": "A"}],
    ["b", "a", "n", "k"], [0, 0, 0, 0],
    ["b", "a", "n", "x"], [0, 0, 0, 0])
check("只错 1 个音(25%) → 不报（低于双阈值）", len(errs), 0)

# 只错 1 个音但该词很短（2 音里错 1 = 50% ≥ 40%）→ 要报
errs = cp.compute_word_errors(
    [{"word": "at", "角色": "A"}],
    ["æ", "t"], [0, 0],
    ["æ", "b"], [0, 0])
check("2音词错1个(50%) → 报（比例超阈值）", len(errs), 1)

# 原音频与模仿都是空段 → 跳过，不崩不报
errs = cp.compute_word_errors([{"word": "x", "角色": "A"}], [], [], [], [])
check("双方全空 → 不报不崩", len(errs), 0)


# ==========================================================================
# 5. 长音频句子分配 assign_mine_to_sentences
#    锚定：长音频模式的核心 —— 全局对齐后把模仿音素分配进各句，
#    区间内一个都没对上的句 = 漏句（不计分）。
# ==========================================================================
group("5. 长音频逐句分配与漏句判定")

text_ph = ["a", "b", "c", "d", "e", "f", "g", "h"]
sentences = [
    {"句号": 1, "文本": "ab", "a": 0, "b": 2},
    {"句号": 2, "文本": "cd", "a": 2, "b": 4},
    {"句号": 3, "文本": "ef", "a": 4, "b": 6},
    {"句号": 4, "文本": "gh", "a": 6, "b": 8},
]
import difflib

# 完整模仿：每句 2 音全对上 → 无漏句
sm = difflib.SequenceMatcher(a=text_ph, b=list(text_ph), autojunk=False)
assign = cl.assign_mine_to_sentences(sm.get_opcodes(), sentences, 8, 8)
check("完整模仿 → 4 句都有分配", sorted(assign.keys()), [1, 2, 3, 4])
check("完整模仿 → 无空句", sum(1 for v in assign.values() if not v), 0)

# 只读前 2 句 → 后 2 句空 → 判漏句
mine_half = text_ph[:4]
sm = difflib.SequenceMatcher(a=text_ph, b=mine_half, autojunk=False)
assign = cl.assign_mine_to_sentences(sm.get_opcodes(), sentences, 8, 4)
check("只读前2句 → 句3无分配", assign.get(3), [])
check("只读前2句 → 句4无分配", assign.get(4), [])

# 逐句评测：完整句不该判漏句（10-30 改：evaluate_sentence 签名加了 sent_susp，
# 返回不再有 匹配/替换/缺失/多余，差异来自强制对齐可疑音）
sent = {"句号": 1, "文本": "ab", "a": 0, "b": 2, "_mine": [0, 1]}
wb_local = [{"word": "ab", "a": 0, "b": 2, "角色": "A"}]
res = cl.evaluate_sentence(sent, text_ph, list(text_ph), [0.0, 0.1], wb_local, [])
check("完整句不判漏句", res["漏句"], False)
check("完整句差异为空（无可疑音）", res["差异"], [])

# 空句（一个模仿音素都没分到）→ 判漏句
# ⚠️ 句子区间必须 >= MIN_SENT_PH(3) 音，否则会被判成「碎句-忽略」而不判漏句。
# 这里用 a=2,b=6（4 音），确认是"有内容的句但完全没跟读"→ 漏句。
sent = {"句号": 3, "文本": "cdef", "a": 2, "b": 6, "_mine": []}
res = cl.evaluate_sentence(sent, text_ph, list(text_ph), [0.0], [], [])
check("空句(4音)判漏句", res["漏句"], True)

# 少于 3 音的空句 → 碎句忽略，不是漏句（10-17踩过：讯飞把 "There. Are" 拆两句）
sent = {"句号": 9, "文本": "g", "a": 6, "b": 7, "_mine": []}
res = cl.evaluate_sentence(sent, text_ph, list(text_ph), [0.0], [], [])
check("碎句(1音)空 → 标忽略不判漏句", (res["忽略"], res["漏句"]), (True, False))


# ==========================================================================
# 6. 综合分公式
# ==========================================================================
group("6. 综合分公式与等级")

# 发音分 10-29 起改强制对齐口径：分母=参考音素数，分子=口音型+其他+读太轻
fa_stat = {"参考音素数": 100, "口音型": 10, "其他": 5, "读太轻": 5, "没读出来": 10}
check_near("发音分 (1-(口音型+其他+读太轻)/参考音素数)×100",
           os_.score_pronunciation(fa_stat), 80.0, 0.01)
check("「没读出来」不扣发音分（归音变）",
      os_.score_pronunciation({"参考音素数": 100, "口音型": 0, "其他": 0,
                               "读太轻": 0, "没读出来": 50}), 100.0)
# 音变分 10-30 起改强制对齐口径：1 - (没读出来+读太轻)/参考音素数。
#   多余音已删除（difflib 错位假象）。
check_near("音变分 (1-(没读出来+读太轻)/参考音素数)×100",
           os_.score_phoneme_change(fa_stat), 85.0, 0.01)
check("「口音型/其他」不扣音变分（归发音）",
      os_.score_phoneme_change({"参考音素数": 100, "口音型": 30, "其他": 20,
                                "读太轻": 0, "没读出来": 0}), 100.0)
check("空输入 → None（不当0 分处理）", os_.score_pronunciation(None), None)
check("空输入音变分 → None", os_.score_phoneme_change(None), None)
check("等级边界 90=优", os_.grade(90), "优")
check("等级边界 59.9=待加强", os_.grade(59.9), "待加强")
check("None → —", os_.grade(None), "—")

# build_align_stat（10-30 改签名）：只留 原生音数 + 模仿音数，difflib 四类已删。
_aln = cl.build_align_stat([{"模仿音素数": 200}, {"模仿音素数": 40}], 280)
check("原生音数 = 参考音素数（传入 n_ref）", _aln["原生音数"], 280)
check("模仿音数 = 各句模仿音素数之和", _aln["模仿音数"], 240)


# ==========================================================================
# 7. history 模式隔离（P1 · 10-18）
#    锚定：63.2（4秒单句）与 67.2（72秒长音频）被写成"↑+4.0"假进步。
# ==========================================================================
group("7. history 跨模式隔离（锚定 P1 假进步）")

hist_mixed = [
    {"时间": "t0", "综合分": 63.2},                                # 旧记录：无模式
    {"时间": "t1", "综合分": 55.0, "模式": "单句"},
    {"时间": "t2", "综合分": 67.2, "模式": "长音频"},
]
check("长音频只跟长音频比（命中 67.2）",
      os_.find_prev_same_mode(hist_mixed, "长音频")["综合分"], 67.2)
check("单句只跟单句比（命中 55.0，不被 67.2 污染）",
      os_.find_prev_same_mode(hist_mixed, "单句")["综合分"], 55.0)
check("全旧记录无模式 → 拒绝出对比",
      os_.find_prev_same_mode(hist_mixed[:1], "长音频"), None)
check("空历史 → None", os_.find_prev_same_mode([], "单句"), None)
check("模式判定：有长音频段", os_.detect_mode({"长音频": {}}), "长音频")
check("模式判定：无长音频段", os_.detect_mode({"相似度": 63.2}), "单句")
# 空 dict 也算长音频（存在即长音频）—— 防止 compare_long 写空段时被误判成单句
check("模式判定：长音频段存在但为空 → 仍算长音频",
      os_.detect_mode({"长音频": {}}), "长音频")


# ==========================================================================
# 8. 重音明细渲染（10-21 · 先摊开依据再下结论）
#    锚定：以前只打印「哪几块突出」，依据全藏在 json 里，读者无法自查判定。
# ==========================================================================
group("8. 重音明细渲染（先依据后结论）")

_d = [
    {"序号": 1, "词": ["where", "are"], "角色": "A", "突出": True,
     "归一值": {"音高": 3.1, "响度": 1.4, "时长": 1.2},
     "落差σ": {"音高": 2.5, "响度": 1.8, "时长": 0.4}, "命中路": ["音高", "响度"]},
    {"序号": 2, "词": ["oh"], "角色": "A", "突出": False,
     "归一值": {"响度": 0.8, "时长": 0.9},       # 音高缺测（None，不给键）
     "落差σ": {"响度": -1.1, "时长": -0.6}, "命中路": []},
    {"序号": 3, "词": ["im"], "角色": "B", "突出": False,
     "归一值": {"响度": 0.9}, "落差σ": {"响度": -0.9}, "命中路": []},
]
_lines = cr.render_stress_detail(_d, "原音频")
_body = "\n".join(_lines)


def _row(text):
    """取一条明细行（不是表头/图例/统计行）：以空格 + 序号开头。"""
    s = text.strip()
    return s[:1].isdigit()


_rows = [x for x in _lines if _row(x)]
check("明细含全部 3 块（不只突出的）", len(_rows), 3)
check("每块恰好一行", sorted(x.strip().split()[0] for x in _rows), ["1", "2", "3"])
check("突出的块标★", any("★" in x and "where+are" in x for x in _lines), True)
check("不突出的块标○", any("○" in x and " oh " in x for x in _lines), True)
check("漏掉的块标 M", any("M" in x and "where+are" in x
                          for x in cr.render_stress_detail(_d, "原音频", miss=[1])), True)
check("多出的块标 X", any("X" in x and "oh" in x
                          for x in cr.render_stress_detail(_d, "我的", extra=[2])), True)
check("缺失该路数据 → 显示 - 而非崩", any("- (-" in x or x.count("-") >= 3
                                    for x in _lines if "im" in x), True)
check("命中路如实显示", any("音高/响度" in x for x in _lines), True)
check("音高缺测统计提示（说明 - 不是 0）",
      any("音高显示为 -" in x and "2/3" in x for x in _lines), True)
check("无缺测时不加该提示",
      any("音高显示为 -" in x for x in cr.render_stress_detail(
          [dict(_d[0])], "原音频")), False)
check("空明细不崩", cr.render_stress_detail([], "原音频")[0], "   （该侧没有词块）")
# 对齐：中文按 2 格算，所有数据行的判定标记列起始位置必须一致
def _colpos(rows):
    pos = set()
    for r in rows:
        pos.add(len(r) - len(r.lstrip()))
    return len(pos) == 1
check("中英混排列左边界一致", _colpos(_rows), True)
check("判定标记列对齐（★/○/M/X 同列）",
      len({len(x.rstrip()) - len(x.rstrip().lstrip("*○MX")) for x in _rows}), 1)

_v = cr.render_stress_verdict({1: "where+are", 2: "oh"}, {1: "oh", 3: "im"}, [1, 2], [3])
check("漏项列出对侧词名", any("漏" in x and "where+are" in x for x in _v), True)
check("多项列出（我多出来的）", any("多" in x and "im" in x for x in _v), True)
check("空集显示「无」而非空串", any(x.strip().endswith("无") for x in
      cr.render_stress_verdict({}, {}, [], [])), True)

# ---- 重点词清单：算法直接给「哪个词 + 怎么改」 ----
# 场景：原音频 #1 "where" 靠音高(+2.0半音)+响度突出；我这边音高只有 -0.2 → 差 2.2 半音
_fr = [{"序号": 1, "词": ["where"], "角色": "A", "突出": True,
        "归一值": {"音高": 2.0, "响度": 1.5, "时长": 1.1},
        "落差σ": {"音高": 2.4, "响度": 1.6, "时长": 0.3}, "命中路": ["音高", "响度"]},
       {"序号": 2, "词": ["thank"], "角色": "A", "突出": False,
        "归一值": {"音高": 0.0, "响度": 0.9, "时长": 0.9},
        "落差σ": {"音高": 0.0, "响度": -0.5, "时长": -0.4}, "命中路": []}]
_fm = [{"序号": 1, "词": ["where"], "角色": "A", "突出": False,
        "归一值": {"音高": -0.2, "响度": 0.8, "时长": 1.0},
        "落差σ": {"音高": -0.5, "响度": -1.4, "时长": 0.1}, "命中路": []},
       {"序号": 2, "词": ["thank"], "角色": "A", "突出": True,
        "归一值": {"音高": 1.0, "响度": 1.4, "时长": 1.0},
        "落差σ": {"音高": 1.2, "响度": 1.5, "时长": 0.2}, "命中路": ["音高", "响度"]}]
_fw = cr.render_focus_words(_fr, _fm, {1: "where", 2: "thank"}, {1: "where", 2: "thank"})
check("重点词清单只列漏的词（where），不列命中的（thank）",
      any("where" in x for x in _fw) and not any("thank" in x for x in _fw), True)
check("给出音高改法（方向+幅度档位）", any("音高" in x and "往上挑" in x for x in _fw), True)
# ⚠️ 方向也要断言：原音频音高更高 → 建议「往上挑」。只查数字的话，
#    把 gap 算反（my-ref）用例照样绿 —— 变异测试实测踩过这个坑。
check("音高改法方向正确（往上挑，不是压低）",
      any("往上挑" in x for x in _fw) and not any("压低" in x for x in _fw), True)
check("响度改法方向正确（原声更响 → 再读响些）",
      any("再读响些" in x for x in _fw) and not any("读轻些" in x for x in _fw), True)
check("幅度用档位而非假精确数字", not any("半音）" in x for x in _fw), True)
check("不输出百分比数字（避免假精确）", not any("%" in x for x in _fw), True)
# 分档本身也要断言：轻微/明显/很大 三档各有代表值
check("幅度分档：轻微（音高差 0.5 半音）",
      cr._magnitude("音高", 0.5), "轻微")
check("幅度分档：明显（音高差 2 半音）",
      cr._magnitude("音高", 2.0), "明显")
check("幅度分档：很大（音高差 5 半音）",
      cr._magnitude("音高", 5.0), "很大")
check("幅度分档：响度 10% → 轻微",
      cr._magnitude("响度", 0.10), "轻微")
check("幅度分档：响度 30% → 明显",
      cr._magnitude("响度", 0.30), "明显")
check("幅度分档：响度 80% → 很大",
      cr._magnitude("响度", 0.80), "很大")
check("幅度分档：负值取绝对值（方向另判）",
      cr._magnitude("音高", -5.0), "很大")
check("清单里带上了档位词",
      any(("轻微" in x or "明显" in x or "很大" in x) for x in _fw), True)
# 音高路是双向的：gap<0 时不能武断说「压低」——原声可能靠降调实现重读
_fr3 = [dict(_fr[0])]
_fr3[0]["归一值"] = dict(_fr[0]["归一值"], 音高=-3.0)
_fm3 = [dict(_fm[0])]
_fm3[0]["归一值"] = dict(_fm[0]["归一值"], 音高=2.0)
_fw3 = cr.render_focus_words(_fr3, _fm3, {1: "where"}, {1: "where"})
check("音高反向时不武断说「压低」",
      not any("压低" in x for x in _fw3), True)
check("音高反向时给出中性说明（降调重读的可能）",
      any("反了" in x or "往哪边走" in x for x in _fw3), True)
# 极端差值 → 判为测量异常，不给方向建议
_fx = [dict(_fr[0])]
_fx[0]["归一值"] = dict(_fr[0]["归一值"], 时长=4.0)   # 原侧 4 倍邻居
_fx[0]["命中路"] = ["音高", "响度", "时长"]          # 时长在命中路里才会给建议
_fxm = [dict(_fm[0])]
_fxm[0]["归一值"] = dict(_fm[0]["归一值"], 时长=1.0)  # 我侧 1 倍 → 差 3 倍
_fxm[0]["命中路"] = []
_fxt = cr.render_focus_words(_fx, _fxm, {1: "where"}, {1: "where"})
check("时长差 3 倍 → 判异常不给「拖长」",
      any("反常" in x and "拖长" not in x for x in _fxt), True)
check("响度差 5 倍 → 判异常",
      "响度差得反常" in cr._cue_advice("响度", 5.0, 1.0), True)
check("时长差 3 倍 → 判异常",
      "时长差得反常" in cr._cue_advice("时长", 4.0, 1.0), True)
check("音高差 13 半音 → 判异常",
      "反常" in cr._cue_advice("音高", 13.0, 0.0), True)
check("正常范围内不判异常",
      "反常" not in cr._cue_advice("时长", 1.5, 1.0), True)
check("带用法说明（照单子练）", any("怎么用" in x for x in _fw), True)
check("清单有一行一个词", sum(1 for x in _fw if x.strip().startswith("·")), 1)
# 全命中时不该列重点词：两侧突出集合完全相同
_fr2 = [dict(_fr[0]), dict(_fr[1])]
_fr2[0]["突出"] = True
_fr2[1]["突出"] = True
_fm2 = [dict(_fm[0]), dict(_fm[1])]
_fm2[0]["突出"] = True
_fw2 = cr.render_focus_words(_fr2, _fm2, {1: "where", 2: "thank"}, {1: "where", 2: "thank"})
check("全命中 → 明说不用改", any("不用改" in x for x in _fw2), True)
check("全命中 → 不列词条", sum(1 for x in _fw2 if x.strip().startswith("·")), 0)
# 多出来的重音
_fe = cr.render_extra_words(_fr, _fm, {1: "where", 2: "thank"}, {1: "where", 2: "thank"})
check("多出重音被单列（不扣分但提示）", any("thank" in x and "不扣分" in x for x in _fe), True)
check("无多出重音 → 返回空", cr.render_extra_words(_fr, _fr, {}, {}), [])


# ==========================================================================
# 9. 词边界重建（锚定 10-26 espeak 连读合并导致区间错位）
# ==========================================================================
group("9. 词边界重建（锚定 10-26 连读合并 → 区间整体错位）")

import gen_ref_phonemes as gp

# 场景：espeak 把 "for the" 连读合并成一段 fɚðə → 段数比词数少 1。
# 旧实现按「第 i 段 = 第 i 词」硬套 → 从 for 开始所有词的区间整体错位一格，
# 报告里「单词 the 实读 X」指到了 lesson 的音上。
# 这里构造最小复现：逐词音素 vs 整句音素（整句里 for+the 合并）。
_wp = [["h", "eɪ"], ["h", "eɪ"], ["ð", "æ", "t"], ["w", "ʌ", "z"],
       ["f", "ʌ", "n"], ["θ", "æ", "ŋ", "k", "s"],
       ["f", "ɔː", "ɹ"], ["ð", "ə"], ["l", "ɛ", "s", "ə", "n"]]
_sp = ["h", "eɪ", "h", "eɪ", "ð", "æ", "t", "w", "ʌ", "z", "f", "ʌ", "n",
       "θ", "æ", "ŋ", "k", "s", "f", "ɚ", "ð", "ə", "l", "ɛ", "s", "ə", "n"]
_rg, _cost = gp.align_words_to_phonemes(_wp, _sp)


def _seg(i):
    r = _rg[i]
    return " ".join(_sp[r[0]:r[1]]) if r else ""


check("连读合并场景：每个词都拿到区间（无一为 None）",
      sum(1 for r in _rg if r is None), 0)
check("连读合并场景：合并前的词不受影响", _seg(0), "h eɪ")
check("连读合并场景：for 拿到整句里的 f ɚ", _seg(6), "f ɚ")
check("连读合并场景：the 拿到 ð ə（不是 lesson 的音）", _seg(7), "ð ə")
check("连读合并场景：lesson 拿到 l ɛ s ə n", _seg(8), "l ɛ s ə n")
check("连读合并场景：DP 代价很小（≤5）", _cost <= 5, True)

# 区间必须单调、不重叠
# ⚠️ 先挡掉 None —— 区间缺失时下面会 TypeError 崩掉，
#    崩掉看不出是哪个用例失败（变异测试时才发现，已踩）。这里显式断言。
check("区间无 None（先判，再谈单调）", any(r is None for r in _rg), False)
if all(r is not None for r in _rg):
    check("区间单调递增（后一个 a ≥ 前一个 b）",
          all(_rg[i][0] >= _rg[i - 1][1] for i in range(1, len(_rg))), True)
else:
    check("区间单调递增（后一个 a ≥ 前一个 b）", False, True)

# 覆盖完整：所有区间并起来 = 整句音素数
# ⚠️ 判断必须写成 `r is not None` —— 写成 `e is not None` 会在解包时就崩
#    （None 解包成 s,e 直接 TypeError，掩盖了前面已经报出的失败）
_cov = sum(e - s for r in _rg if r is not None for s, e in [r] if e > s)
check("区间完整覆盖整句音素（无遗漏）", _cov, len(_sp))
# ⚠️ 区间不能首尾颠倒（a ≤ b）—— 颠倒了下游切音素会切出空串或负长度，
#    而且症状极难查（表现为"这个词没音"）。变异测试发现，已踩。
check("区间首尾不颠倒（a ≤ b）",
      all(r[0] <= r[1] for r in _rg if r is not None), True)
_wp2 = [["h", "eɪ"], ["ð", "ə"], ["l", "ɛ", "s", "ə", "n"]]
_sp2 = ["h", "eɪ", "ð", "ə", "l", "ɛ", "s", "ə", "n"]
_rg2, _c2 = gp.align_words_to_phonemes(_wp2, _sp2)
check("无合并场景：代价为 0", _c2, 0)
# ⚠️ 同样要防 None —— 解包 None 会 TypeError 崩掉，
#    崩掉会让「前面的失败」被掩盖，看不出真问题在哪。
check("无合并场景：区间无 None", any(r is None for r in _rg2), False)
check("无合并场景：词区间逐个对上",
      [" ".join(_sp2[r[0]:r[1]]) for r in _rg2 if r is not None]
      if all(r is not None for r in _rg2) else "<有 None 区间>",
      ["h eɪ", "ð ə", "l ɛ s ə n"])

# 空区间补齐
_filled = gp.fill_empty_ranges([(0, 2), None, (5, 7)], 7)
check("空区间被补成非 None", _filled[1] is not None, True)
check("空区间落在前后邻居之间", _filled[1][0] >= 2 and _filled[1][1] <= 5, True)
check("补齐后不越界", all(0 <= s <= e <= 7 for s, e in _filled), True)
# ⚠️ 补齐**不能侵占邻居的音素** —— 凭空给一个词分配音素，下游会把它当成
#    "这个词有音"，比留空更危险。补法是取空隙中点，宽度为 0。
check("补齐后总覆盖量不变（不侵占邻居）",
      sum(e - s for s, e in _filled), 2 + 0 + 2)
check("补齐点落在空隙内", 2 <= _filled[1][0] <= 5, True)
# ⚠️ 补齐点必须落在空隙**中点** —— 贴在邻居末尾会跟邻居抢同一个位置，
#    下游按区间切音素时会切出重叠（变异测试发现，已踩）。
check("补齐点取空隙中点", _filled[1][0], (2 + 5) // 2)
# ⚠️ 已有宽度的区间必须**原样保留** —— 补齐只动零宽的那些。
#    越界的区间（如 (5,2)）也要被纠正成合法值，不能原样放行。
check("非零宽区间不被改写",
      gp.fill_empty_ranges([(0, 2), (5, 7)], 7), [(0, 2), (5, 7)])
check("越界区间被纠正（不原样放行）",
      gp.fill_empty_ranges([(0, 2), (5, 2)], 7)[1][0] <= gp.fill_empty_ranges([(0, 2), (5, 2)], 7)[1][1],
      True)
check("全部为空时也不崩", gp.fill_empty_ranges([None, None], 4)[0][0] >= 0, True)

# 契约：align 的返回值直接可用 —— 调用方不该再自己补齐
# （补齐放在 align 内部，忘了调就会静默漏词 —— 变异测试发现，已踩）
# 用一个「完全对不上」的词触发真正的空区间：
#   逐词有个 "z"，整句里根本没有 → 对齐不上 → 补齐逻辑必须接手
_rg3, _c3 = gp.align_words_to_phonemes([["a"], ["z"], ["b"]], ["a", "b"])
check("对不上的词 → 区间仍非 None（align 内部已补齐）",
      any(r is None for r in _rg3), False)
# ⚠️ 断言前先挡 None —— 解包 None 会 TypeError 崩掉，掩盖前面的失败。
#    变异测试时才发现：崩溃比失败更难查（已踩）。
check("对不上的词 → 不越界",
      all(0 <= s <= e <= 2 for s, e in _rg3) if all(r is not None for r in _rg3)
      else False, True)
check("对不上的词 → 不侵占邻居音素",
      sum(e - s for s, e in _rg3) if all(r is not None for r in _rg3) else -1, 2)

# 末位补齐也不能越界（没有后邻居可依时，只能贴住末尾）
_edge = gp.fill_empty_ranges([(0, 2), None], 2)
check("末位补齐不越界",
      all(0 <= s <= e <= 2 for s, e in _edge), True)
check("末位补齐贴住末尾", _edge[1][0] <= 2 and _edge[1][1] <= 2, True)


# ==========================================================================
# 10. 口径版本隔离 + 母语者基准 + L2 音变分类（10-28）
# ==========================================================================
group("10. 口径版本隔离（锚定 10-26 修词边界后新旧分数不可横比）")

_V = os_.SCORE_VERSION
# 模拟真实 history：10-21 那条是**旧口径**（修词边界 bug 之前），
# 10-26 之后是新口径
_hist = [
    {"时间": "t0", "综合分": 63.2},                                    # 无模式无版本
    {"时间": "t1", "综合分": 55.0, "模式": "单句"},                     # 无版本
    {"时间": "t2", "综合分": 67.2, "模式": "长音频"},                   # 无版本（旧口径）
    {"时间": "t3", "综合分": 60.6, "模式": "长音频", "口径版本": 1},     # 旧口径
    {"时间": "t4", "综合分": 60.3, "模式": "长音频", "口径版本": _V},   # 新口径
]
# 关键：找新口径的记录时，不能命中旧口径的 60.6
_prev = os_.find_prev_same_mode(_hist, "长音频", _V)
check("新口径只跟新口径比（命中 60.3，不是旧口径的 60.6）",
      _prev["综合分"] if _prev else None, 60.3)
# 只传模式（不限版本）→ 仍能找到最近的（向后兼容旧调用）
check("不传版本 → 只按模式找（向后兼容）",
      os_.find_prev_same_mode(_hist, "长音频")["综合分"], 60.3)
# 旧记录没有版本字段 → 视为未知版本 → 拒绝用来比
_hist_only_old = _hist[:3]
check("全是旧口径记录 → 拒绝出对比",
      os_.find_prev_same_mode(_hist_only_old, "长音频", _V), None)
check("全是旧口径记录 → 可比数 0",
      os_.count_comparable(_hist_only_old, "长音频", _V), 0)
check("全是旧口径记录 → 同模式数 1（t2 是长音频；t0/t1 模式不同）",
      os_.count_comparable(_hist_only_old, "长音频"), 1)
check("版本匹配的可比数正确",
      os_.count_comparable(_hist, "长音频", _V), 1)
check("append_history 会记口径版本",
      os_.SCORE_VERSION >= 2, True)
# ⚠️ 写入层也要测 —— 只测「查找」会漏掉「append_history 忘了写版本字段」。
#    那样每条新记录都缺版本 → 查找永远匹配不上 → 永远无对比（静默失效）。
#    变异测试发现的：删掉 append_history 里的版本字段，用例照样全绿。
_hist_tmp = os_.HISTORY_JSON
_orig_json = os_.json
try:
    os_.HISTORY_JSON = os_.os.path.join(os_.os.path.dirname(_hist_tmp),
                                        "_t_selftest_history.json")
    os_.append_history(70.0, {"发音": 70.0, "音变": 70.0, "节奏": 70.0,
                               "重音位置": 70.0}, "长音频")
    _w = os_.read_history()
    check("写入的记录带口径版本", _w[-1].get("口径版本"), _V)
    # ⚠️ 断言前先挡 None —— 解包 None 会 TypeError 崩掉，
    #    崩掉比"断言失败"更难查：崩在前面的失败会被掩盖（已踩）。
    _found = os_.find_prev_same_mode(_w, "长音频", _V)
    check("写入后能按版本找回自己",
          _found["综合分"] if _found else None, 70.0)
finally:
    if os_.os.path.exists(os_.HISTORY_JSON):
        os_.os.remove(os_.HISTORY_JSON)
    os_.HISTORY_JSON = _hist_tmp

group("11. L2 音变分类（口音型 vs 没读出）")

# 命中 L2 对照表 → 给出针对性提示
check("θ→f 判为口音型并给改法",
      "咬舌" in cp.l2_tip("θ", "f"), True)
check("ð→d 判为口音型",
      "咬舌" in cp.l2_tip("ð", "d"), True)
check("ɹ→l 判为口音型",
      "卷舌" in cp.l2_tip("ɹ", "l"), True)
check("v→w 判为口音型",
      "双唇" in cp.l2_tip("v", "w"), True)
check("z→s 判为口音型",
      "声带" in cp.l2_tip("z", "s"), True)
check("æ→ɛ 判为口音型",
      "下巴" in cp.l2_tip("æ", "ɛ"), True)
# 非 L2 音变 → 不给口音提示
check("非 L2 组合 → 不给口音提示", cp.l2_tip("k", "p"), "")
check("相同音 → 不给口音提示", cp.l2_tip("k", "k"), "")
# 一对多 / 多对一 → 不判定（硬套会给出错误指导）
check("多个音对多个音 → 不判定", cp.l2_tip("θ æ", "f"), "")
check("空实读 → 不判定", cp.l2_tip("θ", ""), "")
# ⚠️ 多音 → 单音（实读只有一个音）**也不能套** ——
#    否则会把"θ æ 读成了 f"说成"θ 读成了 f"，但实际丢的是 æ。
#    变异测试发现的：去掉一对一限制后这里会硬套。
check("多音对单音 → 不判定（不硬套）", cp.l2_tip("θ æ", "f"), "")
check("单音对多音 → 不判定（不硬套）", cp.l2_tip("θ", "f s"), "")
check("多音对多音 → 不判定", cp.l2_tip("θ æ", "f s"), "")

# ---- 大块里的逐音配对（真实跑出来才发现的漏）----
# difflib 把连续几个音的变化挤成一块：「θ æ」→「f ɛ」其实是两个音变。
# 只做整块一对一 → 一个都认不出来 → 报告里"口音型"只有 2 处。
_pairs = cp.align_l2_pairs("θ æ", "f ɛ")
check("块内逐音配对：认出 2 个音变", len(_pairs), 2)
check("块内逐音配对：θ→f 认出来了", _pairs[0][0:2] if _pairs else None, ("θ", "f"))
check("块内逐音配对：æ→ɛ 认出来了", _pairs[1][0:2] if len(_pairs) > 1 else None, ("æ", "ɛ"))
# 长度不等 → 不硬套（对不上号）
check("块内长度不等 → 不判定", cp.align_l2_pairs("θ æ ŋ", "f ɛ"), [])
check("块内空输入 → 空", cp.align_l2_pairs("", ""), [])
# ⚠️ 数据要选**真不在 L2 表里**的 —— p→b 在表里（塞音清浊化），
#    用它当"无音变"的例子会误判（已踩）。
check("块内无音变 → 空", cp.align_l2_pairs("k t", "p b"), [])
# ⚠️ l2_tip 内部不能越界 —— 实读为空时取 gots[0] 会 IndexError。
#    这条断言让它变成"红色的失败"而不是"崩溃"（崩溃会掩盖前面的失败）。
try:
    _empty_res = cp.l2_tip("θ", "")
    check("实读为空时不越界（不崩）", _empty_res, "")
except Exception as e:
    check("实读为空时不越界（不崩）", "崩了: %s" % type(e).__name__, "不崩")
# 通用要领仍然可用
check("通用要领仍能出（θ 有要领）", "咬舌" in cp.phoneme_tips("θ"), True)
check("没有要领的音 → 空串", cp.phoneme_tips("xyzzy"), "")

# ---- 集成层：差异条目必须带「类别」（10-30 改：来自强制对齐可疑音）----
# ⚠️ 只测纯函数不够 —— 变异测试发现：把 compare_long 里生成「类别」字段
#    那行删掉，纯函数用例照样全绿。所以必须测「差异条目长什么样」。
# 新口径：evaluate_sentence 的差异来自 sent_susp（强制对齐可疑音，含「下标」）。
_sent = {"a": 0, "b": 5, "句号": 1, "文本": "that w", "_mine": [0, 1, 2, 3]}
_text_ph = ["θ", "æ", "t", "w", "ʌ"]
_mine = ["f", "æ", "t", "w"]
_wb = [{"word": "that", "a": 0, "b": 3, "角色": "A"},
       {"word": "w", "a": 3, "b": 5, "角色": "A"}]
# 强制对齐可疑音：下标0 θ→f（口音型）、下标4 ʌ 没读出来
_sent_susp = [
    {"音素": "θ", "实读": "f", "类别": "口音型", "下标": 0, "后验": 0.1},
    {"音素": "ʌ", "实读": "", "类别": "没读出来", "下标": 4, "后验": 0.05},
]
_res = cl.evaluate_sentence(_sent, _text_ph, _mine, [0, 20, 40, 60], _wb, _sent_susp)
_diff = _res["差异"]
check("差异条目带类别字段", all("类别" in d for d in _diff), True)
_kinds = set(d.get("类别") for d in _diff)
check("类别里有「口音型」（θ→f 命中 L2 表）", "口音型" in _kinds, True)
check("类别里有「未读出」（漏掉的音）", "未读出" in _kinds, True)
_l2d = [d for d in _diff if d.get("类别") == "口音型"]
check("口音型条目带针对性要领", bool(_l2d and _l2d[0].get("要领")), True)
check("口音型条目的要领提到咬舌",
      "咬舌" in (_l2d[0].get("要领") if _l2d else ""), True)
_missd = [d for d in _diff if d.get("类别") == "未读出"]
check("未读出条目的实读为空", _missd[0]["实读"] if _missd else None, "")
check("未读出条目明说「没读出来」",
      "没读出" in (_missd[0].get("要领") if _missd else ""), True)
check("差异条目挂词标签（θ 属于 that）",
      "that" in (_l2d[0].get("词") if _l2d else ""), True)

# ---- 分类统计函数（main 和报告都要用）----
_cats_t = cl.classify_diffs([{"句号": 1, "差异": _diff}])
check("分类统计：口音型 1 处", len(_cats_t.get("口音型", [])), 1)
check("分类统计：未读出 1 处", len(_cats_t.get("未读出", [])), 1)
check("分类统计：条目带句号", _cats_t["口音型"][0][0], 1)
check("分类统计：空输入 → 空字典",
      cl.classify_diffs([]), {"口音型": [], "未读出": [], "读太轻": [], "替换": []})
check("分类统计：None 输入不崩", cl.classify_diffs(None)["口音型"], [])

group("12. 母语者基准（让分数有参照）")

# 基准必须跟本次同口径：强制对齐可疑率（10-30 改，删 difflib ratio）。
_ref_ph = ["h", "eɪ", "ð", "æ", "t", "w", "ʌ", "z"]   # 8 音
# 原音频强制对齐可疑音（母语者 + 干净录音 = 工具误差底线）：只 1 处口音型
_ref_verdict = [{"音素": "ð", "实读": "d", "类别": "口音型", "下标": 2}]
# 模仿强制对齐可疑音：3 处
_mine_verdict = [
    {"音素": "ð", "实读": "f", "类别": "口音型", "下标": 2},
    {"音素": "ʌ", "实读": "æ", "类别": "其他", "下标": 6},
    {"音素": "z", "实读": "s", "类别": "口音型", "下标": 7},
]

# ---- 基准计算函数本身 ----
_bl = cl.compute_baseline(_ref_ph, _ref_verdict)
check("基准函数算出可疑率（1/8）", _bl["可疑率"] if _bl else None, 12.5)
check("基准函数统计可疑数", _bl["可疑数"] if _bl else None, 1)
check("基准函数统计口音型", _bl["口音型"] if _bl else None, 1)
check("基准函数统计没读出来", _bl["没读出来"] if _bl else None, 0)
# 关键：基准必须用「原音频」而不是「模仿」——
#   用模仿自己的结果当基准，基准就等于本次分数，差距永远是 0（自欺欺人）。
_bl_wrong = cl.compute_baseline(_ref_ph, _mine_verdict)
check("用模仿当基准会得到假基准（可疑率 3/8）",
      _bl_wrong["可疑率"] if _bl_wrong else None, 37.5)
check("正确基准 ≠ 假基准（基准确实用的是原音频）",
      (_bl["可疑率"] != _bl_wrong["可疑率"])
      if (_bl and _bl_wrong) else False, True)
check("基准口径写明了是原音频",
      "原音频" in (_bl.get("口径") or ""), True)
check("空参考音素 → None", cl.compute_baseline([], _ref_verdict), None)
check("空可疑清单（None）→ None", cl.compute_baseline(_ref_ph, None), None)
# 空可疑清单但传了空列表（不是 None）→ 可疑率 0（母语者全对）
_bl_zero = cl.compute_baseline(_ref_ph, [])
check("可疑清单为空列表 → 可疑率 0", _bl_zero["可疑率"] if _bl_zero else None, 0.0)

# ⚠️ 必须**真跑一次生成**看产物，不能查源码里有没有那个词 ——
#    查源码的话，注释里也写着关键字段，删掉真正那行照样绿
#    （变异测试发现：查源码的断言太弱，测不出真问题）。
import json as _json_mod
_gp_out = os.path.join(os_.OUT_DIR, "_t_selftest_gen_out.json")
_gp_orig = gp.OUT_JSON
try:
    import io as _io
    import contextlib as _ctx
    gp.OUT_JSON = _gp_out
    with _ctx.redirect_stdout(_io.StringIO()):     # 生成过程会打印，挡掉
        gp.main()          # 用默认文本生成一次
    check("生成成功，产物存在", os.path.exists(_gp_out), True)
    if os.path.exists(_gp_out):
        with open(_gp_out, encoding="utf-8") as f:
            _gen_out = _json_mod.load(f)
        check("产物里仍有音素和词边界",
              bool(_gen_out.get("音素")) and bool(_gen_out.get("词边界")), True)
finally:
    gp.OUT_JSON = _gp_orig
    if os.path.exists(_gp_out):
        os.remove(_gp_out)

# 空输入不崩
check("空逐词列表 → 返回空", gp.align_words_to_phonemes([], ["a"])[0], [])
check("空整句 → 返回零宽区间", gp.align_words_to_phonemes([["a"]], [])[0], [(0, 0)])


# ==========================================================================
# 按句重音聚合 + 报告渲染（10-28：重音位置从「只报漏了啥」改成「逐句三行」）
# ==========================================================================
group("按句重音聚合与逐句渲染")

# build_per_sentence_stress 的输入：块序号->句号、两侧突出块、块名映射、句子结果
_sent_res = [
    {"句号": 1, "文本": "hey that was fun", "漏句": False, "忽略": False},
    {"句号": 2, "文本": "Thanks for the lesson", "漏句": False, "忽略": False},
    {"句号": 3, "文本": "Really", "漏句": False, "忽略": False},
]
# 句1 块1-4，句2 块5-8，句3 块9
_bmap = {1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 3}
_nr = {1: "hey", 2: "that", 3: "was", 4: "fun", 5: "Thanks", 6: "for",
       7: "the", 8: "lesson", 9: "Really"}
_nm = dict(_nr)
# 原音频突出 fun(4)、lesson(8)；我突出 fun(4)、Thanks(5)
_mr = [4, 8]
_mm = [4, 5]
_per = cl.build_per_sentence_stress(_bmap, _mr, _mm, _nr, _nm, _sent_res)
check("每句都返回（含无重音句）", len(_per), 3)
_s1 = _per[0]
check("句1 应读重音", _s1["应读重音"], ["fun"])
check("句1 实读重音", _s1["实读重音"], ["fun"])
check("句1 漏读重音（空）", _s1["漏读重音"], [])
check("句1 多读重音（空）", _s1["多读重音"], [])
_s2 = _per[1]
check("句2 漏读 lesson", _s2["漏读重音"], ["lesson"])
check("句2 多读 Thanks", _s2["多读重音"], ["Thanks"])
check("句2 应读重音（原突出的是 lesson，Thanks 是我多读的）", _s2["应读重音"], ["lesson"])
_s3 = _per[2]
check("句3 无重音句也应返回", _s3["应读重音"], [])
check("句3 实读重音空", _s3["实读重音"], [])

# _mark_stressed_words：把重读词用【】框进原句，不区分大小写、按词边界
check("框词（大小写不敏感）",
      os_._mark_stressed_words("Thanks for the lesson", ["thanks", "lesson"]),
      "【Thanks】 for the 【lesson】")
check("框词不命中则原样",
      os_._mark_stressed_words("hey that was fun", ["nope"]),
      "hey that was fun")
check("空重音列表返回原句",
      os_._mark_stressed_words("Really", []), "Really")

# _render_stress_by_sentence：三行结构（应读/实读/纠音）
_rh = {"重音_信号": {"按句重音": _per}}
_rlines = os_._render_stress_by_sentence(_rh)
check("渲染非空", bool(_rlines), True)
check("包含应读行", any("应读重音" in ln for ln in _rlines), True)
check("包含实读行", any("你读的重音" in ln for ln in _rlines), True)
check("句2 纠音含「漏了」", any("漏了" in ln and "lesson" in ln for ln in _rlines), True)
check("句2 纠音含「多了」", any("多了" in ln and "Thanks" in ln for ln in _rlines), True)
# 无按句数据（旧 JSON / 单句）→ 返回空列表，报告不崩
check("无按句数据返回空列表", os_._render_stress_by_sentence({}), [])
check("空重音信号返回空列表", os_._render_stress_by_sentence({"重音_信号": {}}), [])


# ==========================================================================
# 强制对齐（10-29：forced_align 单调分配 + 逐音后验判错）
#   锚定：DP 不能用「编辑距离」（允许跳参考音素会走捷径，对齐全崩）
#   必须用「单调分配」，每个参考音素匹配到严格递增的帧。
# ==========================================================================
group("强制对齐（单调分配 + 逐音后验）")

import numpy as np
# ⚠️ 词表约定与真实 wav2vec2-espeak 一致：0=blank，1/2/3=special，
#    音素从 4 开始。测试数据若用 1/2/3 当音素 id，会被当成 special 误判
#    （踩过：b 的实读被判成"没读出来"）。
_fa_id2tok = {0: "<pad>", 1: "<s>", 2: "</s>", 3: "<unk>",
              4: "a", 5: "b", 6: "c", 7: "d", 8: "e"}
_fa_logits = np.full((15, 9), -5.0, dtype=np.float32)
for f in range(3): _fa_logits[f, 4] = 5.0         # a: 帧 0-2
for f in range(3, 6): _fa_logits[f, 5] = 5.0      # b: 帧 3-5
for f in range(6, 9): _fa_logits[f, 6] = 5.0      # c: 帧 6-8
for f in range(9, 12): _fa_logits[f, 7] = 5.0     # d: 帧 9-11
for f in range(12, 15): _fa_logits[f, 8] = 5.0    # e: 帧 12-14

_fa_target = ["a", "b", "c", "d", "e"]
_fa_aligned = cp.forced_align(_fa_logits, _fa_target, _fa_id2tok)
check("对齐音素数 = 参考音素数", len(_fa_aligned), 5)
check("帧位置精确命中（a→0, b→3, c→6, d→9, e→12）",
      [it["帧"] for it in _fa_aligned], [0, 3, 6, 9, 12])
check("全部后验接近 1", all(it["后验"] > 0.9 for it in _fa_aligned), True)
# 帧严格递增（单调，无跳过）
_frames = [it["帧"] for it in _fa_aligned]
check("帧严格递增", all(_frames[i] < _frames[i + 1] for i in range(len(_frames) - 1)), True)

# 缺失音：b 缺失，后验应极低，且**不拉低其他音的后验**（踩过：用累计 best 当后验）
_fa_miss_logits = np.full((15, 9), -5.0, dtype=np.float32)
for f in range(3): _fa_miss_logits[f, 4] = 5.0        # a
for f in range(3, 6): _fa_miss_logits[f, 6] = 5.0     # c（b 缺失）
for f in range(6, 9): _fa_miss_logits[f, 7] = 5.0     # d
for f in range(9, 12): _fa_miss_logits[f, 8] = 5.0    # e
for f in range(12, 15): _fa_miss_logits[f, 0] = 5.0   # 空白
_fa_aligned2 = cp.forced_align(_fa_miss_logits, _fa_target, _fa_id2tok)
_fa_b = [it for it in _fa_aligned2 if it["音素"] == "b"][0]
_fa_others = [it for it in _fa_aligned2 if it["音素"] != "b"]
check("缺失的 b 后验极低", _fa_b["后验"] < 0.01, True)
check("其他音后验不受影响（仍 >0.9）", all(it["后验"] > 0.9 for it in _fa_others), True)

# 空输入不崩
check("空参考音素 → 空列表", cp.forced_align(_fa_logits, [], _fa_id2tok), [])
check("空 logits → 空列表",
      cp.forced_align(np.zeros((0, 9), dtype=np.float32), _fa_target, _fa_id2tok), [])

# align_phone_verdict：后验低 → 判可疑，并按实读符号分类
_fa_argmax = _fa_miss_logits.argmax(axis=1).tolist()
_verdict = cp.align_phone_verdict(_fa_aligned2, _fa_target, _fa_argmax, _fa_id2tok,
                                  threshold=0.5)
check("判出可疑音（b）", len(_verdict), 1)
check("可疑音是 b", _verdict[0]["音素"], "b")
check("b 类别是「其他」（实读是 a，不在 L2 表）", _verdict[0]["类别"], "其他")
check("b 实读符号是 a", _verdict[0]["实读"], "a")

# 口音型：θ→f 命中 L2 表。
# ⚠️ 必须带上下文音素（θ+æ），否则单个 θ 会对齐到"空白帧"（那里均匀分布
#    概率最高），而不是 f 的帧 —— 单调约束下，后面有 æ 才能把 θ 逼到 f 区域。
_fa_th_id2tok = {0: "<pad>", 1: "<s>", 2: "</s>", 3: "<unk>",
                 4: "θ", 5: "æ", 6: "f"}
_fa_th_logits = np.full((9, 7), -5.0, dtype=np.float32)
for f in range(3): _fa_th_logits[f, 6] = 5.0        # 帧 0-2 读成 f（θ 该在这）
for f in range(3, 6): _fa_th_logits[f, 5] = 5.0     # 帧 3-5 读成 æ
for f in range(6, 9): _fa_th_logits[f, 0] = 5.0     # 帧 6-8 空白
_fa_th_aligned = cp.forced_align(_fa_th_logits, ["θ", "æ"], _fa_th_id2tok)
_fa_th_verdict = cp.align_phone_verdict(_fa_th_aligned, ["θ", "æ"],
                                        _fa_th_logits.argmax(axis=1).tolist(),
                                        _fa_th_id2tok, threshold=0.5)
_th_l2 = [v for v in _fa_th_verdict if v["音素"] == "θ" and v["类别"] == "口音型"]
check("θ→f 判为口音型", len(_th_l2) == 1, True)
check("θ 的实读是 f", _th_l2[0]["实读"] if _th_l2 else None, "f")
check("可疑音带「下标」字段（供长音频归句）", _th_l2[0].get("下标") is not None, True)


# ==========================================================================
# 转写文本修复（缩略词 + 粘连词分割）
# ==========================================================================
group("转写文本修复（缩略词 + 粘连词分割）")

import transcript_fix as tf

_tf1, _tf1c = tf.fix_transcript("Im in charge")
check("缩略词 Im → I'm", _tf1, "I'm in charge")
check("缩略词修复记录 1 条", len(_tf1c), 1)

_tf2, _tf2c = tf.fix_transcript("wellsort of")
check("粘连 wellsort → well sort", _tf2, "well sort of")
check("粘连修复记录 1 条", len(_tf2c), 1)

_tf3, _ = tf.fix_transcript("ohNo problem")
check("大小写粘连 ohNo → oh No（保留旧规则）", _tf3, "oh No problem")

# 不误拆正常词：wordninja 会把 blading → b+lading，必须被 MIN_SEG_LEN 挡住
check("blading 不误拆", tf.split_glued_word("blading"), None)
check("computer 不误拆", tf.split_glued_word("computer"), None)
check("into 不误拆", tf.split_glued_word("into"), None)

# 含撇号的不碰（it'ss 这类听错的留给人工）
_tf4, _ = tf.fix_transcript("it'ss a deal")
check("含撇号词不分割（it'ss 保持原样）", "it'ss" in _tf4, True)

# 空输入不崩
check("空文本返回空", tf.fix_transcript(""), ("", []))
check("None 文本不崩", tf.fix_transcript(None), (None, []))


# ==========================================================================
# 标点恢复（subword 合并 + 词尾标点；只测纯函数，不加载 1.1GB 模型）
# ==========================================================================
group("标点恢复（subword 合并 + 词尾标点）")

import punctuate as pu

# 词尾标点：每个词后面跟它词尾 token 的标签
check("词尾标点插入", pu._merge_and_punct(
    ["▁specialist", "▁well", "▁sort"], [".", "0", "."]),
    ["specialist.", "well", "sort."])
# subword 合并：special + ist → specialist，取最后一个 subword 的标签
check("subword 合并取词尾标签", pu._merge_and_punct(
    ["▁special", "ist", "▁well"], ["0", ".", "0"]),
    ["specialist.", "well"])
# 无标点标签（0/O/空）不加标点
check("无标点标签不加", pu._merge_and_punct(
    ["▁hello", "▁world"], ["0", "O"]), ["hello", "world"])
# 特殊 token 跳过
check("特殊 token 跳过", pu._merge_and_punct(
    ["<s>", "▁hello", "</s>"], ["0", ".", "0"]), ["hello."])
# 空输入
check("空 token 返回空", pu._merge_and_punct([], []), [])


# ==========================================================================
# 原音频缓存失效检测（10-30：换原音频后旧缓存不自动失效）
# ==========================================================================
group("原音频缓存失效检测（锚定换素材错乱不报错的坑）")

import audio_paths as ap

# 指纹解析：list/tuple 都认；旧缓存无字段 → None
check("指纹从 list 读回", ap.reference_cache_fingerprint({"原音频指纹": [1, 2]}), (1, 2))
check("指纹从 tuple 读回", ap.reference_cache_fingerprint({"原音频指纹": (3, 4)}), (3, 4))
check("旧缓存无指纹字段 → None", ap.reference_cache_fingerprint({"转写文本": "x"}), None)
check("空缓存 → None", ap.reference_cache_fingerprint(None), None)
check("指纹字段损坏 → None", ap.reference_cache_fingerprint({"原音频指纹": "abc"}), None)

# 过期判定：用真实临时文件测（os.stat 拿真实 mtime/size）
import tempfile as _tempfile
_fp = os.path.join(os_.OUT_DIR, "_t_selftest_ref.mp3")
try:
    with open(_fp, "wb") as f:
        f.write(b"x" * 100)
    _real_fp = ap._file_fingerprint(_fp)
    check("真实文件能算出指纹（mtime+size）", _real_fp is not None and len(_real_fp) == 2, True)
    # 同一文件 → 未过期
    check("指纹一致 → 未过期",
          ap.reference_cache_stale({"原音频指纹": list(_real_fp)}, _fp), False)
    # mtime 变（换文件内容+改大小）→ 过期
    check("size 变了 → 过期",
          ap.reference_cache_stale({"原音频指纹": [_real_fp[0], _real_fp[1] + 1]}, _fp), True)
    check("mtime 变了 → 过期",
          ap.reference_cache_stale({"原音频指纹": [_real_fp[0] + 5, _real_fp[1]]}, _fp), True)
finally:
    if os.path.exists(_fp):
        os.remove(_fp)

check("无指纹字段 → 无法判断（None）",
      ap.reference_cache_stale({"转写文本": "x"}, _fp if os.path.exists(_fp) else "whatever"), None)
check("空缓存 → None", ap.reference_cache_stale(None, "whatever"), None)
check("无原音频路径 → None", ap.reference_cache_stale({"原音频指纹": [10, 100]}, None), None)


# ==========================================================================
# 单词级纠音（10-30：转写模仿音频 → 逐词对齐）
# ==========================================================================
group("单词级纠音（tokenize_words / align_words / split_sentences_text）")

check("tokenize 去标点并小写",
      cl.tokenize_words("Thanks, for the lesson."), ["thanks", "for", "the", "lesson"])
check("tokenize 保留词内撇号",
      cl.tokenize_words("I'm sure."), ["i'm", "sure"])
check("tokenize 空输入 → 空列表", cl.tokenize_words(""), [])
check("tokenize 纯标点 → 空列表", cl.tokenize_words("... !!!"), [])

# 对齐：完全一致 → 全读对
_al = cl.align_words(["thanks", "for", "the", "lesson"], ["thanks", "for", "the", "lesson"])
check("全对 → 4 条全读对",
      (len(_al), all(x["类型"] == "读对" for x in _al)), (4, True))

# 对齐：读错一个词（thanks → sanks）
_al = cl.align_words(["thanks", "for", "the"], ["sanks", "for", "the"])
_types = [x["类型"] for x in _al]
check("读错一个词 → 读错/读对/读对", _types, ["读错", "读对", "读对"])
check("读错的实读是 sanks", _al[0]["实读"], "sanks")

# 对齐：漏读一个词（少了 for）
_al = cl.align_words(["thanks", "for", "the"], ["thanks", "the"])
_types = [x["类型"] for x in _al]
check("漏读一个词 → 读对/漏读/读对", _types, ["读对", "漏读", "读对"])
check("漏读词的原词是 for", [x for x in _al if x["类型"] == "漏读"][0]["原词"], "for")

# 对齐：多读一个词（多了 yeah）
_al = cl.align_words(["thanks", "the"], ["thanks", "yeah", "the"])
_types = [x["类型"] for x in _al]
check("多读一个词 → 读对/多读/读对", _types, ["读对", "多读", "读对"])
check("多读词的实读是 yeah", [x for x in _al if x["类型"] == "多读"][0]["实读"], "yeah")

# 对齐：两边都空 → 空列表
check("两边都空 → 空列表", cl.align_words([], []), [])

# 对齐：大小写不敏感（讯飞统一小写，原词保留大写，Hey vs hey 应判读对）
_al = cl.align_words(["Hey", "thanks"], ["hey", "thanks"])
_types = [x["类型"] for x in _al]
check("大小写不同判读对（Hey vs hey）", _types, ["读对", "读对"])

# 断句：按句末标点切分
check("断句按句号切",
      cl.split_sentences_text("Thanks for the lesson. So tell me. What do you do?"),
      ["Thanks for the lesson.", "So tell me.", "What do you do?"])
check("断句空输入 → 空列表", cl.split_sentences_text(""), [])
check("断句无标点 → 单句", cl.split_sentences_text("hello world"), ["hello world"])


# ==========================================================================
# 转写确认环节（任务 3：confirm_transcript 的纯逻辑分支）
# ==========================================================================
group("转写确认（confirm_transcript 短路/读回分支）")
import analyze_reference as ar

# 1. AUTO_CONFIRM=1 → 跳过确认，直接返回原文本、无改动
_saved_auto = ar._AUTO_CONFIRM
ar._AUTO_CONFIRM = True
try:
    _r = ar.confirm_transcript("hello world")
    check("AUTO_CONFIRM=1 → 直接返回原文本", _r, ("hello world", False))
finally:
    ar._AUTO_CONFIRM = _saved_auto

# 2. 待确认文件已存在 → 看 should_reconfirm 的决策
check("should_reconfirm：空 → 要弹", ar.should_reconfirm("", "x"), True)
check("should_reconfirm：一致 → 不弹", ar.should_reconfirm("same text", "same text"), False)
check("should_reconfirm：不一致 → 要弹", ar.should_reconfirm("old text", "new text"), True)

# 3. 端到端：文件存在且内容与本次一致 → 直接读回、不弹编辑器
_saved_confirm = ar.CONFIRM_TXT
_conf = os.path.join(os_.OUT_DIR, "_t_selftest_confirm.txt")
ar.CONFIRM_TXT = _conf
try:
    with open(_conf, "w", encoding="utf-8") as f:
        f.write("same text\n")
    _r = ar.confirm_transcript("same text")
    check("已有且内容相同 → 读回、无改动", _r, ("same text", False))
finally:
    ar.CONFIRM_TXT = _saved_confirm
    if os.path.exists(_conf):
        os.remove(_conf)

# split_lines / _collapse（每句一行展示 + 读回压单行）
check("split_lines 按句末标点切行",
      ar.split_lines("Thanks for the lesson. So tell me. What do you do?"),
      ["Thanks for the lesson.", "So tell me.", "What do you do?"])
check("split_lines 空输入 → 空列表", ar.split_lines(""), [])
check("_collapse 换行压回单空格",
      ar._collapse("Thanks.\nSo tell me.\nWhat?"), "Thanks. So tell me. What?")


# ==========================================================================
# 汇总
# ==========================================================================
total = len(_PASS) + len(_FAIL)
print("\n" + "=" * 62)
print("自检结果：%d/%d 通过" % (len(_PASS), total))
if _FAIL:
    print("失败用例：")
    for g, d in _FAIL:
        print("   · [%s] %s" % (g, d))
    print("=" * 62)
    sys.exit(1)
print("全部通过。")
print("=" * 62)
sys.exit(0)