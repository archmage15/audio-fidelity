# -*- coding: utf-8 -*-
"""离线预生成参考音素表 —— 调 espeak-ng，产出纯数据 JSON。

【为什么必须"离线预生成"】
espeak-ng 是 GPL-3.0（copyleft，有传染性）。按计划书决策 3：
运行时脚本**绝不 import / 链接**它 —— 只在本机离线跑一次，把结果落成
纯数据文件（参考音素.json），之后所有脚本都只读这个 JSON。

【输入】参考文本，必须写**标准形式**（`want to` 而不是 `wanna`）。
    角色用 | 分隔 —— | 之前是角色 A，之后是角色 B（对齐用，见计划书 §5.5）。
    例：  so Where are you working now Terry|oh im still at the bank

【输出】..\audio\参考音素.json
    {
      "文本": "...",
      "音素": ["s", "oʊ", ...],                              # 整句音素序列
      "词边界": [{"word": "so", "a": 0, "b": 2, "角色": "A"}, ...]
    }                                                        # a/b 是音素下标

【用法】
    python gen_ref_phonemes.py --probe              # 只看 espeak 原始输出（换文本时用）
    python gen_ref_phonemes.py                      # 用 DEFAULT_TEXT 生成
    python gen_ref_phonemes.py "your text here"     # 指定文本

【⚠️ 为什么是"整句跑"而不是"逐词跑"（09-24 实测确定）】
espeak-ng 的 `--ipa` 输出**按空格分的是"词"，词内音素是连续串、无分隔符**，
且**重音标记 ˈ ˌ 夹在音素中间**。实测整句跑的输出段数与输入词数一一对应：
    'so Where are you working now'  →  'sˌoʊ wˌɛɹ ɑːɹ juː wˈɜːkɪŋ nˈaʊ'   (6 段对 6 词)
所以整句跑一次就能同时拿到「词边界」和「句中真实发音」——
比逐词调用更准（逐词会把 so 读成 sˈoʊ，句中其实是 sˌoʊ，重音位置都不同）。
词内音素切分用**最长匹配**，词典直接取模型自己的符号表 ——
保证切出来的 token 一定在模型词表里，后续 diff 不会因符号集不同而误配。

【⚠️⚠️ 10-26 修：espeak 的"段"不等于"词"（连读合并会错位）】
上面那条"段数 = 词数"**只在简单句成立**。espeak 会做英语连读合并，
把弱读功能词挂到前一个词上：
    'for the lesson'  →  'fɚðə lɛsən'    ← 2 个词挤成 1 段！
于是段数(86) ≠ 词数(87)，**从合并点开始所有词的音素区间整体错位一格**：
    词"the" 的区间装的是 lesson 的音、词"lesson" 的区间装的是 oh 的音……
后果：报告里「单词 X 实读 Y」的指控**指错了词**；句子边界跟着一起错；
节奏/重音的词级归因也不可信。实测首音吻合率只有 6%（2/35 个功能词）。

修法（两路取长补短）：
  1) **整句跑**  → 拿音素**本体**（保留句中真实发音，如 for 读 f ɚ 不是 f ɔːɹ）
  2) **逐词跑**  → 拿「词 → 音素」的对应关系
  3) **DP 对齐**把 2 对到 1 上，重建每个词在整句音素序列里的区间
实测代价 7/267、区间完整覆盖无空隙、首音吻合率 6% → 84%。
剩下的 16% 不是 bug，是整句/逐词读法的**真实差异**（如 for：f ɚ vs f ɔːɹ）。

⚠️ 逐词跑必须**先剥重音标记再切分** —— 否则 'ˈ' 会被当单字符切进音素里，
   DP 代价从 7 飙到 222，对齐全乱。
"""
import json
import os
import re
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_JSON = os.path.join(os.environ.get("AUDIO_TEST_WORK_DIR") or os.path.join(ROOT, "audio"), "参考音素.json")
MODEL_DIR = os.environ.get("PHONEME_MODEL_DIR") or os.path.join(ROOT, "models", "wav2vec2-espeak")
VOCAB_JSON = os.path.join(MODEL_DIR, "vocab.json")


# espeak-ng 可执行文件的候选位置（msi 装完通常不写 PATH）
from platform_tools import espeak_candidates
CANDIDATES = espeak_candidates()

# 从 espeak 输出里剥掉的"非音素"标记：主重音 / 次重音 / 撇号
STRIP_CHARS = "ˈˌ'"

# espeak 的 -v 音色（美音，与现有素材一致）
VOICE = "en-us"

# 模型词表里的特殊 token，不能当切分词典
SPECIAL_TOKENS = {"<pad>", "<s>", "</s>", "<unk>", "??"}


def find_espeak():
    """返回 (可执行路径, 版本号)。找不到返回 (None, None)。"""
    for cand in CANDIDATES:
        try:
            r = subprocess.run([cand, "--version"], capture_output=True, timeout=15)
            if r.returncode == 0:
                return cand, r.stdout.decode("utf-8", "replace").strip()
        except Exception:
            continue
    return None, None


def raw_phonemize(exe, text):
    """原样返回 espeak-ng 的 IPA 输出（不做解析）—— 供 probe 用。"""
    r = subprocess.run([exe, "-q", "--ipa", "-v", VOICE, text],
                       capture_output=True, timeout=20)
    return (r.returncode,
            r.stdout.decode("utf-8", "replace"),
            r.stderr.decode("utf-8", "replace"))


def load_keyset():
    """读模型符号表，返回 (词典集合, 最长 token 长度, 全集) —— 切分用。"""
    if not os.path.isfile(VOCAB_JSON) or os.path.getsize(VOCAB_JSON) == 0:
        from compare_phoneme import ensure_model
        ensure_model()
    with open(VOCAB_JSON, encoding="utf-8") as f:
        vocab = json.load(f)
    keyset = {k for k in vocab.keys() if k not in SPECIAL_TOKENS and k.strip()}
    max_len = max((len(k) for k in keyset), default=1)
    return keyset, max_len, set(vocab.keys())


def strip_marks(s):
    for ch in STRIP_CHARS:
        s = s.replace(ch, "")
    return s


def split_phonemes(s, keyset, max_len):
    """把"词内连续音素串"切成音素 token —— 动态规划，保证每个 token 都在词典里。

    ⚠️ 为什么不能用贪心最长匹配：实测 "juː" 会被切成 "ju" + "ː"，
       而 "ː" 不在模型符号表里 —— 贪心不会回溯。DP 会退一步试 "j" + "uː" ✓

    无可行切分时返回 (单字符列表, 那些字符)：不静默丢字符 ——
    制造"看起来对、其实漏了音"的假象，比直接报出来危险得多。
    """
    n = len(s)
    memo = {}

    def solve(i):
        if i >= n:
            return ()
        if i in memo:
            return memo[i]
        memo[i] = None
        for L in range(min(max_len, n - i), 0, -1):
            t = s[i:i + L]
            if t in keyset:
                rest = solve(i + L)
                if rest is not None:
                    memo[i] = (t,) + rest
                    return memo[i]
        return None

    res = solve(0)
    if res is None:
        return [c for c in s], list(s)
    return list(res), []


def clean_word(w):
    """去掉首尾标点，只留字母与撇号。"""
    return re.sub(r"^[^\w']+|[^\w']+$", "", w, flags=re.UNICODE)


# ---- 词边界重建（10-26 修：espeak 连读合并会导致区间错位）----------------

# DP 对齐的带宽（占序列长度的比例）+ 绝对下限。
# 带状 = 只在「对角线附近」算，省时间也防跑飞。实测 267 音时带 53 够用。
BAND_RATIO = 0.20
BAND_MIN = 40


def align_words_to_phonemes(word_phons, sent_phons, band_ratio=BAND_RATIO,
                            band_min=BAND_MIN):
    """把「逐词音素」对齐到「整句音素」，重建每个词在整句里的区间。

    输入
      word_phons : list[list[str]]  逐词跑 espeak 得到的音素（每个词一段）
      sent_phons : list[str]        整句跑 espeak 得到的音素序列

    输出
      (ranges, cost)
        ranges : list[(start, end)] —— 每个词在 sent_phons 里的闭开区间；
                 没有对齐上的词给 (x, x)（零宽，由调用方补齐）
        cost   : DP 总代价（对齐错得越离谱越大；实测正常素材 ≤ 20）

    为什么用 DP 而不是贪心：
      连读合并是「一对多」的映射（2 词挤成 1 段），贪心会在第一个不匹配处
      就崩掉，后面全错位。DP 允许「替换 / 跳过 / 多出」三种代价，
      能把错位吸收在少数几个音上。

    代价定义（Levenshtein 变体）：
      替换 = 1（音素不同）
      跳过整句音素 = 1（整句多出来的）
      跳过逐词音素 = 1（逐词多出来的）
    """
    flat = []
    owner = []
    for wi, toks in enumerate(word_phons):
        for t in toks:
            flat.append(t)
            owner.append(wi)
    n, m = len(flat), len(sent_phons)
    if n == 0 or m == 0:
        return [(0, 0)] * len(word_phons), 0

    INF = 10 ** 9
    band = max(band_min, int(n * band_ratio))
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    bt = [[0] * (m + 1) for _ in range(n + 1)]
    # 回溯动作常量（写清语义，避免「跳过谁」搞反 —— 搞反了会静默错位）
    MATCH = 1       # 消耗 flat[i-1] 和 sent[j-1]
    SKIP_WORD = 2   # 只消耗 flat[i-1]（逐词多出来的音）
    SKIP_SENT = 3   # 只消耗 sent[j-1]（整句多出来的音）
    dp[0] = list(range(m + 1))          # 整句多出来的音全跳过
    for j in range(1, m + 1):
        bt[0][j] = SKIP_SENT
    for i in range(1, n + 1):
        dp[i][0] = i                     # 逐词多出来的音全跳过
        bt[i][0] = SKIP_WORD
        center = int(round(i * m / float(n)))
        lo = max(1, center - band)
        hi = min(m, center + band)
        row, prow = dp[i], dp[i - 1]
        brow = bt[i]
        ai = flat[i - 1]
        for j in range(lo, hi + 1):
            best = prow[j - 1] + (0 if ai == sent_phons[j - 1] else 1)
            move = MATCH
            if prow[j] + 1 < best:       # dp[i-1][j] + 1 → 跳过逐词音素
                best = prow[j] + 1
                move = SKIP_WORD
            if row[j - 1] + 1 < best:    # dp[i][j-1] + 1 → 跳过整句音素
                best = row[j - 1] + 1
                move = SKIP_SENT
            row[j] = best
            brow[j] = move

    # 回溯出「逐词音素 ↔ 整句音素」的配对
    pairs = []
    i, j = n, m
    while i > 0 and j > 0:
        mv = bt[i][j]
        if mv == MATCH:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif mv == SKIP_WORD:
            i -= 1
        else:
            j -= 1
    pairs.reverse()

    ranges = [None] * len(word_phons)
    for (ii, jj) in pairs:
        wi = owner[ii]
        cur = ranges[wi]
        if cur is None:
            ranges[wi] = (jj, jj + 1)
        else:
            ranges[wi] = (min(cur[0], jj), max(cur[1], jj + 1))
    # 空区间在**这里**补齐，不留给调用方 ——
    # 契约：返回的区间列表里不会有 None，调用方直接用就行。
    # （把补齐放在外面的话，忘了调就静默漏词 —— 变异测试发现这个风险，已踩）
    ranges = fill_empty_ranges(ranges, len(sent_phons))
    return ranges, dp[n][m]


def fill_empty_ranges(ranges, total):
    """把零宽区间补齐 —— 没有对齐上的词，按前后邻居推断它该占哪。

    补法：该词的区间取「前一个词的结束」到「后一个词的开始」的中点，
    尽量少动别的词的区间。没邻居就用 0 或 total。
    """
    out = list(ranges)
    for i in range(len(out)):
        if out[i] is not None and out[i][1] > out[i][0]:
            continue
        prev_b = 0
        for k in range(i - 1, -1, -1):
            if out[k] and out[k][1] > out[k][0]:
                prev_b = out[k][1]
                break
        next_a = None
        for k in range(i + 1, len(out)):
            if out[k] and out[k][1] > out[k][0]:
                next_a = out[k][0]
                break
        if next_a is None:
            a = b = min(prev_b, total)
        elif prev_b >= next_a:
            a = b = next_a
        else:
            a = b = (prev_b + next_a) // 2
        out[i] = (a, b)
    return out


def phonemize_words(exe, words, keyset, max_len):
    """逐词跑 espeak，拿每个词的音素列表。

    一次调用传全部词（每词一行）—— espeak 按行分开处理，返回行数 == 词数。
    比逐词调用快几十倍（87 词：0.4 秒 vs 几秒）。

    ⚠️ 必须先剥重音标记再切分 —— 否则 'ˈ' 会作为单字符混进音素里，
       DP 代价从 7 飙到 222，对齐全乱（10-26 实测）。
    """
    if not words:
        return []
    payload = "\n".join(words)
    try:
        r = subprocess.run([exe, "-q", "--ipa", "-v", VOICE], input=payload,
                           capture_output=True, text=True, encoding="utf-8",
                           timeout=60)
        lines = [ln.strip() for ln in (r.stdout or "").splitlines() if ln.strip()]
    except Exception:
        lines = []
    # 行数对不上（espeak 吞了某行/拆了某行）→ 退回逐词调用
    if len(lines) != len(words):
        lines = []
        for w in words:
            rc, out, err = raw_phonemize(exe, w)
            seg = strip_marks(out).strip()
            lines.append(seg)
    result = []
    for ln in lines:
        toks, unk = split_phonemes(strip_marks(ln), keyset, max_len)
        result.append(toks)
    return result


def no_espeak():
    print("X 找不到 espeak-ng。请先安装，或把它所在目录加进 PATH。候选位置：")
    for c in CANDIDATES:
        print("    " + c)
    return 1


def probe(exe, ver):
    print("espeak-ng : %s" % exe)
    print("version   : %s" % ver)
    print()
    for w in ["so", "working", "Terry", "bank", "the", "think"]:
        rc, out, err = raw_phonemize(exe, w)
        print("  %-9s rc=%d  repr=%s" % (w, rc, repr(out)))
    rc, out, err = raw_phonemize(exe, "so Where are you working now")
    print()
    print("  整句 repr=%s" % repr(out))
    return 0


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    exe, ver = find_espeak()
    if exe is None:
        return no_espeak()

    if "--probe" in sys.argv:
        return probe(exe, ver)

    print("espeak-ng : %s" % exe)
    print("version   : %s" % ver.strip())

    if not args or not args[0].strip():
        print("X 必须传入本次原音频的转写文本；请先运行原音频分析。")
        return 2
    text = args[0]
    print("文本      : %s" % text)
    print()

    keyset, max_len, vocab_all = load_keyset()
    print("切分词典  : 模型符号表 %d 项（已排除特殊 token，最长 %d 字符）" % (len(keyset), max_len))
    print()

    segments = [s.strip() for s in text.split("|") if s.strip()]
    role_names = "ABCDEFGH"

    phonemes, word_bounds = [], []
    all_unknown, problems = [], []
    sentences = []          # 句子边界：[{"句号":1, "文本":"...", "a":0, "b":32, "角色":"A"}]
                          # a/b 是音素下标（闭开区间）。用于长音频按句评测：
                          # 参考序列按句切段，模仿音素按内容对齐进各句区间。

    for si, seg in enumerate(segments):
        role = role_names[si] if si < len(role_names) else "?"
        words = [w for w in (clean_word(x) for x in seg.split()) if w]
        # 按句末标点（. ? ! ；;）把词切成句子 —— 只切文本，不用任何时间戳。
        # ⚠️ 必须在**原始词**（未剥标点）上判断句末：clean_word 已把标点去掉，
        #    拿剥完的词判断会一句都切不出来（实测踩过）。
        raw_words = [x for x in seg.split() if clean_word(x)]
        sent_groups, cur = [], []
        for rw in raw_words:
            cur.append(clean_word(rw))
            if rw and rw[-1] in ".?!;":
                sent_groups.append(cur)
                cur = []
        if cur:
            sent_groups.append(cur)
        rc, out, err = raw_phonemize(exe, seg)
        if rc != 0:
            print("X espeak-ng 失败：%s" % err.strip())
            return 2
        parts = strip_marks(out).split()

        if len(parts) != len(words):
            # ⚠️ 段数对不上：espeak 把相邻词连读合并了（如 for the -> fɚðə）。
            #    这是**英语连读规则**，不是错误；但意味着「第 i 段 = 第 i 词」不成立，
            #    按索引硬套会让后面所有词的音素区间整体错位（10-26 实测）。
            #    修法：下面用「整句音素 + 逐词音素 + DP 对齐」重建区间。
            problems.append("角色 %s：输入 %d 词，espeak 输出 %d 段（连读合并）"
                            "—— 已用 DP 对齐重建词边界" % (role, len(words), len(parts)))
        print("角色 %s：%d 词，%d 句" % (role, len(words), len(sent_groups)))

        # 词 -> 句号 映射（按词序累加，与 espeak 段错位时也保持句子结构）
        word2sent = {}
        wi_sent = 0
        for gi, g in enumerate(sent_groups):
            for _ in g:
                if wi_sent < len(words):
                    word2sent[wi_sent] = gi
                    wi_sent += 1

        # ---- 音素本体：整句跑的音素（保留句中真实发音）----
        seg_phonemes = []
        for p in parts:
            toks, unk = split_phonemes(p, keyset, max_len)
            all_unknown.extend(unk)
            seg_phonemes.extend(toks)

        # ---- 词边界：逐词跑 + DP 对齐（10-26 修）----
        word_phons = phonemize_words(exe, words, keyset, max_len)
        ranges, dp_cost = align_words_to_phonemes(word_phons, seg_phonemes)

        # DP 代价高 = 对齐不可信（词边界可能还是错的）。这时候要**报出来**，
        # 不能装作没事 —— 词边界错位会让整份报告的指控全部指错词。
        if dp_cost > max(10, len(seg_phonemes) * 0.10):
            problems.append("角色 %s：DP 对齐代价 %d（音素 %d 个）偏高 —— "
                            "词边界可能不准，请人工核对" % (role, dp_cost, len(seg_phonemes)))

        seg_base = len(phonemes)
        for wi, w in enumerate(words):
            a, b = ranges[wi]
            a = min(max(a, 0), len(seg_phonemes))
            b = min(max(b, a), len(seg_phonemes))
            gi = word2sent.get(wi, len(sent_groups) - 1 if sent_groups else 0)
            word_bounds.append({"word": w, "a": seg_base + a, "b": seg_base + b,
                                "角色": role, "句号": gi + 1})
            print("  %-10s %s" % (w, " ".join(seg_phonemes[a:b]) if b > a else "（无）"))
        phonemes.extend(seg_phonemes)

        # 汇总每句的音素区间
        for gi, g in enumerate(sent_groups):
            idxs = [wi for wi in range(len(words)) if word2sent.get(wi) == gi]
            if not idxs:
                continue
            # 直接用该句首尾词的音素下标
            first_wb = word_bounds[len(word_bounds) - len(words) + idxs[0]]
            last_wb = word_bounds[len(word_bounds) - len(words) + idxs[-1]]
            sentences.append({"句号": len(sentences) + 1,
                              "文本": " ".join(g),
                              "a": first_wb["a"], "b": last_wb["b"],
                              "角色": role})

    payload = {"文本": text, "音素": phonemes, "词边界": word_bounds,
               "句子边界": sentences}
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    print()
    print("共 %d 个词，%d 个音素" % (len(word_bounds), len(phonemes)))
    print("已写出：%s" % OUT_JSON)

    # ---- 两道自检 ----
    print()
    missing = sorted(set(phonemes) - vocab_all)
    if missing:
        print("⚠️ 以下符号模型符号表里没有（会影响对比，需 normalize_tok 处理）：")
        print("   " + "  ".join(missing))
    else:
        print("✓ 用到的音素符号模型表里全有")

    if all_unknown:
        print("⚠️ 切分时遇到词典外的字符（已按单字符保留）：")
        print("   " + "  ".join(sorted(set(all_unknown))))

    # 词边界自检：必须单调递增、完整覆盖音素序列、首尾不越界。
    # 这道检查护的是「词边界错位」这个 bug —— 它不会让程序崩，
    # 但会让报告里所有「单词 X」的指控指错词（10-26 踩过）。
    print()
    wb_ok = True
    prev_b = 0
    for i, w in enumerate(word_bounds):
        if w["a"] < prev_b:
            print("⚠️ 词边界 #%d (%s) a=%d 小于上一个词的 b=%d —— 区间不单调"
                  % (i, w["word"], w["a"], prev_b))
            wb_ok = False
        if w["b"] < w["a"]:
            print("⚠️ 词边界 #%d (%s) b=%d < a=%d —— 区间反了"
                  % (i, w["word"], w["b"], w["a"]))
            wb_ok = False
        prev_b = w["b"]
    if word_bounds and prev_b != len(phonemes):
        print("⚠️ 词边界末尾 b=%d ≠ 音素总数 %d —— 有音素没归到任何词"
              % (prev_b, len(phonemes)))
        wb_ok = False
    if wb_ok:
        print("✓ 词边界单调递增，完整覆盖 %d 个音素" % len(phonemes))

    if problems:
        print()
        print("⚠️ 词/段对应问题（需人工确认）：")
        for p in problems:
            print("   " + p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
