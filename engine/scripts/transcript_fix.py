# -*- coding: utf-8 -*-
"""转写文本修复：缩略词规则 + 粘连词分割。

讯飞 ASR 的两类典型错字（都不是拼错，是「丢分隔符」）：
  1. 撇号丢失：Im → I'm（讯飞常把撇号吞掉）
  2. 词粘连：  wellsort → well sort（丢空格，两词黏成一个）

为什么不能裸用 wordninja：
  实测 wordninja 会把正常词误拆（blading → b + lading、roller 也易被拆）。
  所以这里：
    - 缩略词规则先跑（无歧义、整词替换）
    - 粘连分割只对「疑似粘连」的纯字母词跑，且只接受
      「拆成 ≥2 段、且每段都 ≥ MIN_SEG_LEN」的结果，
      单字符碎片（b + lading）一律拒绝、保持原词。
"""

import re

try:
    import wordninja
    _HAS_WORDNINJA = True
except ImportError:
    _HAS_WORDNINJA = False

# 撇号丢失的常见缩略词（只列**无歧义**的；its / cant / ill / well 有歧义不列）。
# 每项 (错写, 正确)，显式列大小写变体（句首大写 vs 句中全小写）。
CONTRACTIONS = [
    ("Im", "I'm"), ("im", "I'm"),
    ("dont", "don't"), ("Dont", "Don't"),
    ("wont", "won't"), ("Wont", "Won't"),
    ("youre", "you're"), ("Youre", "You're"),
    ("weve", "we've"), ("Weve", "We've"),
    ("theyre", "they're"), ("Theyre", "They're"),
    ("isnt", "isn't"), ("Isnt", "Isn't"),
    ("wasnt", "wasn't"), ("Wasnt", "Wasn't"),
    ("doesnt", "doesn't"), ("Doesnt", "Doesn't"),
    ("didnt", "didn't"), ("Didnt", "Didn't"),
    ("ive", "I've"),
    ("youve", "you've"), ("Youve", "You've"),
    ("hes", "he's"), ("Hes", "He's"),
    ("shes", "she's"), ("Shes", "She's"),
    ("theyll", "they'll"), ("Theyll", "They'll"),
    ("youll", "you'll"), ("Youll", "You'll"),
    ("thats", "that's"), ("Thats", "That's"),
]

# wordninja 分割的最小片段长度：挡住 blading → b + lading 这种单字符误拆。
MIN_SEG_LEN = 2
# 只对 >= 这个长度的纯字母词尝试分割（粘连词通常是两个 ≥2 字符词，总长 ≥5）。
MIN_GLUE_LEN = 5


def split_glued_word(word):
    """对疑似粘连词做分割。返回拆分后的空格串，或 None（不拆/拆不开）。

    只接受「拆成 ≥2 段、每段都 ≥ MIN_SEG_LEN」的结果；否则返回 None。
    """
    if not _HAS_WORDNINJA:
        return None
    if not word or not word.isalpha() or len(word) < MIN_GLUE_LEN:
        return None
    parts = wordninja.split(word)
    if len(parts) < 2:
        return None
    if any(len(p) < MIN_SEG_LEN for p in parts):
        return None
    return " ".join(parts)


def fix_transcript(text):
    """修复转写文本。返回 (修复后文本, 修复记录列表)。

    修复记录：list[ (类型, 错写, 修正, 处数) ]，供上层打印给用户看。
    """
    changes = []
    if not text:
        return text, changes

    # 1. 大小写粘连拆开（ohNo → oh No）—— 与旧版同规则，保留
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    # 2. 缩略词规则：整词替换（前后都不能是字母，避免误伤词内子串）
    for wrong, right in CONTRACTIONS:
        pattern = r"(?<![A-Za-z])" + re.escape(wrong) + r"(?![A-Za-z])"
        new_text, n = re.subn(pattern, right, text)
        if n:
            changes.append(("缩略词", wrong, right, n))
            text = new_text

    # 3. 粘连词分割（wellsort → well sort）
    words = text.split()
    fixed_words = []
    for w in words:
        # 含撇号 / 含非字母 / 过短 → 不碰
        if "'" in w or not w.isalpha() or len(w) < MIN_GLUE_LEN:
            fixed_words.append(w)
            continue
        glued = split_glued_word(w)
        if glued:
            changes.append(("粘连拆分", w, glued, 1))
            fixed_words.append(glued)
        else:
            fixed_words.append(w)
    text = " ".join(fixed_words)

    # 4. 压空白
    text = re.sub(r"\s+", " ", text).strip()
    return text, changes
