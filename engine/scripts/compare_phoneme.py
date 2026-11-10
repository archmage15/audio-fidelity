# -*- coding: utf-8 -*-
r"""还原度对比 · 第二块：音素

把「原音频」和「你的模仿」分别喂给音素识别模型，得到两条音素序列，
对齐后告诉你：哪些音没还原、哪里多了/少了音（连读、弱化、吞音就看这里）。

模型：facebook/wav2vec2-lv-60-espeak-cv-ft
  - 首次运行会自动从 HuggingFace 下载（约 1.2GB），之后走本地缓存，不再下
  - 输出 eSpeak 风格音素（接近 IPA）

用法：
  1. 把两个音频放进 audio\ 文件夹，命名成：原音频.* 和 模仿.*
     （扩展名随意：mp3 / wav / flac / ogg 都可以，m4a / aac 不行）
  2. 双击 scripts\run_phoneme.bat（或根目录的 一键跑还原度.bat）
  3. 屏幕出对比结果，结果写到 output\ 文件夹

目录约定（本脚本在 scripts\ 里，用 ROOT 指回上一层）：
  audio\   放录音（你手动放，只读）
  output\  放本次结果（自动生成，可随时清空）
  models\  放音素模型（只读，缺了自动下载）

可选（给音素标"单词"）：
  把 讯飞配置.json（appid / secret_key）放在 ROOT，脚本会额外调用讯飞转写，
  把原音频转成"词 + 时间"，从而在每条纠错后面标出对应的单词。
  没有该文件 / 断网 / 无额度时自动跳过，只给音素级结果。

依赖：torch / transformers / soundfile / numpy
      （解码直接用模型自带 vocab.json，不需要额外装 phonemizer）
"""

from cpu_runtime import configure_cpu_runtime
configure_cpu_runtime()
import difflib
import json
import os
import sys
import unicodedata

import numpy as np

from cpu_runtime import configure_torch_cpu
configure_torch_cpu()

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 国内访问 HuggingFace：官方 huggingface.co 常常连不上，默认走国内镜像。
# 想用官方站，运行前设环境变量 HF_ENDPOINT=https://huggingface.co
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 关掉 xet 传输协议：它和镜像站不兼容，会让下载卡住。
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# ---- 目录约定 ----
# 本脚本位于 <项目>\scripts\ 下，ROOT = 项目根（上一层）
#   audio\  你放录音     output\  脚本写结果     models\  模型权重     根目录  讯飞配置.json
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC_DIR = os.environ.get("AUDIO_TEST_WORK_DIR") or os.path.join(ROOT, "audio")
OUT_DIR = os.environ.get("AUDIO_TEST_OUTPUT_DIR") or os.path.join(ROOT, "output")
for _d in (SRC_DIR, OUT_DIR):
    os.makedirs(_d, exist_ok=True)

REF_FILE = None      # 运行时按格式探测（见 audio_paths），可能是 .mp3 / .wav / .flac
MY_FILE = None
OUT_JSON = os.path.join(OUT_DIR, "音素对比.json")
# 给人看的报告（json 留给机器/AI，这份是纯文本、人读的）
OUT_REPORT = os.path.join(OUT_DIR, "音素对比_报告.txt")

MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
# 模型存放目录：默认放在 ROOT\models\ 里（首次运行自动下载）。
# 想改到别处，设环境变量 PHONEME_MODEL_DIR 指向目标目录即可。
LOCAL_DIR = os.environ.get("PHONEME_MODEL_DIR") or os.path.join(ROOT, "models", "wav2vec2-espeak")
SR = 16000

# wav2vec2 特征提取下采样步长 320 采样点 @16kHz -> 每帧 20ms
FRAME_MS = 20

# ---- 连读区判定 ----
# 讯飞在连读处给的"词边界"本身不可靠：两个词黏成一团时，它会把整团划给前一个词，
# 后一个词只剩几十毫秒的碎屑（实测 "are" 只有 30ms）。
# 连读在语言上本来就是一整团、切不开的东西，所以这里不硬切，而是标成"连读区"。
MIN_WORD_MS = 80        # 短于此长度的词，视为"碎片词"，可能与邻居构成连读
MERGE_GAP_MS = 60       # 碎片词与邻居间隔小于此值才合并（FRAME_MS * 3）

# 「参考音素表」：由 gen_ref_phonemes.py 离线预生成（espeak-ng 是 GPL，只在本机跑一次）
# 运行时只读这个纯数据文件 —— 绝不 import / 链接 espeak-ng
REF_PHONEME_JSON = os.path.join(SRC_DIR, "参考音素.json")

# 判"这个词读错了"的阈值（compute_word_errors 用）
WORD_ERR_RATIO = 0.40   # 错音占该词音素比例 ≥ 此值
WORD_ERR_MIN = 2        # 或错音个数 ≥ 此值


# 「应读音素 → 一句话发音要领」：命中就附在差异后面，帮学生知道怎么改。
# 只覆盖中国学习者最常错的音；没覆盖到的跳过（不硬凑）。
PHONEME_TIPS = {
    # 双元音（要滑过去，别读成单元音）
    "oʊ": "双元音：/o/ 滑到 /ʊ/，嘴唇由开到圆收，别读成单元音",
    "eɪ": "双元音：/e/ 滑到 /ɪ/，嘴角向两边展开，别读成单元音",
    "aɪ": "双元音：/a/ 滑到 /ɪ/，开口要大、下巴下压，别读成单元音",
    "aʊ": "双元音：/a/ 滑到 /ʊ/，别读成单元音",
    "ɔɪ": "双元音：/ɔ/ 滑到 /ɪ/",
    # 长短元音
    "iː": "长元音：拉长，舌位高前、嘴角向两边（如 see）",
    "ɪ": "短元音：短促，舌位略低（如 sit），别和 iː 混",
    "uː": "长元音：圆唇拉长（如 too）",
    "ʊ": "短元音：圆唇短促（如 book），别和 uː 混",
    "ɔː": "长元音：圆唇、下巴下压（如 law）",
    "ɜː": "长元音：舌中位、嘴角向两边、卷舌拉长（如 bird）",
    "ɑː": "长元音：大开口（如 father）",
    "ɑːɹ": "r 化元音：发 ɑː 的同时卷舌（美音 car 的元音）",
    "æ": "短元音：大开口、下巴用力下压（如 cat），别和 ɛ 混",
    "ɛ": "短元音：开口适中（如 bed），别和 æ 混",
    "e": "短元音：开口适中（如 bed），别和 æ 混",
    "ʌ": "短元音：舌中低、放松（如 cup）",
    "ə": "弱读央元音：最放松的 /ə/（如 about 的 a）",
    # 辅音难点
    "θ": "咬舌音：舌尖轻抵上下齿间送气（清音，如 think），别读成 s",
    "ð": "咬舌音：同 θ 但声带振动（浊音，如 this），别读成 z/d",
    "ŋ": "鼻音：舌根抵软腭（如 sing 的 ng），别漏掉也别加 g",
    "ɹ": "卷舌音：舌尖卷起但不碰硬腭（美音 r），别和 l 混",
    "l": "边音：舌尖抵上齿龈、气流从舌两侧出，别和 r 混",
    "t": "爆破音：舌尖抵上齿龈、送气爆破（如 top），别漏掉",
    "p": "爆破音：双唇闭合、送气爆破",
    "k": "爆破音：舌根抵软腭、爆破",
    "v": "唇齿音：上齿轻咬下唇、声带振动，别和 w 混",
    "w": "双唇圆拢、声带振动，别和 v 混",
    "z": "齿龈摩擦音：声带振动（如 zoo），别读成 s",
    "s": "齿龈摩擦音：清音（如 see）",
}


# ---- L2（中国学习者）典型音变对照表（10-28 加）----
# 为什么需要它：
#   报告原来只说「应读 θ，实读 f」，用户看到的是"你把 θ 读成了 f"，
#   但**没说这正是中国学习者最典型的音变**，也不知道该往哪改。
#   加上对照后，差异会被标成"口音型错误"，并直接给针对性的改法。
#
# 键是「应读音 → 实读音」，值是「这个音变成这样说明什么、怎么改」。
# 只收英语教学里公认的 L2 音变，不收模型自己的识别噪声。
L2_CONFUSIONS = {
    # 咬舌音汉化 —— 中国学习者第一大难
    ("θ", "f"): "咬舌音读成了 f：舌尖没伸到上下齿之间，改成轻轻咬住送气",
    ("θ", "s"): "咬舌音读成了 s：舌尖没伸到齿间，改成轻轻咬住送气",
    ("θ", "t"): "咬舌音读成了 t：舌尖没伸到齿间，改成轻轻咬住送气",
    ("ð", "d"): "浊咬舌音读成了 d：声带没振动，改成像 this 那样咬舌带声",
    ("ð", "t"): "浊咬舌音读成了 t：声带没振动，改成像 this 那样咬舌带声",
    ("ð", "z"): "浊咬舌音读成了 z：改成像 this 那样咬舌带声",
    ("ð", "v"): "浊咬舌音读成了 v：别用上齿咬下唇，要咬舌",
    # 卷舌音不到位
    ("ɹ", "l"): "卷舌音读成了 l：舌尖要卷起但不能碰上颚，别用舌尖顶上齿龈",
    ("ɹ", "w"): "卷舌音读成了 w：别用双唇圆拢，要卷舌",
    # 塞擦化 / 摩擦清化
    ("dʒ", "t"): "dʒ 读成了 t：舌尖要堵住上齿龈再放，不要直接送气",
    ("dʒ", "tʃ"): "dʒ 读成了 tʃ：改成浊的（声带要振动）",
    ("dʒ", "z"): "dʒ 读成了 z：塞擦音要有一个爆破的成分",
    ("tʃ", "t"): "tʃ 读成了 t：漏了摩擦，要读出持续的 sh",
    ("z", "s"): "z 读成了 s：声带要振动（如 zoo）",
    ("v", "w"): "v 读成了 w：别用双唇圆拢，要上齿轻咬下唇",
    ("v", "f"): "v 读成了 f：声带要振动",
    # 元音常见偏差
    ("æ", "ɛ"): "æ 读成了 ɛ：下巴要再往下压、开口再大一点（如 cat）",
    ("æ", "e"): "æ 读成了 e：开口不够大",
    ("ɪ", "iː"): "短 i 读成了长 iː：要短促，不能拖长（如 sit）",
    ("eɪ", "e"): "eɪ 读成了单元音 e：要滑过去（嘴角向两边展开）",
    ("eɪ", "ɛ"): "eɪ 读成了 ɛ：要滑过去，不能停在 e",
    ("oʊ", "aʊ"): "oʊ 读成了 aʊ：嘴唇要由圆变开，不能一直撅着",
    ("ʌ", "ɑː"): "ʌ 读成了 ɑː：口腔要放松，别张大嘴",
    ("ʌ", "æ"): "ʌ 读成了 æ：嘴巴放松，不要用力张开",
    ("ɜː", "ɚ"): "ɜː 读成了 ɚ：卷舌的 r 色彩要更明显（美音 bird）",
    ("uː", "ʊ"): "uː 读成了 ʊ：要拉长、嘴唇更圆",
    ("ŋ", "n"): "ŋ 读成了 n：舌根要抵住软腭，不要用舌尖",
    # 塞音清浊化（清读成浊是常态，浊读成清较少见，故只收前者）
    ("p", "b"): "p 读成了 b：送气的，别用声带振动",
    ("t", "d"): "t 读成了 d：送气的，别用声带振动",
    ("k", "g"): "k 读成了 g：送气的，别用声带振动",
}


def l2_tip(ref_str, got_str):
    """判断是不是 L2 典型音变，命中就给针对性提示。

    输入：应读音素串、实读音素串（空格分隔）
    返回：提示串；没命中返回空串。

    ⚠️ 一对多 / 多对一时**不判定** —— 硬套会给出错误的指导
       （比如「θ æ 读成了 f」套成"θ 读成 f"，但实际丢的是 æ）。
       所以大块替换里的音变识别交给 align_l2_pairs()。
    """
    refs = ref_str.split()
    gots = got_str.split()
    # ⚠️ 必须先判空 —— 直接取 gots[0] 会 IndexError 崩掉。
    #    崩在自检里比"断言失败"更难查（会掩盖前面的失败）。
    if not refs or not gots:
        return ""
    if len(refs) != 1 or len(gots) != 1:
        return ""
    tip = L2_CONFUSIONS.get((refs[0], gots[0]))
    if not tip:
        return ""
    return "【口音型】%s → %s：%s" % (refs[0], gots[0], tip)


def align_l2_pairs(ref_str, got_str):
    """在大块替换里**逐音配对**找出 L2 音变。

    为什么需要它（10-28 真实跑出来才发现）：
      difflib 会把连续几个音的变化挤成一个大块替换，例如
          参考「θ æ ŋ」→ 实读「f ɛ」   （其实是 θ→f、æ→ɛ 两个音变）
      原来只做「整块一对一」判定 → 这种块里**一个音变都识别不出来**
      → 报告里"口音型"只有 2 处，而实际有几十处。

    做法：块内两边长度相同时，逐位配对；
          长度不同时不硬套（对不上号，硬套会给出错误指导）。

    返回 [(参考音, 实读音, 提示串), ...]
    """
    refs = ref_str.split()
    gots = got_str.split()
    if not refs or not gots or len(refs) != len(gots):
        return []
    out = []
    for r, g in zip(refs, gots):
        tip = L2_CONFUSIONS.get((r, g))
        if tip:
            out.append((r, g, "【口音型】%s → %s：%s" % (r, g, tip)))
    return out


def phoneme_tips(ref_str):
    """给「应读」音素串附发音要领（命中 PHONEME_TIPS 的音素；没命中返回空串）。"""
    tips = []
    for ph in ref_str.split():
        if ph in PHONEME_TIPS:
            tips.append("%s:%s" % (ph, PHONEME_TIPS[ph]))
    if not tips:
        return ""
    return "【要领】" + "；".join(tips)


# ---- 日志（统一模块，带时间戳；详细模式见 logkit.py 顶部说明）----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logkit import log, out, log_setup, log_done, log_step, log_error, VERBOSE
from audio_paths import find_audio_pair, missing_hint, short_name

REF_FILE, MY_FILE = find_audio_pair(SRC_DIR)


def load_ref_phonemes():
    """读离线预生成的参考音素表。返回 dict 或 None（读不到 → 下游降级）。"""
    if not os.path.exists(REF_PHONEME_JSON):
        return None
    try:
        with open(REF_PHONEME_JSON, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        log("⚠ 参考音素.json 读失败：%s" % e, show=True)
        return None
    if not data.get("音素") or not data.get("词边界"):
        log("⚠ 参考音素.json 内容不完整（缺 音素 / 词边界）", show=True)
        return None
    return data


def load_mono(path):
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SR:
        n_out = int(round(len(audio) * SR / sr))
        x_in = np.linspace(0, 1, len(audio), endpoint=False)
        x_out = np.linspace(0, 1, n_out, endpoint=False)
        audio = np.interp(x_out, x_in, audio)
    return audio.astype(np.float32)


def load_vocab(model_path):
    """读模型自带符号表，得到 id -> 音素 的映射"""
    with open(os.path.join(model_path, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    return {int(v): k for k, v in vocab.items()}


def ctc_decode(ids, id2tok, blank=0, specials=(1, 2, 3)):
    """CTC 解码：合并重复 -> 去空白/特殊符 -> 映射成音素。
    同时返回每个音素所在的帧号（用于按时间归属到单词）。"""
    out = []
    frames = []
    prev = None
    for pos, i in enumerate(ids):
        i = int(i)
        if i == prev:
            continue
        prev = i
        if i == blank or i in specials:
            continue
        tok = id2tok.get(i, "")
        if tok:
            out.append(tok)
            frames.append(pos)
    return out, frames


def forced_align(logits, target_phones, id2tok):
    """强制对齐：把参考音素序列（该读什么）对齐到音频的帧级 logits。

    与 difflib 序列比对不同：这是「已知要读什么，去音频里找它在哪」。
    读快读慢不影响，因为每个参考音素在音频里找到「最像它的那一帧」即可。

    ⚠️ 必须用「单调分配」，不能用「编辑距离 DP」——
    允许"跳过参考音素"那条边时，DP 会走捷径（把 N 个参考音全跳掉、
    随便匹配总代价更低）→ 对齐全崩、后验全是 -inf、判错率 100%。
    正确：`best[j][t] = logp[t][id_j] + max_{t'<t} best[j-1][t']`，
    每个参考音**必须**匹配到严格递增的帧。缺失的音表现为后验低，
    而不是被 DP 跳过。

    参数：
      logits         [T, V] 帧级原始 logits（未 softmax，numpy 或 torch）
      target_phones  参考音素序列 list[str]（espeak 生成的"该读的音素"）
      id2tok         {id: 音素符号}

    返回 list[dict]，逐音对齐结果：
      {"音素", "帧", "logp", "后验"}  后验 = exp(logp)，0~1 的"像不像"分
    """
    if hasattr(logits, "detach"):          # torch tensor → numpy
        logits = logits.detach().cpu().numpy()
    if logits.ndim == 3:
        logits = logits[0]                  # [1, T, V] → [T, V]
    logits = np.asarray(logits, dtype=np.float32)
    T, V = logits.shape
    if T == 0 or not target_phones:
        return []

    # log_softmax（数值稳定）
    logp = logits - np.max(logits, axis=1, keepdims=True)
    logp = logp - np.log(np.sum(np.exp(logp), axis=1, keepdims=True))

    # 音素 → id；词表里没有的音素跳过（正常情况不该出现，防御性处理）
    tok2id = {v: k for k, v in id2tok.items()}
    phone_ids = [tok2id[p] for p in target_phones if p in tok2id]
    J = len(phone_ids)
    if J == 0:
        return []

    # 单调分配 Viterbi：best[j][t] = logp[t][id_j] + max_{t'<t} best[j-1][t']
    best = np.full((J, T), -np.inf, dtype=np.float32)
    best[0] = logp[:, phone_ids[0]]
    for j in range(1, J):
        prefix_max = np.maximum.accumulate(best[j - 1])        # 到每帧为止的前缀最大
        shifted = np.concatenate(([-np.inf], prefix_max[:-1]))  # t' < t（严格递增）
        best[j] = logp[:, phone_ids[j]] + shifted

    # 回溯：最后一个音素取全帧最大，往前逐个取「前一帧之前」的最大
    frames = [0] * J
    t = int(np.argmax(best[J - 1]))
    frames[J - 1] = t
    for j in range(J - 2, -1, -1):
        t = int(np.argmax(best[j][:t]))        # 严格在前一帧之前
        frames[j] = t

    # ⚠️ 后验要用「单帧」logp[t][id_j]，不能用累计 best[j][t]——
    #    best 是含前缀的累计值，若前一个音缺失（后验极低），会一路拉低
    #    后面所有音的后验（踩过：b 缺失时 c/d/e 后验全变 0.000045）。
    result = []
    for j in range(J):
        pid = phone_ids[j]
        t = frames[j]
        lp_single = float(logp[t, pid])
        result.append({
            "音素": id2tok.get(pid, "?"),
            "帧": t,
            "logp": lp_single,
            "后验": float(np.exp(lp_single)),
        })
    return result


def align_phone_verdict(aligned, target_phones, argmax_ids, id2tok,
                        threshold=0.5):
    """在强制对齐结果上判「可疑音」：后验低于阈值 = 这个音读得不像。

    参数：
      aligned        forced_align 的输出（逐音 帧/logp/后验）
      target_phones  参考音素序列（与 aligned 一一对应）
      argmax_ids     每帧的贪心 id（argmax(logits)，含 blank/special）
      id2tok         {id: 音素符号}
      threshold      后验低于此值判"可疑"（0~1，先取 0.5，待标定）

    返回 list[dict]，可疑音：
      {"音素", "后验", "实读", "类别", "下标"}
      类别 ∈ {"口音型", "没读出来", "其他"}
      「实读」= 对齐帧处贪心出来的符号（若仍等于参考音素，则只是读太轻）
      「下标」= 该音在 target_phones 里的位置（10-30 加，供长音频按句归因用）
    """
    if len(aligned) != len(target_phones):
        return []
    blank, specials = 0, (1, 2, 3)
    out = []
    for idx, (item, ref_ph) in enumerate(zip(aligned, target_phones)):
        post = item.get("后验", 0.0)
        if post >= threshold:
            continue
        t = item.get("帧", 0)
        got_id = int(argmax_ids[t]) if 0 <= t < len(argmax_ids) else blank
        got = id2tok.get(got_id, "")
        if got_id in (blank,) + tuple(specials):
            got = ""                       # 该帧是静音/特殊符 → 没读出
        if not got:
            out.append({"音素": ref_ph, "后验": round(post, 3),
                        "实读": "", "类别": "没读出来", "下标": idx})
        elif got == ref_ph:
            # 对齐帧贪心还是它自己，只是概率低 → 读太轻，不算读错
            out.append({"音素": ref_ph, "后验": round(post, 3),
                        "实读": got, "类别": "读太轻", "下标": idx})
        else:
            cat = "口音型" if (ref_ph, got) in L2_CONFUSIONS else "其他"
            out.append({"音素": ref_ph, "后验": round(post, 3),
                        "实读": got, "类别": cat, "下标": idx})
    return out


def load_feature_extractor(model_path):
    try:
        from transformers import AutoFeatureExtractor
        return AutoFeatureExtractor.from_pretrained(model_path)
    except Exception:
        from transformers import Wav2Vec2FeatureExtractor
        return Wav2Vec2FeatureExtractor.from_pretrained(model_path)


def model_ready(path):
    """判断模型目录是否真的完整。

    只看文件在不在不够 —— HF 缓存异常时会出现 0 字节的坏文件，
    加载时会报 "is not a valid JSON file"。所以这里顺带查大小。
    """
    need = ["config.json", "preprocessor_config.json", "vocab.json"]
    for name in need:
        p = os.path.join(path, name)
        if not (os.path.exists(p) and os.path.getsize(p) > 0):
            return False
    weight = os.path.join(path, "pytorch_model.bin")
    if os.path.exists(weight):
        return os.path.getsize(weight) > 0
    # 有些仓库只提供 safetensors 格式
    return any(f.endswith(".safetensors") and os.path.getsize(os.path.join(path, f)) > 0
               for f in os.listdir(path))


def bad_files(path):
    """列出目录里 0 字节的坏文件（HF 缓存符号链接落盘失败会留下这些）。"""
    if not os.path.isdir(path):
        return []
    out = []
    for name in os.listdir(path):
        p = os.path.join(path, name)
        if os.path.isfile(p) and os.path.getsize(p) == 0:
            out.append(name)
    return out


def repair_model():
    """删掉坏文件 + 清掉缓存索引里对应记录，让下次下载真的重下。

    背景：HF 缓存会记"这个文件我下过了"。若上次下成 0 字节，
    重跑时它会跳过不重下，坏文件就一直留着 —— 必须连缓存记录一起删。
    """
    broken = bad_files(LOCAL_DIR)
    if not broken:
        return []
    log("")
    log("发现 %d 个 0 字节的坏文件，正在自动修复：" % len(broken))
    cache_dir = os.path.join(LOCAL_DIR, ".cache", "huggingface", "download")
    for name in broken:
        os.remove(os.path.join(LOCAL_DIR, name))
        log("   已删除坏文件：" + name)
        meta = os.path.join(cache_dir, name + ".metadata")
        if os.path.exists(meta):
            os.remove(meta)
            log("   已清除其缓存记录，确保会重新下载")
    return broken


def ensure_model():
    """模型不存在或损坏则修复/下载（走 local_dir 模式）。返回本地模型目录。"""
    repair_model()                      # 先清坏文件，避免"已存在故跳过"的误判
    if model_ready(LOCAL_DIR):
        return LOCAL_DIR
    log("")
    log("首次运行：下载音素模型（约 1.2GB，走 hf-mirror 镜像，请耐心等）...")
    log("存放位置：" + LOCAL_DIR)
    os.makedirs(LOCAL_DIR, exist_ok=True)
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=MODEL_ID, local_dir=LOCAL_DIR)
    repair_model()                      # 下载后若仍有坏文件，再修一轮
    if not model_ready(LOCAL_DIR):
        broken = bad_files(LOCAL_DIR)
        raise RuntimeError(
            "模型文件不完整，自动修复后仍未通过检查。\n"
            "  仍然为 0 字节的文件：" + (", ".join(broken) if broken else "（无，可能是缺文件）") + "\n"
            "  解决办法：删掉整个目录后重跑本脚本，让它重新下载：\n"
            "    " + LOCAL_DIR)
    log("模型下载完成。")
    return LOCAL_DIR


# ---------------- 连读区识别（用参考音素表词边界 + 序号对齐）----------------

def build_text_spans(word_bounds, owner, frame_times):
    """用「参考文本给的词边界 + CTC 帧给的时间」拼出 spans。

    ⚠️ 与旧 build_word_spans **同构**（同样的键），所以 pair_fragments 逐字不改就能接：
        {idx, word, start_ms, end_ms, span_ms, n_ph, short}

    参数
      word_bounds : 参考音素.json 的 词边界 [{word, a, b, 角色}]
      owner       : map_words_to_audio 的输出 —— 每个「实际音素」归属的词下标
      frame_times : 每个实际音素的时间(ms)，与 owner 等长

    沿用了旧实现的两个修补（首尾补边、单调修正）—— 换数据源不等于可以少护栏。
    """
    if not word_bounds:
        return []

    spans = []
    for wi, wb in enumerate(word_bounds):
        ts = [frame_times[j] for j in range(len(owner)) if owner[j] == wi]
        spans.append({"idx": wi, "word": wb["word"],
                      "start_ms": int(min(ts)) if ts else None,
                      # frame_times 存的是音素的**起始**帧，末尾要补一帧（20ms）
                      "end_ms": (int(max(ts)) + FRAME_MS) if ts else None,
                      "n_ph": len(ts)})

    # ⚠️ 不做"首尾补边"（旧 build_word_spans 有，这里实测**不该有**）：
    # 旧实现的时间来自讯飞词表，补边只是填边角空白；
    # 新实现的时间**直接来自音素帧** —— 把首词 start 硬拉成 0 会让它的时长虚高
    # （实测：so 真实 740→1030ms，补边后变成 0→1000ms）。那会污染节奏维度。

    # 1) 一个音素都没分到的词（如原音频里的感叹词 oh）：用邻居边界补一个区间。
    #    绝不能给 0 宽 —— 那会被 pair_fragments 当成"碎片词"并进连读区，纯属误报。
    for i, sp in enumerate(spans):
        if sp["start_ms"] is not None and sp["end_ms"] is not None:
            continue
        prev_end = None
        for k in range(i - 1, -1, -1):
            if spans[k]["end_ms"] is not None:
                prev_end = spans[k]["end_ms"]
                break
        next_start = None
        for k in range(i + 1, len(spans)):
            if spans[k]["start_ms"] is not None:
                next_start = spans[k]["start_ms"]
                break
        if prev_end is None and next_start is None:
            sp["start_ms"] = sp["end_ms"] = 0
        elif prev_end is None:
            sp["start_ms"], sp["end_ms"] = next_start - 1, next_start
        elif next_start is None:
            sp["start_ms"], sp["end_ms"] = prev_end, prev_end + 1
        else:
            mid = (prev_end + next_start) // 2
            sp["start_ms"], sp["end_ms"] = mid, max(mid + 1, next_start)

    # 2) 单调修正：start 必须严格递增（帧边界会抖，可能压出 start >= end）
    for i in range(1, len(spans)):
        if spans[i]["start_ms"] <= spans[i - 1]["start_ms"]:
            spans[i]["start_ms"] = spans[i - 1]["start_ms"] + 1
        if spans[i]["start_ms"] >= spans[i]["end_ms"]:
            spans[i]["end_ms"] = spans[i]["start_ms"] + 1

    for sp in spans:
        sp["span_ms"] = sp["end_ms"] - sp["start_ms"]
        sp["short"] = sp["span_ms"] < MIN_WORD_MS

    return spans


def pair_fragments(spans):
    """把"碎片词"和它的邻居并成"连读区"。

    三条限制，都是为了防止"越并越大"：
      1) 只并"时长过短"的碎片词（不看 n_ph —— 有些长词可能一个音都没分到，
         那是别的问题，不是连读，不该并）。
      2) 一个词一旦进了某个连读区，就不再作为下一轮的合并起点（防连锁传染）。
      3) 连读区最多 2 个词 —— 真连读是"两个词黏在一起"，三个词整体糊成一团
         不是典型连读，并进去只会让报告更糊。

    返回 (主区间列表, 连读区列表, 区间下标 -> 连读区下标 映射)
    """
    if not spans:
        return [], [], {}

    zone_of = {}          # 主区间下标 -> 连读区下标
    zones = []
    consumed = set()      # 已被并进连读区、不再单独出现的主区间下标

    for i, sp in enumerate(spans):
        if not sp["short"] or i in consumed:
            continue
        prev_sp = spans[i - 1] if i > 0 else None
        next_sp = spans[i + 1] if i + 1 < len(spans) else None

        # 邻居是否"紧挨着"（判断连读的关键：够近才可能黏在一起）
        prev_near = (prev_sp is not None and (i - 1) not in consumed
                     and (sp["start_ms"] - prev_sp["end_ms"]) <= MERGE_GAP_MS)
        next_near = (next_sp is not None and (i + 1) not in consumed
                     and (next_sp["start_ms"] - sp["end_ms"]) <= MERGE_GAP_MS)

        if prev_near:
            # 优先朝前并（连读多是"前词拖住了后词"）
            members = [i - 1, i]
        elif next_near:
            members = [i, i + 1]
        else:
            continue          # 孤立短词：不并，保持独立

        if any(m in consumed for m in members):
            continue

        zi = len(zones)
        for m in members:
            consumed.add(m)
            zone_of[m] = zi
        zones.append({
            "zone_idx": zi,
            "member_idx": members,
            "words": [spans[m]["word"] for m in members],
            "start_ms": spans[members[0]]["start_ms"],
            "end_ms": spans[members[-1]]["end_ms"],
            "n_ph": sum(spans[m]["n_ph"] for m in members),
            "why": " ".join(spans[m]["word"] for m in members),
        })

    main = [sp for i, sp in enumerate(spans) if i not in consumed]
    return main, zones, zone_of


def map_words_to_audio(text_ph, audio_ph, word_bounds):
    """按「参考音素序列」与「实际音素序列」的对齐关系，给每个实际音素定词归属。

    返回：list，长度 = len(audio_ph)，每项是 word_bounds 的下标（int）。

    ⚠️ 这是替掉 assign_word_owner 的关键改动。旧实现拿音素**时间点**去撞词区间，
    跨词边界时会把音归错词 —— 实测：`still at the` 里 at 的 /t/ 在 /ð/ 前省掉了，
    却被算到 the 头上，报告写成「[缺音] the：少了 t」（正确答案是 at）。
    改成按**音素序号**对齐 —— 正是旧文档 §7.3 那句
    「比较单位必须是音素序号 / 句子内相对位置，绝不能拿绝对时间戳直接相减」。
    """
    if not audio_ph:
        return []
    if not text_ph or not word_bounds:
        return [0] * len(audio_ph)

    pos2word = {}
    for wi, wb in enumerate(word_bounds):
        for i in range(wb["a"], wb["b"]):
            pos2word[i] = wi

    owner = [None] * len(audio_ph)
    sm = difflib.SequenceMatcher(a=text_ph, b=audio_ph)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for k in range(min(i2 - i1, j2 - j1)):
                owner[j1 + k] = pos2word.get(i1 + k)
        elif op == "replace":
            n, m = i2 - i1, j2 - j1
            for k in range(m):
                i = i1 + (k * n // m) if m else i1
                owner[j1 + k] = pos2word.get(min(i, i2 - 1))
        # delete：文本有、实际没有 → 没有实际音素要归属
        # insert：实际多出来的音 → 交给下面的"就近继承"

    # 未定的（多读出来的音）：继承前一个词；开头就多读的，取后面第一个
    last = None
    for j in range(len(owner)):
        if owner[j] is None:
            owner[j] = last
        else:
            last = owner[j]
    nxt = None
    for j in range(len(owner) - 1, -1, -1):
        if owner[j] is None:
            owner[j] = nxt
        else:
            nxt = owner[j]
    return [0 if o is None else o for o in owner]


def detect_partial_coverage(ref, mine):
    """检测「模仿只覆盖了原音频的一部分」（半截模仿 / 缺句漏句 / 跳开头）。

    背景：学生可能只跟读了原音频的**前半句**（后面整段没录），也可能**从中间
    开始读**（跳过开头几句）。旧逻辑把这些「没读到的部分」当成逐音的「缺失」
    扣分 —— 把"没读到"错判成"读错了"。这里改成：先判断模仿实际覆盖到原音频
    的哪个区间，区间之外没读到的头/尾段不算错。

    判据（用 matching_blocks 的覆盖跨度，排除孤立巧合匹配）：
      1. 前提：len(mine) <= len(ref) * LEN_RATIO —— 模仿明显短一截才可能是半截，
         排除"读完整句只是中间读错"的情况。
      2. 找出所有 size>0 的匹配块；**剔除孤立巧合块**（size==1 且前后 gap>=3 的
         单音素匹配，如跳开头后模仿首音 oʊ 碰巧对上原音频开头的 oʊ）。
      3. 用剩下的"有效匹配块"算覆盖区间 [min_a, max_a)：
         头部漏读 = min_a，尾部漏读 = n - max_a。
      4. 头/尾漏读段 >= MIN_TAIL 音 且 占原长 >= MIN_TAIL_RATIO，即判半截并截断。

    为什么不用 opcodes 的 delete 段：
      跳开头时，模仿开头音素和原音频开头「不同但都有音」，difflib 会判成
      replace（原音频开头 s oʊ w... 被"替换"成模仿的 oʊ æ...），而不是 delete，
      所以"开头 delete"抓不到漏头。匹配块的覆盖跨度 + 剔除孤立巧合更稳。

    返回 dict：
      {
        "半截": bool,                       # 是否判定为半截模仿（漏头或漏尾）
        "覆盖到原音频位置": int,            # 截断区间末尾音素下标（max_a）
        "覆盖起始位置": int,                # 截断区间起始音素下标（min_a）
        "覆盖比例": float,                  # (end-start) / len(ref)
        "尾巴漏读音数": int,                # 原音频尾部未被覆盖的音素数
        "头部漏读音数": int,                # 原音频头部未被覆盖的音素数
        "尾巴漏读词": [str, ...],           # 尾部漏读段对应的词（调用处填）
        "头部漏读词": [str, ...],           # 头部漏读段对应的词（调用处填）
      }
    不判定为半截时，覆盖区间 = [0, n)，覆盖比例 = 1.0。
    """
    MIN_TAIL = 3            # 漏读段至少这么多音，才算"整段没读"（小于这个算零星漏读）
    MIN_TAIL_RATIO = 0.10   # 漏读段占原音频的比例达到此值才截断
    LEN_RATIO = 0.75        # 模仿音素数 <= 原长 × 此值，才可能是半截（明显短一截）
    GAP_ISOLATED = 3        # 单音素匹配块与其它匹配块的间隔 >= 此值，视为孤立巧合
    # ⚠️ 实测结论（selftest.py 变异测试 + 边界扫描，2026-10-18）：
    #    三个门槛里**只有 LEN_RATIO 真正生效**。逐点扫描 40 音参考：
    #      mine=30（漏 10 音/25%）→ 判半截；mine=31（漏 9 音/22.5%）→ 不判。
    #    边界正好落在 LEN_RATIO=0.75 上——因为要触发 MIN_TAIL_RATIO=0.10
    #    只需漏 4 音，而漏 4 音时 len(mine)=36 > 30，早被 LEN_RATIO 挡在门外。
    #    即 MIN_TAIL / MIN_TAIL_RATIO 在任何规模下都是**死门槛**（改了不起作用）。
    #    保留它们是作为"语义说明"与将来放宽 LEN_RATIO 后的兜底；若要精简，
    #    删这两行不影响当前任何行为——但要先改 selftest 的对应断言。

    n = len(ref)
    empty = {"半截": False, "覆盖到原音频位置": n, "覆盖起始位置": 0,
             "覆盖比例": 1.0, "尾巴漏读音数": 0, "头部漏读音数": 0,
             "尾巴漏读词": [], "头部漏读词": []}
    if n == 0 or not mine:
        return empty

    # 前提①：模仿不短一截 → 不可能是"只读了一半"
    if len(mine) > n * LEN_RATIO:
        return empty

    sm = difflib.SequenceMatcher(a=ref, b=mine, autojunk=False)
    blocks = [(a, b, size) for a, b, size in sm.get_matching_blocks() if size > 0]
    if not blocks:
        return empty

    # 剔除孤立巧合块（size==1 且前后都离其它块 >= GAP_ISOLATED）
    def is_isolated(i):
        a, _b, size = blocks[i]
        if size != 1:
            return False
        prev_gap = a - (blocks[i - 1][0] + blocks[i - 1][2]) if i > 0 else 999
        next_gap = blocks[i + 1][0] - (a + size) if i + 1 < len(blocks) else 999
        return prev_gap >= GAP_ISOLATED and next_gap >= GAP_ISOLATED

    kept = [blk for i, blk in enumerate(blocks) if not is_isolated(i)]
    if not kept:
        return empty

    min_a = min(blk[0] for blk in kept)
    max_a = max(blk[0] + blk[2] for blk in kept)

    head_len = min_a
    tail_len = n - max_a
    head_partial = (head_len >= MIN_TAIL) and ((head_len / float(n)) >= MIN_TAIL_RATIO)
    tail_partial = (tail_len >= MIN_TAIL) and ((tail_len / float(n)) >= MIN_TAIL_RATIO)
    partial = head_partial or tail_partial

    covered_start = head_len if head_partial else 0
    covered_end = n - tail_len if tail_partial else n
    covered_len = covered_end - covered_start
    ratio = covered_len / float(n) if n > 0 else 1.0

    return {
        "半截": partial,
        "覆盖到原音频位置": covered_end,
        "覆盖起始位置": covered_start,
        "覆盖比例": round(ratio, 4),
        "尾巴漏读音数": tail_len,
        "头部漏读音数": head_len,
        "尾巴漏读词": [],
        "头部漏读词": [],
    }


def collapse_owner(codes):
    """把归属编码压成 (连读区名列表, 单词名列表)，保持首次出现顺序。"""
    zones_seen, words_seen = [], []
    for c in codes:
        if not c:
            continue
        kind, _, name = c.partition(":")
        if kind == "zone":
            z = int(name)
            if z not in zones_seen:
                zones_seen.append(z)
        elif kind == "word":
            if name not in words_seen:
                words_seen.append(name)
    return zones_seen, words_seen


def compute_word_errors(word_bounds, ref_ph, ref_owner, my_ph, my_owner):
    """逐词比较「原音频实际音素」与「模仿实际音素」，标出读错的词。

    ★ 基准（09-23 定）：**以原音频实际音素为准**，参考文本只用来定词边界。
      理由：判据是「还原度」（像不像原音频），不是「达标度」（像不像字典标准音）。
      连读吞音这类 —— 原音频也吞、学生也吞 → **不算错**。
      反例：若拿参考文本 `want to` 当基准，学生把 want to 正确连读成 wanna
      会被误报成"缺音"，那就把对的判成错的了。

    返回：[{词序, 词, 角色, 类型, 期望音素, 实读音素, 错误音数, 错误比例}]
    """
    kind_cn = {"replace": "替换", "delete": "缺失", "insert": "多余"}
    out = []
    for wi, wb in enumerate(word_bounds):
        ref_seg = [ref_ph[j] for j in range(len(ref_ph)) if ref_owner[j] == wi]
        my_seg = [my_ph[j] for j in range(len(my_ph)) if my_owner[j] == wi]
        if not ref_seg and not my_seg:
            continue
        sm = difflib.SequenceMatcher(a=ref_seg, b=my_seg)
        wrong, kinds = 0, []
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == "equal":
                continue
            wrong += max(i2 - i1, j2 - j1)
            kinds.append(kind_cn.get(op, op))
        if not wrong:
            continue
        ratio = wrong / float(max(len(ref_seg), len(my_seg), 1))
        if wrong >= WORD_ERR_MIN or ratio >= WORD_ERR_RATIO:
            out.append({
                "词序": wi + 1,
                "词": wb["word"],
                "角色": wb.get("角色"),
                "类型": "+".join(sorted(set(kinds))),
                "期望音素": " ".join(ref_seg),
                "实读音素": " ".join(my_seg),
                "错误音数": wrong,
                "错误比例": round(ratio, 3),
            })
    return out


def format_words(word_ids, words, zone_names=None):
    """生成词标签，附在每条差异后面。

    注意（这是一个隐式契约，不能破）：
      overall_score.py 会反向解析本函数产出的字符串 —— 它把【】里的内容按双引号
      切开、取奇数位当"问题词"。所以：
        - 双引号必须成对
        - 被双引号包住的必须是"单个词"，不能是 "Where are" 这种拼接串
          （拼成一串的话，那边拿 "Where are" 去跟单个词 "Where" 比，一个都匹配不上）
      连读区的两个词因此各自单独加引号：`"Where" "are"`，抠词得 ["Where","are"]。
    """
    if zone_names:
        # 连读区：连读本来切不开，直接告诉学生"这里是一个连读"。
        # 每个词单独加引号（而不是拼成一串放进一个引号里）—— 这样上游按
        # split('"')[1::2] 抠词时，能分别拿到 "Where"、"are"，原句里两个词都能框出。
        all_words = []
        for nm in zone_names:
            all_words.extend(nm.split(" "))
        names = '" "'.join(all_words)
        return "【连读区 \"%s\"】⚠ 这两个词黏在一起读，按整体练，不必逐个拆" % names
    if not words or not word_ids:
        return ""
    if len(word_ids) == 1:
        return "【单词 \"%s\"】" % words[word_ids[0]]["word"]
    return "【单词 \"%s\" → \"%s\" 之间】" % (words[word_ids[0]]["word"],
                                            words[word_ids[-1]]["word"])


def dwidth(s):
    """按东亚宽度估算显示宽度：宽/全角算 2，其余算 1（配合等宽字体）。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def pad_to(s, width):
    """右侧补空格到指定显示宽度。"""
    return s + " " * max(0, width - dwidth(s))


def center_to(s, width):
    """在指定显示宽度里居中。"""
    total = max(0, width - dwidth(s))
    left = total // 2
    return " " * left + s + " " * (total - left)


def build_aligned(ref, mine, sm):
    """按对齐关系把两条音素序列展开成 (原音频音, 你的音) 列表；缺/多处为空串。"""
    pairs = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        r, m = ref[i1:i2], mine[j1:j2]
        for k in range(max(len(r), len(m))):
            pairs.append((r[k] if k < len(r) else "", m[k] if k < len(m) else ""))
    return pairs


def write_report(ref, mine, sm, words, ratio, diff_items, problem_words):
    """写一份给人读的纯文本报告（json 留给机器/AI）。"""
    L = []
    L.append("=" * 60)
    L.append("   音素对比报告（发音 + 音变）")
    L.append("=" * 60)
    L.append("")
    L.append("%s  vs  %s" % (short_name(REF_FILE), short_name(MY_FILE)))
    L.append("整体音素相似度：%.1f / 100" % ratio)
    L.append("")
    # 一、逐音对照：横排，上行=原音频，下行=我的，★=不一样
    L.append("-" * 60)
    L.append("一、逐音对照   （上=原音频 应读，下=我的 实读，★=不一样）")
    L.append("-" * 60)
    L.append("")
    pairs = [(r if r else "—", m if m else "—") for r, m in build_aligned(ref, mine, sm)]
    if pairs:
        cell = max(3, max(dwidth(p) for pr in pairs for p in pr)) + 2
        la, lm = "原音频:", "我的:"
        pw = max(dwidth(la), dwidth(lm)) + 1
        step = 12                      # 每行最多放多少个音，放不下就换块
        for s0 in range(0, len(pairs), step):
            chunk = pairs[s0:s0 + step]
            L.append((pad_to(la, pw) + "".join(center_to(r, cell) for r, m in chunk)).rstrip())
            L.append((pad_to(lm, pw) + "".join(center_to(m, cell) for r, m in chunk)).rstrip())
            L.append((" " * pw + "".join(center_to("★", cell) if r != m else " " * cell
                                         for r, m in chunk)).rstrip())
            L.append("")
    # 二、问题清单
    L.append("-" * 60)
    L.append("二、问题清单")
    L.append("-" * 60)
    if diff_items:
        for it in diff_items:
            if it.get("zone_names"):
                who = "、".join(it["zone_names"])          # 连读区：直接报词组
            elif it["words"]:
                who = "、".join(it["words"])
            else:
                who = "（未定位到词）"
            if it["kind"] == "替换":
                line = "    [替换] %s：应读 %s  →  实读 %s" % (who, it["ref"], it["my"])
                tip = phoneme_tips(it["ref"])
                if tip:
                    line += "   " + tip
                L.append(line)
            elif it["kind"] == "缺失":
                line = "    [缺音] %s：少了 %s" % (who, it["ref"])
                tip = phoneme_tips(it["ref"])
                if tip:
                    line += "   " + tip
                L.append(line)
            else:
                L.append("    [多音] %s：多了 %s" % (who, it["my"]))
        L.append("")
        L.append("    ※ [缺音]/[多音] 多出现在连读、弱化、吞音处——这就是“音变”")
        if any(it.get("zone") for it in diff_items):
            L.append("    ※ 标了「连读区」的地方，是两个词黏在一起读（如 Where are）——")
            L.append("      可能是连读，也可能是词太短、系统测不准，总之按整体练，不必逐个拆。")
    else:
        L.append("    （音素序列完全一致，很棒）")
    # 原句（问题词用【】框出）
    if words:
        bad = set(w for w, _ in problem_words)
        sent = " ".join(("【%s】" % w["word"]) if w["word"] in bad else w["word"] for w in words)
        L.append("")
        L.append("-" * 60)
        L.append("原句（有问题的词已用【】框出）")
        L.append("-" * 60)
        L.append("    " + sent)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main():
    # 先把耗时大户 import 掉，再开日志 —— 否则 log_setup 的时间戳会比真正开始早十几秒
    import torch
    from transformers import AutoModelForCTC
    log_setup("音素对比")

    out("=" * 62)
    out("   还原度对比 · 音素")
    out("=" * 62)

    if not (REF_FILE and MY_FILE):
        out("")
        out("X 没找到音频文件：")
        out(missing_hint([("原音频", REF_FILE), ("模仿", MY_FILE)], SRC_DIR))
        log_done("（音频缺失，提前退出）")
        return

    log_step("加载音素识别模型")
    model_path = ensure_model()
    fe = load_feature_extractor(model_path)
    model = AutoModelForCTC.from_pretrained(model_path)
    model.eval()
    id2tok = load_vocab(model_path)
    log("模型就绪（%s）" % os.path.basename(model_path), verbose=True)

    results = {}
    frames_map = {}
    for tag, path in (("原音频", REF_FILE), ("我的模仿", MY_FILE)):
        log_step("识别：%s" % tag)
        audio = load_mono(path)
        log("   音频时长 %.2f 秒（%d 采样点）" % (len(audio) / float(SR), len(audio)),
            verbose=True)
        inputs = fe(audio, sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        log("   模型输出 %d 帧（每帧 %dms）" % (logits.shape[1], FRAME_MS), verbose=True)
        ids = torch.argmax(logits, dim=-1)[0].tolist()
        tokens, frames = ctc_decode(ids, id2tok)
        results[tag] = tokens
        frames_map[tag] = frames
        log("   识别出的音素共 %d 个" % len(tokens))

    ref = results["原音频"]
    mine = results["我的模仿"]
    log("原音频 %d 音 vs 我的 %d 音" % (len(ref), len(mine)), verbose=True)

    # 读「参考音素表」（由 gen_ref_phonemes.py 离线预生成）—— 不再调讯飞。
    # 词边界来自参考文本；音素归属按「序号对齐」定（见 map_words_to_audio）
    log_step("读取参考音素表")
    ref_data = load_ref_phonemes()
    text_ph = ref_data["音素"] if ref_data else []
    word_bounds = ref_data["词边界"] if ref_data else []
    if ref_data:
        log("   参考音素 %d 个，词边界 %d 个（角色 %s）"
            % (len(text_ph), len(word_bounds),
               "".join(sorted({wb.get("角色") or "?" for wb in word_bounds}))))
    else:
        log("   ⚠ 参考音素.json 不可用 → 本次只出音素级结果，不标词", show=True)

    # ---- 半截模仿检测（模仿只覆盖原音频一部分时，只按覆盖范围评分）----
    log_step("检测模仿覆盖范围")
    coverage = detect_partial_coverage(ref, mine)
    if coverage["半截"]:
        start = coverage["覆盖起始位置"]
        end = coverage["覆盖到原音频位置"]
        # 定位漏读的词：头部 = 下标 < start 的词；尾部 = 下标 >= end 的词
        head_words = [wb["word"] for wb in word_bounds if wb["b"] <= start]
        tail_words = [wb["word"] for wb in word_bounds if wb["a"] >= end]
        coverage["头部漏读词"] = head_words
        coverage["尾巴漏读词"] = tail_words
        # 截断原音频音素序列：只保留被实际覆盖到的 [start, end) 区间
        ref = ref[start:end]
        # 词边界截断：只保留落在 [start, end) 内的词，并把下标平移 -start
        word_bounds = [dict(wb, a=wb["a"] - start, b=wb["b"] - start)
                       for wb in word_bounds if wb["a"] >= start and wb["b"] <= end]
        # 参考音素序列（text_ph）同步截断，保证词边界下标仍落在范围内
        text_ph = text_ph[start:end]
        log("   ⚠ 检测到半截模仿：模仿覆盖原音频第 %d~%d 个音素（%.1f%%），"
            "已按覆盖范围截断重算" % (start, end, coverage["覆盖比例"] * 100), show=True)
        if head_words:
            log("   头部漏读词：%s" % " ".join(head_words), show=True)
        if tail_words:
            log("   尾部漏读词：%s" % " ".join(tail_words), show=True)
    else:
        log("   模仿覆盖完整（%.1f%%），不截断" % (coverage["覆盖比例"] * 100), verbose=True)

    # 音素时间轴：帧号 × FRAME_MS
    ref_frame_times = [f * FRAME_MS for f in frames_map["原音频"]]
    my_frame_times = [f * FRAME_MS for f in frames_map["我的模仿"]]

    # 音素归属：按「参考音素序列 ↔ 实际音素序列」的**对齐关系**定，
    # 不再按时间点撞词区间（那正是"跨词归错词"的根因）
    ref_owner = map_words_to_audio(text_ph, ref, word_bounds)
    my_owner = map_words_to_audio(text_ph, mine, word_bounds)

    # 两侧词块：节奏脚本的唯一计时输入，也是连读区检测的入口
    log_step("连读区识别")
    spans = build_text_spans(word_bounds, ref_owner, ref_frame_times)
    spans_my = build_text_spans(word_bounds, my_owner, my_frame_times)
    main_spans, zones, zone_of = pair_fragments(spans)          # ★ 逐字不改
    _main_my, zones_my, _zone_my = pair_fragments(spans_my)
    log("   词区间 %d 段；碎片词 %d 个；连读区 %d 处"
        % (len(spans), sum(1 for s in spans if s["short"]), len(zones)), verbose=True)
    for z in zones:
        log("   连读区：%s（%d-%dms）" % (" ".join(z["words"]), z["start_ms"], z["end_ms"]))

    # 词下标 -> 连读区下标（差异分析里判断"这个词在连读区里吗"）
    zone_by_word = {}
    for z in zones:
        for m in z["member_idx"]:
            zone_by_word[m] = z["zone_idx"]
    my_zone_words = set()
    for z in zones_my:
        for m in z["member_idx"]:
            my_zone_words.add(m)

    log_step("对齐与差异分析")
    sm = difflib.SequenceMatcher(a=ref, b=mine)
    ratio = sm.ratio() * 100

    # 对齐统计：把音素序列的对齐关系数成"几个替换 / 几个缺失 / 几个多余"。
    # 综合评分器据此直接算「发音分（替换）」和「音变分（缺/多）」，
    # 不必再重跑 wav2vec2 —— 数据在这一步就已算好。
    align_stat = {"原生音数": len(ref), "模仿音数": len(mine),
                  "匹配音数": 0, "替换音数": 0, "缺失音数": 0, "多余音数": 0}
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            align_stat["匹配音数"] += (i2 - i1)
        elif op == "replace":
            align_stat["替换音数"] += max(i2 - i1, j2 - j1)
        elif op == "delete":
            align_stat["缺失音数"] += (i2 - i1)
        else:  # insert
            align_stat["多余音数"] += (j2 - j1)

    def resolve_by_words(ph_idx_list):
        """把一组「原音频音素下标」解析成 (连读区名列表, 词下标列表, 词名列表)。

        规则：只要落到的词里有在连读区里的，就整体按连读区呈现 ——
        连读本来就切不开，硬拆反而误导（与旧 resolve 同旨）。
        """
        wis = sorted({ref_owner[j] for j in ph_idx_list if 0 <= j < len(ref_owner)})
        if not wis:
            return [], [], []
        zone_ids = sorted({zone_by_word[wi] for wi in wis if wi in zone_by_word})
        if zone_ids:
            return [" ".join(zones[z]["words"]) for z in zone_ids], [], []
        return [], wis, [word_bounds[wi]["word"] for wi in wis]

    diff_lines = []
    problem_words = []      # (词, 类型) 汇总
    diff_items = []         # 给人类报告用的结构化差异
    zone_hits = {}          # 连读区名 -> {"词": [...], "n": 次数}，写进 json 供后期复用
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        if op == "replace":
            line = "  [替换] 应该读 %s  →  你读成了 %s" % (" ".join(ref[i1:i2]), " ".join(mine[j1:j2]))
            kind = "替换"
            zn, idxs, names = resolve_by_words(list(range(i1, i2)))
        elif op == "delete":
            line = "  [缺失] 你没读 %s（原音频里有）" % " ".join(ref[i1:i2])
            kind = "缺失"
            zn, idxs, names = resolve_by_words(list(range(i1, i2)))
        else:  # insert
            # 多读出来的音在原音频里没有对应位置 → 就近挂到它前面那个音所属的词
            line = "  [多余] 你多读了 %s（原音频里没有）" % " ".join(mine[j1:j2])
            kind = "多余"
            zn, idxs, names = resolve_by_words([max(0, i1 - 1)])
        label = format_words(idxs, word_bounds, zone_names=zn if zn else None)
        if label:
            line = line + "   " + label
            for i in idxs:
                problem_words.append((word_bounds[i]["word"], kind))
            for nm in zn:
                zone_hits.setdefault(nm, {"词": nm.split(" "), "n": 0})
                zone_hits[nm]["n"] += 1
                # 连读区涉及的两个词都算"有问题"，原句里一并框出（并集框）
                for wname in nm.split(" "):
                    problem_words.append((wname, kind))
        diff_items.append({"kind": kind,
                           "ref": " ".join(ref[i1:i2]),
                           "my": " ".join(mine[j1:j2]),
                           "words": names,
                           "zone": bool(zn),
                           "zone_names": zn})
        # 附发音要领（针对「应读」音素；[多余] 无「应读」音，不给要领）
        if kind in ("替换", "缺失"):
            tip = phoneme_tips(" ".join(ref[i1:i2]))
            if tip:
                line = line + "   " + tip
        diff_lines.append(line)

    out("")
    out("=" * 62)
    out("   音素对比结果")
    out("=" * 62)
    out("整体音素相似度：%.1f / 100" % ratio)
    if diff_lines:
        out("差异 %d 处（明细见报告）" % len(diff_lines))
    else:
        out("音素序列完全一致。")
    log("对齐统计：匹配 %d ｜ 替换 %d ｜ 缺失 %d ｜ 多余 %d"
        % (align_stat["匹配音数"], align_stat["替换音数"],
           align_stat["缺失音数"], align_stat["多余音数"]), verbose=True)

    # 写一份"给人看"的纯文本报告（json 留给机器/AI）
    log_step("写报告与数据")
    try:
        write_report(ref, mine, sm, word_bounds, ratio, diff_items, problem_words)
        out("")
        out("报告（人看）：音素对比_报告.txt")
        out("数据（机器）：音素对比.json")
    except Exception as e:
        log("（写报告失败：%s）" % e, show=True)

    # 词切分：连读区 / 碎片词 / 每个词分到几个音素。
    word_split = {}
    if word_bounds:
        word_split = {
            "最小词长ms": MIN_WORD_MS,
            "碎片词": [{"word": sp["word"], "start_ms": sp["start_ms"],
                        "end_ms": sp["end_ms"], "span_ms": sp["span_ms"],
                        "n_ph": sp["n_ph"],
                        "原因": "时长不足 %dms" % MIN_WORD_MS}
                       for sp in spans if sp["short"]],
            "连读区": [{"词": z["words"], "start_ms": z["start_ms"], "end_ms": z["end_ms"],
                        "n_ph": z["n_ph"],
                        "归词音素数": [[spans[m]["word"], spans[m]["n_ph"]] for m in z["member_idx"]],
                        "说明": "%s 黏连成块（连读或短词测量不足），按整体看" % " ".join(z["words"])}
                       for z in zones],
            "命中差异的连读区": list(zone_hits.values()),
            "词音素分布": [{"word": sp["word"], "n_ph": sp["n_ph"],
                            "span_ms": sp["span_ms"]} for sp in spans],
        }

    # 单词错误（标注层，不进综合分。基准 = 原音频实际音素，见 compute_word_errors）
    word_errors = compute_word_errors(word_bounds, ref, ref_owner, mine, my_owner) if word_bounds else []

    # 词块：节奏脚本的唯一计时输入（两侧都要）
    word_blocks = {
        "原音频": [{"词": sp["word"], "start_ms": sp["start_ms"], "end_ms": sp["end_ms"],
                    "连读": sp["idx"] in zone_by_word,
                    "角色": word_bounds[sp["idx"]].get("角色") if sp["idx"] < len(word_bounds) else None}
                   for sp in spans],
        "我的": [{"词": sp["word"], "start_ms": sp["start_ms"], "end_ms": sp["end_ms"],
                  "连读": sp["idx"] in my_zone_words,
                  "角色": word_bounds[sp["idx"]].get("角色") if sp["idx"] < len(word_bounds) else None}
                 for sp in spans_my],
    }

    try:
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump({"原音频": ref, "我的模仿": mine,
                       "相似度": ratio, "对齐统计": align_stat,
                       "差异": diff_lines,
                       # 词表去掉 start_ms/end_ms —— 时间上移到「词块」
                       "词表": [{"word": wb["word"], "角色": wb.get("角色")} for wb in word_bounds],
                       "词切分": word_split,
                       "词块": word_blocks,
                       "单词错误": word_errors or None,
                       "半截模仿": coverage,
                       "参考音素可用": bool(ref_data)}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        # 写盘失败必须让人看见：下游 compare_rhythm / overall_score 都读这个 json，
        # 它没写出来 = 整条链断。旧版 except:pass 会静默吞掉，这里改成显式告警。
        log("（写 音素对比.json 失败，下游将读不到数据：%s）" % e, show=True)

    log_done("音素对比")


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
