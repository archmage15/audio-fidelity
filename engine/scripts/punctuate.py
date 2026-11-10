# -*- coding: utf-8 -*-
"""标点恢复：讯飞 ASR 常把句号标成逗号，导致断句不准（三句并一句）。

用 xlm-roberta 标点模型对转写文本重新标点，复用 transformers 环境。
模型：oliverguhr/fullstop-punctuation-multilingual-base（xlm-roberta-base，英文句号 F1≈0.94）
标签：0=无标点  . 句号  , 逗号  ? 问号  - 连字符  : 冒号

用法：
    from punctuate import restore_punctuation
    text = restore_punctuation("specialist well sort of yeah I guess so that's great")
    # -> "specialist. well sort of yeah I guess so. that's great."

为什么单独一个模块（不与 transcript_fix 合并）：
  标点模型约 500MB，加载/推理都重，跟 wordninja（几十 KB）不是一回事。
  拆开便于独立测试、也便于以后换模型。
"""

from cpu_runtime import configure_cpu_runtime
configure_cpu_runtime()
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# 走 hf-mirror 镜像（与 compare_phoneme 一致）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

MODEL_ID = "oliverguhr/fullstop-punctuation-multilingual-base"
LOCAL_DIR = os.environ.get("PUNCT_MODEL_DIR") or os.path.join(ROOT, "models", "punct-multilang")

# 句末标点：去掉后让模型重标（讯飞的逗号/句号不可信）。词内撇号/连字符保留。
_STRIP_RE = re.compile(r"[,.?!:;]")
# 真正需要插入的标点标签（0/O/空 = 无标点）
_PUNCT_LABELS = {".", ",", "?", "!", "-", ":", ";"}
_SPECIAL = {"<s>", "</s>", "<pad>", "<unk>", "[CLS]", "[SEP]", "[PAD]", "[UNK]",
            "<cls>", "<sep>", "<pad>", "<unk>"}

_model = None
_tokenizer = None
_id2label = None


def ensure_punct_model():
    """加载标点模型（首次由调用方下载好约 500MB）。返回 (model, tokenizer, id2label)。"""
    global _model, _tokenizer, _id2label
    from cpu_runtime import configure_torch_cpu
    configure_torch_cpu()
    if _model is not None:
        return _model, _tokenizer, _id2label
    import torch  # noqa: F401  确保 torch 可用
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    if not os.path.exists(os.path.join(LOCAL_DIR, "config.json")):
        raise RuntimeError(
            "标点模型未下载到 " + LOCAL_DIR + "。\n"
            "  请先下载：snapshot_download(repo_id='" + MODEL_ID + "', local_dir='" + LOCAL_DIR + "')")
    _tokenizer = AutoTokenizer.from_pretrained(LOCAL_DIR)
    _model = AutoModelForTokenClassification.from_pretrained(LOCAL_DIR)
    _model.eval()
    # id2label 可能是 {int: str} 或 {str: str}，统一成 {int: str}
    raw = _model.config.id2label or {}
    _id2label = {}
    for k, v in raw.items():
        _id2label[int(k)] = v
    return _model, _tokenizer, _id2label


def _merge_and_punct(tokens, labels):
    """把 subword token 合并回词，按「词尾 token」的标签在词后插标点。

    xlm-roberta 用 sentencepiece，词首 token 以 '▁' 开头；续 token 不带前缀。
    标点标签语义 = 「这个 token 后面跟什么标点」，所以取词最后一个 subword 的标签。
    """
    result = []
    cur_word = ""
    cur_label = "0"
    for tok, lab in zip(tokens, labels):
        if tok in _SPECIAL:
            continue
        if tok.startswith("▁"):
            if cur_word:
                result.append(cur_word + (cur_label if cur_label in _PUNCT_LABELS else ""))
            cur_word = tok[1:]
        else:
            cur_word += tok
        cur_label = lab
    if cur_word:
        result.append(cur_word + (cur_label if cur_label in _PUNCT_LABELS else ""))
    return result


def _trailing_punct(word):
    """取词尾的句末标点（. ? !），没有则返回空串。"""
    m = re.search(r"[.?!]$", word)
    return m.group(0) if m else ""


def restore_punctuation(text):
    """重新断句，但**只加句号、不改原句号**（「或」策略）：

    讯飞的句号/问号是对的（断句位置对），讯飞的逗号才可能把该断句的地方连起来。
    所以：原句末标点（. ? !）原样保留；只有「讯飞标逗号/无标点」的位置，
    才听标点模型的（模型说该断就断）。这样既修好「三句并一句」，又不改坏原句号。
    """
    model, tokenizer, id2label = ensure_punct_model()
    import torch

    # 1. 记录原词（带标点）→ 去标点得到纯词序列（词数一致，用于模型输入）
    orig_words = text.split()
    clean_words = [_STRIP_RE.sub("", w) for w in orig_words]
    clean_text = " ".join(clean_words)
    if not clean_text.strip():
        return text

    # 2. 模型对纯词序列预测标点
    enc = tokenizer(clean_text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**enc).logits
    pred_ids = logits[0].argmax(-1).tolist()
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist())
    labels = [id2label.get(p, "0") for p in pred_ids]
    predicted_words = _merge_and_punct(tokens, labels)

    # 3. 词数对不上（subword 边界异常，如 it'ss）→ 退回整段重标
    if len(predicted_words) != len(clean_words):
        return " ".join(predicted_words)

    # 4. 「或」策略：原句末标点保留，其余听模型的
    result = []
    for orig, pred in zip(orig_words, predicted_words):
        tp = _trailing_punct(orig)
        if tp:                                   # 原词尾有句末标点 → 保留
            result.append(_STRIP_RE.sub("", orig) + tp)
        else:                                    # 逗号/无标点 → 听模型的
            result.append(pred)
    return " ".join(result)
