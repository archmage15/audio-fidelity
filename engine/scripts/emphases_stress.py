# -*- coding: utf-8 -*-
r"""用 emphases 找重读词（替代邻域法，2026-11-02 大法师拍板）。

emphases 是 Georgia Tech 的单词重音（prominence）识别模型，用真人母语者
众包标注训练，靠声学信号判断每个词"母语者听起来重不重"。输出逐词
prominence 分数（0~1，越高越重读）。

本模块只做一件事：给一段音频 + 词边界，返回每个词的重读判定。

依赖：torch / pypar / emphases / torchutil（这些只在 .venv 里，正式评测
已改用 .venv，见 一键跑还原度.bat）。

⚠️ emphases 加载模型约 12 秒（import torch + 加载权重），模块级缓存，
   同一次评测只加载一次。
"""

from cpu_runtime import configure_cpu_runtime
configure_cpu_runtime()
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CHECKPOINT = os.path.join(ROOT, "models", "emphases", "model.pt")

# prominence 分数阈值：> 阈值判「重读」。当前是经验值（n=1），待多样本标定。
#   实测：实词重读约 0.3~0.8，虚词/弱读约 0.0~0.1，0.25 能把两者分得较干净。
THRESHOLD = 0.25

_model = None  # 模块级缓存：加载一次，复用


def _ensure_model():
    """加载 emphases 模型（缓存，只加载一次）。"""
    global _model
    from cpu_runtime import configure_torch_cpu
    configure_torch_cpu()
    if _model is None:
        import torchutil
        import emphases
        _model = emphases.Model()
        torchutil.checkpoint.load(CHECKPOINT, _model)
    return _model


def stress_scores(audio_wave, sample_rate, blocks):
    """给一段音频 + 词边界，返回每个词的 prominence 分数（numpy 数组）。

    参数：
      audio_wave   一维 numpy 波形（建议 16kHz，其他采样率 emphases 内部会重采样）
      sample_rate  采样率
      blocks       list[dict]，每个词含 start_ms / end_ms / 词（列表）

    返回：len(blocks) 长的 float 数组，逐词 prominence 分数（0~1）。
    """
    import torch
    import pypar
    import emphases

    words = []
    for b in blocks:
        start_s = b["start_ms"] / 1000.0
        end_s = b["end_ms"] / 1000.0
        if end_s <= start_s:
            end_s = start_s + 0.01  # 兜底：极端短词给个最小时长
        phoneme = pypar.Phoneme(pypar.SILENCE, start_s, end_s)
        # 词文本取列表第一个（词块契约：词 是 list[str]）
        wtext = b["词"][0] if isinstance(b.get("词"), (list, tuple)) and b["词"] else "?"
        words.append(pypar.Word(wtext, [phoneme]))

    alignment = pypar.Alignment(words)
    audio_t = torch.from_numpy(np.asarray(audio_wave, dtype=np.float32))[None, :]
    scores = emphases.from_alignment_and_audio(
        alignment, audio_t, sample_rate, CHECKPOINT, None, None)
    # bfloat16 -> float32 -> numpy
    return scores.cpu().float().numpy().ravel()


def stress_marks(audio_wave, sample_rate, blocks, threshold=THRESHOLD):
    """返回 (重读词下标集合[1-based], 逐词分数数组)。

    重读 = 分数 > threshold。下标从 1 起，与邻域法 marked 契约一致。
    """
    scores = stress_scores(audio_wave, sample_rate, blocks)
    marks = [i + 1 for i in range(len(blocks)) if scores[i] > threshold]
    return marks, scores
