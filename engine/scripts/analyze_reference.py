# -*- coding: utf-8 -*-
r"""原音频分析 · 一次性跑完原音频的所有分析，产出缓存文件

为什么单独这个脚本（10-28 加）：
  原音频是固定范本，它的转写文本、参考音素表、CTC 识别音素、节奏/重音特征
  全都不会变。但 compare_long 每次都要对原音频重新跑一遍这些（尤其 1.2GB
  模型的 CTC 推理 + RMS/音高计算），白花时间。本脚本把原音频的完整分析
  结果一次性算好、存成缓存，compare_long 之后直接读缓存，只分析模仿音频。

产出：audio\原音频\原音频缓存.json，包含：
  转写文本、参考音素表（音素/词边界/句子边界）、原音频 CTC 音素序列 + 帧、
  原音频 RMS、原音频音高。

用法：python scripts\analyze_reference.py
      （被 audio\原音频\分析原音频.bat 调用）
"""

from cpu_runtime import configure_cpu_runtime
configure_cpu_runtime()
import json
import os
import subprocess
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC_DIR = os.environ.get("AUDIO_TEST_WORK_DIR") or os.path.join(ROOT, "audio")
REF_DIR = os.path.join(SRC_DIR, "原音频")          # 原音频放这里
REF_PHONEME_JSON = os.path.join(SRC_DIR, "参考音素.json")
TRANSCRIBE_TXT = os.path.join(SRC_DIR, "转写文本.txt")
CACHE_JSON = os.path.join(REF_DIR, "原音频缓存.json")
XUNFEI_CONFIG = os.path.join(ROOT, "讯飞配置.json")

sys.path.insert(0, HERE)
from logkit import log, out, log_setup, log_done, log_step, log_error
from audio_paths import find_reference, short_name, _file_fingerprint, file_content_hash, reference_cache_stale
import compare_phoneme as cp
import compare_rhythm as cr


def run_script(name, args=()):
    cmd = [sys.executable, os.path.join(HERE, name)] + list(args)
    r = subprocess.run(cmd, cwd=ROOT, stdin=subprocess.DEVNULL)
    return r.returncode


def clean_transcript(text):
    """转写文本清理 + 修复：大小写粘连拆开 + 缩略词（Im→I'm）+ 粘连词分割
    （wellsort→well sort）。返回 (清理后文本, 修复记录列表)。"""
    import transcript_fix
    return transcript_fix.fix_transcript(text)


# 转写确认环节的待确认文件（人工兜底，任务 3）
CONFIRM_TXT = os.path.join(REF_DIR, "转写文本_待确认.txt")
# 跳过确认开关：环境变量 AUTO_CONFIRM=1 或命令行 --auto，无人值守时不弹编辑器
_AUTO_CONFIRM = (os.environ.get("AUTO_CONFIRM", "").strip() in ("1", "true", "True", "yes"))


def split_lines(text):
    """按句末标点（. ? !）把转写文本切成一行一句，方便人工逐句核对。

    只作展示用（写进待确认文件），读回时会压回单行，不影响后续生成参考音素表。
    """
    import re
    # 在句末标点后切，但保留标点；去掉多余空格
    parts = re.split(r"(?<=[.?!])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _collapse(text):
    """把文本里的换行/多余空白压成单空格（读回待确认文件时用）。

    待确认文件是「每句一行」给人看的，读回后要并回单行才能进后续流程
    （gen_ref_phonemes 等按空格分词）。
    """
    import re
    return re.sub(r"\s+", " ", text).strip()


def should_reconfirm(existing, text):
    """判断是否要重新弹确认。

    existing 为空 → 首次，要弹；
    existing 与 text 一致 → 已确认过，不弹；
    existing 与 text 不一致 → 换过原音频或上次改过，要弹。
    """
    if not existing:
        return True
    return existing != text


def _open_in_editor(path):
    from platform_tools import open_text_editor
    return open_text_editor(path)


def confirm_transcript(text):
    """转写确认环节（人工兜底，任务 3）。

    把标点恢复后的转写文本写到 CONFIRM_TXT，用编辑器打开让用户过目/修改，
    改完保存关闭后读回。兜底两类自动修不掉的错：
      1. 标点模型误判（句边界偏差、个别逗号误判）
      2. 讯飞听错词（it'ss 应为 it's a）

    返回 (确认后文本, 是否有改动)。

    AUTO_CONFIRM=1 时跳过确认，直接用传入文本（无人值守/批量重跑不卡住）。
    """
    if _AUTO_CONFIRM:
        log("   AUTO_CONFIRM=1，跳过转写确认，直接用转写结果", show=True)
        return text, False

    # 待确认文件已存在 → 看它和本次转写结果是否一致：
    #   - 一致：这文件就是本次转写写的、且已确认过，直接采用，不重弹
    #   - 不一致：换过原音频（旧文件是上个音频的）或上次手动改过 → 重弹确认
    existing = ""
    if os.path.exists(CONFIRM_TXT):
        try:
            with open(CONFIRM_TXT, encoding="utf-8") as f:
                existing = _collapse(f.read())
        except Exception:
            existing = ""
    if not should_reconfirm(existing, text):
        log("   转写文本已确认过（与本次转写一致，直接采用 %d 字符）" % len(existing), show=True)
        return existing, False
    if existing:
        log("   待确认文件内容与本次转写不一致（换过原音频或上次改过），重新确认",
            show=True)

    # 首次确认：写待确认文件（每句一行，方便逐句核对）+ 开编辑器 + 轮询等保存
    try:
        os.makedirs(REF_DIR, exist_ok=True)
        with open(CONFIRM_TXT, "w", encoding="utf-8") as f:
            f.write("\n".join(split_lines(text)) + "\n")
    except Exception as e:
        log("   ⚠ 无法写待确认文件（%s），跳过确认直接继续" % e, show=True)
        return text, False

    out("")
    out("  ┌─ 转写确认（人工兜底）──────────────────────────")
    out("  │ 已把转写文本写进：")
    out("  │   " + CONFIRM_TXT)
    out("  │ 已用编辑器打开。请过目，有错直接改，保存后关闭窗口。")
    out("  │ 不改直接关闭 = 确认无误。")
    out("  └────────────────────────────────────────────")

    opened = _open_in_editor(CONFIRM_TXT)
    if not opened and os.name != "nt":
        raise RuntimeError("无法打开确认编辑器。服务器批量缓存请去掉 --confirm-text；"
                           "人工确认请在交互终端配置 EDITOR，或使用桌面 xdg-open。"
                           "待确认文件：" + CONFIRM_TXT)
    if not opened:
        log("   ⚠ 无法自动打开编辑器，请手动编辑：%s" % CONFIRM_TXT, show=True)

    # 等用户改完保存并关闭编辑器。这里没法可靠感知「编辑器已关闭」，
    # 改为轮询「文件内容变化后，等待一个稳定期」。
    import time as _time
    last_text = text
    stable = 0
    while True:
        try:
            with open(CONFIRM_TXT, encoding="utf-8") as f:
                cur = f.read().strip()
        except Exception:
            cur = ""
        if cur and cur != last_text:
            last_text = cur
            stable = 0            # 内容变了 → 重置稳定计数
        else:
            stable += 1
        if stable >= 30:          # 约 30 秒内容不变 → 视为已确认（含「直接关闭不改」）
            break
        _time.sleep(1)

    # 读回最终文本。用户可能删空（误删）→ 删空则沿用原文本并提示。
    try:
        with open(CONFIRM_TXT, encoding="utf-8") as f:
            confirmed = _collapse(f.read())
    except Exception:
        confirmed = ""
    if not confirmed:
        log("   ⚠ 待确认文件被清空，沿用转写结果继续", show=True)
        confirmed = text
    changed = (confirmed != text)
    if changed:
        log("   已采用人工修改后的文本（%d 字符 → %d 字符）" % (len(text), len(confirmed)),
            show=True)
    else:
        log("   转写文本确认无误", show=True)
    return confirmed, changed


def main():
    global _AUTO_CONFIRM
    # 命令行 --auto 跳过转写确认（与 AUTO_CONFIRM=1 环境变量等价）
    if "--auto" in sys.argv:
        _AUTO_CONFIRM = True

    log_setup("原音频分析")
    out("=" * 62)
    out("   原音频分析 · 一次性跑完并存缓存")
    out("=" * 62)

    ref_path = find_reference()
    if not ref_path:
        out("X 没找到原音频，请放到 audio\\原音频\\ 目录：")
        out("      audio\\原音频\\原音频.mp3（或 wav / flac / ogg）")
        log_done("（缺原音频，提前退出）")
        return 1
    out("原音频：%s" % short_name(ref_path))

    # Prepare the vocabulary before transcription and reference generation.
    log_step("检查音素模型（首次缺失时下载）")
    cp.ensure_model()

    # ---- 1. 讯飞转写（拿台词文本当真值）----
    log_step("讯飞转写原音频")
    if not os.path.exists(XUNFEI_CONFIG):
        out("X 没有 讯飞配置.json，无法转写。请放好再跑。")
        log_done("（缺讯飞配置，提前退出）")
        return 1
    if run_script("xunfei_transcribe.py", [ref_path]) != 0:
        out("X 没识别到文本（转写失败或音频无人声）。")
        log_done("（转写失败，提前退出）")
        return 1
    text = None
    if os.path.exists(TRANSCRIBE_TXT):
        with open(TRANSCRIBE_TXT, encoding="utf-8") as f:
            text = f.read().strip()
    if not text:
        out("X 没识别到文本（转写结果为空）。")
        log_done("（转写为空，提前退出）")
        return 1
    text, fixes = clean_transcript(text)
    for kind, wrong, right, n in fixes:
        out("   修复 [%s]：%r → %r（%d 处）" % (kind, wrong, right, n))

    # ---- 1.5 标点恢复（重新断句）----
    # 讯飞常把句号标成逗号，导致三句并一句 → 用标点模型重新断句。
    # 失败（如模型没下载）就沿用原标点继续，不中断。
    log_step("标点恢复（重新断句，首次加载模型约 1.1GB 需十几秒）")
    import punctuate
    try:
        text = punctuate.restore_punctuation(text)
        log("   重新标点完成：%s" % (text[:60] + ("..." if len(text) > 60 else "")))
    except Exception as _e:
        log("   ⚠ 标点恢复失败（%s），沿用原标点继续" % _e, show=True)
    log("   台词 %d 字符" % len(text))

    # ---- 1.6 转写确认（人工兜底，任务 3）----
    # 标点恢复后、生成参考音素表前，把文本给用户过目/修改，兜底标点误判和听错词。
    log_step("转写确认（人工兜底）")
    text, _confirmed_changed = confirm_transcript(text)

    # ---- 2. 生成参考音素表 ----
    log_step("生成参考音素表（音素/词边界/句子边界）")
    if run_script("gen_ref_phonemes.py", [text]) != 0:
        out("X 生成参考音素表失败，请查看上方具体错误（模型词表或文本转换）")
        log_done("（生成失败，提前退出）")
        return 1
    with open(REF_PHONEME_JSON, encoding="utf-8") as f:
        ref_data = json.load(f)
    log("   参考 %d 音素 / %d 词 / %d 句"
        % (len(ref_data.get("音素") or []), len(ref_data.get("词边界") or []),
           len(ref_data.get("句子边界") or [])))

    # ---- 3. CTC 识别原音频 ----
    log_step("CTC 识别原音频（跑音素模型）")
    os.makedirs(REF_DIR, exist_ok=True)
    _logits_npy = os.path.join(REF_DIR, "原音频_logits.npy")
    reuse_logits = False
    if os.path.exists(_logits_npy) and os.environ.get("AUDIO_TEST_FORCE_REFERENCE_ANALYSIS") != "1":
        try:
            with open(CACHE_JSON, encoding="utf-8") as previous_file:
                previous_cache = json.load(previous_file)
            reuse_logits = reference_cache_stale(previous_cache, ref_path) is False
        except (OSError, ValueError, TypeError):
            pass
    if reuse_logits:
        # 原音频没变 → 复用 logits 缓存，跳过 1.2GB 模型加载。
        # （10-30 前踩过 3 次：加载模型要 30~60 秒，用户以为卡住关了窗口 →
        #   缓存写不进来，转写/参考音素表更新了但缓存还是旧的）
        _logits_arr = np.load(_logits_npy)
        ids = _logits_arr.argmax(axis=1).tolist()
        id2tok = cp.load_vocab(cp.ensure_model())
        ref_toks, ref_frames = cp.ctc_decode(ids, id2tok)
        log("   原音频 %d 音素（复用 logits 缓存，跳过模型加载）" % len(ref_toks))
    else:
        out("   ⚠ 首次运行要加载音素模型（约 1.2GB），需 30~60 秒。")
        out("     请耐心等，这一步完成前**不要关窗口**，否则缓存写不进来。")
        import torch
        model_path = cp.ensure_model()
        fe = cp.load_feature_extractor(model_path)
        from transformers import AutoModelForCTC
        model = AutoModelForCTC.from_pretrained(model_path)
        model.eval()
        id2tok = cp.load_vocab(model_path)
        audio = cp.load_mono(ref_path)
        inputs = fe(audio, sampling_rate=cp.SR, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        ids = torch.argmax(logits, dim=-1)[0].tolist()
        ref_toks, ref_frames = cp.ctc_decode(ids, id2tok)
        log("   原音频 %d 音素" % len(ref_toks))
        # 帧级 logits 存 .npy（强制对齐用）。⚠️ 存二进制 .npy 而不是塞 JSON——
        #   JSON 存 float 数组会膨胀 5~10 倍（约 2.6MB → 十几 MB），.npy 原样存。
        _logits_arr = logits[0].detach().cpu().numpy().astype("float32")
        np.save(_logits_npy, _logits_arr)
        log("   原音频 logits 已存：%s（%d 帧 × %d 符号）"
            % (os.path.basename(_logits_npy), _logits_arr.shape[0], _logits_arr.shape[1]))

    # ---- 4. 节奏/重音特征（RMS + 音高）----
    log_step("算节奏/重音特征（RMS + 音高）")
    _ref_wave = cp.load_mono(ref_path)  # 只加载一次，RMS/音高共用
    env = cr.frame_rms(_ref_wave)
    f0 = cr.pitch_contour(_ref_wave)
    log("   RMS %d 帧 / 音高 %d 帧" % (len(env), len(f0)))

    # ---- 5. 打包缓存 ----
    log_step("写缓存")
    cache = {
        "转写文本": text,
        "参考音素": ref_data,
        "原音频_CTC音素": ref_toks,
        "原音频_帧": ref_frames,
        "原音频_RMS": [float(x) for x in env],
        "原音频_音高": [float(x) for x in f0],
        "原音频路径": ref_path,
        "原音频SHA256": file_content_hash(ref_path),
        # 指纹（10-30 加）：换原音频后旧缓存要能被识别出来并拒绝使用，
        # 否则会用旧参考音素/logits 评测新音频、结果错乱且不报错。
        "原音频指纹": list(_file_fingerprint(ref_path)),
        "生成时间": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(REF_DIR, exist_ok=True)
    with open(CACHE_JSON, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)

    out("")
    out("原音频分析完成，缓存已存：")
    out("    " + CACHE_JSON)
    out("以后一键跑还原度会直接读这份缓存，不再重跑原音频分析。")
    log_done("原音频分析")
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
