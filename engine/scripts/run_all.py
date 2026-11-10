# -*- coding: utf-8 -*-
r"""一键跑还原度 · 统一走长文本逐句评测（单句 = 长文本切 1 句）

被 长文本模式一键跑还原度.bat / 单句模式一键跑还原度.bat 调用。

模式（10-28 并轨：删掉独立的单句链路，单句并入长文本）：
  所有音频统一走：
      1. 每次先调讯飞转写取台词文本当真值（不依赖手动放 转写文本.txt）
         - 转写空/失败 → 报「没识别到文本」直接退出
      2. 转写文本做轻量清理（andWhat → and What 这类粘连）后生成参考音素表
      3. compare_long.py 按标点切句、逐句评测（一句文本切出 1 句，自然只跑 1 次）
  最后跑 overall_score.py

用法：python scripts\run_all.py
"""

from cpu_runtime import configure_cpu_runtime
configure_cpu_runtime()
import os
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC_DIR = os.environ.get("AUDIO_TEST_WORK_DIR") or os.path.join(ROOT, "audio")

REF_PHONEME_JSON = os.path.join(SRC_DIR, "参考音素.json")
TRANSCRIBE_TXT = os.path.join(SRC_DIR, "转写文本.txt")
XUNFEI_CONFIG = os.path.join(ROOT, "讯飞配置.json")
REF_DIR = os.path.join(SRC_DIR, "原音频")
REF_CACHE_JSON = os.path.join(REF_DIR, "原音频缓存.json")   # 原音频分析缓存（10-28 加）

sys.path.insert(0, HERE)
from logkit import log, out, log_setup, log_done, log_step, log_error
from audio_paths import find_audio, missing_hint, short_name, reference_cache_stale

# 音频格式无关：mp3 / wav / flac / ogg 都认，认不出返回 None
# 原音频可能放在 audio\原音频\ 子目录（10-28 拆缓存后）
REF_WAV = find_audio("原音频", REF_DIR) or find_audio("原音频", SRC_DIR)
MY_WAV = find_audio("模仿", SRC_DIR)


def audio_duration(path):
    """音频时长（秒）；读不到返回 None。"""
    try:
        import soundfile as sf
        return sf.info(path).duration
    except Exception:
        return None


def run_script(name, args=()):
    """调另一个脚本（stdin 关掉，避免结尾 input() 卡住）。返回退出码。"""
    cmd = [sys.executable, os.path.join(HERE, name)] + list(args)
    log("   运行：%s %s" % (name, " ".join(args[:1]) + ("..." if args else "")), verbose=True)
    try:
        r = subprocess.run(cmd, cwd=ROOT, stdin=subprocess.DEVNULL)
        return r.returncode
    except Exception as e:
        log("（运行 %s 失败：%s）" % (name, e), show=True)
        return 1


def ensure_reference():
    """确保参考音素表就绪（优先用原音频缓存，跳过转写+生成）。

    返回 True=就绪，False=没缓存且无法生成（调用方应直接退出）。

    10-28 改：原音频分析结果已缓存（audio\\原音频\\原音频缓存.json）时，
    参考音素表/转写文本都是缓存里现成的，直接跳过讯飞转写和 espeak 生成。
    没缓存才需要先跑「分析原音频.bat」。

    10-30 加：缓存里存了原音频指纹，换过原音频则缓存过期 —— 此时
    **拒绝用旧缓存**（否则会用旧参考音素/logits 评测新音频、结果错乱
    且不报错），提示重新跑「分析原音频.bat」。
    """
    # 有缓存 → 先做失效检测，过期则拒绝使用（10-30 加）
    if os.path.exists(REF_CACHE_JSON):
        cache = None
        try:
            import json as _json
            with open(REF_CACHE_JSON, encoding="utf-8") as f:
                cache = _json.load(f)
        except Exception:
            cache = None
        stale = reference_cache_stale(cache, REF_WAV)
        if stale is True:
            out("")
            out("X 原音频缓存已过期：原音频文件换过了。")
            out("  旧缓存里的参考音素/CTC 结果对应的是旧原音频，")
            out("  直接拿来评测会得出错误结论。")
            out("  请重新双击 audio\\原音频\\分析原音频.bat 重建缓存。")
            return False
        if stale is None:
            # 旧版缓存没有指纹字段，无法判断是否过期 —— 保守起见提示但继续
            log("   ⚠ 原音频缓存缺少指纹字段（旧版缓存），无法判断是否过期；"
                "若已换过原音频请先重跑 分析原音频.bat", show=True)
        log("   有原音频缓存 → 跳过转写与参考音素表生成", show=True)
        return True
    # 没缓存：看看参考音素表是否已经在（旧流程残留），在也能跑，但要提示补缓存
    if os.path.exists(REF_PHONEME_JSON):
        log("   没缓存但已有参考音素表，继续（建议先跑 分析原音频.bat 建缓存）", show=True)
        return True
    out("")
    out("X 没找到原音频缓存，也没有参考音素表。")
    out("  请先双击 audio\\原音频\\分析原音频.bat，把原音频分析一遍生成缓存，")
    out("  之后一键跑还原度才会只分析模仿音频。")
    return False


def main():
    log_setup("一键路由")
    out("=" * 62)
    out("   还原度测试 · 一键跑（统一走长文本逐句评测）")
    out("=" * 62)

    if not (REF_WAV and MY_WAV):
        out("X 没找到音频文件，请把两个音频放进 audio\\：")
        out(missing_hint([("原音频", REF_WAV), ("模仿", MY_WAV)], SRC_DIR))
        log_done("（缺音频，提前退出）")
        return 1
    # 单句 = 长文本切出恰好 1 句的特例，不再有独立的单句链路（10-28 并轨）。
    #   所有音频统一走「讯飞转写 → 按标点切句 → compare_long 逐句评测」。
    #   一句文本切出来就是 1 句，逐句评测循环自然只跑 1 次，无需单独分支。
    dur = audio_duration(REF_WAV) or 0.0
    out("输入音频：%s（%.1fs） vs %s" % (short_name(REF_WAV), dur, short_name(MY_WAV)))
    out("本次模式：长文本逐句评测（单句/多句统一）")

    log_step("[1/2] 讯飞转写 + 逐句评测（发音/音变/节奏/重音）")
    if not ensure_reference():
        log_done("（没识别到文本，提前退出）")
        return 1
    if run_script("compare_long.py") != 0:
        out("X 逐句评测失败，详见上方日志")
        log_done("（评测失败）")
        return 1
    log_step("[2/2] 综合评分 + 报告")
    if run_script("overall_score.py") != 0:
        out("X 综合评分失败，详见上方日志")
        log_done("（评分失败）")
        return 1
    out("")
    out("=" * 62)
    out("全部完成。结果在：" + os.path.abspath(os.environ.get("AUDIO_TEST_OUTPUT_DIR") or os.path.join(ROOT, "output")))
    out("    还原度综合报告.txt   ← 先看这份（人看）")
    out("    音素对比_报告.txt    ← 逐句/逐音明细")
    out("    （逐句明细 + 漏句清单见 音素对比_报告.txt）")
    out("=" * 62)
    log_done("一键跑还原度")
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
