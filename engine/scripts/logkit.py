# -*- coding: utf-8 -*-
r"""还原度测试 · 统一日志模块

三个脚本（compare_phoneme / compare_rhythm / overall_score）共用这一份日志逻辑，
避免同一套代码抄三遍、以后改一处要改三处。

产出两个文件（都在 <项目>\output\logs\ 里）：

  全部日志.txt   —— 追加模式。跨多次运行累积，能看历史。
                    每次运行会在开头插一段分隔线，标明时间与脚本名。

  本次运行.txt   —— 覆盖模式。每次运行重写，只看最新一次。
                    三步（音素 → 节奏 → 综合）会依次追加进同一个文件，
                    所以打开它就是本次的完整时间线。

    为什么是"依次追加"而不是各自覆盖？
      一键 bat 会连着跑三个脚本。若每个脚本都覆盖，后一个会把前一个冲掉，
      只剩最后一步。所以约定：**第一个脚本负责清空，后面的都追加**。

开关（环境变量）：

  LOG_VERBOSE=1   详细模式。除关键步骤外，还记录中间数据（音素数、词数、
                  各子项分、耗时明细等）。默认关闭，只记关键步骤。
                  用法：set LOG_VERBOSE=1 && 一键跑还原度.bat

  LOG_OFF=1       完全关闭日志写盘（终端仍照常打印）。
                  适合临时跑一下、不想留痕的场景。

  LOG_QUIET=1     终端只显示"结果"，步骤类日志行只写进文件。
                  终端会安静很多，但结果块（相似度、分数、文件名）照常显示。
                  适合每天用的时候不想被过程日志刷屏。

用法（三个脚本里都一样）：

  from logkit import log, log_setup, log_done

  log_setup("音素对比")      # 在 main() 开场调用一次：决定本次是否清空单次日志
  log("加载模型 ...")        # 日志行：自动带时间戳、自动落盘；静默模式下不打印
  log("识别出 %d 个音素" % n, verbose=True)   # 只在详细模式下出现
  out("整体音素相似度：71.9 / 100")           # 结果行：始终打印，静默模式下也显示
  log_done()                 # 在 main() 收尾调用一次：记录总耗时
"""

from cpu_runtime import configure_cpu_runtime, cpu_runtime_description
configure_cpu_runtime()

import datetime
import os
import sys
import time

from console_runtime import prevent_console_pause
prevent_console_pause()

# ---- 目录约定 ----
# 本文件在 <项目>\scripts\ 下，ROOT = 项目根（上一层）
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
LOG_DIR = os.path.join(os.environ.get("AUDIO_TEST_OUTPUT_DIR") or os.path.join(_ROOT, "output"), "logs")
LOG_ALL = os.path.join(LOG_DIR, "全部日志.txt")
LOG_THIS = os.path.join(LOG_DIR, "本次运行.txt")

# ---- 开关 ----
VERBOSE = os.environ.get("LOG_VERBOSE", "").strip() in ("1", "true", "True", "yes")
OFF = os.environ.get("LOG_OFF", "").strip() in ("1", "true", "True", "yes")
QUIET = os.environ.get("LOG_QUIET", "").strip() in ("1", "true", "True", "yes")

# ---- 同一轮判断阈值（秒）----
# 三个脚本由一键 bat 顺序跑，同一轮间隔 < 几秒。但 compare_phoneme 首次下载
# 模型（1.2GB）时，下载期间日志不写盘、mtime 停在下载前；慢网下下载会超过
# 60 秒，导致 compare_rhythm 误判成"新一轮"、把音素那步的日志冲掉。
# 放宽到 10 分钟，覆盖最长的单步耗时。
SESSION_GAP_S = 600

# ---- 内部状态 ----
_START = None          # 本次 main() 的起始时刻（算总耗时用）
_warned = False        # 写盘失败只提示一次，不刷屏


def _now():
    """当前时刻，精确到毫秒。"""
    now = datetime.datetime.now()   # 只取一次，避免时分秒与毫秒来自两个跨秒的 now
    return now.strftime("%H:%M:%S.") + "%03d" % (now.microsecond // 1000)


def _ensure_dir():
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        return True
    except Exception:
        return False


def _write(path, text, append=True):
    """写日志文件。任何失败都不许影响主流程。"""
    global _warned
    if OFF:
        return
    if not _ensure_dir():
        if not _warned:
            print("[日志] 无法创建目录 %s，本次不写日志" % LOG_DIR, flush=True)
            _warned = True
        return
    try:
        mode = "a" if append else "w"
        with open(path, mode, encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        if not _warned:
            print("[日志] 写入失败（不影响分析）：%s" % e, flush=True)
            _warned = True


def _bar(ch="=", n=62):
    return ch * n


def log_setup(tag):
    """本轮开场：决定是否清空"本次运行"日志，并写分隔线。

    约定：**第一个跑的脚本清空，后面的追加**。
    怎么判断谁是第一个？看"本次运行.txt"里有没有本轮的标记。
    这里用更简单的办法：由调用方声明 is_first；但三个脚本各自独立、
    无法互相知道 —— 所以改用"一键 bat 传环境变量"的方式太重。

    最终方案：**按时间判断**。若"本次运行.txt"的最后修改时间距现在
    超过 SESSION_GAP_S 秒，就认为这是一轮新的运行，清空重写；否则视为
    同一轮的后续脚本，追加。这样既不用改 bat，也能让三步串在同一个文件里。
    """
    global _START
    _START = time.time()

    fresh = True
    if os.path.exists(LOG_THIS):
        try:
            age = time.time() - os.path.getmtime(LOG_THIS)
            if age < SESSION_GAP_S:
                fresh = False          # 阈值内有脚本写过 → 同一轮，追加
        except Exception:
            fresh = True

    head = []
    if fresh:
        head.append(_bar())
        head.append("本轮运行开始    %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        head.append(_bar())
    head.append("")
    head.append("--- [%s] %s %s ---" % (_now(), tag, "开始"))
    if VERBOSE:
        head.append("    （详细模式 LOG_VERBOSE=1）")
    block = "\n".join(head) + "\n"

    _write(LOG_THIS, block, append=not fresh)
    _write(LOG_ALL, block, append=True)
    log(cpu_runtime_description())


def log(msg="", verbose=False, show=None):
    """日志行：带时间戳，写两个日志文件。

    终端是否打印，取决于 show：
      show=False  —— 一定不打印（静默模式下所有日志行都走这条）
      show=True   —— 一定打印
      show=None   —— 看全局开关：LOG_QUIET=1 时不打印，默认打印

    verbose=True 的行只在详细模式下记录（终端与文件都不出）。
    """
    text = str(msg)
    if verbose and not VERBOSE:
        return

    to_terminal = (not QUIET) if show is None else bool(show)
    if to_terminal:
        print("[%s] %s" % (_now(), text), flush=True)

    # 文件里始终带时间戳，即使终端不显示 —— 便于事后追时间线
    line = "[%s] %s" % (_now(), text)
    _write(LOG_THIS, line + "\n", append=True)
    _write(LOG_ALL, line + "\n", append=True)


def out(msg=""):
    """结果行：一定会打印到终端，同时也写进日志文件（便于回看）。

    和 log() 的区别只有一个 —— 静默模式（LOG_QUIET=1）下 log() 不打印，
    out() 照常打印。用它包住"给用户看的结果块"：
    相似度、分数、文件名、缺少文件的提示等。
    """
    log(msg, show=True)


def log_step(msg):
    """关键步骤：带一个 ▶ 标记，方便在日志里一眼找到阶段分界。"""
    log("▶ " + str(msg))


def log_done(note=""):
    """本轮收尾：记录总耗时。"""
    global _START
    if _START is None:
        log("完成 %s" % note)
        return
    cost = time.time() - _START
    tail = "完成 %s（本脚本耗时 %.1f 秒）" % (note, cost)
    log(tail)
    _write(LOG_THIS, "\n", append=True)
    _write(LOG_ALL, "\n", append=True)


def log_error(exc_text):
    """记录错误。错误必须让人看见，所以无论是否静默都打印。"""
    log("!! 出错：", show=True)
    for line in str(exc_text).rstrip().splitlines():
        log("   " + line, show=True)
