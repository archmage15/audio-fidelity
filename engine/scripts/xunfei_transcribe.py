# -*- coding: utf-8 -*-
r"""讯飞录音文件转写（LFASR 标准版）· 长音频自动转写

用途：还原度测试的「第 0 步」——原音频没有台词文本时，先用讯飞把它转成
文字（带句级时间戳），产出：
  - audio/转写文本.txt      人读：整段转写文本
  - audio/转写结果.json     机器读：整段文本 + 逐句时间戳（供切句对齐用）

认证：从 ROOT\讯飞配置.json 读 {"app_id": "...", "secret_key": "..."}
     （该文件被 .gitignore 忽略，不进版本库。LFASR 只需这两把。）

接口（官方 v1，host raasr.xfyun.cn/api，认证 HmacSHA1(MD5(appid+ts), sk)）：
  prepare → upload(分片) → merge → getProgress(轮询至状态9) → getResult
  文档：https://www.xfyun.cn/doc/asr/lfasr/API.html

音频要求：wav/flac/opus/m4a/mp3，≤500M、≤5h，8k/16k 采样率。
     44.1k 立体声会先降成 16k 单声道（体积砍到 1/5，转写更快）。

用法：python scripts\xunfei_transcribe.py              # 转写 audio\原音频.*
     python scripts\xunfei_transcribe.py <wav路径>     # 指定其他音频
"""

from cpu_runtime import configure_cpu_runtime
configure_cpu_runtime()
import base64
import hashlib
import hmac
import json
import os
import sys
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC_DIR = os.environ.get("AUDIO_TEST_WORK_DIR") or os.path.join(ROOT, "audio")
CONFIG_JSON = os.path.join(ROOT, "讯飞配置.json")
OUT_TXT = os.path.join(SRC_DIR, "转写文本.txt")
OUT_JSON = os.path.join(SRC_DIR, "转写结果.json")

BASE = "https://raasr.xfyun.cn/api"
POLL_INTERVAL_S = 10       # 官方建议 10 分钟一次，实测短音频几十秒就出，这里 10s 轮询
POLL_MAX_TIMES = 60        # 最多轮询 10 分钟

# ---- 日志 ----
sys.path.insert(0, HERE)
from logkit import log, out, log_setup, log_done, log_step, log_error


def load_keys():
    """读讯飞配置。返回 (app_id, secret_key) 或抛异常（报错指路）。"""
    if not os.path.exists(CONFIG_JSON):
        raise RuntimeError(
            "找不到 " + CONFIG_JSON + "\n"
            "  请在 ROOT 下建 讯飞配置.json（已被 .gitignore 忽略）：\n"
            '  {"app_id": "你的APPID", "secret_key": "你的SecretKey"}\n'
            "  两把钥匙在讯飞控制台 → 应用 → Websocket服务接口认证信息。")
    with open(CONFIG_JSON, encoding="utf-8") as f:
        cfg = json.load(f)
    app_id = (cfg.get("app_id") or "").strip()
    secret_key = (cfg.get("secret_key") or "").strip()
    if not app_id or not secret_key:
        raise RuntimeError("讯飞配置.json 里 app_id / secret_key 是空的，请填好再跑。")
    return app_id, secret_key


def sign(app_id, secret_key, ts):
    """LFASR 签名（官方文档 §signa生成）：
    ① baseString = appid + ts
    ② md5 = MD5(baseString)
    ③ signa = base64(HmacSHA1(md5, secret_key))
    ⚠️ 中间必须先 MD5 一次，直接 HmacSHA1(appid+ts) 会被报 26610 appId illegal。"""
    base_string = app_id + str(ts)
    md5_str = hashlib.md5(base_string.encode()).hexdigest()
    dig = hmac.new(secret_key.encode(), md5_str.encode(), hashlib.sha1).digest()
    return base64.b64encode(dig).decode()


def api_call(app_id, secret_key, action, params):
    """POST form + 签名（upload 分片除外）。返回 dict；ok!=0 抛异常。"""
    ts = int(time.time())
    body = {"app_id": app_id, "ts": str(ts), "signa": sign(app_id, secret_key, ts)}
    body.update(params)
    req = Request(BASE + "/" + action, data=urlencode(body).encode())
    with urlopen(req, timeout=60) as r:
        resp = json.loads(r.read().decode())
    if resp.get("ok") != 0:
        raise RuntimeError("%s 失败（err_no=%s）：%s"
                           % (action, resp.get("err_no"), resp.get("failed")))
    return resp


def prep_audio(src_path):
    """降成 16k 单声道，返回实际用于上传的 wav 路径。
    已是 16k 单声道则直接用原文件。"""
    import soundfile as sf
    info = sf.info(src_path)
    if info.samplerate == 16000 and info.channels == 1:
        return src_path
    import numpy as np
    audio, sr = sf.read(src_path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != 16000:
        n_out = int(round(len(audio) * 16000 / sr))
        x_in = np.linspace(0, 1, len(audio), endpoint=False)
        x_out = np.linspace(0, 1, n_out, endpoint=False)
        audio = np.interp(x_out, x_in, audio)
    import tempfile
    fd, tmp = tempfile.mkstemp(prefix="xfyun_", suffix=".wav")
    os.close(fd)
    try:
        sf.write(tmp, audio.astype("float32"), 16000)
    except Exception:
        os.remove(tmp)
        raise
    log("   预处理：%.1fkHz/%dch -> 16kHz/1ch（%.1fMB -> %.1fMB）"
        % (info.samplerate / 1000, info.channels,
           os.path.getsize(src_path) / 1e6, os.path.getsize(tmp) / 1e6), verbose=True)
    return tmp


def upload_whole(app_id, secret_key, task_id, upload_path):
    """单分片上传（音频 < 10M 时 slice_num=1，无需真正切片）。"""
    ts = int(time.time())
    boundary = "----LFASRBoundary%d" % ts
    file_name = os.path.basename(upload_path)
    fields = [
        ("app_id", app_id),
        ("ts", str(ts)),
        ("signa", sign(app_id, secret_key, ts)),
        ("task_id", task_id),
        ("slice_id", "aaaaaaaaaa"),        # 单分片的固定序号
    ]
    with open(upload_path, "rb") as f:
        audio_bytes = f.read()
    parts = []
    for k, v in fields:
        parts.append("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n"
                     % (boundary, k, v))
    parts.append("--%s\r\nContent-Disposition: form-data; name=\"content\"; filename=\"%s\"\r\n"
                 "Content-Type: application/octet-stream\r\n\r\n" % (boundary, file_name))
    payload = ("".join(parts)).encode() + audio_bytes + ("\r\n--%s--\r\n" % boundary).encode()
    req = Request(BASE + "/upload", data=payload,
                  headers={"Content-Type": "multipart/form-data; boundary=%s" % boundary})
    with urlopen(req, timeout=300) as r:
        resp = json.loads(r.read().decode())
    if resp.get("ok") != 0:
        raise RuntimeError("upload 失败（err_no=%s）：%s"
                           % (resp.get("err_no"), resp.get("failed")))


def transcribe_audio(src):
    """转写任意音频，返回 (整段文本, 句子列表, 任务号)。

    句子列表：list[{"文本", "开始ms", "结束ms", "说话人"}]。
    不写文件、不落盘，供其他脚本直接复用（10-30 抽出来给 compare_long
    转写「模仿」音频做单词级纠音用）。
    """
    if not src or not os.path.exists(src):
        raise RuntimeError("找不到音频文件" + ("：" + str(src) if src else ""))

    app_id, secret_key = load_keys()
    upload_path = prep_audio(src)

    try:
        file_size = os.path.getsize(upload_path)
        resp = api_call(app_id, secret_key, "prepare", {
            "file_len": str(file_size),
            "file_name": os.path.basename(upload_path),
            "slice_num": "1",                 # <10M 单分片
            "lfasr_type": "0",                # 标准版
            "language": "en",                 # 英文（跟读素材）
        })
        task_id = resp["data"]

        upload_whole(app_id, secret_key, task_id, upload_path)
        api_call(app_id, secret_key, "merge", {"task_id": task_id})

        for attempt in range(1, POLL_MAX_TIMES + 1):
            time.sleep(POLL_INTERVAL_S)
            resp = api_call(app_id, secret_key, "getProgress", {"task_id": task_id})
            prog = json.loads(resp["data"]) if isinstance(resp.get("data"), str) else resp.get("data", {})
            status = prog.get("status")
            if status == 9:
                break
            if status in (4, -1):
                raise RuntimeError("转写失败：%s" % prog)
        else:
            raise RuntimeError("等待超时（%d 秒），任务号 %s"
                               % (POLL_INTERVAL_S * POLL_MAX_TIMES, task_id))

        resp = api_call(app_id, secret_key, "getResult", {"task_id": task_id})
        data = json.loads(resp["data"]) if isinstance(resp.get("data"), str) else resp["data"]
        sentences = [{"文本": seg.get("onebest", ""),
                      "开始ms": int(seg.get("bg", 0)),
                      "结束ms": int(seg.get("ed", 0)),
                      "说话人": seg.get("speaker", "0")}
                     for seg in data if seg.get("onebest")]
        full_text = " ".join(s["文本"] for s in sentences).strip()
        if not full_text:
            raise RuntimeError("转写结果为空（音频里可能没有可识别的语音）")
        return full_text, sentences, task_id
    finally:
        # 清理临时 16k 文件：无论成功/失败/异常都要删（原来只在成功后删，
        # 中途异常会残留 .16k.wav，见 11-02 目录整理）。
        if upload_path != src:
            try:
                os.remove(upload_path)
            except Exception:
                pass


def main():
    log_setup("讯飞转写")
    out("=" * 62)
    out("   讯飞录音文件转写（LFASR）")
    out("=" * 62)

    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if args:
        src = args[0]
    else:
        from audio_paths import find_audio
        src = find_audio("原音频", SRC_DIR)
    if not src or not os.path.exists(src):
        out("X 找不到音频文件" + ("：" + src if src else "：请把音频放进 " + SRC_DIR))
        log_done("（缺音频，提前退出）")
        return

    log_step("讯飞转写")
    full_text, sentences, task_id = transcribe_audio(src)

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(full_text + "\n")
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"文本": full_text, "句子": sentences,
                   "任务号": task_id, "来源音频": os.path.basename(src)},
                  f, ensure_ascii=False, indent=2)

    out("")
    out("转写完成：%s" % os.path.basename(src))
    out("识别文本：%s" % (full_text[:90] + ("..." if len(full_text) > 90 else "")))
    out("句段数  ：%d" % len(sentences))
    out("")
    out("文本（人看）：" + OUT_TXT)
    out("数据（机器）：" + OUT_JSON)
    log_done("讯飞转写")


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
