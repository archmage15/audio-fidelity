"""FFprobe inspection and FFmpeg normalization to mono 16 kHz PCM-16 WAV.

Only local files are read: every FFmpeg/FFprobe call uses an argument list
(no shell), a protocol whitelist of ``file`` and a timeout. Container formats
outside the allow-list (playlists, concat scripts, images …) are rejected.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

ALLOWED_FORMATS = {'wav', 'mp3', 'flac', 'ogg', 'mov', 'mp4', 'm4a', 'matroska', 'webm', 'aac'}
ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.oga', '.opus', '.m4a', '.mp4',
                      '.webm', '.aac', '.bin'}
TARGET_RATE = 16000


class MediaError(Exception):
    def __init__(self, code: str, message: str, http_status: int = 422):
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status


@dataclass
class ProbeResult:
    format_names: set[str]
    codec: str
    sample_rate: int
    channels: int
    duration: float | None

    @property
    def is_target_wav(self) -> bool:
        return ('wav' in self.format_names and self.codec == 'pcm_s16le'
                and self.channels == 1 and self.sample_rate == TARGET_RATE)


def probe(path: Path, timeout: int = 30) -> ProbeResult:
    cmd = ['ffprobe', '-v', 'error', '-protocol_whitelist', 'file', '-print_format', 'json',
           '-show_format', '-show_streams', str(path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired as error:
        raise MediaError('decode_failed', '音频解析超时，请换一个文件重试。') from error
    if proc.returncode != 0:
        raise MediaError('unsupported_format', '无法识别音频格式，请上传 WAV/MP3/FLAC/OGG/M4A/WebM 音频。', 415)
    try:
        data = json.loads(proc.stdout or b'{}')
    except ValueError as error:
        raise MediaError('decode_failed', '音频信息读取失败。') from error
    fmt = data.get('format') or {}
    names = {name.strip() for name in str(fmt.get('format_name', '')).split(',') if name.strip()}
    if not names & ALLOWED_FORMATS:
        raise MediaError('unsupported_format', '不支持此文件格式，请上传常见音频文件。', 415)
    audio = [s for s in data.get('streams') or [] if s.get('codec_type') == 'audio']
    if not audio:
        raise MediaError('unsupported_format', '文件中没有音轨。', 415)
    stream = audio[0]
    duration = None
    for candidate in (stream.get('duration'), fmt.get('duration')):
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            continue
        if value > 0:
            duration = value
            break
    return ProbeResult(format_names=names, codec=str(stream.get('codec_name', '')),
                       sample_rate=int(stream.get('sample_rate') or 0),
                       channels=int(stream.get('channels') or 0), duration=duration)


def wav_info(path: Path) -> tuple[float, int, int, int]:
    """(duration seconds, sample rate, channels, sample width) from the WAV header."""
    try:
        with wave.open(str(path), 'rb') as stream:
            frames, rate = stream.getnframes(), stream.getframerate()
            return frames / rate, rate, stream.getnchannels(), stream.getsampwidth()
    except (wave.Error, EOFError, ZeroDivisionError) as error:
        raise MediaError('decode_failed', '转换后的音频无法读取。') from error


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_duration(duration: float | None, minimum: float, maximum: float) -> None:
    if duration is None:
        return
    if duration < minimum:
        raise MediaError('duration_out_of_range', f'音频太短（{duration:.1f} 秒），至少需要 {minimum:g} 秒。')
    if duration > maximum + 0.05:
        raise MediaError('duration_out_of_range',
                         f'音频太长（{duration:.1f} 秒），最长 {maximum:g} 秒，请剪短后重传。')


def normalize(source: Path, target: Path, *, timeout: int = 180) -> dict:
    """Create ``target`` (mono/16 kHz/PCM-16 WAV) from ``source`` atomically.

    Returns {'duration', 'sha256', 'transcoded'}; raises MediaError when the
    input cannot be decoded. No resampling tricks beyond the fixed format
    conversion: no speed change, trimming or loudness processing.
    """
    info = probe(source)
    partial = target.with_name(target.stem + '.part.wav')
    partial.unlink(missing_ok=True)
    transcoded = not info.is_target_wav
    try:
        if transcoded:
            cmd = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error',
                   '-protocol_whitelist', 'file,pipe', '-i', str(source),
                   '-map', '0:a:0', '-vn', '-ac', '1', '-ar', str(TARGET_RATE), '-c:a', 'pcm_s16le',
                   '-map_metadata', '-1', '-fflags', '+bitexact', '-flags:a', '+bitexact',
                   '-f', 'wav', str(partial)]
            try:
                proc = subprocess.run(cmd, capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL)
            except subprocess.TimeoutExpired as error:
                raise MediaError('decode_failed', '音频转码超时。') from error
            if proc.returncode != 0:
                raise MediaError('decode_failed', '音频解码失败，文件可能已损坏。')
        else:
            shutil.copyfile(source, partial)
        duration, rate, channels, width = wav_info(partial)
        if rate != TARGET_RATE or channels != 1 or width != 2 or duration <= 0:
            raise MediaError('decode_failed', '转换后的音频格式异常。')
        digest = file_sha256(partial)
        os.replace(partial, target)
        return {'duration': duration, 'sha256': digest, 'transcoded': transcoded}
    finally:
        partial.unlink(missing_ok=True)
