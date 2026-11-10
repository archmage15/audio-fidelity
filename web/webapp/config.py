"""Runtime settings read from environment variables (see .env.example)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


def _int(name: str, default: int) -> int:
    value = os.environ.get(name, '').strip()
    return int(value) if value else default


def _float(name: str, default: float) -> float:
    value = os.environ.get(name, '').strip()
    return float(value) if value else default


def _bool(name: str, default: bool) -> bool:
    value = os.environ.get(name, '').strip().lower()
    if not value:
        return default
    return value in {'1', 'true', 'yes', 'on'}


@dataclass(frozen=True)
class Settings:
    engine_root: Path
    engine_python: Path
    data_root: Path
    reference_cache_dir: Path
    redis_url: str
    public_origin: str
    cookie_secure: bool
    session_days: int
    max_upload_bytes: int
    min_audio_seconds: float
    max_audio_seconds: float
    max_reference_seconds: float
    max_active_per_user: int
    max_active_global: int
    engine_timeout_seconds: int
    rq_job_timeout: int
    min_free_bytes: int
    retention_days: int
    idle_upload_hours: int
    hf_offline: bool
    asr_provider_default: str
    tingwu_config: Path
    engine_mode: str
    engine_socket: Path
    engine_server_max_rss_mb: int
    queue_name: str = 'audio'
    transcode_version: str = 'ffmpeg-pcm16-mono-16k-v1'
    engine_version: str = field(default='')

    # ---- derived paths -------------------------------------------------
    @property
    def db_path(self) -> Path:
        return self.data_root / 'app.sqlite3'

    @property
    def assets_dir(self) -> Path:
        return self.data_root / 'assets'

    @property
    def runs_dir(self) -> Path:
        return self.data_root / 'runs'

    @property
    def cache_logs_dir(self) -> Path:
        return self.data_root / 'cache-logs'

    @property
    def temp_dir(self) -> Path:
        return self.data_root / 'temp'

    @property
    def library_dir(self) -> Path:
        return self.data_root / 'reference-library'

    @property
    def asr_media_dir(self) -> Path:
        return self.data_root / 'asr-media'

    @property
    def xunfei_config(self) -> Path:
        return self.engine_root / '讯飞配置.json'

    @property
    def cache_version(self) -> str:
        return self.reference_cache_dir.name

    def ensure_dirs(self) -> None:
        for path in (self.data_root, self.assets_dir, self.runs_dir, self.cache_logs_dir,
                     self.temp_dir, self.temp_dir / 'uploads', self.temp_dir / 'engine',
                     self.temp_dir / 'sys', self.reference_cache_dir, self.library_dir,
                     self.asr_media_dir, self.engine_socket.parent):
            path.mkdir(parents=True, exist_ok=True)


def _engine_version(engine_root: Path) -> str:
    explicit = os.environ.get('ENGINE_VERSION', '').strip()
    if explicit:
        return explicit
    try:
        info = json.loads((engine_root / 'BUILD_INFO.json').read_text(encoding='utf-8'))
        return f"{info.get('name', 'engine')}@{info.get('release_date', 'unknown')}"
    except (OSError, ValueError):
        return 'engine@unknown'


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # The API container and the host worker read the same .env; paths are
    # identical inside and outside the container (same-path bind mounts).
    engine_root = Path(os.environ.get('ENGINE_ROOT', '/srv/audio-demo/engine')).resolve()
    data_root = Path(os.environ.get('DATA_ROOT', '/srv/audio-demo/data')).resolve()
    engine_python = Path(os.environ.get('ENGINE_PYTHON') or engine_root / '.venv-linux/bin/python')
    namespace = os.environ.get('CACHE_NAMESPACE', '').strip() or 'v1'
    cache_dir = Path(os.environ.get('REFERENCE_CACHE_DIR') or data_root / 'reference_cache' / namespace)
    redis_url = os.environ.get('REDIS_URL', '').strip()
    if not redis_url:
        password = os.environ.get('REDIS_PASSWORD', '')
        auth = f':{password}@' if password else ''
        redis_url = (f"redis://{auth}{os.environ.get('REDIS_HOST', '127.0.0.1')}:"
                     f"{os.environ.get('REDIS_PORT', '6379')}/0")
    return Settings(
        engine_root=engine_root,
        engine_python=engine_python,
        data_root=data_root,
        reference_cache_dir=cache_dir.resolve(),
        redis_url=redis_url,
        public_origin=os.environ.get('PUBLIC_ORIGIN', 'http://localhost:8000').rstrip('/'),
        cookie_secure=_bool('COOKIE_SECURE', True),
        session_days=_int('SESSION_DAYS', 30),
        max_upload_bytes=_int('MAX_UPLOAD_BYTES', 20 * 1024 * 1024),
        min_audio_seconds=_float('MIN_AUDIO_SECONDS', 2.0),
        max_audio_seconds=_float('MAX_AUDIO_SECONDS', 60.0),
        max_reference_seconds=_float('MAX_REFERENCE_SECONDS', _float('MAX_AUDIO_SECONDS', 60.0)),
        max_active_per_user=_int('MAX_ACTIVE_PER_USER', 2),
        max_active_global=_int('MAX_ACTIVE_GLOBAL', 20),
        engine_timeout_seconds=_int('ENGINE_TIMEOUT_SECONDS', 840),
        rq_job_timeout=_int('RQ_JOB_TIMEOUT', 900),
        min_free_bytes=_int('MIN_FREE_BYTES', 2 * 1024 ** 3),
        retention_days=_int('RETENTION_DAYS', 7),
        idle_upload_hours=_int('IDLE_UPLOAD_HOURS', 24),
        hf_offline=_bool('HF_OFFLINE', True),
        asr_provider_default=os.environ.get('ASR_PROVIDER', '').strip().lower() or 'xunfei',
        engine_mode=os.environ.get('ENGINE_MODE', '').strip().lower() or 'server',
        engine_socket=Path(os.environ.get('ENGINE_SOCKET') or data_root / 'run' / 'engine.sock'),
        engine_server_max_rss_mb=_int('ENGINE_SERVER_MAX_RSS_MB', 3200),
        tingwu_config=Path(os.environ.get('TINGWU_CONFIG') or engine_root / '听悟配置.json'),
        engine_version=_engine_version(engine_root),
    )
