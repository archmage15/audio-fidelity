"""Transcription provider selection (讯飞 / 通义听悟) shared by API and worker.

The engine itself only knows 讯飞. When 通义听悟 is selected the worker puts
``asr_hook/`` on the engine's PYTHONPATH; its sitecustomize reroutes both
engine transcription call sites to ``afd_tingwu`` (see that module).
Config files are only checked for presence/fields here; values never leave.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path

from . import storage
from .config import get_settings

PROVIDERS = {'xunfei': '讯飞', 'tingwu': '通义听悟'}
REQUIRED_FIELDS = {'xunfei': ('app_id', 'secret_key'),
                   'tingwu': ('access_key_id', 'access_key_secret', 'app_key')}
SETTING_KEY = 'asr_provider'
HOOK_DIR = Path(__file__).resolve().parent / 'asr_hook'
MEDIA_NAME = re.compile(r'^[0-9a-f]{32}\.(wav|mp3|m4a|flac|ogg|opus|aac|webm|mp4)$')
MEDIA_MAX_AGE_SECONDS = 3600


def config_path(provider: str) -> Path:
    settings = get_settings()
    return settings.tingwu_config if provider == 'tingwu' else settings.xunfei_config


def check_config(provider: str) -> dict:
    """{'configured': bool, 'problem': str|None} without exposing any value."""
    path = config_path(provider)
    if not path.is_file():
        return {'configured': False, 'problem': f'缺少配置文件 {path.name}'}
    try:
        raw = json.loads(path.read_text(encoding='utf-8-sig'))
    except PermissionError:
        return {'configured': False, 'problem': f'{path.name} 没有读取权限'}
    except (OSError, ValueError):
        return {'configured': False, 'problem': f'{path.name} 不是有效的 JSON'}
    if not isinstance(raw, dict):
        return {'configured': False, 'problem': f'{path.name} 不是 JSON 对象'}
    missing = [k for k in REQUIRED_FIELDS[provider] if not str(raw.get(k) or '').strip()]
    if missing:
        return {'configured': False, 'problem': f"{path.name} 缺少字段：{', '.join(missing)}"}
    return {'configured': True, 'problem': None}


def default_provider() -> str:
    value = get_settings().asr_provider_default
    return value if value in PROVIDERS else 'xunfei'


def current_provider(conn: sqlite3.Connection | None = None) -> str:
    if conn is None:
        with storage.db() as own:
            return current_provider(own)
    value = storage.get_app_setting(conn, SETTING_KEY)
    return value if value in PROVIDERS else default_provider()


def status(conn: sqlite3.Connection) -> dict:
    row = conn.execute('SELECT s.updated_at, u.name FROM app_settings s LEFT JOIN users u '
                       'ON u.id = s.updated_by WHERE s.key = ?', (SETTING_KEY,)).fetchone()
    return {
        'asr_provider': current_provider(conn),
        'updated_at': row[0] if row else None,
        'updated_by': row[1] if row else None,
        'providers': [{'id': key, 'name': name, 'config_file': config_path(key).name, **check_config(key)}
                      for key, name in PROVIDERS.items()],
    }


def engine_env(provider: str) -> dict[str, str]:
    settings = get_settings()
    env = {'AFD_ASR_PROVIDER': provider}
    if provider == 'tingwu':
        env.update({
            'PYTHONPATH': str(HOOK_DIR),
            'AFD_TINGWU_CONFIG': str(settings.tingwu_config),
            'AFD_ASR_MEDIA_DIR': str(settings.asr_media_dir),
            'AFD_PUBLIC_ORIGIN': settings.public_origin,
        })
    return env


def media_file(name: str) -> Path | None:
    """Temporary audio handed to 通义听悟 by URL; None unless valid and fresh."""
    if not MEDIA_NAME.match(name):
        return None
    path = get_settings().asr_media_dir / name
    try:
        if time.time() - path.stat().st_mtime > MEDIA_MAX_AGE_SECONDS:
            return None
    except OSError:
        return None
    return path


def cleanup_media() -> int:
    removed = 0
    directory = get_settings().asr_media_dir
    if not directory.is_dir():
        return 0
    cutoff = time.time() - MEDIA_MAX_AGE_SECONDS
    for path in directory.iterdir():
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            pass
    return removed
