"""SQLite business records and id → controlled path mapping.

SQLite runs in WAL mode on local disk; the API and the single worker share it.
Writes that must be atomic (quota checks, idempotency, job claims) use
``BEGIN IMMEDIATE`` so only one writer evaluates the condition at a time.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from .config import get_settings

ACTIVE_STATUSES = ('pending_enqueue', 'queued', 'running')
FINAL_STATUSES = ('succeeded', 'failed', 'timed_out')

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    is_admin INTEGER NOT NULL DEFAULT 0,
    invite_hash TEXT NOT NULL UNIQUE,
    disabled INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS assets (
    id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL REFERENCES users(id),
    display_name TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'upload',
    source_mime TEXT,
    source_format TEXT,
    size_bytes INTEGER NOT NULL,
    duration_seconds REAL,
    sha256 TEXT,
    status TEXT NOT NULL,
    error TEXT,
    transcode_version TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS assets_owner ON assets(owner_id, created_at);
CREATE TABLE IF NOT EXISTS refs (
    id TEXT PRIMARY KEY,
    asset_id TEXT NOT NULL REFERENCES assets(id),
    owner_id TEXT NOT NULL REFERENCES users(id),
    is_public INTEGER NOT NULL DEFAULT 0,
    display_name TEXT NOT NULL,
    status TEXT NOT NULL,
    cache_version TEXT NOT NULL,
    cache_sha256 TEXT,
    transcript TEXT,
    prepare_job_id TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL REFERENCES users(id),
    type TEXT NOT NULL,
    reference_id TEXT REFERENCES refs(id),
    imitation_asset_id TEXT REFERENCES assets(id),
    status TEXT NOT NULL,
    stage TEXT,
    attempt_no INTEGER NOT NULL DEFAULT 1,
    attempt_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    queued_at TEXT,
    started_at TEXT,
    finished_at TEXT,
    error_code TEXT,
    error_message TEXT,
    error_detail TEXT,
    output_dir TEXT,
    result_json TEXT,
    files_purged INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_owner ON jobs(owner_id, created_at);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status);
CREATE TABLE IF NOT EXISTS inference_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs(id),
    attempt_id TEXT NOT NULL,
    task_type TEXT NOT NULL,
    reference_asset_id TEXT,
    imitation_asset_id TEXT,
    reference_duration_seconds REAL,
    imitation_duration_seconds REAL,
    inference_elapsed_seconds REAL,
    inference_started_at TEXT,
    inference_finished_at TEXT,
    status TEXT NOT NULL,
    cache_hit INTEGER,
    error_code TEXT,
    engine_version TEXT NOT NULL,
    UNIQUE (job_id, attempt_id)
);
CREATE INDEX IF NOT EXISTS inference_runs_started ON inference_runs(inference_started_at);
CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    updated_by TEXT
);
CREATE TABLE IF NOT EXISTS idempotency (
    owner_id TEXT NOT NULL,
    key TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    job_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (owner_id, key)
);
"""


# ---- helpers -------------------------------------------------------------

def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: datetime | None = None) -> str:
    return (value or utcnow()).isoformat(timespec='milliseconds').replace('+00:00', 'Z')


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def new_id(prefix: str) -> str:
    return f'{prefix}_{uuid.uuid4().hex[:20]}'


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def new_invite_code() -> str:
    alphabet = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'
    raw = ''.join(secrets.choice(alphabet) for _ in range(12))
    return '-'.join(raw[i:i + 4] for i in range(0, 12, 4))


def normalize_invite(code: str) -> str:
    return ''.join(ch for ch in code.upper() if ch.isalnum())


def invite_hash(code: str) -> str:
    return sha256_text('invite:' + normalize_invite(code))


def connect() -> sqlite3.Connection:
    settings = get_settings()
    conn = sqlite3.connect(settings.db_path, timeout=15, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA busy_timeout = 15000')
    conn.execute('PRAGMA foreign_keys = ON')
    conn.execute('PRAGMA synchronous = NORMAL')
    return conn


@contextmanager
def db() -> Iterator[sqlite3.Connection]:
    conn = connect()
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def transaction(conn: sqlite3.Connection, immediate: bool = True) -> Iterator[sqlite3.Connection]:
    conn.execute('BEGIN IMMEDIATE' if immediate else 'BEGIN')
    try:
        yield conn
    except BaseException:
        conn.execute('ROLLBACK')
        raise
    else:
        conn.execute('COMMIT')


def init_db() -> None:
    settings = get_settings()
    settings.ensure_dirs()
    with db() as conn:
        conn.execute('PRAGMA journal_mode = WAL')
        conn.executescript(SCHEMA)
        columns = {row[1] for row in conn.execute('PRAGMA table_info(inference_runs)')}
        if 'asr_provider' not in columns:
            conn.execute('ALTER TABLE inference_runs ADD COLUMN asr_provider TEXT')
        if 'engine_warm' not in columns:
            # 1 = persistent engine with models already loaded, 0 = loaded during this run,
            # NULL = one-shot engine process (loads everything every time).
            conn.execute('ALTER TABLE inference_runs ADD COLUMN engine_warm INTEGER')


def get_app_setting(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute('SELECT value FROM app_settings WHERE key = ?', (key,)).fetchone()
    return row[0] if row else None


def set_app_setting(conn: sqlite3.Connection, key: str, value: str, user_id: str | None) -> None:
    conn.execute('INSERT INTO app_settings (key, value, updated_at, updated_by) VALUES (?, ?, ?, ?) '
                 'ON CONFLICT (key) DO UPDATE SET value = excluded.value, '
                 'updated_at = excluded.updated_at, updated_by = excluded.updated_by',
                 (key, value, iso(), user_id))


def row_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


# ---- controlled paths ----------------------------------------------------

def asset_dir(asset_id: str) -> Path:
    if not asset_id.startswith('asset_') or not asset_id[6:].isalnum():
        raise ValueError('invalid asset id')
    return get_settings().assets_dir / asset_id


def asset_source_path(asset_id: str) -> Path:
    return asset_dir(asset_id) / 'source.bin'


def asset_wav_path(asset_id: str) -> Path:
    return asset_dir(asset_id) / 'audio.wav'


def run_output_dir(job_id: str, attempt_id: str) -> Path:
    return get_settings().runs_dir / job_id / attempt_id


# ---- users and sessions --------------------------------------------------

def create_user(conn: sqlite3.Connection, name: str, is_admin: bool) -> tuple[dict, str]:
    code = new_invite_code()
    user = {'id': new_id('user'), 'name': name, 'is_admin': int(is_admin),
            'invite_hash': invite_hash(code), 'disabled': 0, 'created_at': iso()}
    conn.execute('INSERT INTO users (id, name, is_admin, invite_hash, disabled, created_at) '
                 'VALUES (:id, :name, :is_admin, :invite_hash, :disabled, :created_at)', user)
    return user, code


def reset_invite(conn: sqlite3.Connection, user_id: str) -> str:
    code = new_invite_code()
    conn.execute('UPDATE users SET invite_hash = ? WHERE id = ?', (invite_hash(code), user_id))
    conn.execute('DELETE FROM sessions WHERE user_id = ?', (user_id,))
    return code


def create_session(conn: sqlite3.Connection, user_id: str, days: int) -> str:
    token = secrets.token_urlsafe(32)
    now = utcnow()
    conn.execute('INSERT INTO sessions (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)',
                 (sha256_text(token), user_id, iso(now), iso(now + timedelta(days=days))))
    return token


def session_user(conn: sqlite3.Connection, token: str) -> dict | None:
    row = conn.execute(
        'SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id '
        'WHERE s.token_hash = ? AND s.expires_at > ? AND u.disabled = 0',
        (sha256_text(token), iso())).fetchone()
    return row_dict(row)


# ---- jobs ------------------------------------------------------------------

def get_job(conn: sqlite3.Connection, job_id: str) -> dict | None:
    return row_dict(conn.execute('SELECT * FROM jobs WHERE id = ?', (job_id,)).fetchone())


def get_asset(conn: sqlite3.Connection, asset_id: str) -> dict | None:
    return row_dict(conn.execute('SELECT * FROM assets WHERE id = ?', (asset_id,)).fetchone())


def get_ref(conn: sqlite3.Connection, ref_id: str) -> dict | None:
    return row_dict(conn.execute('SELECT * FROM refs WHERE id = ?', (ref_id,)).fetchone())


def count_active(conn: sqlite3.Connection, owner_id: str | None = None) -> int:
    marks = ','.join('?' * len(ACTIVE_STATUSES))
    if owner_id is None:
        return conn.execute(f'SELECT COUNT(*) FROM jobs WHERE status IN ({marks})',
                            ACTIVE_STATUSES).fetchone()[0]
    return conn.execute(f'SELECT COUNT(*) FROM jobs WHERE owner_id = ? AND status IN ({marks})',
                        (owner_id, *ACTIVE_STATUSES)).fetchone()[0]


def new_attempt_id(attempt_no: int) -> str:
    return f'attempt-{attempt_no}-{uuid.uuid4().hex[:6]}'


def insert_job(conn: sqlite3.Connection, *, owner_id: str, job_type: str,
               reference_id: str | None, imitation_asset_id: str | None) -> dict:
    now = iso()
    job = {'id': new_id('job'), 'owner_id': owner_id, 'type': job_type,
           'reference_id': reference_id, 'imitation_asset_id': imitation_asset_id,
           'status': 'pending_enqueue', 'stage': None, 'attempt_no': 1,
           'attempt_id': new_attempt_id(1), 'created_at': now, 'updated_at': now}
    conn.execute(
        'INSERT INTO jobs (id, owner_id, type, reference_id, imitation_asset_id, status, stage, '
        'attempt_no, attempt_id, created_at, updated_at) VALUES (:id, :owner_id, :type, '
        ':reference_id, :imitation_asset_id, :status, :stage, :attempt_no, :attempt_id, '
        ':created_at, :updated_at)', job)
    return job


def mark_queued(conn: sqlite3.Connection, job_id: str, attempt_no: int) -> None:
    now = iso()
    conn.execute("UPDATE jobs SET status = 'queued', queued_at = COALESCE(queued_at, ?), updated_at = ? "
                 "WHERE id = ? AND attempt_no = ? AND status = 'pending_enqueue'",
                 (now, now, job_id, attempt_no))


def claim_job(job_id: str, attempt_no: int) -> dict | None:
    """Atomically take ownership of one attempt; None when someone else has it."""
    with db() as conn, transaction(conn):
        now = iso()
        cur = conn.execute(
            "UPDATE jobs SET status = 'running', stage = 'normalizing', started_at = ?, "
            "finished_at = NULL, queued_at = COALESCE(queued_at, ?), updated_at = ? "
            "WHERE id = ? AND attempt_no = ? AND status IN ('pending_enqueue', 'queued')",
            (now, now, now, job_id, attempt_no))
        if cur.rowcount != 1:
            return None
        return get_job(conn, job_id)


def set_stage(job: dict, stage: str) -> None:
    with db() as conn:
        conn.execute("UPDATE jobs SET stage = ?, updated_at = ? WHERE id = ? AND attempt_id = ? "
                     "AND status = 'running'", (stage, iso(), job['id'], job['attempt_id']))


def finish_job(job: dict, status: str, *, stage: str | None = None, result: dict | None = None,
               output_dir: str | None = None, error_code: str | None = None,
               error_message: str | None = None, error_detail: str | None = None) -> bool:
    """Write the final state only while this attempt still owns the job."""
    now = iso()
    with db() as conn:
        cur = conn.execute(
            'UPDATE jobs SET status = ?, stage = COALESCE(?, stage), finished_at = ?, updated_at = ?, '
            'result_json = ?, output_dir = COALESCE(?, output_dir), error_code = ?, '
            "error_message = ?, error_detail = ? WHERE id = ? AND attempt_id = ? AND status = 'running'",
            (status, stage, now, now, json.dumps(result, ensure_ascii=False) if result else None,
             output_dir, error_code, error_message, error_detail, job['id'], job['attempt_id']))
        return cur.rowcount == 1


# ---- inference metrics -----------------------------------------------------

def record_attempt_started(job: dict, task_type: str, metrics: dict, *, reference_asset_id: str,
                           imitation_asset_id: str | None, asr_provider: str | None = None) -> None:
    with db() as conn:
        conn.execute(
            'INSERT INTO inference_runs (job_id, attempt_id, task_type, reference_asset_id, '
            'imitation_asset_id, reference_duration_seconds, imitation_duration_seconds, '
            "inference_started_at, status, engine_version, asr_provider) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running', ?, ?) "
            'ON CONFLICT (job_id, attempt_id) DO UPDATE SET asr_provider = excluded.asr_provider, '
            'reference_duration_seconds = excluded.reference_duration_seconds, '
            'imitation_duration_seconds = excluded.imitation_duration_seconds, '
            "inference_started_at = excluded.inference_started_at, status = 'running'",
            (job['id'], job['attempt_id'], task_type, reference_asset_id, imitation_asset_id,
             metrics.get('reference_duration_seconds'), metrics.get('imitation_duration_seconds'),
             iso(), get_settings().engine_version, asr_provider))


def finish_attempt(job: dict, status: str, elapsed: float | None, *, started_at: str | None = None,
                   finished_at: str | None = None, cache_hit: bool | None = None,
                   error_code: str | None = None, engine_warm: bool | None = None) -> None:
    with db() as conn:
        conn.execute(
            'UPDATE inference_runs SET status = ?, inference_elapsed_seconds = ?, '
            'inference_started_at = COALESCE(?, inference_started_at), inference_finished_at = ?, '
            'cache_hit = ?, error_code = ?, engine_warm = ? WHERE job_id = ? AND attempt_id = ?',
            (status, None if elapsed is None else max(0.0, float(elapsed)), started_at,
             finished_at or iso(), None if cache_hit is None else int(cache_hit), error_code,
             None if engine_warm is None else int(engine_warm), job['id'], job['attempt_id']))


def attempt_metrics(conn: sqlite3.Connection, job_id: str, attempt_id: str) -> dict | None:
    row = conn.execute(
        'SELECT reference_duration_seconds, imitation_duration_seconds, inference_elapsed_seconds, '
        'status, cache_hit FROM inference_runs WHERE job_id = ? AND attempt_id = ?',
        (job_id, attempt_id)).fetchone()
    return row_dict(row)
