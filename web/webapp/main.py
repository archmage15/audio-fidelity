"""FastAPI routes: sessions, uploads, references, evaluations, jobs, admin."""
from __future__ import annotations

import csv
import io
import json
import logging
import os
import shutil
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from . import asr, maintenance, queueing, storage
from .config import get_settings
from .engine_adapter import CHART_FILES, REPORT_FILES, sanitize_text
from .media import ALLOWED_EXTENSIONS, MediaError, check_duration, probe

log = logging.getLogger('audio.api')
STATIC_DIR = Path(__file__).resolve().parent / 'static'
COOKIE = 'afd_session'
UNSAFE = {'POST', 'PUT', 'PATCH', 'DELETE'}


class APIError(Exception):
    def __init__(self, status: int, code: str, message: str, **extra: Any):
        super().__init__(message)
        self.status, self.code, self.message, self.extra = status, code, message, extra


# ---- lifespan: schema + maintenance loop --------------------------------

def _maintenance_loop(stop: threading.Event) -> None:
    while not stop.wait(60):
        maintenance.run_once()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    storage.init_db()
    stop = threading.Event()
    thread = threading.Thread(target=_maintenance_loop, args=(stop,), name='maintenance', daemon=True)
    if os.environ.get('DISABLE_MAINTENANCE') != '1':
        thread.start()
    yield
    stop.set()


app = FastAPI(title='Audio fidelity demo', lifespan=lifespan, docs_url=None, redoc_url=None,
              openapi_url=None)


@app.exception_handler(APIError)
async def _api_error(_request: Request, error: APIError):
    return JSONResponse({'error': {'code': error.code, 'message': error.message, **error.extra}},
                        status_code=error.status)


@app.middleware('http')
async def guard(request: Request, call_next):
    settings = get_settings()
    path = request.url.path
    if request.method in UNSAFE and path.startswith('/api/'):
        origin = request.headers.get('origin')
        if origin and origin.rstrip('/') != settings.public_origin:
            return JSONResponse({'error': {'code': 'bad_origin', 'message': '请求来源不被允许。'}}, 403)
        if request.headers.get('x-requested-with') != 'fetch':
            return JSONResponse({'error': {'code': 'csrf', 'message': '缺少请求校验头。'}}, 403)
    if request.method == 'POST' and path == '/api/assets':
        length = request.headers.get('content-length')
        if length and length.isdigit() and int(length) > settings.max_upload_bytes + 256 * 1024:
            return JSONResponse({'error': {'code': 'file_too_large', 'message': _too_large_text()}}, 413)
    response = await call_next(request)
    response.headers.setdefault('X-Content-Type-Options', 'nosniff')
    response.headers.setdefault('Referrer-Policy', 'same-origin')
    if path.startswith('/api/'):
        response.headers.setdefault('Cache-Control', 'no-store')
    return response


def _too_large_text() -> str:
    return f'文件超过 {get_settings().max_upload_bytes // (1024 * 1024)} MiB 上限。'


# ---- sessions --------------------------------------------------------------

_login_attempts: dict[str, deque] = defaultdict(deque)
_login_lock = threading.Lock()


def client_ip(request: Request) -> str:
    return request.headers.get('x-real-ip') or (request.client.host if request.client else '?')


def current_user(request: Request) -> dict:
    token = request.cookies.get(COOKIE)
    if token:
        with storage.db() as conn:
            user = storage.session_user(conn, token)
        if user:
            return user
    raise APIError(401, 'unauthenticated', '请先使用邀请码登录。')


def admin_user(user: dict = Depends(current_user)) -> dict:
    if not user['is_admin']:
        raise APIError(403, 'forbidden', '仅管理员可用。')
    return user


class SessionIn(BaseModel):
    invite_code: str = Field(min_length=4, max_length=64)


@app.post('/api/session')
def login(body: SessionIn, request: Request):
    ip = client_ip(request)
    now = time.monotonic()
    with _login_lock:
        attempts = _login_attempts[ip]
        while attempts and now - attempts[0] > 600:
            attempts.popleft()
        if len(attempts) >= 10:
            raise APIError(429, 'too_many_attempts', '尝试次数过多，请 10 分钟后再试。')
        attempts.append(now)
    settings = get_settings()
    with storage.db() as conn:
        row = conn.execute('SELECT * FROM users WHERE invite_hash = ? AND disabled = 0',
                           (storage.invite_hash(body.invite_code),)).fetchone()
        if not row:
            raise APIError(401, 'invalid_invite', '邀请码无效。')
        token = storage.create_session(conn, row['id'], settings.session_days)
    response = JSONResponse({'user': user_view(dict(row))})
    response.set_cookie(COOKIE, token, max_age=settings.session_days * 86400, httponly=True,
                        secure=settings.cookie_secure, samesite='lax', path='/')
    return response


@app.delete('/api/session')
def logout(request: Request):
    token = request.cookies.get(COOKIE)
    if token:
        with storage.db() as conn:
            conn.execute('DELETE FROM sessions WHERE token_hash = ?', (storage.sha256_text(token),))
    response = JSONResponse({'ok': True})
    response.delete_cookie(COOKIE, path='/')
    return response


def user_view(user: dict) -> dict:
    settings = get_settings()
    return {'id': user['id'], 'name': user['name'], 'is_admin': bool(user['is_admin']),
            'limits': {'max_upload_bytes': settings.max_upload_bytes,
                       'min_audio_seconds': settings.min_audio_seconds,
                       'max_audio_seconds': settings.max_audio_seconds,
                       'max_reference_seconds': settings.max_reference_seconds,
                       'max_active_jobs': settings.max_active_per_user}}


@app.get('/api/me')
def me(user: dict = Depends(current_user)):
    return {'user': user_view(user)}


# ---- assets ------------------------------------------------------------------

def ensure_disk_space() -> None:
    settings = get_settings()
    if shutil.disk_usage(settings.data_root).free < settings.min_free_bytes:
        raise APIError(503, 'storage_low', '存储不足，请稍后重试。')


def asset_view(asset: dict) -> dict:
    return {'asset_id': asset['id'], 'display_name': asset['display_name'], 'kind': asset['kind'],
            'status': asset['status'], 'size_bytes': asset['size_bytes'],
            'duration_seconds': asset['duration_seconds'], 'created_at': asset['created_at'],
            'audio_url': f"/api/assets/{asset['id']}/audio"}


def _clean_display_name(name: str | None, fallback: str) -> str:
    name = Path(name or '').name.strip().replace('\x00', '')
    return (name or fallback)[:120]


@app.post('/api/assets', status_code=201)
async def upload_asset(file: UploadFile = File(...), kind: str = Form('upload'),
                       purpose: str = Form('imitation'), user: dict = Depends(current_user)):
    settings = get_settings()
    if kind not in ('upload', 'recording') or purpose not in ('imitation', 'reference'):
        raise APIError(422, 'invalid_field', '上传参数无效。')
    if purpose == 'reference' and not user['is_admin']:
        raise APIError(403, 'forbidden', '只有管理员可以上传原音频。')
    display = _clean_display_name(file.filename, 'recording' if kind == 'recording' else 'audio')
    suffix = Path(display).suffix.lower()
    mime = (file.content_type or '').split(';')[0].strip().lower()
    if suffix not in ALLOWED_EXTENSIONS and not mime.startswith('audio/'):
        raise APIError(415, 'unsupported_format', '不支持此文件类型，请上传 WAV/MP3/FLAC/OGG/M4A/WebM 音频。')
    ensure_disk_space()
    part = settings.temp_dir / 'uploads' / f'{uuid.uuid4().hex}.part'
    size = 0
    try:
        with part.open('wb') as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise APIError(413, 'file_too_large', _too_large_text())
                out.write(chunk)
        if size == 0:
            raise APIError(422, 'empty_file', '文件为空。')
        asset_id = storage.new_id('asset')
        directory = storage.asset_dir(asset_id)
        directory.mkdir(parents=True)
        source = storage.asset_source_path(asset_id)
        os.replace(part, source)
    finally:
        part.unlink(missing_ok=True)
        await file.close()
    try:
        info = await run_in_threadpool(probe, source)
        limit = settings.max_reference_seconds if purpose == 'reference' else settings.max_audio_seconds
        check_duration(info.duration, settings.min_audio_seconds, limit)
    except MediaError as error:
        shutil.rmtree(directory, ignore_errors=True)
        raise APIError(error.http_status, error.code, error.message)
    now = storage.iso()
    asset = {'id': asset_id, 'owner_id': user['id'], 'display_name': display, 'kind': kind,
             'source_mime': mime or None, 'source_format': ','.join(sorted(info.format_names)),
             'size_bytes': size, 'duration_seconds': info.duration, 'status': 'uploaded',
             'created_at': now, 'updated_at': now}
    with storage.db() as conn:
        conn.execute('INSERT INTO assets (id, owner_id, display_name, kind, source_mime, source_format, '
                     'size_bytes, duration_seconds, status, created_at, updated_at) VALUES (:id, :owner_id, '
                     ':display_name, :kind, :source_mime, :source_format, :size_bytes, :duration_seconds, '
                     ':status, :created_at, :updated_at)', asset)
    return asset_view(asset)


def _can_read_asset(conn, asset: dict, user: dict) -> bool:
    if asset['owner_id'] == user['id'] or user['is_admin']:
        return True
    return conn.execute("SELECT 1 FROM refs WHERE asset_id = ? AND is_public = 1 AND status = 'ready'",
                        (asset['id'],)).fetchone() is not None


@app.get('/api/assets/{asset_id}')
def get_asset(asset_id: str, user: dict = Depends(current_user)):
    with storage.db() as conn:
        asset = storage.get_asset(conn, asset_id)
        if not asset or asset['status'] == 'deleted' or not _can_read_asset(conn, asset, user):
            raise APIError(404, 'not_found', '音频不存在。')
    return asset_view(asset)


_SOURCE_MIME = {'wav': 'audio/wav', 'mp3': 'audio/mpeg', 'flac': 'audio/flac', 'ogg': 'audio/ogg',
                'webm': 'audio/webm', 'matroska': 'audio/webm', 'mp4': 'audio/mp4', 'mov': 'audio/mp4',
                'aac': 'audio/aac'}


@app.get('/api/assets/{asset_id}/audio')
def asset_audio(asset_id: str, user: dict = Depends(current_user)):
    with storage.db() as conn:
        asset = storage.get_asset(conn, asset_id) if asset_id.startswith('asset_') else None
        if not asset or asset['status'] == 'deleted' or not _can_read_asset(conn, asset, user):
            raise APIError(404, 'not_found', '音频不存在。')
    wav = storage.asset_wav_path(asset_id)
    if wav.is_file():
        return FileResponse(wav, media_type='audio/wav')
    source = storage.asset_source_path(asset_id)
    if not source.is_file():
        raise APIError(404, 'not_found', '音频文件已清理。')
    formats = (asset['source_format'] or '').split(',')
    mime = next((_SOURCE_MIME[f] for f in formats if f in _SOURCE_MIME), 'application/octet-stream')
    return FileResponse(source, media_type=mime)


@app.delete('/api/assets/{asset_id}')
def delete_asset(asset_id: str, user: dict = Depends(current_user)):
    with storage.db() as conn, storage.transaction(conn):
        asset = storage.get_asset(conn, asset_id)
        if not asset or asset['status'] == 'deleted' or (asset['owner_id'] != user['id'] and not user['is_admin']):
            raise APIError(404, 'not_found', '音频不存在。')
        if conn.execute('SELECT 1 FROM refs WHERE asset_id = ?', (asset_id,)).fetchone():
            raise APIError(409, 'asset_in_use', '该音频是原音频素材，不能删除。')
        marks = ','.join('?' * len(storage.ACTIVE_STATUSES))
        if conn.execute(f'SELECT 1 FROM jobs WHERE imitation_asset_id = ? AND status IN ({marks})',
                        (asset_id, *storage.ACTIVE_STATUSES)).fetchone():
            raise APIError(409, 'asset_in_use', '该音频正在评测中，完成后才能删除。')
        conn.execute("UPDATE assets SET status = 'deleted', updated_at = ? WHERE id = ?", (storage.iso(), asset_id))
    shutil.rmtree(storage.asset_dir(asset_id), ignore_errors=True)
    return {'ok': True}


# ---- references ----------------------------------------------------------------

def ref_view(conn, ref: dict, user: dict) -> dict:
    asset = storage.get_asset(conn, ref['asset_id']) or {}
    view = {'reference_id': ref['id'], 'display_name': ref['display_name'], 'status': ref['status'],
            'is_public': bool(ref['is_public']), 'transcript': ref['transcript'],
            'duration_seconds': asset.get('duration_seconds'), 'created_at': ref['created_at'],
            'audio_url': f"/api/assets/{ref['asset_id']}/audio", 'job_id': ref['prepare_job_id'],
            'error': ref['error'] if ref['status'] == 'failed' else None}
    if user['is_admin']:
        view['cache_version'] = ref['cache_version']
    return view


def _visible_ref(conn, ref_id: str, user: dict) -> dict:
    ref = storage.get_ref(conn, ref_id)
    if not ref or ref['status'] == 'archived' and not user['is_admin']:
        raise APIError(404, 'not_found', '原音频不存在。')
    if not (ref['is_public'] or ref['owner_id'] == user['id'] or user['is_admin']):
        raise APIError(404, 'not_found', '原音频不存在。')
    return ref


@app.get('/api/references')
def list_references(include_pending: bool = False, user: dict = Depends(current_user)):
    with storage.db() as conn:
        if user['is_admin'] and include_pending:
            rows = conn.execute("SELECT * FROM refs WHERE status != 'archived' ORDER BY created_at DESC")
        else:
            rows = conn.execute("SELECT * FROM refs WHERE status = 'ready' AND (is_public = 1 OR owner_id = ?) "
                                'ORDER BY display_name COLLATE NOCASE', (user['id'],))
        return {'items': [ref_view(conn, dict(r), user) for r in rows.fetchall()]}


@app.get('/api/references/{ref_id}')
def get_reference(ref_id: str, user: dict = Depends(current_user)):
    with storage.db() as conn:
        return ref_view(conn, _visible_ref(conn, ref_id, user), user)


class ReferenceIn(BaseModel):
    asset_id: str
    display_name: str = Field(min_length=1, max_length=120)
    is_public: bool = True


class ReferencePatch(BaseModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=120)
    is_public: bool | None = None
    archived: bool | None = None


def _request_hash(payload: dict) -> str:
    return storage.sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def _idempotent_job(conn, user: dict, key: str | None, request_hash: str) -> dict | None:
    if not key:
        return None
    if len(key) > 100:
        raise APIError(422, 'invalid_idempotency_key', 'Idempotency-Key 过长。')
    row = conn.execute('SELECT * FROM idempotency WHERE owner_id = ? AND key = ?', (user['id'], key)).fetchone()
    if not row:
        return None
    if row['request_hash'] != request_hash:
        raise APIError(409, 'idempotency_conflict', '同一个请求键对应了不同的参数。')
    return storage.get_job(conn, row['job_id'])


def _check_quota(conn, user: dict) -> None:
    settings = get_settings()
    # Per-user limit applies to testers; admins (batch reference preparation) only hit the global cap.
    if not user['is_admin'] and storage.count_active(conn, user['id']) >= settings.max_active_per_user:
        raise APIError(429, 'user_quota', f'你已有 {settings.max_active_per_user} 个任务在处理，请等待完成后再提交。')
    if storage.count_active(conn) >= settings.max_active_global:
        raise APIError(503, 'queue_full', '当前排队人数较多，请稍后再试。')


def _remember_key(conn, user: dict, key: str | None, request_hash: str, job_id: str) -> None:
    if key:
        conn.execute('INSERT INTO idempotency (owner_id, key, request_hash, job_id, created_at) '
                     'VALUES (?, ?, ?, ?, ?)', (user['id'], key, request_hash, job_id, storage.iso()))


def _enqueue_response(job: dict, extra: dict | None = None) -> JSONResponse:
    if not queueing.enqueue(job['id'], job['attempt_no']):
        raise APIError(503, 'queue_unavailable', '排队服务暂不可用，任务已记录，将自动补投；也可稍后重试。',
                       job_id=job['id'])
    return JSONResponse({'job_id': job['id'], 'status': 'queued', 'poll_after_ms': 2000, **(extra or {})},
                        status_code=202)


@app.post('/api/references', status_code=202)
def create_reference(body: ReferenceIn, request: Request, user: dict = Depends(admin_user)):
    settings = get_settings()
    key = request.headers.get('idempotency-key')
    request_hash = _request_hash({'op': 'reference', **body.model_dump()})
    ensure_disk_space()
    with storage.db() as conn, storage.transaction(conn):
        existing = _idempotent_job(conn, user, key, request_hash)
        if existing:
            return JSONResponse({'job_id': existing['id'], 'reference_id': existing['reference_id'],
                                 'status': existing['status'], 'poll_after_ms': 2000}, status_code=202)
        asset = storage.get_asset(conn, body.asset_id)
        if not asset or asset['owner_id'] != user['id'] or asset['status'] in ('deleted', 'invalid'):
            raise APIError(404, 'not_found', '音频不存在或不可用。')
        _check_quota(conn, user)
        now = storage.iso()
        ref_id = storage.new_id('ref')
        conn.execute('INSERT INTO refs (id, asset_id, owner_id, is_public, display_name, status, cache_version, '
                     "created_at, updated_at) VALUES (?, ?, ?, ?, ?, 'preparing', ?, ?, ?)",
                     (ref_id, asset['id'], user['id'], int(body.is_public), body.display_name.strip(),
                      settings.cache_version, now, now))
        job = storage.insert_job(conn, owner_id=user['id'], job_type='prepare_reference',
                                 reference_id=ref_id, imitation_asset_id=None)
        conn.execute('UPDATE refs SET prepare_job_id = ? WHERE id = ?', (job['id'], ref_id))
        _remember_key(conn, user, key, request_hash, job['id'])
    return _enqueue_response(job, {'reference_id': ref_id})


@app.patch('/api/references/{ref_id}')
def update_reference(ref_id: str, body: ReferencePatch, user: dict = Depends(admin_user)):
    with storage.db() as conn:
        ref = storage.get_ref(conn, ref_id)
        if not ref:
            raise APIError(404, 'not_found', '原音频不存在。')
        if body.display_name is not None:
            conn.execute('UPDATE refs SET display_name = ? WHERE id = ?', (body.display_name.strip(), ref_id))
        if body.is_public is not None:
            conn.execute('UPDATE refs SET is_public = ? WHERE id = ?', (int(body.is_public), ref_id))
        if body.archived is True:
            conn.execute("UPDATE refs SET status = 'archived' WHERE id = ? AND status != 'preparing'", (ref_id,))
        conn.execute('UPDATE refs SET updated_at = ? WHERE id = ?', (storage.iso(), ref_id))
        return ref_view(conn, storage.get_ref(conn, ref_id), user)


# ---- evaluations & jobs -----------------------------------------------------------

class EvaluationIn(BaseModel):
    reference_id: str
    imitation_asset_id: str


@app.post('/api/evaluations', status_code=202)
def create_evaluation(body: EvaluationIn, request: Request, user: dict = Depends(current_user)):
    key = request.headers.get('idempotency-key')
    request_hash = _request_hash({'op': 'evaluate', **body.model_dump()})
    ensure_disk_space()
    with storage.db() as conn, storage.transaction(conn):
        existing = _idempotent_job(conn, user, key, request_hash)
        if existing:
            return JSONResponse({'job_id': existing['id'], 'status': existing['status'], 'poll_after_ms': 2000},
                                status_code=202)
        ref = _visible_ref(conn, body.reference_id, user)
        if ref['status'] != 'ready':
            raise APIError(409, 'reference_not_ready', '原音频尚未准备好。')
        asset = storage.get_asset(conn, body.imitation_asset_id)
        if not asset or asset['owner_id'] != user['id'] or asset['status'] == 'deleted':
            raise APIError(404, 'not_found', '模仿音频不存在，请重新上传。')
        if asset['status'] == 'invalid':
            raise APIError(422, 'decode_failed', '模仿音频无法解码，请重新上传。')
        _check_quota(conn, user)
        job = storage.insert_job(conn, owner_id=user['id'], job_type='evaluate', reference_id=ref['id'],
                                 imitation_asset_id=asset['id'])
        _remember_key(conn, user, key, request_hash, job['id'])
    return _enqueue_response(job)


def _seconds_between(start: str | None, end: str | None) -> float | None:
    a, b = storage.parse_iso(start), storage.parse_iso(end) if end else storage.utcnow()
    if not a or not b:
        return None
    return round(max(0.0, (b - a).total_seconds()), 3)


def job_view(conn, job: dict, user: dict) -> dict:
    ref = storage.get_ref(conn, job['reference_id']) if job['reference_id'] else None
    imitation = storage.get_asset(conn, job['imitation_asset_id']) if job['imitation_asset_id'] else None
    metrics = storage.attempt_metrics(conn, job['id'], job['attempt_id']) or {}
    view: dict[str, Any] = {
        'job_id': job['id'], 'type': job['type'], 'status': job['status'], 'stage': job['stage'],
        'attempt': job['attempt_no'], 'created_at': job['created_at'], 'queued_at': job['queued_at'],
        'started_at': job['started_at'], 'finished_at': job['finished_at'],
        'queue_wait_seconds': _seconds_between(job['queued_at'] or job['created_at'], job['started_at'])
        if job['started_at'] else None,
        'elapsed_seconds': _seconds_between(job['started_at'], job['finished_at']) if job['started_at'] else None,
        'reference': {'reference_id': ref['id'], 'display_name': ref['display_name'],
                      'audio_url': f"/api/assets/{ref['asset_id']}/audio"} if ref else None,
        'imitation': {'asset_id': imitation['id'], 'display_name': imitation['display_name'],
                      'audio_url': f"/api/assets/{imitation['id']}/audio" if imitation['status'] != 'deleted' else None}
        if imitation else None,
        'metrics': {
            'reference_duration_seconds': _round3(metrics.get('reference_duration_seconds')),
            'imitation_duration_seconds': _round3(metrics.get('imitation_duration_seconds')),
            'inference_elapsed_seconds': _round3(metrics.get('inference_elapsed_seconds')),
            **({'cache_hit': bool(metrics['cache_hit'])} if metrics.get('cache_hit') is not None else {}),
        },
        'result': None, 'error': None,
    }
    if job['status'] in storage.ACTIVE_STATUSES:
        view['poll_after_ms'] = 2000
    if job['status'] in ('pending_enqueue', 'queued'):
        anchor = job['queued_at'] or job['created_at']
        view['queue_ahead'] = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE status = 'running' OR (status IN ('pending_enqueue','queued') "
            'AND COALESCE(queued_at, created_at) < ?)', (anchor,)).fetchone()[0]
    if job['status'] == 'succeeded' and job['result_json']:
        result = json.loads(job['result_json'])
        result.pop('metrics', None)
        if job['type'] == 'evaluate':
            available = not job['files_purged']
            result['report_url'] = f"/api/jobs/{job['id']}/report" if available else None
            result['report_urls'] = {name: f"/api/jobs/{job['id']}/report?kind={name}"
                                     for name in result.get('reports', [])} if available else {}
            result['chart_urls'] = {name: f"/api/jobs/{job['id']}/charts/{name}"
                                    for name in result.get('charts', [])} if available else {}
            result['files_available'] = available
        view['result'] = result
    if job['status'] in ('failed', 'timed_out'):
        view['error'] = {'code': job['error_code'], 'message': job['error_message']}
        if user['is_admin']:
            view['error']['detail'] = job['error_detail']
    return view


def _round3(value):
    return None if value is None else round(float(value), 3)


def _own_job(conn, job_id: str, user: dict) -> dict:
    job = storage.get_job(conn, job_id)
    if not job or (job['owner_id'] != user['id'] and not user['is_admin']):
        raise APIError(404, 'not_found', '任务不存在。')
    return job


@app.get('/api/jobs/{job_id}')
def get_job(job_id: str, user: dict = Depends(current_user)):
    with storage.db() as conn:
        return job_view(conn, _own_job(conn, job_id, user), user)


@app.get('/api/jobs')
def list_jobs(limit: int = 20, before: str | None = None, type: str | None = None,
              user: dict = Depends(current_user)):
    limit = max(1, min(limit, 50))
    clauses, params = ['owner_id = ?'], [user['id']]
    if before:
        clauses.append('created_at < ?')
        params.append(before)
    if type in ('evaluate', 'prepare_reference'):
        clauses.append('type = ?')
        params.append(type)
    with storage.db() as conn:
        rows = conn.execute(f"SELECT * FROM jobs WHERE {' AND '.join(clauses)} ORDER BY created_at DESC LIMIT ?",
                            (*params, limit + 1)).fetchall()
        items = [job_view(conn, dict(r), user) for r in rows[:limit]]
    return {'items': items, 'next_before': items[-1]['created_at'] if len(rows) > limit else None}


def _job_output(conn, job_id: str, user: dict) -> tuple[dict, Path]:
    job = _own_job(conn, job_id, user)
    if job['type'] != 'evaluate' or job['status'] != 'succeeded' or not job['output_dir']:
        raise APIError(404, 'not_found', '报告不存在。')
    if job['files_purged']:
        raise APIError(410, 'expired', '报告已超过保留期限被清理。')
    output = Path(job['output_dir'])
    if storage.run_output_dir(job['id'], job['attempt_id']) != output:
        raise APIError(404, 'not_found', '报告不存在。')
    return job, output


@app.get('/api/jobs/{job_id}/report')
def job_report(job_id: str, kind: str = 'summary', download: bool = False, user: dict = Depends(current_user)):
    settings = get_settings()
    if kind not in REPORT_FILES:
        raise APIError(404, 'not_found', '报告不存在。')
    with storage.db() as conn:
        job, output = _job_output(conn, job_id, user)
        ref = storage.get_ref(conn, job['reference_id'])
    path = output / REPORT_FILES[kind]
    if not path.is_file():
        raise APIError(404, 'not_found', '报告不存在。')
    replacements = {str(storage.asset_wav_path(ref['asset_id'])): '原音频.wav',
                    str(storage.asset_wav_path(job['imitation_asset_id'])): '模仿音频.wav',
                    str(output): '〈本次报告目录〉', str(settings.reference_cache_dir): '〈缓存库〉',
                    str(settings.engine_root): '〈评测程序〉', str(settings.data_root): '〈数据目录〉'}
    text = sanitize_text(path.read_text(encoding='utf-8', errors='replace'), replacements)
    headers = {}
    if download:
        if kind == 'json':
            name = f"结构化评测报告-{job_id}.json"
            headers['Content-Disposition'] = f"attachment; filename=\"report-{job_id}.json\"; filename*=UTF-8''{quote(name)}"
        else:
            name = f"{'还原度综合报告' if kind == 'summary' else '音素对比报告'}-{job_id}.txt"
            headers['Content-Disposition'] = f"attachment; filename=\"report-{job_id}.txt\"; filename*=UTF-8''{quote(name)}"
    if kind == 'json':
        try:
            payload = json.loads(text)
        except ValueError:
            raise APIError(500, 'report_invalid', '结构化报告无法读取。') from None
        return JSONResponse(payload, headers=headers)
    return PlainTextResponse(text, headers=headers)


@app.get('/api/jobs/{job_id}/charts/{name}')
def job_chart(job_id: str, name: str, user: dict = Depends(current_user)):
    if name not in CHART_FILES:
        raise APIError(404, 'not_found', '图表不存在。')
    with storage.db() as conn:
        _job, output = _job_output(conn, job_id, user)
    path = output / CHART_FILES[name]
    if not path.is_file():
        raise APIError(404, 'not_found', '图表不存在。')
    return FileResponse(path, media_type='image/png')


@app.post('/api/jobs/{job_id}/retry', status_code=202)
def retry_job(job_id: str, user: dict = Depends(current_user)):
    ensure_disk_space()
    with storage.db() as conn, storage.transaction(conn):
        job = _own_job(conn, job_id, user)
        if job['status'] not in ('failed', 'timed_out'):
            raise APIError(409, 'not_retryable', '只有失败或超时的任务可以重试。')
        if job['type'] == 'prepare_reference':
            if not user['is_admin']:
                raise APIError(403, 'forbidden', '仅管理员可用。')
        else:
            ref = storage.get_ref(conn, job['reference_id'])
            if not ref or ref['status'] != 'ready':
                raise APIError(409, 'reference_not_ready', '原音频尚未准备好。')
            asset = storage.get_asset(conn, job['imitation_asset_id'])
            if not asset or asset['status'] in ('deleted', 'invalid'):
                raise APIError(409, 'asset_unavailable', '模仿音频已删除或无法解码，请重新上传后提交。')
        _check_quota(conn, user)
        attempt_no = job['attempt_no'] + 1
        conn.execute(
            "UPDATE jobs SET status = 'pending_enqueue', stage = NULL, attempt_no = ?, attempt_id = ?, "
            'queued_at = NULL, started_at = NULL, finished_at = NULL, error_code = NULL, error_message = NULL, '
            'error_detail = NULL, result_json = NULL, output_dir = NULL, updated_at = ? WHERE id = ?',
            (attempt_no, storage.new_attempt_id(attempt_no), storage.iso(), job_id))
        if job['type'] == 'prepare_reference':
            conn.execute("UPDATE refs SET status = 'preparing', error = NULL, updated_at = ? WHERE id = ?",
                         (storage.iso(), job['reference_id']))
        job = storage.get_job(conn, job_id)
    return _enqueue_response(job)


# ---- admin -------------------------------------------------------------------------

def _runs_query(start: str | None, end: str | None, task_type: str | None):
    clauses, params = [], []
    if start:
        clauses.append('inference_started_at >= ?')
        params.append(start)
    if end:
        clauses.append('inference_started_at < ?')
        params.append(end)
    if task_type:
        clauses.append('task_type = ?')
        params.append(task_type)
    where = ('WHERE ' + ' AND '.join(clauses)) if clauses else ''
    return f'SELECT * FROM inference_runs {where} ORDER BY id', params


@app.get('/api/admin/inference-runs')
def admin_runs(start: str | None = None, end: str | None = None, task_type: str | None = None,
               format: str = 'json', _admin: dict = Depends(admin_user)):
    sql, params = _runs_query(start, end, task_type)
    with storage.db() as conn:
        rows = [dict(r) for r in conn.execute(sql, params)]
    if format == 'csv':
        buffer = io.StringIO()
        columns = ['job_id', 'attempt_id', 'task_type', 'reference_duration_seconds',
                   'imitation_duration_seconds', 'inference_elapsed_seconds', 'inference_started_at',
                   'inference_finished_at', 'status', 'cache_hit', 'error_code', 'engine_version',
                   'asr_provider', 'engine_warm']
        writer = csv.DictWriter(buffer, columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
        return Response('﻿' + buffer.getvalue(), media_type='text/csv; charset=utf-8',
                        headers={'Content-Disposition': 'attachment; filename="inference-runs.csv"'})
    return {'items': rows}


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return round(values[low] + (values[high] - values[low]) * (position - low), 3)


@app.get('/api/admin/stats')
def admin_stats(start: str | None = None, end: str | None = None, _admin: dict = Depends(admin_user)):
    sql, params = _runs_query(start, end, 'evaluate')
    with storage.db() as conn:
        rows = [dict(r) for r in conn.execute(sql, params)]
        jobs = dict(conn.execute('SELECT status, COUNT(*) FROM jobs GROUP BY status').fetchall())
    by_status: dict[str, int] = defaultdict(int)
    for row in rows:
        by_status[row['status']] += 1
    ok = [r['inference_elapsed_seconds'] for r in rows
          if r['status'] == 'succeeded' and r['inference_elapsed_seconds'] is not None]
    finished = sum(by_status[s] for s in ('succeeded', 'failed', 'timed_out', 'abandoned'))
    return {'evaluate_attempts': dict(by_status),
            'success_rate': round(by_status['succeeded'] / finished, 4) if finished else None,
            'inference_elapsed_seconds': {'count': len(ok), 'mean': round(sum(ok) / len(ok), 3) if ok else None,
                                          'p50': _percentile(ok, 0.5), 'p95': _percentile(ok, 0.95)},
            'jobs_by_status': jobs}


class UserIn(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    is_admin: bool = False


class UserPatch(BaseModel):
    disabled: bool


def _user_admin_view(row) -> dict:
    row = dict(row)
    return {'id': row['id'], 'name': row['name'], 'is_admin': bool(row['is_admin']),
            'disabled': bool(row['disabled']), 'created_at': row['created_at'],
            'last_login_at': row.get('last_login_at'), 'evaluations': row.get('evaluations') or 0}


_USERS_SQL = ('SELECT u.*, (SELECT MAX(s.created_at) FROM sessions s WHERE s.user_id = u.id) AS last_login_at, '
              "(SELECT COUNT(*) FROM jobs j WHERE j.owner_id = u.id AND j.type = 'evaluate') AS evaluations "
              'FROM users u')


@app.get('/api/admin/users')
def admin_users(_admin: dict = Depends(admin_user)):
    with storage.db() as conn:
        rows = conn.execute(_USERS_SQL + ' ORDER BY u.created_at DESC').fetchall()
    return {'items': [_user_admin_view(r) for r in rows]}


@app.post('/api/admin/users', status_code=201)
def admin_create_user(body: UserIn, _admin: dict = Depends(admin_user)):
    with storage.db() as conn:
        user, code = storage.create_user(conn, body.name.strip(), body.is_admin)
        row = conn.execute(_USERS_SQL + ' WHERE u.id = ?', (user['id'],)).fetchone()
    # The plain invite code is only returned here; the database keeps its hash.
    return {'user': _user_admin_view(row), 'invite_code': code}


@app.post('/api/admin/users/{user_id}/reset-invite')
def admin_reset_invite(user_id: str, admin: dict = Depends(admin_user)):
    with storage.db() as conn:
        if not conn.execute('SELECT 1 FROM users WHERE id = ?', (user_id,)).fetchone():
            raise APIError(404, 'not_found', '用户不存在。')
        if user_id == admin['id']:
            raise APIError(409, 'self_reset', '不能在网页上重置自己的邀请码，以免当前会话失效。')
        code = storage.reset_invite(conn, user_id)
        row = conn.execute(_USERS_SQL + ' WHERE u.id = ?', (user_id,)).fetchone()
    return {'user': _user_admin_view(row), 'invite_code': code}


@app.patch('/api/admin/users/{user_id}')
def admin_update_user(user_id: str, body: UserPatch, admin: dict = Depends(admin_user)):
    with storage.db() as conn, storage.transaction(conn):
        target = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
        if not target:
            raise APIError(404, 'not_found', '用户不存在。')
        if body.disabled and user_id == admin['id']:
            raise APIError(409, 'self_disable', '不能停用自己的账号。')
        conn.execute('UPDATE users SET disabled = ? WHERE id = ?', (int(body.disabled), user_id))
        if body.disabled:
            conn.execute('DELETE FROM sessions WHERE user_id = ?', (user_id,))
        row = conn.execute(_USERS_SQL + ' WHERE u.id = ?', (user_id,)).fetchone()
    return {'user': _user_admin_view(row)}


class SettingsPatch(BaseModel):
    asr_provider: str


@app.get('/api/admin/settings')
def admin_settings(_admin: dict = Depends(admin_user)):
    with storage.db() as conn:
        return asr.status(conn)


@app.put('/api/admin/settings')
def admin_update_settings(body: SettingsPatch, admin: dict = Depends(admin_user)):
    if body.asr_provider not in asr.PROVIDERS:
        raise APIError(422, 'invalid_provider', '未知的转写服务。')
    check = asr.check_config(body.asr_provider)
    if not check['configured']:
        raise APIError(409, 'provider_not_configured',
                       f"{asr.PROVIDERS[body.asr_provider]}还没有配置好：{check['problem']}。")
    with storage.db() as conn:
        storage.set_app_setting(conn, asr.SETTING_KEY, body.asr_provider, admin['id'])
        log.info('asr provider set to %s by %s', body.asr_provider, admin['id'])
        return asr.status(conn)


@app.get('/api/asr-media/{name}')
def asr_media(name: str):
    # Temporary audio fetched by 通义听悟 (no session): random 128-bit name, 1 h lifetime.
    path = asr.media_file(name)
    if path is None:
        raise APIError(404, 'not_found', '文件不存在或已过期。')
    return FileResponse(path, headers={'Cache-Control': 'no-store'})


def _engine_server_health() -> dict:
    from .engine_adapter import server_status
    status = server_status(timeout=2.0)
    if status is None:
        return {'reachable': False, 'note': '启动中（加载模型/预热）、正在处理任务或已停止'}
    keep = ('ready', 'pid', 'jobs', 'started_at', 'last_job_at', 'warmup', 'model_loads', 'rss_mb')
    return {'reachable': True, **{k: status.get(k) for k in keep}}


@app.get('/api/admin/health')
def admin_health(_admin: dict = Depends(admin_user)):
    settings = get_settings()
    models = settings.engine_root / 'models'
    return {
        'redis': queueing.redis_ok(),
        'engine_root': settings.engine_root.is_dir(),
        'engine_python': settings.engine_python.exists(),
        'xunfei_config': asr.check_config('xunfei')['configured'],
        'tingwu_config': asr.check_config('tingwu')['configured'],
        'engine_mode': settings.engine_mode,
        'engine_server': _engine_server_health() if settings.engine_mode == 'server' else None,
        'asr_provider': asr.current_provider(),
        'models': {name: (models / name).is_dir() and not any((models / name).glob('*.partial'))
                   for name in ('wav2vec2-espeak', 'emphases', 'punct-multilang')},
        'disk_free_bytes': shutil.disk_usage(settings.data_root).free,
        'engine_version': settings.engine_version,
        'cache_version': settings.cache_version,
    }


@app.get('/api/health')
def health():
    try:
        with storage.db() as conn:
            conn.execute('SELECT 1').fetchone()
        db_ok = True
    except Exception:  # noqa: BLE001
        db_ok = False
    redis_ok = queueing.redis_ok()
    return JSONResponse({'ok': db_ok and redis_ok, 'db': db_ok, 'redis': redis_ok},
                        status_code=200 if db_ok else 503)


# ---- static page ---------------------------------------------------------------------

@app.get('/', include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / 'index.html', headers={'Cache-Control': 'no-cache'})


app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')
