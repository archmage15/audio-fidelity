"""Operator CLI: ``python -m webapp.admin <command>`` (run inside the api container).

Commands
  create-user NAME [--admin]       create a tester (or admin) and print the invite code
  list-users                       list users
  reset-invite USER_ID             issue a new invite code and log out old sessions
  disable-user USER_ID             block login and end sessions
  import-library [DIR] [--recursive] [--private] --owner ADMIN_ID
                                   batch-register reference audio files and queue caching
  references                       list reference materials and their status
  sync-transcripts                 refresh reference text from cache 转写文本.txt
  retry-failed [--type T]          queue a new attempt for failed jobs
  export-runs [--start ISO] [--end ISO] [--task-type evaluate|prepare_reference]
                                   print inference_runs as CSV
  reconcile                        run queue reconciliation and cleanup once
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

from . import maintenance, queueing, storage
from .config import get_settings
from .engine_adapter import read_transcript
from .media import MediaError, check_duration, probe

LIBRARY_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.mp4', '.webm', '.opus', '.oga'}


def create_user(args) -> None:
    with storage.db() as conn:
        user, code = storage.create_user(conn, args.name, args.admin)
    role = '管理员' if args.admin else '试用者'
    print(f'{role} {user["name"]} 已创建：{user["id"]}\n邀请码：{code}\n（邀请码只显示这一次，请妥善转交）')


def list_users(_args) -> None:
    with storage.db() as conn:
        for row in conn.execute('SELECT id, name, is_admin, disabled, created_at FROM users ORDER BY created_at'):
            flags = ('admin ' if row['is_admin'] else '') + ('disabled' if row['disabled'] else '')
            print(f"{row['id']}\t{row['name']}\t{flags.strip() or '-'}\t{row['created_at']}")


def reset_invite(args) -> None:
    with storage.db() as conn:
        if not conn.execute('SELECT 1 FROM users WHERE id = ?', (args.user_id,)).fetchone():
            sys.exit('用户不存在')
        print('新邀请码：' + storage.reset_invite(conn, args.user_id))


def disable_user(args) -> None:
    with storage.db() as conn:
        conn.execute('UPDATE users SET disabled = 1 WHERE id = ?', (args.user_id,))
        conn.execute('DELETE FROM sessions WHERE user_id = ?', (args.user_id,))
    print('已禁用')


def import_library(args) -> None:
    settings = get_settings()
    source = Path(args.directory or settings.library_dir).resolve()
    if not source.is_dir():
        sys.exit(f'目录不存在：{source}')
    with storage.db() as conn:
        owner = conn.execute('SELECT * FROM users WHERE id = ? AND is_admin = 1', (args.owner,)).fetchone()
    if not owner:
        sys.exit('--owner 必须是管理员用户编号（list-users 查看）')
    pattern = source.rglob('*') if args.recursive else source.iterdir()
    files = sorted(p for p in pattern if p.is_file() and p.suffix.lower() in LIBRARY_EXTENSIONS)
    if not files:
        sys.exit('未找到音频文件')
    queued = 0
    for path in files:
        try:
            info = probe(path)
            check_duration(info.duration, settings.min_audio_seconds, settings.max_reference_seconds)
        except MediaError as error:
            print(f'跳过 {path.name}：{error.message}')
            continue
        asset_id = storage.new_id('asset')
        storage.asset_dir(asset_id).mkdir(parents=True)
        shutil.copyfile(path, storage.asset_source_path(asset_id))
        now = storage.iso()
        with storage.db() as conn, storage.transaction(conn):
            conn.execute("INSERT INTO assets (id, owner_id, display_name, kind, source_format, size_bytes, "
                         "duration_seconds, status, created_at, updated_at) VALUES (?, ?, ?, 'library', ?, ?, ?, "
                         "'uploaded', ?, ?)", (asset_id, owner['id'], path.name[:120],
                                              ','.join(sorted(info.format_names)), path.stat().st_size,
                                              info.duration, now, now))
            ref_id = storage.new_id('ref')
            conn.execute("INSERT INTO refs (id, asset_id, owner_id, is_public, display_name, status, cache_version, "
                         "created_at, updated_at) VALUES (?, ?, ?, ?, ?, 'preparing', ?, ?, ?)",
                         (ref_id, asset_id, owner['id'], int(not args.private), path.stem[:120],
                          settings.cache_version, now, now))
            job = storage.insert_job(conn, owner_id=owner['id'], job_type='prepare_reference',
                                     reference_id=ref_id, imitation_asset_id=None)
            conn.execute('UPDATE refs SET prepare_job_id = ? WHERE id = ?', (job['id'], ref_id))
        state = '已排队' if queueing.enqueue(job['id'], 1) else '待补投（Redis 不可用）'
        queued += 1
        print(f'{path.name} → {ref_id} / {job["id"]} {state}')
    print(f'共登记 {queued} 个原音频；准备任务逐个执行，新素材会调用讯飞转写。')


def list_references(_args) -> None:
    with storage.db() as conn:
        for row in conn.execute('SELECT id, display_name, status, is_public, cache_version, created_at '
                                'FROM refs ORDER BY created_at'):
            print(f"{row['id']}\t{row['status']}\t{'public' if row['is_public'] else 'private'}\t"
                  f"{row['cache_version']}\t{row['display_name']}")


def export_runs(args) -> None:
    clauses, params = [], []
    for column, op, value in (('inference_started_at', '>=', args.start), ('inference_started_at', '<', args.end),
                              ('task_type', '=', args.task_type)):
        if value:
            clauses.append(f'{column} {op} ?')
            params.append(value)
    where = ('WHERE ' + ' AND '.join(clauses)) if clauses else ''
    with storage.db() as conn:
        rows = [dict(r) for r in conn.execute(f'SELECT * FROM inference_runs {where} ORDER BY id', params)]
    writer = csv.DictWriter(sys.stdout, list(rows[0]) if rows else ['job_id'])
    writer.writeheader()
    writer.writerows(rows)


def sync_transcripts(_args) -> None:
    """Refresh each ready reference's display text from its cache 转写文本.txt."""
    settings = get_settings()
    changed = 0
    with storage.db() as conn:
        refs = [dict(r) for r in conn.execute(
            "SELECT id, display_name, cache_sha256, transcript FROM refs WHERE status = 'ready' "
            'AND cache_sha256 IS NOT NULL')]
        for ref in refs:
            text = read_transcript(settings.reference_cache_dir / ref['cache_sha256'])
            if text and text != ref['transcript']:
                conn.execute('UPDATE refs SET transcript = ?, updated_at = ? WHERE id = ?',
                             (text, storage.iso(), ref['id']))
                changed += 1
                print(f"{ref['display_name']}: 已更新")
            elif not text:
                print(f"{ref['display_name']}: 缓存中没有转写文本")
    print(f'共检查 {len(refs)} 个原音频，更新 {changed} 个。')


def retry_failed(args) -> None:
    """Queue a new attempt for failed/timed-out jobs (e.g. after fixing 讯飞配置.json)."""
    settings = get_settings()
    with storage.db() as conn:
        jobs = [dict(r) for r in conn.execute(
            "SELECT * FROM jobs WHERE type = ? AND status IN ('failed', 'timed_out') ORDER BY created_at",
            (args.type,))]
    if not jobs:
        print('没有需要重试的任务')
        return
    for job in jobs:
        with storage.db() as conn, storage.transaction(conn):
            if storage.count_active(conn) >= settings.max_active_global:
                print('全局活动任务已满，稍后再运行本命令')
                return
            attempt_no = job['attempt_no'] + 1
            cur = conn.execute(
                "UPDATE jobs SET status = 'pending_enqueue', stage = NULL, attempt_no = ?, attempt_id = ?, "
                'queued_at = NULL, started_at = NULL, finished_at = NULL, error_code = NULL, error_message = NULL, '
                "error_detail = NULL, result_json = NULL, output_dir = NULL, updated_at = ? "
                "WHERE id = ? AND status IN ('failed', 'timed_out')",
                (attempt_no, storage.new_attempt_id(attempt_no), storage.iso(), job['id']))
            if cur.rowcount != 1:
                continue
            if job['type'] == 'prepare_reference':
                conn.execute("UPDATE refs SET status = 'preparing', error = NULL, updated_at = ? WHERE id = ?",
                             (storage.iso(), job['reference_id']))
        state = '已排队' if queueing.enqueue(job['id'], attempt_no) else '待补投（Redis 不可用）'
        print(f"{job['id']} 第 {attempt_no} 次尝试 {state}")


def reconcile(_args) -> None:
    maintenance.run_once()
    print('完成')


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog='python -m webapp.admin', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('create-user')
    p.add_argument('name')
    p.add_argument('--admin', action='store_true')
    p.set_defaults(func=create_user)
    sub.add_parser('list-users').set_defaults(func=list_users)
    p = sub.add_parser('reset-invite')
    p.add_argument('user_id')
    p.set_defaults(func=reset_invite)
    p = sub.add_parser('disable-user')
    p.add_argument('user_id')
    p.set_defaults(func=disable_user)
    p = sub.add_parser('import-library')
    p.add_argument('directory', nargs='?')
    p.add_argument('--recursive', action='store_true')
    p.add_argument('--private', action='store_true')
    p.add_argument('--owner', required=True)
    p.set_defaults(func=import_library)
    sub.add_parser('references').set_defaults(func=list_references)
    p = sub.add_parser('export-runs')
    p.add_argument('--start')
    p.add_argument('--end')
    p.add_argument('--task-type', choices=['evaluate', 'prepare_reference'])
    p.set_defaults(func=export_runs)
    sub.add_parser('sync-transcripts').set_defaults(func=sync_transcripts)
    p = sub.add_parser('retry-failed')
    p.add_argument('--type', choices=['prepare_reference', 'evaluate'], default='prepare_reference')
    p.set_defaults(func=retry_failed)
    sub.add_parser('reconcile').set_defaults(func=reconcile)
    args = parser.parse_args(argv)
    storage.init_db()
    args.func(args)


if __name__ == '__main__':
    main()
