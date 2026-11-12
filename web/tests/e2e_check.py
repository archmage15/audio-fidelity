"""End-to-end check against a running deployment.

Usage:
  python tests/e2e_check.py BASE_URL --admin-code CODE --user-code CODE --user2-code CODE \
      [--reference FILE] [--imitation FILE] [--reference-id ID]

Without --reference-id the admin uploads --reference (or a generated tone) and
waits for caching — on a real engine this calls Xunfei once.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import httpx

H = {'X-Requested-With': 'fetch'}
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, info: str = '') -> bool:
    results.append((name, ok, info))
    print(('PASS ' if ok else 'FAIL ') + name + (f'  [{info}]' if info else ''), flush=True)
    return ok


def client(base: str, code: str) -> httpx.Client:
    c = httpx.Client(base_url=base, headers={**H, 'Origin': base}, timeout=60, follow_redirects=False)
    r = c.post('/api/session', json={'invite_code': code})
    r.raise_for_status()
    return c


def tone(path: Path, seconds: float, extra: list[str]) -> Path:
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', f'sine=frequency=440:duration={seconds}',
                    *extra, str(path)], check=True)
    return path


def wait_job(c: httpx.Client, job_id: str, limit: float = 900) -> dict:
    start = time.time()
    while time.time() - start < limit:
        job = c.get(f'/api/jobs/{job_id}').json()
        if job['status'] in ('succeeded', 'failed', 'timed_out'):
            return job
        time.sleep(2)
    raise TimeoutError(job_id)


def upload(c: httpx.Client, path: Path, **form) -> httpx.Response:
    with path.open('rb') as fh:
        return c.post('/api/assets', files={'file': (path.name, fh)}, data=form)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('base')
    p.add_argument('--admin-code', required=True)
    p.add_argument('--user-code', required=True)
    p.add_argument('--user2-code', required=True)
    p.add_argument('--reference')
    p.add_argument('--imitation')
    p.add_argument('--reference-id')
    a = p.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix='afd-e2e-'))
    admin, user, user2 = (client(a.base, code) for code in (a.admin_code, a.user_code, a.user2_code))

    # ---- auth & CSRF
    anon = httpx.Client(base_url=a.base, timeout=30)
    check('unauthenticated → 401', anon.get('/api/jobs').status_code == 401)
    check('bad invite → 401', anon.post('/api/session', json={'invite_code': 'NOPE-NOPE'}, headers=H).status_code == 401)
    check('missing CSRF header → 403', user.post('/api/evaluations', json={}, headers={'X-Requested-With': ''}).status_code == 403)
    check('foreign Origin → 403', user.post('/api/evaluations', json={}, headers={'Origin': 'https://evil.example'}).status_code == 403)

    # ---- reference
    ref_id = a.reference_id
    if not ref_id:
        ref_file = Path(a.reference) if a.reference else tone(tmp / 'ref.mp3', 4, [])
        check('user cannot upload reference', upload(user, ref_file, purpose='reference').status_code == 403)
        r = upload(admin, ref_file, purpose='reference')
        check('admin uploads reference → 201', r.status_code == 201, r.text[:200])
        asset = r.json()
        r = admin.post('/api/references', json={'asset_id': asset['asset_id'], 'display_name': 'E2E 原音频 ' + time.strftime('%H%M%S'),
                                                'is_public': True}, headers={'Idempotency-Key': str(uuid.uuid4())})
        check('create reference → 202', r.status_code == 202, r.text[:200])
        ref_id = r.json()['reference_id']
        job = wait_job(admin, r.json()['job_id'])
        check('reference prepared', job['status'] == 'succeeded', str(job.get('error')))
        check('prepare metrics recorded', job['metrics']['inference_elapsed_seconds'] is not None, str(job['metrics']))
    ref = user.get(f'/api/references/{ref_id}').json()
    check('reference ready & visible to user', ref.get('status') == 'ready', str(ref)[:200])
    check('reference has transcript', bool(ref.get('transcript')))
    check('user lists reference', any(i['reference_id'] == ref_id for i in user.get('/api/references').json()['items']))

    # ---- upload validation
    bad = tmp / 'bad.mp3'
    bad.write_bytes(b'not audio at all' * 100)
    check('garbage file → 415', upload(user, bad).status_code == 415)
    check('too short → 422', upload(user, tone(tmp / 'short.wav', 0.5, [])).status_code == 422)
    check('too long → 422', upload(user, tone(tmp / 'long.mp3', 75, [])).status_code == 422)

    # ---- imitation upload (WebM/Opus like a browser recording) and evaluation
    imit_file = Path(a.imitation) if a.imitation else tone(tmp / 'recording.webm', 4, ['-c:a', 'libopus'])
    r = upload(user, imit_file, kind='recording')
    check('upload imitation → 201', r.status_code == 201, r.text[:200])
    imit = r.json()
    key = str(uuid.uuid4())
    body = {'reference_id': ref_id, 'imitation_asset_id': imit['asset_id']}
    r1 = user.post('/api/evaluations', json=body, headers={'Idempotency-Key': key})
    r2 = user.post('/api/evaluations', json=body, headers={'Idempotency-Key': key})
    check('evaluation → 202', r1.status_code == 202, r1.text[:200])
    check('same key → same job', r1.json().get('job_id') == r2.json().get('job_id'))
    r3 = user.post('/api/evaluations', json={**body, 'reference_id': 'ref_other'}, headers={'Idempotency-Key': key})
    check('same key, different body → 409', r3.status_code == 409, r3.text[:120])
    job_id = r1.json()['job_id']
    check('other user cannot see job → 404', user2.get(f'/api/jobs/{job_id}').status_code == 404)
    check('other user cannot fetch audio → 404', user2.get(f"/api/assets/{imit['asset_id']}/audio").status_code == 404)
    job = wait_job(user, job_id)
    ok = check('evaluation succeeded', job['status'] == 'succeeded', str(job.get('error')))
    if ok:
        res = job['result']
        check('overall score numeric', isinstance(res['overall'], (int, float)), str(res['overall']))
        check('four parts present', set(res['parts']) == {'pronunciation', 'sound_changes', 'rhythm', 'stress'}, str(res['parts']))
        m = job['metrics']
        check('metrics: durations + inference time', all(m.get(k) is not None for k in (
            'reference_duration_seconds', 'imitation_duration_seconds', 'inference_elapsed_seconds')), str(m))
        report = user.get(res['report_url'])
        check('report readable', report.status_code == 200 and '综合还原度' in report.text)
        leaked = [line for line in report.text.splitlines() if any(s in line for s in ('/data', '/srv', '/www', '/tmp', '/root'))]
        check('report has no server paths', not leaked, ' | '.join(leaked[:2]))
        detail = user.get(res['report_urls'].get('detail', res['report_url']))
        check('detail report readable', detail.status_code == 200)
        structured_url = res['report_urls'].get('json')
        check('structured report URL available', bool(structured_url))
        structured = user.get(structured_url) if structured_url else None
        if structured:
            body = structured.json() if structured.status_code == 200 else {}
            check('structured report schema', body.get('schema_version') == 1 and
                  {'focus_sounds', 'top_issues', 'sentences'} <= body.keys())
            check('structured report content and path redaction',
                  body.get('sentences', [{}])[0].get('words', [{}])[0].get('issues', [{}])[0].get('how')
                  == '路径 〈服务器路径〉')
            downloaded = user.get(structured_url + '&download=1')
            check('structured report downloads as JSON', downloaded.status_code == 200 and
                  'report-' in downloaded.headers.get('content-disposition', '') and
                  '.json' in downloaded.headers.get('content-disposition', ''))
            check('other user cannot fetch structured report', user2.get(structured_url).status_code == 404)
        for name, url in res.get('chart_urls', {}).items():
            r = user.get(url)
            check(f'chart {name} png', r.status_code == 200 and r.content[:4] == b'\x89PNG')
        r = user.get(job['imitation']['audio_url'], headers={'Range': 'bytes=0-99'})
        check('audio Range → 206', r.status_code == 206 and len(r.content) == 100)
        listed = user.get('/api/jobs').json()['items']
        check('job in recent list', any(i['job_id'] == job_id for i in listed))
        check('asset in use by job cannot be... deleted after finish', user.delete(f"/api/assets/{imit['asset_id']}").status_code == 200)

    # ---- per-user quota (2 active)
    ids = []
    for _ in range(3):
        r = upload(user, imit_file, kind='upload')
        r = user.post('/api/evaluations', json={'reference_id': ref_id, 'imitation_asset_id': r.json()['asset_id']},
                      headers={'Idempotency-Key': str(uuid.uuid4())})
        ids.append(r)
    check('third concurrent job → 429', [r.status_code for r in ids] == [202, 202, 429], str([r.status_code for r in ids]))
    for r in ids[:2]:
        wait_job(user, r.json()['job_id'])

    passed = sum(ok for _, ok, _ in results)
    print(f'\n{passed}/{len(results)} checks passed')
    return 0 if passed == len(results) else 1


if __name__ == '__main__':
    sys.exit(main())
