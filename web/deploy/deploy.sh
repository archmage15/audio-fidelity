#!/usr/bin/env bash
# Sync the website to the server, rebuild/start the API + Redis containers and
# (re)install/restart the host inference worker.
# Usage: deploy/deploy.sh [ssh-key] [user@host]
set -euo pipefail
KEY="${1:-hongyan_new_login_key}"
HOST="${2:-root@112.124.25.152}"
REMOTE=/srv/audio-demo
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RELEASE="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M)"

rsync -az --delete -e "ssh -i $KEY -o StrictHostKeyChecking=accept-new" \
  --exclude .git --exclude .venv --exclude .venv-worker --exclude data --exclude .env \
  --exclude '__pycache__' --exclude .pytest_cache \
  "$ROOT/" "$HOST:$REMOTE/"
ssh -i "$KEY" "$HOST" bash -s -- "$REMOTE" "$RELEASE" <<'REMOTE_SCRIPT'
set -euo pipefail
cd "$1"
[[ -f .env ]] || { echo ".env missing in $1 (copy .env.example and fill it)"; exit 1; }
sed -i "s/^RELEASE=.*/RELEASE=$2/" .env
set -a; source <(grep -E '^(DATA_ROOT|APP_UID|APP_GID|ENGINE_ROOT)=' .env); set +a
mkdir -p "$DATA_ROOT"
chown -R "${APP_UID}:${APP_GID}" "$DATA_ROOT"
chmod 640 .env && chgrp "$APP_GID" .env   # readable by the www worker, not by others

# API + Redis
docker compose build api
docker compose up -d --remove-orphans

# Host worker: small venv with the queue client only (engine env stays untouched)
if [[ ! -x .venv-worker/bin/python ]]; then
  /usr/bin/python3.12 -m venv .venv-worker
fi
.venv-worker/bin/pip install -q -i https://mirrors.aliyun.com/pypi/simple -r requirements-worker.txt
install -m 644 deploy/systemd/audio-fidelity-worker.service /etc/systemd/system/audio-fidelity-worker.service
install -m 644 deploy/systemd/audio-fidelity-engine.service /etc/systemd/system/audio-fidelity-engine.service
systemctl daemon-reload
systemctl enable --quiet audio-fidelity-engine audio-fidelity-worker
# Both stop gracefully (finish the current job). The engine reloads models and
# warms up after the restart; jobs submitted meanwhile wait for it.
systemctl restart audio-fidelity-engine
systemctl restart audio-fidelity-worker
sleep 3
docker compose ps
systemctl --no-pager --lines=3 status audio-fidelity-engine audio-fidelity-worker
REMOTE_SCRIPT
