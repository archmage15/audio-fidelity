# API image (FastAPI + ffprobe for upload checks). Inference does not run here:
# the worker runs on the host with the engine's own .venv-linux.
FROM ubuntu:24.04

ARG APT_MIRROR=http://mirrors.aliyun.com/ubuntu
ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH=/opt/web/bin:$PATH

RUN sed -i "s|http://archive.ubuntu.com/ubuntu|${APT_MIRROR}|g; s|http://security.ubuntu.com/ubuntu|${APT_MIRROR}|g" \
        /etc/apt/sources.list.d/ubuntu.sources \
 && apt-get update \
 && apt-get install -y --no-install-recommends python3.12 python3.12-venv ffmpeg ca-certificates tini \
 && rm -rf /var/lib/apt/lists/*

COPY requirements-web.txt /tmp/requirements-web.txt
RUN python3.12 -m venv /opt/web \
 && /opt/web/bin/pip install --no-cache-dir -i "${PIP_INDEX_URL}" -r /tmp/requirements-web.txt

WORKDIR /app
COPY webapp /app/webapp
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "webapp.main:app", "--host", "0.0.0.0", "--port", "8000"]
