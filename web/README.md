# 音频还原度测试 · Demo 网站

线上地址：<https://audio-fidelity.xinghexingsui.com>（邀请码登录）

在现有评测程序 `audio-fidelity-linux` 之外增加的试用网站：选择已缓存的原音频 → 上传或录制模仿音频 → 排队评测 → 查看综合分、四个分项、改进建议、图表与报告。保留“先缓存原音频、再评测”的机制。

## 架构

```
浏览器 ──HTTPS──> 宝塔 Nginx (443→8443, SNI 分流)
                    └─> 127.0.0.1:8896  api 容器 (FastAPI + ffprobe)
                                         │  SQLite / 文件: /srv/audio-demo/data  (同路径挂载)
                                         └─> redis 容器 (RQ 队列, AOF, 仅 127.0.0.1:6380)
                                                 │
宿主机 systemd: audio-fidelity-worker  <──────────┘  (单个 RQ Worker)
   ├─ FFmpeg 转 16 kHz/单声道/PCM16 WAV
   └─ ENGINE_PYTHON = <引擎>/.venv-linux/bin/python  webapp/engine_runner.py
        └─ 现有 path_jobs.prepare_reference() / evaluate()（独立进程组，840 s 硬超时）
```

- **API + Redis 在 Docker**，**推理 Worker 在宿主机**：直接使用服务器上已装好的引擎环境、模型和 `讯飞配置.json`，不在镜像里重复安装 torch。
- 数据目录、引擎目录在容器内外是**同一个绝对路径**，数据库里记录的路径两边通用。
- Worker 启动时调用引擎的 `cpu_runtime` 设定 CPU 亲和性（最多 3 核、留 1 核），FFmpeg 与所有引擎子进程继承同一组 CPU。

## 目录

| 路径 | 说明 |
| --- | --- |
| `webapp/main.py` | HTTP 接口、会话、权限、上传、管理接口、定期核对线程 |
| `webapp/tasks.py` | RQ 任务入口 `webapp.tasks.execute`、任务认领与状态同步 |
| `webapp/engine_adapter.py` | 进程组/超时管理、引擎环境变量、结果映射、报告脱敏 |
| `webapp/engine_runner.py` | 由 ENGINE_PYTHON 执行，调用现有引擎并计时 |
| `webapp/media.py` | FFprobe 检查、FFmpeg 统一转码 |
| `webapp/storage.py` | SQLite 表结构、编号→受控路径、埋点读写 |
| `webapp/maintenance.py` | 待入队补投、失联任务关闭、保留期清理 |
| `webapp/worker.py` | Worker 入口（CPU 亲和性 + 启动核对） |
| `webapp/admin.py` | 运维命令：建用户、批量导入素材、重试、导出埋点 |
| `webapp/static/` | 单页前端（上传、录音、轮询、结果、最近任务） |
| `compose.yml` / `Dockerfile` | api + redis 容器 |
| `deploy/` | Nginx 站点、Redis 配置、systemd Worker 单元、部署脚本 |
| `tests/` | 端到端检查脚本、桩引擎（本地开发用） |

文档：[部署与运维](docs/DEPLOY.md) · [HTTP 接口](docs/API.md)

## 本地开发（不需要推理依赖）

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-web.txt httpx soundfile filelock
# 用真实引擎源码 + 桩阶段脚本搭一个秒级返回的测试引擎
bash tests/fake_engine/make_fake_engine.sh /path/to/audio-fidelity-linux /tmp/fake-engine
docker run -d --name afd-redis -p 127.0.0.1:6390:6379 redis:8-alpine
export DATA_ROOT=/tmp/afd-data ENGINE_ROOT=/tmp/fake-engine ENGINE_PYTHON=$PWD/.venv/bin/python \
       REDIS_URL=redis://127.0.0.1:6390/0 PUBLIC_ORIGIN=http://127.0.0.1:8899 COOKIE_SECURE=0
.venv/bin/python -m webapp.admin create-user 管理员 --admin
.venv/bin/uvicorn webapp.main:app --port 8899 &   .venv/bin/python -m webapp.worker &
.venv/bin/python tests/e2e_check.py http://127.0.0.1:8899 --admin-code ... --user-code ... --user2-code ...
```
