# 部署与运维

目标服务器：`root@112.124.25.152`（Ubuntu 24.04，4 核 / 7 GB，宝塔 Nginx），域名 `audio-fidelity.xinghexingsui.com`。

## 1. 组成与位置

| 组件 | 运行方式 | 位置 / 端口 |
| --- | --- | --- |
| 网站代码、compose、`.env` | — | `/srv/audio-demo` |
| 业务数据（SQLite、音频、缓存、报告） | 宿主目录，属主 `www`(1001) | `/srv/audio-demo/data` |
| API | Docker `audio-fidelity-demo-api-1`，uid 1001 | `127.0.0.1:8896` |
| Redis | Docker `audio-fidelity-demo-redis-1`，AOF，`noeviction`，密码认证 | `127.0.0.1:6380`（仅本机） |
| 推理 Worker（队列消费） | systemd `audio-fidelity-worker`，用户 `www` | 宿主机 |
| 常驻推理进程（模型常驻内存） | systemd `audio-fidelity-engine`，用户 `www`，引擎 venv | 宿主机，`data/run/engine.sock` |
| 引擎（源码、`.venv-linux`、`models/`、`讯飞配置.json`） | 只读使用 | `/www/wwwroot/projectRepo/Cor-Pron/audio-fidelity-linux` |
| Nginx 站点 | 宝塔 | `/www/server/panel/vhost/nginx/audio-fidelity.xinghexingsui.com.conf` |
| 证书 | Let's Encrypt（webroot），自动续期后 reload Nginx | `/etc/letsencrypt/live/audio-fidelity.xinghexingsui.com/` |

## 2. 依赖清单

宿主机（已具备）：Docker + Compose v2、Python 3.12（`/usr/bin/python3.12`）、ffmpeg/ffprobe、espeak-ng、中文字体、rsync、certbot；引擎环境按 `audio-fidelity-linux/Linux使用说明.md` 安装（torch 2.11 CPU、transformers 5.17 等，见其 `requirements-linux.txt`），模型完整放在 `models/`。

网站新增 Python 包（版本已固定）：
- API 镜像 `requirements-web.txt`：fastapi 0.162.0、starlette 1.10.0、pydantic 2.14.2、uvicorn 0.54.0、python-multipart 0.0.33、redis 8.1.0、rq 3.0.0
- 宿主 Worker `requirements-worker.txt`（`/srv/audio-demo/.venv-worker`）：redis 8.1.0、rq 3.0.0。引擎环境本身不做任何改动。

镜像：`ubuntu:24.04`（apt/pip 走阿里云镜像）、`redis:8-alpine`。

## 3. 环境变量（`/srv/audio-demo/.env`，权限 640 root:www）

API 容器（compose `env_file`）和宿主 Worker（systemd `EnvironmentFile`）读同一份。模板见 `.env.example`：

| 变量 | 含义 | 默认 |
| --- | --- | --- |
| `ENGINE_ROOT` | 引擎目录（容器内同路径只读挂载） | — |
| `ENGINE_PYTHON` | 推理解释器 | `$ENGINE_ROOT/.venv-linux/bin/python` |
| `DATA_ROOT` | 数据目录（容器内同路径挂载） | — |
| `CACHE_NAMESPACE` | 原音频缓存命名空间；转码规则或模型变更时改为 v2 | `v1` |
| `REDIS_PASSWORD` / `REDIS_PORT` | Redis 密码 / 宿主映射端口 | — / 6380 |
| `HTTP_PORT` | API 映射到本机的端口 | 8896 |
| `PUBLIC_ORIGIN` | 站点来源，写接口校验 Origin | — |
| `MAX_UPLOAD_BYTES` / `MIN_AUDIO_SECONDS` / `MAX_AUDIO_SECONDS` / `MAX_REFERENCE_SECONDS` | 上传限制 | 20 MiB / 2 / 60 / 60 |
| `MAX_ACTIVE_PER_USER` / `MAX_ACTIVE_GLOBAL` | 活动任务上限（管理员不受每人上限） | 2 / 20 |
| `ENGINE_TIMEOUT_SECONDS` / `RQ_JOB_TIMEOUT` | 引擎进程组硬超时 / RQ 超时 | 840 / 900 |
| `MIN_FREE_BYTES` | 磁盘低水位，低于时拒绝上传与新任务 | 2 GiB |
| `RETENTION_DAYS` / `IDLE_UPLOAD_HOURS` | 试用者结果保留 / 未提交上传清理 | 7 天 / 24 小时 |
| `HF_OFFLINE` | 推理时禁止联网下载模型（缺模型明确失败） | true |
| `ENGINE_MODE` | `server` 交给常驻推理进程；`oneshot` 每个任务新起引擎进程（旧方式，备用） | server |
| `ENGINE_SERVER_MAX_RSS_MB` | 常驻进程内存超过此值时，在任务间自动重启释放内存 | 3200 |
| `ASR_PROVIDER` | 管理员未在网页切换前使用的转写服务：`xunfei` / `tingwu` | xunfei |
| `TINGWU_CONFIG` | 通义听悟配置文件路径 | `$ENGINE_ROOT/听悟配置.json` |
| `AUDIO_TEST_CPU_THREADS` | 可选，降低引擎数值线程数 | 引擎默认 |

## 4. 部署 / 升级

在开发机项目根目录：

```bash
bash deploy/deploy.sh /path/to/hongyan_new_login_key root@112.124.25.152
```

脚本会：rsync 代码（不覆盖 `.env`、`data/`、`.venv-worker/`）→ 写入 `RELEASE=<git 短哈希>` → 构建并启动 api/redis → 创建/更新宿主 Worker venv → 安装 systemd 单元并重启 Worker。Worker 停止是**平滑**的：先等当前任务完成（最长 900 s），再结束残留进程。

首次部署额外步骤（已完成，留作记录）：
1. `cp .env.example .env`，生成 `REDIS_PASSWORD`（`openssl rand -hex 24`）。
2. Nginx：先放 `deploy/nginx-audio-fidelity.conf` 的 80 端口部分 → `nginx -s reload` →
   `certbot certonly --webroot -w /www/wwwroot/acme-challenge -d audio-fidelity.xinghexingsui.com` →
   放完整配置 → `nginx -t && nginx -s reload`；在 `/etc/letsencrypt/renewal/audio-fidelity...conf` 的 `[renewalparams]` 加 `renew_hook = /www/server/nginx/sbin/nginx -s reload`。

## 5. 账号与素材

**日常发邀请直接用网页**：管理员登录后页面底部“邀请管理”可创建邀请（复制邀请码或带链接的邀请消息，链接打开自动填好邀请码）、重置邀请码、停用/恢复用户。以下命令行作为备用。

所有命令在 `/srv/audio-demo` 下执行：

```bash
A="docker compose exec -T api python -m webapp.admin"
$A create-user 张三            # 试用者，输出邀请码（只显示一次）
$A create-user 李四 --admin    # 管理员
$A list-users | $A reset-invite USER_ID | $A disable-user USER_ID

# 批量登记原音频并排队缓存（每个新素材调用一次讯飞转写）
cp 素材/*.mp3 data/reference-library/课程A/ && chown -R www:www data/reference-library
$A import-library /srv/audio-demo/data/reference-library/课程A --owner <管理员USER_ID> [--private] [--recursive]
$A references                  # 查看素材状态
$A retry-failed                # 修好讯飞配置等之后批量重试失败的准备任务
$A sync-transcripts            # 从缓存 转写文本.txt 刷新原音频的展示原文
$A export-runs --start 2026-11-01 --task-type evaluate > runs.csv   # 推理埋点
```

管理员也可在网页“上传新原音频”入口上传单个素材。邀请码即个人身份，不同邀请码之间文件、任务、报告完全隔离；公开素材所有人可用。

## 6. 常驻推理进程

`audio-fidelity-engine`（`webapp/engine_server.py`）启动时设置 CPU（3 核）、导入 torch/transformers、加载 Wav2Vec2，再用最短的已缓存原音频对自己完整评测一次预热（不调用转写服务），然后在 `data/run/engine.sock` 上逐个处理任务。启动到就绪约 30 秒，常驻内存约 1.5–1.9 GB。

- **引擎源码不改**：引擎各阶段原本用 `subprocess.run(python 脚本)` 串起来，常驻进程把这些调用改为进程内执行，并像子进程一样隔离：环境变量、工作目录、argv、sys.path 和引擎模块（`scripts/*.py` 在导入时读路径）每次都换新并在结束后恢复；`cpu_runtime` 是进程级设置，只在启动时配置一次。`transformers` 的 `from_pretrained` 按路径缓存，emphases 库自带进程内缓存。espeak、ffmpeg 等仍是子进程。
- **任务流程**：Worker 先 ping，服务启动/重启期间会等它就绪；任务超时时 Worker 结束该进程，systemd 自动重启（重新加载模型 + 预热）。
- **结果一致性**（2026-11-03 上线前验证）：3 组线上输入分别用旧方式和常驻进程评测，`还原度综合评分.json`、`音素对比.json`、`节奏对比.json` 全部字段、两份报告逐行、节奏图逐字节相同。
- **耗时**（同 3 组，转写关闭）：旧方式引擎耗时 43–48 s；常驻热态 3.6–9.4 s。转写服务（听悟约 6 s）另计。
- 埋点 `inference_runs.engine_warm`：1 = 模型已在内存，0 = 本次加载了模型，空 = 旧方式。
- 回退：`.env` 设 `ENGINE_MODE=oneshot` 后 `systemctl restart audio-fidelity-worker`，可停掉 `audio-fidelity-engine` 释放内存。
- 改了引擎代码：常驻进程每个任务都重新导入引擎模块，新代码下一个任务即生效；换了模型或依赖则需 `systemctl restart audio-fidelity-engine`。

## 7. 转写服务（讯飞 / 通义听悟）

引擎只会调用讯飞。转写在两处用到：准备原音频时转写原文（失败则原音频不可用）；评测时转写模仿录音做单词级纠音（失败只降级，不影响打分）。

- **切换**：管理员在网页“转写服务”卡片里选择，对之后开始的任务生效，已缓存的原音频不会重新转写。每次推理所用的服务记录在 `inference_runs.asr_provider`。
- **通义听悟配置**：`$ENGINE_ROOT/听悟配置.json`（格式见 `deploy/听悟配置.example.json`），属主 `www`、权限 `640`。需要阿里云 AccessKey（RAM 用户授予 `AliyunTingwuFullAccess`）和听悟项目的 AppKey；源语言默认 `en`，可加 `"language"` 字段修改。
- **原理**：选用听悟时，Worker 把 `webapp/asr_hook/` 加入引擎进程的 `PYTHONPATH`，其中的 `sitecustomize.py` 拦截引擎对 `xunfei_transcribe.py` 的两种调用，改用 `afd_tingwu.py`；引擎源码不做任何改动。听悟只接受音频 URL，所以音频会临时复制到 `data/asr-media/`，经 `/api/asr-media/<随机名>` 提供给听悟下载，任务结束即删除，1 小时后链接失效。

## 8. 日常运维

```bash
docker compose ps && docker compose logs -f api            # API
journalctl -u audio-fidelity-worker -f                      # Worker
journalctl -u audio-fidelity-engine -f                      # 常驻推理进程（加载、预热、每个任务耗时/内存）
systemctl restart audio-fidelity-engine                     # 平滑重启（等当前任务结束），约 30 s 后就绪
systemctl restart audio-fidelity-worker                     # 平滑重启（等当前任务结束）
curl -s 127.0.0.1:8896/api/health                           # db/redis
```

- 管理员接口：`/api/admin/health`（引擎、模型、讯飞配置、磁盘）、`/api/admin/stats`（成功率、推理耗时均值/P50/P95）、`/api/admin/inference-runs?format=csv`。
- 单个任务的引擎日志：评测 `data/runs/<job_id>/<attempt_id>.log`，原音频准备 `data/cache-logs/<job_id>/<attempt_id>.log`。失败原因全文对管理员在网页任务详情中可见。
- API 进程每 60 秒核对：补投 `pending_enqueue`、重投 Redis 中丢失的排队任务、关闭 Worker 已失联的 `running` 任务（埋点记 `abandoned`，耗时保持空值）、清理过期会话/上传/结果。Worker 启动时把遗留的 `running` 尝试标记失败，可在网页点“重试”。
- 备份：`data/app.sqlite3`（建议 `sqlite3 data/app.sqlite3 ".backup ..."`）与 `data/reference_cache/`；Redis 不是业务历史的唯一存储。

## 9. 已知限制

- 常驻进程单并发（一次一个任务），systemd `MemoryMax=4G`；任务超时或进程重启后约 30 s 不可用，期间的任务排队等待。
- 网站按**转码后的 WAV**建缓存（`reference_cache/v1`），与引擎目录 `tmp/reference_cache` 中按原始 mp3 建的旧缓存不通用，素材导入时会重新转写一次。
- 时长低于 `MIN_AUDIO_SECONDS`（2 秒）的素材会被拒绝（如 `L1.5.1_07.mp3` 1.9 秒）；如需收录可调低该值。
