# xhxs-audio-fidelity

音频还原度测试 Demo 分发版：英语跟读的音素识别评测引擎 + 网页前后端。

| 目录 | 内容 | 说明文档 |
| --- | --- | --- |
| `engine/` | 评测引擎（Linux 分发包 `audio-fidelity-linux`，release 2026-11-03）：Wav2Vec2 音素识别、强制对齐、节奏/重音（emphases）、综合评分与报告；原音频转写（讯飞） | `engine/README.md`、`engine/Linux使用说明.md`、`engine/动态路径使用说明.md` |
| `web/` | 网页 Demo：FastAPI + 原生 JS 前端、Redis/RQ 队列、常驻推理进程、讯飞/通义听悟转写切换、邀请码账号、管理员功能、部署脚本 | `web/README.md`、`web/docs/DEPLOY.md`、`web/docs/API.md` |

## 参与开发

先读 [docs/开发规范.md](docs/开发规范.md)：从 `develop` 切出 `feature/*` 分支开发，PR 合回 `develop`；只有 `develop` 和 `hotfix/*` 能合入 `main`。AI 智能体的入口说明见 `AGENTS.md` / `CLAUDE.md`。

## 仓库里没有的东西

- **模型**（音素模型约 1.26 GB，另有重音、标点模型；完整环境建议预留 5 GB）：`cd engine && bash linux.sh setup --system-deps` 安装依赖后会调用 `scripts/download_models.py` 下载到 `engine/models/`。
- **转写服务凭据**：`engine/讯飞配置.json`、`engine/听悟配置.json`。格式见 `engine/讯飞配置.example.json`、`web/deploy/听悟配置.example.json`。
- **网站环境变量**：`web/.env`，从 `web/.env.example` 复制后修改。
- 录音素材、原音频缓存、评测输出、Python 虚拟环境、网站数据目录。

以上路径都已写进 `.gitignore`。

## 两部分如何配合

网站不修改引擎源码，只通过环境变量 `ENGINE_ROOT` 指向引擎目录（例如本仓库的 `engine/`），用引擎自己的虚拟环境 `engine/.venv-linux` 运行推理：

1. 按 `engine/Linux使用说明.md` 在目标机器上装好引擎（依赖、模型），`bash linux.sh check` 通过。
2. 按 `web/docs/DEPLOY.md` 部署网站：API 和 Redis 在 Docker 里，Worker 和常驻推理进程（systemd）在宿主机上。
3. 本地开发不需要模型：`web/tests/fake_engine/make_fake_engine.sh engine <目标目录>` 生成一个把重计算换成桩的测试引擎，`web/tests/e2e_check.py` 跑端到端检查。
