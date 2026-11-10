# Linux 使用说明

支持 Linux x86_64，Python 3.12 或 3.13，默认使用 CPU。推荐 Ubuntu 24.04 及以上。
保留“先缓存原音频、再评测”的流程；Windows bat 入口仍可使用。

## 准备项目

把项目源码完整复制到 Linux，进入项目目录。不要直接使用 Windows 的 `.venv`。
Linux 脚本会独立创建 `.venv-linux`。

```bash
cd /path/to/音频还原度测试
bash linux.sh setup --system-deps
```

`--system-deps` 在 Debian/Ubuntu 上安装 eSpeak NG、音频库和中文字体，需要 root 或 sudo。
其他发行版请先用包管理器安装这些系统组件，再执行 `bash linux.sh setup`。
安装会创建虚拟环境、安装固定版本依赖、应用重音推理兼容补丁，并下载缺失的模型。
如果已复制 `models` 目录，会直接复用有效文件。音素模型约 1.26 GB；完整安装建议预留至少 5 GB。

如果使用已有 Python 3.12/3.13 的完整路径：

```bash
PYTHON_BIN=/opt/python3.13/bin/python3.13 bash linux.sh setup --system-deps
```

Ubuntu 22.04 默认 Python 3.10 不满足依赖要求，需要先自行准备 Python 3.12+。
脚本会明确报错，不会替换系统 Python，也不会自动添加第三方系统软件源。

把自己的 `讯飞配置.json` 放在项目根目录，包含 `app_id` 和 `secret_key`。
配置示例文件为 `讯飞配置.example.json`；请填写自己的服务凭据，不要把真实配置公开。

## 缓存与评测

```bash
# 缓存文件夹；加 --recursive 可包含子目录
bash linux.sh cache "/data/范本" --recursive

# 或只缓存一个范本
bash linux.sh cache "/data/范本/lesson 01.wav"

# 用任意两个路径评测；原音频必须已经缓存
bash linux.sh evaluate --reference "/data/范本/lesson 01.wav" --imitation "/data/跟读/我的录音.wav"

# 指定一个新的空输出目录
bash linux.sh evaluate "/data/范本.wav" "/data/跟读.wav" --output "/data/报告/本次"
```

默认只读取输入文件，不需要重命名或搬动音频。支持 wav/mp3/flac/ogg；m4a/aac 请先使用 FFmpeg 转换。
批量缓存默认自动采用转写文本，适用于服务器和无图形界面环境。
需要人工确认时可增加 `--confirm-text`：Linux 桌面使用 xdg-open，SSH 终端可设置 `EDITOR=nano` 或 `EDITOR=vim`。
没有编辑器或交互终端时会明确提示，不会静默确认。

缓存按音频内容保存至 `tmp/reference_cache`，已缓存文件会跳过。
同名不同内容的音频不会串用缓存。迁移到 Linux 后，可复用带内容哈希的旧缓存。
报告和日志默认保存在 `output/runs/时间_编号`；批量缓存清单保存在 `output/cache_batches`。

## 原来的固定目录流程

```bash
# 范本：audio/原音频/原音频.wav；跟读：audio/模仿.wav
bash linux.sh analyze --auto
bash linux.sh run
```

## 环境检查、模型与回归检查

```bash
bash linux.sh doctor
bash linux.sh models
bash linux.sh models --punctuation  # 可选标点恢复模型
bash linux.sh test
```

只检查安装前提、不安装依赖：`bash linux.sh setup --check`。
只安装 Python 环境、暂不下载模型：`bash linux.sh setup --skip-models`。
标点恢复模型未安装时，会沿用转写原标点，不会阻断评测。
服务器绘图使用 Agg 后端；已提供中文字体回退，系统依赖安装会包含 Noto CJK 字体。

可用 `AUDIO_TEST_VENV=/path/to/venv` 指定 Linux 虚拟环境；相对路径以项目目录为基准。
模型或服务请求需要可访问 Hugging Face/讯飞；有自己的合法镜像时可通过 `HF_ENDPOINT` 设置。

官方参考：[PyTorch 安装](https://docs.pytorch.org/get-started/locally/)、[eSpeak NG Linux 安装](https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md#linux)。

## CPU 使用限制

Windows 和 Linux 的缓存、评测入口会自动设置 CPU 亲和性。
最多使用 3 个逻辑核心；首次启动按当前可用逻辑核心数减 1 计算，最少使用 1 个：

| 当前可用逻辑核心 | 本程序最多使用 |
| --- | --- |
| 4 个及以上 | 3 个 |
| 3 个 | 2 个 |
| 2 个 | 1 个 |
| 1 个 | 1 个，无法留出空闲核心 |

同一流水线的子进程继承同一组核心，避免每个阶段各占一组核心。
同时限制 PyTorch、OpenMP、BLAS 等计算库的工作线程，启动日志会显示核心编号。
这里限制的是可执行计算的逻辑核心数，任务管理器中的进程总线程数仍可能大于 3。
留出的核心不供本程序计算，但其他程序和系统仍可使用；不保证其始终处于空闲状态。
例如共 4 个逻辑核心时，本程序满载占整机 CPU 的理论上限约为 75%。

可将计算线程进一步降低，但不能通过该变量突破核心上限：

```bash
AUDIO_TEST_CPU_THREADS=1 bash linux.sh evaluate /data/ref.wav /data/mine.wav
```

Windows PowerShell 示例：`$env:AUDIO_TEST_CPU_THREADS='1'`，再启动评测。
限制从下次启动生效。若手动同时开启多条独立流水线，各自受限，不构成整机的统一资源配额。
