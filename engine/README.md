# 音频还原度测试：Linux 分发包

面向 Linux x86_64，Python 3.12/3.13，默认 CPU 推理。
这是轻量源码包，不含模型、预装依赖、录音、缓存或真实服务密钥。

## 开始使用

解压后进入 audio-fidelity-linux 文件夹：

```bash
# Ubuntu 24.04+；在目标 Linux 机器上安装
bash linux.sh setup --system-deps

# 复制配置模板，填写自己的 app_id 和 secret_key
cp 讯飞配置.example.json 讯飞配置.json
# 用编辑器打开并填写后再运行下面的命令

# 先批量缓存原音频
bash linux.sh cache "/data/原音频目录" --recursive

# 再评测指定的两段音频
bash linux.sh evaluate --reference "/data/范本.wav" --imitation "/data/跟读.wav"
```

首次安装会联网下载 Python 依赖和约 1.26 GB 的音素模型。
完整环境建议预留至少 5 GB；Ubuntu 22.04 默认 Python 3.10 不满足要求，需另行准备 3.12+。
现有模型可直接复制至 models/，完整有效的模型会复用。
支持 wav/mp3/flac/ogg；m4a/aac 请先转换。

默认报告位于 output/runs/时间_编号/，终端会打印完整报告路径。
缓存按原音频内容区分，源文件不需要重命名或搬动。

缓存和评测自动限制为最多使用 3 个逻辑核心，并在可用核心数允许时至少留出 1 个。
可用核心只有 3 个时用 2 个，只有 2 个时用 1 个；单核环境无法留出空闲核心。
子进程继承同一组 CPU 亲和性；这不要求进程总线程数小于等于 3。
日志显示核心编号，可用 AUDIO_TEST_CPU_THREADS=1 进一步降低计算库并行度。

更多命令：`bash linux.sh help`。详细说明见 Linux使用说明.md。
Windows 的虚拟环境不能直接用于 Linux；请在目标机器重新创建环境。

## 验证范围

此前已完成 Bash 语法与部分 Linux 路径检查。
完整 Linux 依赖安装/端到端推理验证未完成；本次仅打包，没有继续安装或运行测试。
包内保留 tests/，供需要时在目标 Linux 机器执行。

## 包内文件

- linux.sh：安装、模型准备、缓存、评测和环境检查入口。
- requirements-linux.txt：固定版本的 Linux CPU 依赖清单。
- scripts/、evaluate.py、cache_audio.py：完整程序源码。
- tmp/patch_emphases.py：重音推理兼容补丁。
- 讯飞配置.example.json：空白配置模板，不包含任何真实凭据。
- BUILD_INFO.json：文件清单及 SHA256。
