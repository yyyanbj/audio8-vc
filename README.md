# Audio8-VC (`a8vc`)

基于 **Edge0-AI** 开源 0.1B 轻量语音模型（[`Edge0/Audio8-TTS-Preview-0.1b`](https://huggingface.co/Edge0/Audio8-TTS-Preview-0.1b) 与官方低占用 ONNX INT8 运行时 [`Edge0/audio8-TTS-0.1B-ONNX-INT8`](https://huggingface.co/Edge0/audio8-TTS-0.1B-ONNX-INT8)）构建的**高推理效率、低资源占用、支持独立 `.a8voice` 音色文件**的实时变声器项目。

## 为什么采用双轨架构？

Edge0-AI 发布的 `Audio8-TTS-0.1B` 原生架构是 **Falcon-H1 混合主干（Mamba-2 SSM + Attention）+ DualAR（Slow AR + Fast AR）+ 10-Codebook 44.1kHz Codec**，原生支持基于参考音频与文本的零样本 TTS 音色克隆（ONNX INT8 纯 CPU 内存仅约 **0.4 GiB**）。

为兼顾**开箱即用**与**后续帧级原生低延迟变声**，本项目采用解耦的双轨架构：

- **Track 1（已落地，零训练开箱即用）：`CascadedVCEngine`**
  - 流水线：`麦克风 / PCM 流 → 轻量 VAD 切分 → 轻量流式/低延迟 ASR (sherpa-onnx) → Audio8-0.1B ONNX INT8 流式合成 → 44.1kHz PCM 输出`
  - **关键低延迟优化 — 内存级前缀 KV/SSM 状态缓存**：Audio8 的 Prompt 结构中，系统提示与参考音色编码（`system` turn）严格位于目标文本（`user` turn）之前。本引擎在加载 `.a8voice` 音色文件时仅需执行一次参考前缀 Prefill，并在内存中缓存截断切片后的 KV Cache（`[:prefix_len]`）与 Mamba `conv_states`/`ssm_states`（每个音色仅约 11 MB RAM），后续每次实时发声跳过 80%+ 的 Prefill 步骤，同时保持 `.a8voice` 文件本身极其轻巧（~5 KB/音色）。
  - **总内存占用**：CPU 模式下约 **0.6 GiB RAM**（无 PyTorch / Transformers 运行时依赖）。
- **Track 2（预留微调工作区）：`CodecS2SEngine` + `training/`**
  - 基于 Audio8 的 10-Codebook 声学编码器（Codebook 0 语义码 + Codebook 1–9 声学残差码），在 `src/a8vc/engine/codec_s2s.py` 与 `training/` 预留原生端到端 Speech-to-Speech（S2S）微调与推理接口，便于在 GPU 环境（如多卡 H200）对 `Edge0/Audio8-TTS-Preview-0.1b` 进行语音转换 SFT 微调，实现保留原声韵律的帧级变声。

## `.a8voice` 单文件音色包规范

官方 0.1B ONNX 运行时通过目录中的 `codes.npy` + `meta.json` 管理音色。本项目将其扩展并封装为单文件便携格式 **`.a8voice`**（标准 ZIP 容器，严禁 Pickle，内置 SHA-256 校验与模型指纹）：

```text
my_voice.a8voice (典型体积 ~3–8 KB)
├── manifest.json   # schema_version, voice_id, name, language, tags, reference_text,
│                   # num_codebooks(10), codebook_size(4096), sample_rate(44100),
│                   # codes_sha256, source_sha256, model_fingerprint, created_at
└── codes.npy       # shape=[10, T], dtype=uint16, 44.1kHz 编码后的离散音色码
```

同时支持与官方 `Audio8_TTS` ONNX 目录格式双向无损互转（`a8vc voice import-official` / `export-official`）。

## 目录结构

```text
.
├── configs/default.toml          # 默认推理、音频、VAD、ASR、服务配置
├── scripts/download_models.py    # 一键下载 Audio8-0.1B ONNX INT8 与可选 ASR/VAD 模型
├── src/a8vc/
│   ├── paths.py                  # 源码 / 便携模式 / Windows 冻结安装目录自动解析
│   ├── models_manager.py         # 模型状态检查与后台断点/镜像一键下载管理
│   ├── desktop.py                # Windows / 跨平台原生桌面客户端启动器 (WebView2 + 本地服务)
│   ├── audio/                    # 重采样/2048对齐/PCM编解码、轻量 VAD、sounddevice 本地音频直通
│   ├── voices/                   # .a8voice 单文件格式、音色库管理、一键音色提取(含可选 ASR 自动转写)
│   ├── asr/                      # 可插拔轻量 ASR (sherpa-onnx SenseVoice/Zipformer/Paraformer + Mock)
│   ├── runtime/                  # Audio8-0.1B ONNX INT8 引擎、拆分式 Prompt 构造、Gumbel-max 采样、KV/SSM 前缀缓存
│   ├── engine/                   # 统一 VoiceConversionEngine 抽象、Track 1 级联流式引擎、本机实时会话控制器、Track 2 Codec S2S 骨架
│   ├── server/                   # FastAPI HTTP/WebSocket 流式服务 + 内置桌面/Web 控制台
│   └── cli.py                    # 统一 CLI (a8vc)
├── packaging/windows/            # Windows 一键打包（PyInstaller 6 + Inno Setup 6 安装包 + 便携版 ZIP）
├── .github/workflows/            # GitHub Actions Windows 自动构建与 Release 发布流水线
├── training/                     # Track 2 原生 S2S Codec 变声微调脚手架与配置模板
└── tests/                        # 单元测试与无权重端到端测试
```

## 快速开始

### 1. 安装依赖（推荐使用 `uv`）

```bash
# 基础运行环境（ONNX Runtime + FastAPI 服务 + 音色包管理）
uv sync --extra dev

# 完整实时变声环境（增加本地音频直通 sounddevice、sherpa-onnx ASR、模型下载工具）
uv sync --extra all --extra dev
```

### 2. 下载模型权重

```bash
# 下载 Audio8-0.1B ONNX INT8 主模型 + registration 音色编码器
uv run python scripts/download_models.py --audio8-onnx

# （可选）同时下载轻量中英双语 SenseVoice INT8 ASR 与 Silero VAD
uv run python scripts/download_models.py --audio8-onnx --asr-sensevoice --silero-vad
```

### 3. 开箱即用的内置开源音色 & 自定义 `.a8voice` 音色制作

项目内置了 **15 款精选开源 `.a8voice` 音色包**（总体积仅约 **47 KB**，涵盖二次元角色、游戏角色、中文多风格与英文榜首音色，均源自 OpenBMB/VoxCPM、FunAudioLLM/CosyVoice、hexgrad/Kokoro-82M、kyutai/tts-voices、Qwen3-TTS ITA-Corpus 与 Edge0-AI/Audio8_TTS 等开源项目，详见 [`src/a8vc/voices/builtins/ATTRIBUTION.md`](src/a8vc/voices/builtins/ATTRIBUTION.md)）。

- 首次启动 `a8vc serve` 或 Windows 桌面客户端（`Audio8-VC.exe`）时，内置音色会自动初始化到 `voices/` 目录；也可在 Web/桌面界面「内置开源音色库」页签按分类筛选、一键安装或恢复。
- 命令行管理内置音色与自定义音色：

```bash
# 查看内置开源音色目录并安装到当前音色库
uv run a8vc voice builtins
uv run a8vc voice install-builtins

# 从 3~15 秒人声参考音频一键打包生成自定义 .a8voice 音色文件
uv run a8vc voice pack \
  --audio samples/target_speaker.wav \
  --id anime_girl \
  --name "二次元少女" \
  --text "你好，欢迎使用实时变声器。" \
  --lang zh

# 若已配置 ASR，可省略 --text 自动转写参考文本
uv run a8vc voice pack --audio samples/target_speaker.wav --id custom_voice --auto-asr

# 兼容导入官方 Audio8 ONNX 音色目录（含 codes.npy + meta.json）
uv run a8vc voice import-official /path/to/official/voices/default --id default

# 查看 / 校验音色库
uv run a8vc voice list
uv run a8vc voice inspect voxcpm_tsundere
uv run a8vc voice verify voices/voxcpm_tsundere.a8voice
```

### 4. 启动实时变声

```bash
# 方式 A：本地音频设备实时直通（麦克风 → 扬声器 / 虚拟声卡，支持终端输入命令热切音色）
uv run a8vc devices
uv run a8vc live --voice anime_girl

# 方式 B：启动 HTTP + WebSocket 实时变声服务与内置 Web 控制台
uv run a8vc serve --host 127.0.0.1 --port 8025
# 浏览器打开 http://127.0.0.1:8025 即可体验浏览器麦克风实时变声、一键热切音色与音色打包

# 方式 C：单句快速试听 & 延迟基准测试（对比冷启动与前缀缓存命中耗时）
uv run a8vc synth --voice anime_girl --text "今天天气真不错！" --out outputs/demo.wav
uv run a8vc benchmark --voice anime_girl --runs 3

# 方式 D：直接启动原生桌面客户端窗口（WebView2，不可用时自动回退浏览器）
uv run a8vc desktop
```

## Windows 可安装软件打包（安装向导 `.exe` + 免安装便携版 `.zip`）

项目已内置完整的 **Windows x64 桌面应用打包流水线**（PyInstaller 6 `onedir` + Inno Setup 6 安装程序），产物同时包含：

- `Audio8-VC.exe`：无黑框原生桌面窗口（基于系统内置 Microsoft Edge WebView2，支持免安装运行、应用内一键下载模型含国内镜像、本机麦克风与 VB-Cable / VoiceMeeter 虚拟声卡直通、热切换 `.a8voice` 音色、打开数据目录）
- `a8vc.exe`：共享同一 `_internal/` 运行时的命令行工具（开始菜单提供 `Audio8-VC (CLI Prompt)` 快捷方式）
- `.a8voice` 文件关联：安装后在资源管理器中**双击 `.a8voice` 音色文件**即可自动导入并激活
- 无需管理员权限：默认按当前用户安装至 `%LOCALAPPDATA%\Programs\Audio8-VC`（也可在向导中选择为所有用户安装），用户数据、模型与日志存放于 `%LOCALAPPDATA%\Audio8-VC`；便携版 ZIP 根目录含 `portable.flag`，所有数据保存在同级 `.\data`

### 方式 1：在 Windows 本机一键打包

前提：Windows 10/11 x64，已安装 Python 3.10+（推荐安装 `uv`），如需生成安装向导 `.exe` 请安装 [Inno Setup 6](https://jrsoftware.org/isdl.php)（未安装时仍会自动产出便携版 `.zip`）。

```powershell
# 1. 标准 CPU 版（安装包 + 便携版 ZIP，首次运行在界面一键下载模型，安装包轻量）
.\packaging\windows\build_windows.ps1
# 或直接双击 packaging\windows\build_windows.bat

# 2. DirectML GPU 通用加速版（推荐：支持 NVIDIA / AMD / Intel 显卡，免装 CUDA，自动启用 dml）
.\packaging\windows\build_windows.ps1 -Provider dml

# 3. NVIDIA CUDA GPU 加速版（onnxruntime-gpu，自动探测系统 CUDA 12 / cuDNN 9 与 cuda_libs）
.\packaging\windows\build_windows.ps1 -Provider cuda
# 如需在本地把 NVIDIA CUDA 12 + cuDNN 9 运行库一并打进自包含包，可追加 -BundleCudaRuntime

# 4. 离线全量版（打包时预置 Audio8-0.1B + SenseVoice ASR + Silero VAD 模型，开箱断网可用）
.\packaging\windows\build_windows.ps1 -Provider dml -BundleModels -UseMirror
```

构建产物输出至 `dist\release\`：
- `Audio8-VC-<version>-win64-<variant>-setup.exe`（Windows 安装向导程序，`<variant>` 为 `cpu` / `dml` / `cuda`）
- `Audio8-VC-<version>-win64-<variant>-portable.zip`（解压即用的免安装便携版）
- `SHA256SUMS.txt`（校验和）

### 方式 2：在 Linux / macOS 下通过 GitHub Actions 云端自动构建 Windows 安装包

仓库已配置 [`.github/workflows/build-windows.yml`](.github/workflows/build-windows.yml)：

- 推送版本标签（如 `git tag v0.1.0 && git push origin v0.1.0`）自动在 `windows-latest` 并行构建 **CPU + DirectML GPU (`dml`) + NVIDIA CUDA GPU (`cuda`)** 三个版本，完成单元测试、PyInstaller 打包、Inno Setup 编译、静默安装/Provider 校验/卸载烟雾测试，并将全部安装包、便携 ZIP 与合并后的 `SHA256SUMS.txt` 发布至 GitHub Releases。
- 也可在 GitHub Actions 页面手动触发 `Build Windows Installer`（支持选择 `all` / `cpu` / `dml` / `cuda` 与是否预打包离线模型 `bundle_models`）并直接下载构建产物。
- 运行 `a8vc info` 可随时查看当前加速后端、可用 ONNX Execution Providers 与 CUDA 运行库探测结果。

### 在 Windows 游戏 / 语音软件中接入实时变声

1. 安装免费虚拟声卡驱动（如 **VB-CABLE** 或 **VoiceMeeter**）。
2. 在 Audio8-VC 桌面端「**本机声卡 / 虚拟声卡直通**」页签中：
   - 输入麦克风：选择你的物理麦克风
   - 输出设备：选择带 `[推荐:虚拟声卡]` 标签的 **`CABLE Input (VB-Audio Virtual Cable)`**
3. 在 Discord / OBS / QQ / 微信 / 游戏语音设置中，将麦克风输入设为 **`CABLE Output (VB-Audio Virtual Cable)`** 即可实时输出变声人声。

## 许可证与合规提示

- 本项目自身代码采用 **Apache-2.0** 协议。
- 上游 **Edge0-AI Audio8-TTS-0.1B 模型权重（含 ONNX INT8）** 采用 **Audio8 Community License v1.0**（年营收 200 万美元以下可免费商用，超过需另行授权），请在使用与分发模型权重前查阅 [Edge0-AI/Audio8_TTS](https://github.com/Edge0-AI/Audio8_TTS) 官方许可条款。
- 请勿在未经本人授权的情况下克隆他人声音或用于欺诈、冒充等用途。
