# Track 2: 基于 Audio8-0.1B Codec 空间的原生 S2S 变声微调路线

本目录面向 **Track 2**（原生端到端 Speech-to-Speech 变声），利用当前 GPU 环境（如多卡 NVIDIA H200）在 [`Edge0/Audio8-TTS-Preview-0.1b`](https://huggingface.co/Edge0/Audio8-TTS-Preview-0.1b) 上进行微调，使模型直接从**源语音语义码 + 目标 `.a8voice` 音色文件**生成目标音色的声学码，彻底移除 ASR 依赖并保留原声韵律与停顿。

## 模型结构切入点

`Edge0/Audio8-TTS-Preview-0.1b` 的核心组成：

1. **Codec (`ArkttsCodec` / `codec.pth`, 44.1 kHz, hop=2048 ≈ 21.53 Hz)**：将波形编码为 `[10, T]` 离散码本：
   - **Codebook 0（Semantic）**：高层语义/音素 token（映射到词表 `[65537, 69632]`）。
   - **Codebooks 1–9（Acoustic Residuals）**：细节声学与说话人音色残差。
2. **Slow AR（24 层 Falcon-H1 Mamba-2 SSM + Attention, hidden=512）**：处理多模态序列（系统音色参考 + 条件输入），输出每帧隐状态 `hidden` 与语义码预测。
3. **Fast AR（4 层 Transformer, intermediate=4864）**：以 `slow_hidden` 和 Codebook 0 为条件，在单帧内自回归预测 Codebooks 1–9。

## S2S 变声微调核心思路

将训练目标从「`Target Text + Reference Voice Codes -> Target [10, T]`」改造为「`Source Semantic Codes (with speaker-perturbation) + Reference Voice Codes -> Target [10, T]`」：

1. **数据自重构（Self-Reconstruction with Timbre Perturbation）**：
   - 不需要昂贵的跨说话人平行语料！
   - 对同一说话人的任意两条干净语音 `(utt_ref, utt_target)`：
     - `utt_ref` 作为目标音色提示（即 `.a8voice` 中的 `[10, T_ref]`）。
     - 对 `utt_target` 施加**音色扰动**（基频随机移调 ±3~6 半音 + 共振峰偏移/声道长度扰动 VTLP，但保持时间长度与语义不变），再经过 `codec_encoder` 提取扰动后的 **Codebook 0 语义流** `src_sem [1, T_tgt]`。
     - 监督目标：还原**未扰动**的 `utt_target` 完整 10 路编码 `tgt_codes [10, T_tgt]`！
2. **为什么有效**：
   - 因为输入语义流来自变调/变共振峰后的“伪源说话人”，模型必须忽略源音色残留，完全依赖 `utt_ref`（目标 `.a8voice`）恢复目标说话人的共振峰与音色，同时逐帧对齐 `src_sem` 的节奏与语调！
3. **训练与导出步骤**：
   - 第一步：下载 PyTorch 基座权重：
     ```bash
     uv sync --extra train --extra download
     uv run python scripts/download_models.py --audio8-pt
     ```
   - 第二步：使用 `training/prepare_vc_pairs.py` 预编码多说话人语音为 `[10, T]` 样本清单与 `.a8voice` 目标音色对。
   - 第三步：参考 `training/configs/s2s_vc_0_1b.yaml`（基于上游 `Edge0-AI/Audio8_TTS` 的 `audio8_tts_sft.py` + DeepSpeed ZeRO-2，`transformers>=4.57.0,<5`）启动 SFT；随后导出为 ONNX INT8 并接入 `src/a8vc/engine/codec_s2s.py`。

> **注意**：上游要求 `transformers>=4.57.0,<5`（版本不匹配会导致全零输出 token），且 0.1B 主干支持 `GRADIENT_CHECKPOINTING=true` 与可选 `mamba-ssm`/`causal-conv1d` CUDA 内核加速。
