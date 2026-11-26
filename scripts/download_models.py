#!/usr/bin/env python3
"""Download model weights for Audio8-VC.

Supports:
- Audio8-0.1B ONNX INT8 runtime + registration voice encoder (Edge0/audio8-TTS-0.1B-ONNX-INT8)
- Audio8-0.1B PyTorch base model for Track 2 SFT (Edge0/Audio8-TTS-Preview-0.1b)
- sherpa-onnx SenseVoice INT8 multilingual ASR (zh/en/ja/ko/yue)
- Silero VAD ONNX (~2 MB)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running directly as `python scripts/download_models.py` without editable install
_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from a8vc.models_manager import (  # noqa: E402
    AUDIO8_ONNX_REPO,
    AUDIO8_PT_REPO,
    HF_MIRROR_ENDPOINT,
    download_hf_snapshot,
    download_sensevoice_asr,
    download_silero_vad,
)


def _print_license_notice() -> None:
    print(
        "\n[Notice] Edge0-AI Audio8-TTS-0.1B weights (including ONNX INT8) are licensed under\n"
        "         Audio8 Community License v1.0 (royalty-free under US$2M annual revenue).\n"
        "         See https://github.com/Edge0-AI/Audio8_TTS for full terms.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Audio8-VC runtime and training models")
    parser.add_argument(
        "--audio8-onnx",
        action="store_true",
        help="Download Edge0/audio8-TTS-0.1B-ONNX-INT8 (includes registration encoder)",
    )
    parser.add_argument(
        "--no-registration",
        action="store_true",
        help="Skip optional registration/codec_encoder_fp16.onnx when downloading --audio8-onnx",
    )
    parser.add_argument(
        "--audio8-pt",
        action="store_true",
        help="Download Edge0/Audio8-TTS-Preview-0.1b PyTorch checkpoint (for Track 2 SFT)",
    )
    parser.add_argument(
        "--asr-sensevoice",
        action="store_true",
        help="Download sherpa-onnx SenseVoice INT8 bilingual ASR model",
    )
    parser.add_argument(
        "--silero-vad",
        action="store_true",
        help="Download Silero VAD ONNX (~2 MB)",
    )
    parser.add_argument(
        "--onnx-repo",
        default=AUDIO8_ONNX_REPO,
        help=f"Override HF repo ID for ONNX INT8 model (default: {AUDIO8_ONNX_REPO})",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help=f"Use Hugging Face mirror ({HF_MIRROR_ENDPOINT}) for faster downloads in China",
    )
    parser.add_argument(
        "--hf-endpoint",
        default="",
        help="Custom Hugging Face endpoint URL (overrides --mirror)",
    )
    parser.add_argument(
        "--models-root",
        default="models",
        help="Root directory for downloaded models (default: models)",
    )
    args = parser.parse_args()

    if not any([args.audio8_onnx, args.audio8_pt, args.asr_sensevoice, args.silero_vad]):
        parser.print_help()
        print("\nTip: run with `--audio8-onnx --asr-sensevoice --silero-vad` for the full Track 1 stack.")
        return 0

    root = Path(args.models_root)
    endpoint = args.hf_endpoint.strip() or (HF_MIRROR_ENDPOINT if args.mirror else None)

    if args.audio8_onnx:
        _print_license_notice()
        ignore = ["registration/*"] if args.no_registration else None
        dest = root / "audio8-tts-0.1b-onnx-int8"
        print(f"Downloading Hugging Face repo `{args.onnx_repo}` -> {dest} ...")
        download_hf_snapshot(
            args.onnx_repo,
            dest,
            ignore_patterns=ignore,
            hf_endpoint=endpoint,
        )
        print(f"Done: {dest}")

    if args.audio8_pt:
        _print_license_notice()
        dest = root / "audio8-tts-preview-0.1b"
        print(f"Downloading Hugging Face repo `{AUDIO8_PT_REPO}` -> {dest} ...")
        download_hf_snapshot(AUDIO8_PT_REPO, dest, hf_endpoint=endpoint)
        print(f"Done: {dest}")

    if args.asr_sensevoice:
        asr_dir = root / "asr"
        print(f"Downloading & extracting SenseVoice INT8 ASR -> {asr_dir} ...")
        download_sensevoice_asr(asr_dir)
        print(f"ASR ready under {asr_dir}")

    if args.silero_vad:
        vad_path = root / "vad" / "silero_vad.onnx"
        print(f"Downloading Silero VAD ONNX -> {vad_path} ...")
        download_silero_vad(vad_path)
        print(f"Done: {vad_path} ({vad_path.stat().st_size / (1024 * 1024):.2f} MiB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
