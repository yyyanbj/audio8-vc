"""Parser and path resolver for Audio8-0.1B ONNX `runtime_manifest.json`."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RuntimeManifest:
    """Validated runtime parameters from `Edge0/audio8-TTS-0.1B-ONNX-INT8`."""

    model_dir: Path
    slow_model_path: Path
    fast_model_path: Path
    codec_decoder_path: Path
    tokenizer_path: Path
    num_layers: int = 24
    num_fast_layers: int = 4
    n_local_heads: int = 2
    max_seq_len: int = 2048
    head_dim: int = 64
    fast_n_local_heads: int = 2
    fast_head_dim: int = 64
    mamba_d_conv: int = 4
    mamba_d_ssm: int = 768
    mamba_n_heads: int = 24
    mamba_n_groups: int = 1
    mamba_d_head: int = 32
    mamba_d_state: int = 64
    conv_channels: int = 896
    semantic_begin_id: int = 65537
    semantic_end_id: int = 69632
    im_end_id: int = 4096
    num_codebooks: int = 10
    codebook_size: int = 4096
    slow_logits_layout: str = "relative_semantic_then_eos"
    sample_rate: int = 44100
    codec_hop_length: int = 2048
    stream_context_frames: int = 128
    stream_guard_frames: int = 1
    model_fingerprint: str = ""
    raw: dict[str, Any] | None = None


def _resolve_model_entry(
    raw: dict[str, Any],
    *,
    mapping_key: str,
    single_key: str,
    precision: str,
    fallback_name: str,
) -> str:
    table = raw.get(mapping_key)
    if isinstance(table, dict):
        if precision in table:
            return str(table[precision])
        if table:
            return str(next(iter(table.values())))
    elif isinstance(table, str) and table.strip():
        return table.strip()
    single = raw.get(single_key)
    if single:
        return str(single)
    return fallback_name


def load_runtime_manifest(
    model_dir: str | Path,
    *,
    precision: str | None = None,
    codec_precision: str | None = None,
) -> RuntimeManifest:
    """Load `runtime_manifest.json` and resolve ONNX & tokenizer file paths."""
    root = Path(model_dir)
    manifest_path = root / "runtime_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Audio8 runtime_manifest.json not found in {root}. "
            "Run `python scripts/download_models.py --audio8-onnx` first."
        )
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))

    chosen_prec = precision or str(raw.get("default_precision", "int8"))
    chosen_codec = codec_precision or str(raw.get("default_codec_precision", "fp16"))

    slow_rel = _resolve_model_entry(
        raw,
        mapping_key="slow_decode_models",
        single_key="slow_decode_model",
        precision=chosen_prec,
        fallback_name=f"slow_ar_{chosen_prec}.onnx",
    )
    fast_rel = _resolve_model_entry(
        raw,
        mapping_key="fast_models",
        single_key="fast_model",
        precision=chosen_prec,
        fallback_name=f"fast_ar_{chosen_prec}.onnx",
    )
    codec_rel = _resolve_model_entry(
        raw,
        mapping_key="codec_models",
        single_key="codec_model",
        precision=chosen_codec,
        fallback_name=f"codec_decoder_{chosen_codec}.onnx",
    )

    tokenizer_rel = str(raw.get("tokenizer_file", "tokenizer/tokenizer.json"))
    tokenizer_path = root / tokenizer_rel
    if not tokenizer_path.is_file():
        alt = root / "tokenizer.json"
        if alt.is_file():
            tokenizer_path = alt

    mamba_d_ssm = int(raw.get("mamba_d_ssm", 768))
    mamba_n_groups = int(raw.get("mamba_n_groups", 1))
    mamba_d_state = int(raw.get("mamba_d_state", 64))
    default_conv_channels = mamba_d_ssm + 2 * mamba_n_groups * mamba_d_state

    return RuntimeManifest(
        model_dir=root,
        slow_model_path=root / slow_rel,
        fast_model_path=root / fast_rel,
        codec_decoder_path=root / codec_rel,
        tokenizer_path=tokenizer_path,
        num_layers=int(raw.get("num_layers", 24)),
        num_fast_layers=int(raw.get("num_fast_layers", 4)),
        n_local_heads=int(raw.get("n_local_heads", 2)),
        max_seq_len=int(raw.get("max_seq_len", 2048)),
        head_dim=int(raw.get("head_dim", 64)),
        fast_n_local_heads=int(raw.get("fast_n_local_heads", 2)),
        fast_head_dim=int(raw.get("fast_head_dim", 64)),
        mamba_d_conv=int(raw.get("mamba_d_conv", 4)),
        mamba_d_ssm=mamba_d_ssm,
        mamba_n_heads=int(raw.get("mamba_n_heads", 24)),
        mamba_n_groups=mamba_n_groups,
        mamba_d_head=int(raw.get("mamba_d_head", 32)),
        mamba_d_state=mamba_d_state,
        conv_channels=int(raw.get("conv_channels", default_conv_channels)),
        semantic_begin_id=int(raw.get("semantic_begin_id", 65537)),
        semantic_end_id=int(raw.get("semantic_end_id", 69632)),
        im_end_id=int(raw.get("im_end_id", 4096)),
        num_codebooks=int(raw.get("num_codebooks", 10)),
        codebook_size=int(raw.get("codebook_size", 4096)),
        slow_logits_layout=str(raw.get("slow_logits_layout", "relative_semantic_then_eos")),
        sample_rate=int(raw.get("sample_rate", raw.get("codec_sample_rate", 44100))),
        codec_hop_length=int(raw.get("codec_hop_length", raw.get("codec_frame_size", 2048))),
        stream_context_frames=int(raw.get("stream_context_frames", 128)),
        stream_guard_frames=int(raw.get("stream_guard_frames", 1)),
        model_fingerprint=str(raw.get("model_fingerprint", "")),
        raw=raw,
    )
