"""Configuration schema and loader for Audio8-VC."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from a8vc.paths import (
    get_data_dir,
    get_default_config_path,
    is_frozen,
    is_portable_mode,
    resolve_readonly_or_writable_path,
)

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


class PathsConfig(BaseModel):
    model_dir: Path = Path("models/audio8-tts-0.1b-onnx-int8")
    voices_dir: Path = Path("voices")
    asr_model_dir: Path = Path("models/asr")
    vad_model_path: Path = Path("models/vad/silero_vad.onnx")


class RuntimeConfig(BaseModel):
    provider: Literal["cpu", "cuda", "coreml", "dml", "auto"] = "cpu"
    threads: int = Field(default=0, ge=0)
    precision: str = "int8"
    codec_precision: str = "fp16"
    prefix_cache_size: int = Field(default=2, ge=0, le=32)
    chunk_frames: int = Field(default=10, ge=2, le=64)
    stream_context_frames: int = Field(default=96, ge=16, le=256)
    stream_guard_frames: int = Field(default=1, ge=0, le=8)
    temperature: float = Field(default=0.7, gt=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1, le=4096)
    max_new_tokens: int = Field(default=512, ge=16, le=2048)


class AudioConfig(BaseModel):
    input_sample_rate: int = Field(default=16000, ge=8000, le=48000)
    output_sample_rate: int = Field(default=44100, ge=16000, le=48000)
    input_block_ms: int = Field(default=20, ge=10, le=100)
    input_device: str | int = ""
    output_device: str | int = ""
    crossfade_ms: int = Field(default=12, ge=0, le=100)


class VADConfig(BaseModel):
    backend: Literal["auto", "energy", "silero"] = "auto"
    energy_threshold_db: float = -38.0
    speech_start_ms: int = Field(default=80, ge=32, le=1000)
    min_silence_ms: int = Field(default=360, ge=100, le=3000)
    pre_speech_ms: int = Field(default=160, ge=0, le=1000)
    min_segment_ms: int = Field(default=250, ge=64, le=2000)
    max_segment_ms: int = Field(default=4500, ge=1000, le=20000)


class ASRConfig(BaseModel):
    backend: Literal["sherpa", "mock"] = "sherpa"
    model_type: Literal["sense_voice", "paraformer", "zipformer"] = "sense_voice"
    language: str = "auto"
    num_threads: int = Field(default=2, ge=1, le=32)
    provider: str = "cpu"


class EngineConfig(BaseModel):
    mode: Literal["cascaded", "codec_s2s"] = "cascaded"
    max_pending_segments: int = Field(default=2, ge=1, le=16)


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8025, ge=1, le=65535)
    default_voice: str = ""


class AppConfig(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    asr: ASRConfig = Field(default_factory=ASRConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _env_overrides() -> dict[str, Any]:
    """Read lightweight A8VC_* environment overrides."""
    overrides: dict[str, Any] = {}
    mapping = {
        "A8VC_MODEL_DIR": ("paths", "model_dir"),
        "A8VC_VOICES_DIR": ("paths", "voices_dir"),
        "A8VC_ASR_MODEL_DIR": ("paths", "asr_model_dir"),
        "A8VC_VAD_MODEL_PATH": ("paths", "vad_model_path"),
        "A8VC_PROVIDER": ("runtime", "provider"),
        "A8VC_THREADS": ("runtime", "threads"),
        "A8VC_ASR_BACKEND": ("asr", "backend"),
        "A8VC_ENGINE_MODE": ("engine", "mode"),
        "A8VC_HOST": ("server", "host"),
        "A8VC_PORT": ("server", "port"),
        "A8VC_DEFAULT_VOICE": ("server", "default_voice"),
    }
    for env_key, (section, field) in mapping.items():
        val = os.environ.get(env_key)
        if val is not None and val != "":
            overrides.setdefault(section, {})[field] = val
    return overrides


def load_config(
    config_path: str | Path | None = None,
    *,
    desktop: bool = False,
) -> AppConfig:
    """Load configuration from TOML file(s) and `A8VC_*` environment variables.

    When running as a frozen Windows app, in portable mode, with `A8VC_DATA_DIR`, or with
    `desktop=True`, user overrides are also loaded from `<data_dir>/config.toml` and relative
    paths under `[paths]` are resolved against the user data directory (with automatic
    fallback to bundled read-only models when present).
    """
    use_data_dir = bool(
        desktop
        or is_frozen()
        or is_portable_mode()
        or os.environ.get("A8VC_DATA_DIR", "").strip()
    )
    data_dir = get_data_dir(desktop=desktop)

    data: dict[str, Any] = {}
    candidates: list[Path] = []
    if config_path is not None:
        candidates.append(Path(config_path))
    else:
        default_cfg = get_default_config_path()
        if default_cfg is not None:
            candidates.append(default_cfg)
        local_cfg = Path("configs/local.toml")
        if local_cfg.is_file():
            candidates.append(local_cfg)
        if use_data_dir:
            user_cfg = data_dir / "config.toml"
            if user_cfg.is_file():
                candidates.append(user_cfg)

    for path in candidates:
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with path.open("rb") as fh:
            parsed = tomllib.load(fh)
        data = _deep_merge(data, parsed)

    data = _deep_merge(data, _env_overrides())
    cfg = AppConfig.model_validate(data)

    if use_data_dir:
        cfg.paths.model_dir = resolve_readonly_or_writable_path(
            cfg.paths.model_dir, data_dir=data_dir, must_be_writable=False
        )
        cfg.paths.asr_model_dir = resolve_readonly_or_writable_path(
            cfg.paths.asr_model_dir, data_dir=data_dir, must_be_writable=False
        )
        cfg.paths.vad_model_path = resolve_readonly_or_writable_path(
            cfg.paths.vad_model_path, data_dir=data_dir, must_be_writable=False
        )
        cfg.paths.voices_dir = resolve_readonly_or_writable_path(
            cfg.paths.voices_dir, data_dir=data_dir, must_be_writable=True
        )

    return cfg
