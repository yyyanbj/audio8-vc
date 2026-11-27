"""Lightweight sherpa-onnx ASR backend (SenseVoice / Paraformer / Zipformer) + Mock backend."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

from a8vc.asr.base import ASRBackend
from a8vc.audio.io import ensure_mono_float32, resample_audio
from a8vc.config import ASRConfig

logger = logging.getLogger("a8vc.asr.sherpa")

# Strip SenseVoice emotion/event tags such as <|zh|><|NEUTRAL|><|Speech|><|woitn|>
_SENSEVOICE_TAG_RE = re.compile(r"<\|[^|]+\|>")


def _clean_asr_text(text: str) -> str:
    cleaned = _SENSEVOICE_TAG_RE.sub("", text)
    return " ".join(cleaned.strip().split())


class MockASR:
    """Deterministic lightweight ASR stub for offline pipeline testing & dry-runs."""

    def __init__(self, phrase: str = "你好，这是实时变声测试。") -> None:
        self._phrase = phrase

    @property
    def name(self) -> str:
        return "mock"

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        if audio.size == 0:
            return ""
        return self._phrase

    def close(self) -> None:
        return None


class SherpaOnnxASR:
    """Low-footprint CPU ASR powered by `sherpa-onnx`."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        model_type: str = "sense_voice",
        language: str = "auto",
        num_threads: int = 2,
        provider: str = "cpu",
    ) -> None:
        # On Windows, DLL loading is process-global by basename (`onnxruntime.dll`).
        # Always load `onnxruntime` before `sherpa_onnx` so the flavor-matched
        # `onnxruntime/capi/onnxruntime.dll` (CPU / DirectML / CUDA) wins over
        # `sherpa_onnx/lib/onnxruntime.dll`.
        try:
            import onnxruntime as _ort  # noqa: F401
        except ImportError:
            pass

        try:
            import sherpa_onnx
        except ImportError as exc:
            raise RuntimeError(
                "SherpaOnnxASR requires `sherpa-onnx`. Install with: uv sync --extra asr"
            ) from exc

        root = Path(model_dir)
        if not root.is_dir():
            raise FileNotFoundError(
                f"ASR model directory not found: {root}. "
                "Run `python scripts/download_models.py --asr-sensevoice` or set asr.backend='mock'."
            )

        self._sherpa = sherpa_onnx
        self._model_type = model_type
        self.active_provider = provider
        try:
            self._recognizer = self._build_recognizer(
                root=root,
                model_type=model_type,
                language=language,
                num_threads=num_threads,
                provider=provider,
            )
        except Exception as exc:
            if provider == "cpu":
                raise
            logger.warning(
                "sherpa-onnx ASR provider=%r failed (%s); falling back to CPU",
                provider,
                exc,
            )
            self.active_provider = "cpu"
            self._recognizer = self._build_recognizer(
                root=root,
                model_type=model_type,
                language=language,
                num_threads=num_threads,
                provider="cpu",
            )

    @property
    def name(self) -> str:
        return f"sherpa-onnx:{self._model_type}"

    def _find_file(self, root: Path, patterns: list[str]) -> str:
        for pat in patterns:
            matches = sorted(root.rglob(pat))
            if matches:
                return str(matches[0])
        raise FileNotFoundError(
            f"Could not locate any of {patterns} under ASR directory {root}"
        )

    def _build_recognizer(
        self,
        *,
        root: Path,
        model_type: str,
        language: str,
        num_threads: int,
        provider: str,
    ) -> Any:
        tokens = self._find_file(root, ["tokens.txt"])
        if model_type == "sense_voice":
            model = self._find_file(root, ["model.int8.onnx", "model.onnx"])
            return self._sherpa.OfflineRecognizer.from_sense_voice(
                model=model,
                tokens=tokens,
                num_threads=num_threads,
                use_itn=True,
                language=language,
                provider=provider,
            )
        if model_type == "paraformer":
            model = self._find_file(root, ["model.int8.onnx", "model.onnx"])
            return self._sherpa.OfflineRecognizer.from_paraformer(
                paraformer=model,
                tokens=tokens,
                num_threads=num_threads,
                provider=provider,
            )
        if model_type == "zipformer":
            encoder = self._find_file(root, ["encoder-*.int8.onnx", "encoder*.onnx"])
            decoder = self._find_file(root, ["decoder-*.int8.onnx", "decoder*.onnx"])
            joiner = self._find_file(root, ["joiner-*.int8.onnx", "joiner*.onnx"])
            return self._sherpa.OfflineRecognizer.from_transducer(
                encoder=encoder,
                decoder=decoder,
                joiner=joiner,
                tokens=tokens,
                num_threads=num_threads,
                provider=provider,
            )
        raise ValueError(f"Unsupported sherpa-onnx model_type: {model_type!r}")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        mono = ensure_mono_float32(audio)
        if mono.size == 0:
            return ""
        if sample_rate != 16000:
            mono = resample_audio(mono, sample_rate, 16000)
        stream = self._recognizer.create_stream()
        stream.accept_waveform(16000, mono)
        self._recognizer.decode_stream(stream)
        return _clean_asr_text(stream.result.text)

    def close(self) -> None:
        self._recognizer = None


def create_asr(cfg: ASRConfig, model_dir: str | Path) -> ASRBackend:
    """Instantiate an ASR backend from application configuration."""
    if cfg.backend == "mock":
        return MockASR()
    if cfg.backend == "sherpa":
        return SherpaOnnxASR(
            model_dir=model_dir,
            model_type=cfg.model_type,
            language=cfg.language,
            num_threads=cfg.num_threads,
            provider=cfg.provider,
        )
    raise ValueError(f"Unknown ASR backend: {cfg.backend!r}")
