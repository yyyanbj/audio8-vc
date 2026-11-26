"""One-click voice enrollment: encode reference audio into a `.a8voice` pack.

Uses Audio8's optional `registration/codec_encoder_fp16.onnx` in lazy-load + immediate-release
mode so the encoder does not stay resident in RAM during live voice conversion.
"""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from a8vc.asr.base import ASRBackend
from a8vc.audio.io import load_audio, load_audio_bytes, pad_to_multiple, resample_audio
from a8vc.voices.format import VoicePack, build_voice_pack, validate_codes_array

MIN_AUDIO_SECONDS = 0.5
MAX_AUDIO_SECONDS = 30.0
CODEC_FRAME_ALIGN = 2048


class VoiceEnroller:
    """Encodes raw human voice recordings into portable `.a8voice` voice packs."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        provider: str = "cpu",
        threads: int = 2,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.provider = provider
        self.threads = max(1, int(threads))

    def _registration_paths(self) -> tuple[Path, Path]:
        reg_dir = self.model_dir / "registration"
        manifest_path = reg_dir / "registration_manifest.json"
        encoder_path = reg_dir / "codec_encoder_fp16.onnx"
        if not encoder_path.is_file():
            # Fallback if user placed fp32 or another filename in registration manifest
            if manifest_path.is_file():
                reg_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                alt = reg_manifest.get("encoder_model") or reg_manifest.get("codec_encoder_model")
                if alt:
                    candidate = reg_dir / alt
                    if candidate.is_file():
                        encoder_path = candidate
        return manifest_path, encoder_path

    def status(self) -> dict[str, Any]:
        manifest_path, encoder_path = self._registration_paths()
        return {
            "available": encoder_path.is_file(),
            "encoder_path": str(encoder_path),
            "manifest_path": str(manifest_path),
            "manifest_exists": manifest_path.is_file(),
        }

    def _read_registration_manifest(self) -> dict[str, Any]:
        manifest_path, _ = self._registration_paths()
        runtime_manifest_path = self.model_dir / "runtime_manifest.json"
        data: dict[str, Any] = {
            "sample_rate": 44100,
            "num_codebooks": 10,
            "codebook_size": 4096,
            "codec_hop_length": CODEC_FRAME_ALIGN,
            "model_fingerprint": "",
        }
        for path in (runtime_manifest_path, manifest_path):
            if path.is_file():
                try:
                    loaded = json.loads(path.read_text(encoding="utf-8"))
                    data.update(loaded)
                except Exception:
                    pass
        return data

    def encode_waveform(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, dict[str, Any]]:
        """Encode 1D float32 audio into `[10, T]` discrete codes using `codec_encoder_fp16.onnx`."""
        import onnxruntime as ort

        _, encoder_path = self._registration_paths()
        if not encoder_path.is_file():
            raise FileNotFoundError(
                f"Voice encoder not found at {encoder_path}. "
                "Run `python scripts/download_models.py --audio8-onnx` to download registration files."
            )

        reg_manifest = self._read_registration_manifest()
        target_sr = int(reg_manifest.get("sample_rate", 44100))
        num_codebooks = int(reg_manifest.get("num_codebooks", 10))
        codebook_size = int(reg_manifest.get("codebook_size", 4096))
        hop_length = int(reg_manifest.get("codec_hop_length", CODEC_FRAME_ALIGN))

        duration = audio.size / float(sample_rate)
        if duration < MIN_AUDIO_SECONDS or duration > MAX_AUDIO_SECONDS:
            raise ValueError(
                f"Reference audio duration must be between {MIN_AUDIO_SECONDS}s and "
                f"{MAX_AUDIO_SECONDS}s, got {duration:.2f}s"
            )

        resampled = resample_audio(audio, sample_rate, target_sr)
        padded = pad_to_multiple(resampled, multiple=hop_length)
        tensor = np.ascontiguousarray(padded.reshape(1, 1, -1))

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = self.threads
        opts.inter_op_num_threads = max(1, self.threads // 2)
        opts.log_severity_level = 3

        available = set(ort.get_available_providers())
        providers = ["CPUExecutionProvider"]
        if self.provider == "cuda" and "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif self.provider == "dml" and "DmlExecutionProvider" in available:
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        elif self.provider == "coreml" and "CoreMLExecutionProvider" in available:
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        elif self.provider == "auto":
            for ep in ("CUDAExecutionProvider", "CoreMLExecutionProvider", "DmlExecutionProvider"):
                if ep in available:
                    providers = [ep, "CPUExecutionProvider"]
                    break

        session = None
        try:
            session = ort.InferenceSession(str(encoder_path), sess_options=opts, providers=providers)
            inp = session.get_inputs()[0]
            if inp.type == "tensor(float16)":
                feed = tensor.astype(np.float16, copy=False)
            else:
                feed = tensor.astype(np.float32, copy=False)
            outputs = session.run(None, {inp.name: feed})
            codes = np.asarray(outputs[0])
            if codes.ndim == 3:
                codes = codes[0]
            validated = validate_codes_array(
                codes,
                num_codebooks=num_codebooks,
                codebook_size=codebook_size,
            )
            return validated, reg_manifest
        finally:
            # Release encoder session promptly to keep memory footprint low
            del session
            gc.collect()

    def enroll_file(
        self,
        audio_path: str | Path,
        *,
        voice_id: str,
        reference_text: str | None = None,
        asr: ASRBackend | None = None,
        name: str | None = None,
        description: str = "",
        language: str = "zh",
        tags: list[str] | None = None,
    ) -> VoicePack:
        """Create a `VoicePack` from an audio file on disk (optionally auto-transcribing text)."""
        path = Path(audio_path)
        raw_bytes = path.read_bytes()
        source_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        audio, orig_sr = load_audio(path)
        return self._enroll_loaded(
            audio=audio,
            orig_sr=orig_sr,
            source_name=path.name,
            source_sha256=source_sha256,
            voice_id=voice_id,
            reference_text=reference_text,
            asr=asr,
            name=name,
            description=description,
            language=language,
            tags=tags,
        )

    def enroll_bytes(
        self,
        raw_bytes: bytes,
        *,
        filename: str,
        voice_id: str,
        reference_text: str | None = None,
        asr: ASRBackend | None = None,
        name: str | None = None,
        description: str = "",
        language: str = "zh",
        tags: list[str] | None = None,
    ) -> VoicePack:
        """Create a `VoicePack` from uploaded audio bytes."""
        source_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        audio, orig_sr = load_audio_bytes(raw_bytes, filename_hint=filename)
        return self._enroll_loaded(
            audio=audio,
            orig_sr=orig_sr,
            source_name=Path(filename).name,
            source_sha256=source_sha256,
            voice_id=voice_id,
            reference_text=reference_text,
            asr=asr,
            name=name,
            description=description,
            language=language,
            tags=tags,
        )

    def _enroll_loaded(
        self,
        *,
        audio: np.ndarray,
        orig_sr: int,
        source_name: str,
        source_sha256: str,
        voice_id: str,
        reference_text: str | None,
        asr: ASRBackend | None,
        name: str | None,
        description: str,
        language: str,
        tags: list[str] | None,
    ) -> VoicePack:
        text = (reference_text or "").strip()
        if not text:
            if asr is None:
                raise ValueError(
                    "reference_text is required unless an ASR backend is configured for auto-transcription."
                )
            audio_16k = resample_audio(audio, orig_sr, 16000)
            text = asr.transcribe(audio_16k, sample_rate=16000).strip()
            if not text:
                raise ValueError(
                    "Auto-transcription returned empty text; please pass --text / reference_text explicitly."
                )

        codes, reg_manifest = self.encode_waveform(audio, orig_sr)
        return build_voice_pack(
            voice_id=voice_id,
            codes=codes,
            reference_text=text,
            name=name,
            description=description,
            language=language,
            tags=tags,
            num_codebooks=int(reg_manifest.get("num_codebooks", 10)),
            codebook_size=int(reg_manifest.get("codebook_size", 4096)),
            sample_rate=int(reg_manifest.get("sample_rate", 44100)),
            codec_hop_length=int(reg_manifest.get("codec_hop_length", CODEC_FRAME_ALIGN)),
            source_audio_name=source_name,
            source_sample_rate=orig_sr,
            source_sha256=source_sha256,
            model_fingerprint=str(reg_manifest.get("model_fingerprint", "")),
        )
