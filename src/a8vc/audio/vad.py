"""Lightweight Voice Activity Detection (VAD) and streaming utterance segmentation."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from a8vc.config import VADConfig

logger = logging.getLogger("a8vc.audio.vad")


class VADBackend(Protocol):
    """Per-frame speech probability estimator operating on 16kHz 512-sample frames."""

    frame_samples: int

    def probability(self, frame: np.ndarray) -> float: ...

    def reset(self) -> None: ...


class EnergyVAD:
    """Zero-dependency adaptive energy VAD with noise floor tracking."""

    frame_samples: int = 512

    def __init__(
        self,
        threshold_db: float = -38.0,
        snr_margin_db: float = 8.0,
        noise_ema: float = 0.96,
    ) -> None:
        self.threshold_db = float(threshold_db)
        self.snr_margin_db = float(snr_margin_db)
        self.noise_ema = float(noise_ema)
        self._noise_floor_rms = 10.0 ** ((threshold_db - 10.0) / 20.0)

    def reset(self) -> None:
        self._noise_floor_rms = 10.0 ** ((self.threshold_db - 10.0) / 20.0)

    def probability(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float32)) + 1e-12))
        db = 20.0 * float(np.log10(rms + 1e-12))
        noise_db = 20.0 * float(np.log10(self._noise_floor_rms + 1e-12))
        effective_threshold = max(self.threshold_db, noise_db + self.snr_margin_db)

        if db < effective_threshold:
            self._noise_floor_rms = self.noise_ema * self._noise_floor_rms + (
                1.0 - self.noise_ema
            ) * rms
        # Smooth sigmoid-like score centered at effective_threshold
        margin = db - effective_threshold
        prob = 1.0 / (1.0 + float(np.exp(-margin / 3.0)))
        return float(np.clip(prob, 0.0, 1.0))


def _resolve_tensor_shape(raw_shape: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    """Resolve static/dynamic ONNX shape metadata into concrete positive integer dims."""
    if not isinstance(raw_shape, (list, tuple)) or len(raw_shape) != len(default):
        return default
    resolved: list[int] = []
    for dim, fallback in zip(raw_shape, default, strict=True):
        if isinstance(dim, int) and dim > 0:
            resolved.append(dim)
        else:
            resolved.append(fallback)
    return tuple(resolved)


class SileroVADOnnx:
    """Optional Silero VAD ONNX backend (~0.6-2 MB).

    Dynamically adapts to all common 16 kHz Silero VAD ONNX exports:
    - sherpa-onnx export: inputs ``['x', 'h', 'c']`` -> outputs ``['prob', 'new_h', 'new_c']``
    - Official Silero v5: inputs ``['input', 'state', 'sr'?]`` -> outputs ``['output', 'stateN']``
    - Official Silero v4: inputs ``['input', 'sr', 'h', 'c']`` -> outputs ``['output', 'hn', 'cn']``
    """

    frame_samples: int = 512

    def __init__(
        self,
        model_path: str | Path,
        sample_rate: int = 16000,
        *,
        session: Any = None,
    ) -> None:
        if sample_rate != 16000:
            raise ValueError("SileroVADOnnx currently expects 16000 Hz input")

        if session is not None:
            self._session = session
        else:
            import onnxruntime as ort

            path = Path(model_path)
            if not path.is_file():
                raise FileNotFoundError(f"Silero VAD model not found: {path}")

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            opts.log_severity_level = 3
            self._session = ort.InferenceSession(
                str(path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )

        inputs = {inp.name: inp for inp in self._session.get_inputs()}
        self._input_names = list(inputs.keys())
        self._has_sr = "sr" in inputs
        self._sr = np.array(sample_rate, dtype=np.int64)

        if "x" in inputs:
            self._audio_input_name = "x"
        elif "input" in inputs:
            self._audio_input_name = "input"
        else:
            raise ValueError(
                f"Unsupported Silero VAD ONNX audio input signature: {self._input_names}"
            )

        if "state" in inputs:
            self._schema = "v5"
            self._state_shape = _resolve_tensor_shape(inputs["state"].shape, (2, 1, 128))
            audio_shape = inputs[self._audio_input_name].shape
            fixed_len = audio_shape[-1] if isinstance(audio_shape, (list, tuple)) and audio_shape else None
            # Official v5 accepts 576 samples (64-sample context + 512 frame) or dynamic length
            self._context_samples = 0 if (isinstance(fixed_len, int) and fixed_len == self.frame_samples) else 64
        elif "h" in inputs and "c" in inputs:
            self._schema = "hc"
            self._h_shape = _resolve_tensor_shape(inputs["h"].shape, (2, 1, 64))
            self._c_shape = _resolve_tensor_shape(inputs["c"].shape, (2, 1, 64))
            self._context_samples = 0
        else:
            raise ValueError(
                f"Unsupported Silero VAD ONNX state signature: {self._input_names}"
            )

        self._context = np.zeros(self._context_samples, dtype=np.float32)
        self.reset()
        # Self-verify tensor feed contract on a silent frame immediately at load time
        _ = self.probability(np.zeros(self.frame_samples, dtype=np.float32))
        self.reset()

    @property
    def schema(self) -> str:
        """Detected ONNX graph variant ('hc' for sherpa/v4 or 'v5')."""
        return self._schema

    def reset(self) -> None:
        self._context = np.zeros(self._context_samples, dtype=np.float32)
        if self._schema == "v5":
            self._state = np.zeros(self._state_shape, dtype=np.float32)
        else:
            self._h = np.zeros(self._h_shape, dtype=np.float32)
            self._c = np.zeros(self._c_shape, dtype=np.float32)

    def probability(self, frame: np.ndarray) -> float:
        x = np.asarray(frame, dtype=np.float32).reshape(-1)
        if x.size != self.frame_samples:
            if x.size < self.frame_samples:
                x = np.pad(x, (0, self.frame_samples - x.size))
            else:
                x = x[: self.frame_samples]
        if self._schema == "v5":
            if self._context_samples > 0:
                model_in = np.concatenate([self._context, x])[None, :]
                self._context = x[-self._context_samples :].copy()
            else:
                model_in = x[None, :]
            feed: dict[str, np.ndarray] = {
                self._audio_input_name: model_in,
                "state": self._state,
            }
            if self._has_sr:
                feed["sr"] = self._sr
            out, self._state = self._session.run(None, feed)
            return float(np.clip(np.asarray(out).reshape(-1)[0], 0.0, 1.0))

        feed = {
            self._audio_input_name: x[None, :],
            "h": self._h,
            "c": self._c,
        }
        if self._has_sr:
            feed["sr"] = self._sr
        out, self._h, self._c = self._session.run(None, feed)
        return float(np.clip(np.asarray(out).reshape(-1)[0], 0.0, 1.0))


def create_vad(cfg: VADConfig, vad_model_path: str | Path | None = None) -> VADBackend:
    """Create a VAD backend according to configuration."""
    resolved_path = Path(vad_model_path) if vad_model_path is not None else None
    if cfg.backend == "silero":
        if resolved_path is None:
            raise ValueError("vad_model_path is required when vad.backend='silero'")
        return SileroVADOnnx(resolved_path)
    if cfg.backend == "auto" and resolved_path is not None and resolved_path.is_file():
        try:
            return SileroVADOnnx(resolved_path)
        except Exception as exc:
            logger.warning(
                "Silero VAD failed to initialize from %s (%s); falling back to EnergyVAD",
                resolved_path,
                exc,
            )
    return EnergyVAD(threshold_db=cfg.energy_threshold_db)


@dataclass(slots=True)
class SpeechSegment:
    """A completed utterance segment emitted by `SpeechSegmenter`."""

    audio: np.ndarray
    sample_rate: int
    start_ms: float
    end_ms: float
    forced_cut: bool = False

    @property
    def duration_ms(self) -> float:
        return (self.audio.size / self.sample_rate) * 1000.0


class SpeechSegmenter:
    """Streaming speech endpoint detector that slices audio into low-latency phrases."""

    def __init__(
        self,
        vad: VADBackend,
        sample_rate: int = 16000,
        speech_threshold: float = 0.5,
        speech_start_ms: int = 80,
        min_silence_ms: int = 360,
        pre_speech_ms: int = 160,
        min_segment_ms: int = 250,
        max_segment_ms: int = 4500,
    ) -> None:
        self.vad = vad
        self.sample_rate = sample_rate
        self.speech_threshold = speech_threshold
        self.frame_samples = vad.frame_samples
        self.frame_ms = (self.frame_samples / sample_rate) * 1000.0

        self._start_frames = max(1, int(round(speech_start_ms / self.frame_ms)))
        self._silence_frames = max(1, int(round(min_silence_ms / self.frame_ms)))
        self._pre_frames = max(0, int(round(pre_speech_ms / self.frame_ms)))
        self._min_frames = max(1, int(round(min_segment_ms / self.frame_ms)))
        self._max_frames = max(self._min_frames, int(round(max_segment_ms / self.frame_ms)))

        self._residual = np.zeros(0, dtype=np.float32)
        self._pre_buffer: deque[np.ndarray] = deque(maxlen=max(1, self._pre_frames))
        self._active_frames: list[np.ndarray] = []
        self._in_speech = False
        self._speech_streak = 0
        self._silence_streak = 0
        self._total_frames = 0
        self._segment_start_frame = 0

    def reset(self) -> None:
        self.vad.reset()
        self._residual = np.zeros(0, dtype=np.float32)
        self._pre_buffer.clear()
        self._active_frames.clear()
        self._in_speech = False
        self._speech_streak = 0
        self._silence_streak = 0
        self._total_frames = 0
        self._segment_start_frame = 0

    def feed(self, audio_chunk: np.ndarray) -> list[SpeechSegment]:
        """Push an arbitrary-length 16kHz float32 audio chunk and return completed segments."""
        chunk = np.asarray(audio_chunk, dtype=np.float32).reshape(-1)
        if chunk.size == 0:
            return []
        buf = np.concatenate([self._residual, chunk]) if self._residual.size > 0 else chunk
        segments: list[SpeechSegment] = []
        offset = 0
        while offset + self.frame_samples <= buf.size:
            frame = buf[offset : offset + self.frame_samples]
            offset += self.frame_samples
            seg = self._process_frame(frame)
            if seg is not None:
                segments.append(seg)
        self._residual = buf[offset:].copy() if offset < buf.size else np.zeros(0, dtype=np.float32)
        return segments

    def flush(self) -> SpeechSegment | None:
        """Flush any active speech segment (e.g. on end-of-stream or user pause)."""
        if not self._in_speech or not self._active_frames:
            self._active_frames.clear()
            self._in_speech = False
            return None
        if len(self._active_frames) < self._min_frames:
            self._active_frames.clear()
            self._in_speech = False
            return None
        return self._finalize_segment(forced_cut=False)

    def _process_frame(self, frame: np.ndarray) -> SpeechSegment | None:
        self._total_frames += 1
        prob = self.vad.probability(frame)
        is_speech = prob >= self.speech_threshold

        if not self._in_speech:
            if self._pre_frames > 0:
                self._pre_buffer.append(frame.copy())
            if is_speech:
                self._speech_streak += 1
                if self._speech_streak >= self._start_frames:
                    self._in_speech = True
                    self._silence_streak = 0
                    self._active_frames = list(self._pre_buffer) if self._pre_frames > 0 else [frame.copy()]
                    self._pre_buffer.clear()
                    self._segment_start_frame = max(0, self._total_frames - len(self._active_frames))
            else:
                self._speech_streak = 0
            return None

        # Currently inside an active speech segment
        self._active_frames.append(frame.copy())
        if is_speech:
            self._silence_streak = 0
        else:
            self._silence_streak += 1

        if self._silence_streak >= self._silence_frames:
            # Trim excess trailing silence while keeping a tiny natural tail (2 frames)
            trim = max(0, self._silence_streak - 2)
            if trim > 0 and len(self._active_frames) > trim:
                self._active_frames = self._active_frames[:-trim]
            if len(self._active_frames) >= self._min_frames:
                return self._finalize_segment(forced_cut=False)
            self._reset_active()
            return None

        if len(self._active_frames) >= self._max_frames:
            return self._finalize_segment(forced_cut=True)

        return None

    def _reset_active(self) -> None:
        self._active_frames.clear()
        self._in_speech = False
        self._speech_streak = 0
        self._silence_streak = 0

    def _finalize_segment(self, forced_cut: bool) -> SpeechSegment:
        audio = np.ascontiguousarray(np.concatenate(self._active_frames), dtype=np.float32)
        start_ms = self._segment_start_frame * self.frame_ms
        end_ms = start_ms + (audio.size / self.sample_rate) * 1000.0
        self._reset_active()
        return SpeechSegment(
            audio=audio,
            sample_rate=self.sample_rate,
            start_ms=start_ms,
            end_ms=end_ms,
            forced_cut=forced_cut,
        )
