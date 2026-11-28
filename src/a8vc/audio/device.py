"""Local audio device enumeration and non-blocking full-duplex real-time audio I/O."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np

from a8vc.audio.io import resample_audio


def _import_sounddevice() -> Any:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "Local audio device I/O requires `sounddevice`. "
            "Install it with: uv sync --extra audio"
        ) from exc
    return sd


_VIRTUAL_CABLE_KEYWORDS = (
    "vb-audio",
    "cable",
    "voicemeeter",
    "virtual",
    "blackhole",
    "vac ",
    "line 1",
)


def _is_virtual_cable_name(name: str) -> bool:
    lower = name.lower()
    return any(k in lower for k in _VIRTUAL_CABLE_KEYWORDS)


def list_audio_devices() -> list[dict[str, Any]]:
    """Return available system audio input/output devices."""
    sd = _import_sounddevice()
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    default_in, default_out = sd.default.device
    rows: list[dict[str, Any]] = []
    for idx, dev in enumerate(devices):
        hostapi_idx = int(dev.get("hostapi", 0))
        hostapi_name = (
            str(hostapis[hostapi_idx]["name"])
            if 0 <= hostapi_idx < len(hostapis)
            else ""
        )
        name = str(dev["name"])
        rows.append(
            {
                "index": idx,
                "name": name,
                "hostapi": hostapi_name,
                "max_input_channels": int(dev["max_input_channels"]),
                "max_output_channels": int(dev["max_output_channels"]),
                "default_samplerate": float(dev["default_samplerate"]),
                "is_default_input": idx == default_in,
                "is_default_output": idx == default_out,
                "is_virtual_cable": _is_virtual_cable_name(name),
            }
        )
    return rows


class AudioPlaybackQueue:
    """Lock-protected float32 playback ring queue fed by inference and drained by audio callbacks."""

    def __init__(self, max_seconds: float = 4.0, sample_rate: int = 44100) -> None:
        self.sample_rate = sample_rate
        self.max_seconds = float(max_seconds)
        self.max_samples = int(self.max_seconds * sample_rate)
        self._lock = threading.Lock()
        self._chunks: deque[np.ndarray] = deque()
        self._queued_samples = 0

    def push(self, audio: np.ndarray, source_sr: int | None = None) -> None:
        arr = np.asarray(audio, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return
        if source_sr is not None and source_sr != self.sample_rate:
            arr = resample_audio(arr, source_sr, self.sample_rate)
        with self._lock:
            self._chunks.append(arr)
            self._queued_samples += arr.size
            while self._queued_samples > self.max_samples and len(self._chunks) > 1:
                dropped = self._chunks.popleft()
                self._queued_samples -= dropped.size

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._queued_samples = 0

    def pop(self, num_samples: int) -> np.ndarray:
        out = np.zeros(num_samples, dtype=np.float32)
        filled = 0
        with self._lock:
            while filled < num_samples and self._chunks:
                head = self._chunks[0]
                need = num_samples - filled
                if head.size <= need:
                    out[filled : filled + head.size] = head
                    filled += head.size
                    self._queued_samples -= head.size
                    self._chunks.popleft()
                else:
                    out[filled:] = head[:need]
                    self._chunks[0] = head[need:]
                    self._queued_samples -= need
                    filled = num_samples
        return out

    @property
    def buffered_ms(self) -> float:
        with self._lock:
            return (self._queued_samples / self.sample_rate) * 1000.0


class LocalAudioBridge:
    """Manages simultaneous microphone capture and speaker/virtual-mic playback."""

    def __init__(
        self,
        on_input_chunk: Callable[[np.ndarray, int], None],
        input_sample_rate: int = 16000,
        output_sample_rate: int = 44100,
        input_block_ms: int = 16,
        input_device: str | int | None = None,
        output_device: str | int | None = None,
    ) -> None:
        self._on_input_chunk = on_input_chunk
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.input_block_ms = input_block_ms
        self.input_block_samples = int(round(input_sample_rate * input_block_ms / 1000.0))
        self.input_device = None if input_device == "" else input_device
        self.output_device = None if output_device == "" else output_device
        self.active_input_sample_rate = input_sample_rate
        self.active_output_sample_rate = output_sample_rate
        self.playback = AudioPlaybackQueue(sample_rate=output_sample_rate)
        self._in_stream: Any = None
        self._out_stream: Any = None

    def _query_device_default_sr(self, sd: Any, device: str | int | None, kind: str) -> int | None:
        try:
            info = sd.query_devices(device, kind)
            sr = int(round(float(info["default_samplerate"])))
            return sr if sr > 0 else None
        except Exception:
            return None

    def start(self) -> None:
        sd = _import_sounddevice()

        # Windows WASAPI / DirectSound microphones & virtual cables often require their
        # native sample rate (e.g. 48000 Hz). Try configured rate first, then fall back.
        in_rates = [self.input_sample_rate]
        dev_in_sr = self._query_device_default_sr(sd, self.input_device, "input")
        if dev_in_sr and dev_in_sr not in in_rates:
            in_rates.append(dev_in_sr)

        out_rates = [self.output_sample_rate]
        dev_out_sr = self._query_device_default_sr(sd, self.output_device, "output")
        if dev_out_sr and dev_out_sr not in out_rates:
            out_rates.append(dev_out_sr)

        in_stream = None
        last_in_err: Exception | None = None
        for sr_in in in_rates:
            block_in = max(64, int(round(sr_in * self.input_block_ms / 1000.0)))

            def _make_in_cb(actual_sr: int) -> Any:
                def _in_cb(indata: np.ndarray, _frames: int, _time_info: Any, _status: Any) -> None:
                    mono = indata[:, 0].copy() if indata.ndim == 2 else indata.copy()
                    self._on_input_chunk(mono, actual_sr)

                return _in_cb

            try:
                in_stream = sd.InputStream(
                    samplerate=sr_in,
                    blocksize=block_in,
                    channels=1,
                    dtype="float32",
                    device=self.input_device,
                    callback=_make_in_cb(sr_in),
                )
                self.active_input_sample_rate = sr_in
                break
            except Exception as exc:
                last_in_err = exc
        if in_stream is None:
            raise RuntimeError(f"Failed to open input audio device: {last_in_err}") from last_in_err

        out_stream = None
        last_out_err: Exception | None = None
        for sr_out in out_rates:
            self.playback.sample_rate = sr_out
            self.playback.max_samples = int(self.playback.max_seconds * sr_out)

            def _out_cb(outdata: np.ndarray, frames: int, _time_info: Any, _status: Any) -> None:
                samples = self.playback.pop(frames)
                outdata[:, 0] = samples

            try:
                out_stream = sd.OutputStream(
                    samplerate=sr_out,
                    blocksize=512,
                    channels=1,
                    dtype="float32",
                    device=self.output_device,
                    callback=_out_cb,
                )
                self.active_output_sample_rate = sr_out
                break
            except Exception as exc:
                last_out_err = exc
        if out_stream is None:
            try:
                in_stream.close()
            except Exception:
                pass
            raise RuntimeError(
                f"Failed to open output audio device: {last_out_err}"
            ) from last_out_err

        self._in_stream = in_stream
        self._out_stream = out_stream
        try:
            self._in_stream.start()
            self._out_stream.start()
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        for stream in (self._in_stream, self._out_stream):
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
        self._in_stream = None
        self._out_stream = None
        self.playback.clear()
