"""Audio decoding, resampling, padding, and PCM conversion utilities."""

from __future__ import annotations

import io
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

MAX_AUDIO_BYTES = 50 * 1024 * 1024


def ensure_mono_float32(audio: np.ndarray) -> np.ndarray:
    """Convert an audio array to contiguous 1D float32 mono and validate finite samples."""
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=1, dtype=np.float32)
    elif arr.ndim != 1:
        raise ValueError(f"Expected 1D or 2D audio array, got shape {arr.shape}")
    if arr.size == 0:
        return np.zeros(0, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError("Audio contains NaN or infinite samples")
    return np.ascontiguousarray(arr, dtype=np.float32)


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """High-quality polyphase resampling matching Audio8 registration behavior."""
    mono = ensure_mono_float32(audio)
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError(f"Sample rates must be positive: {orig_sr} -> {target_sr}")
    if orig_sr == target_sr or mono.size == 0:
        return mono
    divisor = math.gcd(orig_sr, target_sr)
    up = target_sr // divisor
    down = orig_sr // divisor
    resampled = resample_poly(mono, up, down).astype(np.float32, copy=False)
    return np.ascontiguousarray(resampled, dtype=np.float32)


def pad_to_multiple(audio: np.ndarray, multiple: int = 2048) -> np.ndarray:
    """Zero-pad trailing samples so length is a multiple of `multiple` (2048 for Audio8 codec)."""
    mono = ensure_mono_float32(audio)
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}")
    remainder = mono.size % multiple
    if remainder == 0:
        return mono
    pad = multiple - remainder
    return np.ascontiguousarray(np.pad(mono, (0, pad), mode="constant"), dtype=np.float32)


def _decode_with_ffmpeg(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "Unable to decode audio with soundfile and `ffmpeg` was not found in PATH."
        )
    cmd = [
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "pipe:1",
    ]
    run_kwargs: dict[str, object] = {"capture_output": True, "check": False}
    if sys.platform == "win32":
        run_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    proc = subprocess.run(cmd, **run_kwargs)  # type: ignore[arg-type]
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg audio decoding failed: {err}")
    audio = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    return ensure_mono_float32(audio), target_sr


def load_audio(
    path: str | Path,
    target_sr: int | None = None,
) -> tuple[np.ndarray, int]:
    """Load an audio file as 1D float32 mono, optionally resampled to `target_sr`."""
    audio_path = Path(path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    size = audio_path.stat().st_size
    if size <= 0 or size > MAX_AUDIO_BYTES:
        raise ValueError(
            f"Audio file size must be in (0, {MAX_AUDIO_BYTES}] bytes, got {size}"
        )

    try:
        data, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        mono = ensure_mono_float32(data)
    except Exception:
        fallback_sr = target_sr or 44100
        mono, sr = _decode_with_ffmpeg(audio_path, fallback_sr)

    if target_sr is not None and sr != target_sr:
        mono = resample_audio(mono, sr, target_sr)
        sr = target_sr
    return mono, int(sr)


def load_audio_bytes(
    raw_bytes: bytes,
    filename_hint: str = "audio.wav",
    target_sr: int | None = None,
) -> tuple[np.ndarray, int]:
    """Load raw audio bytes (WAV/FLAC/OGG via soundfile, MP3/M4A via ffmpeg fallback)."""
    if not raw_bytes or len(raw_bytes) > MAX_AUDIO_BYTES:
        raise ValueError(
            f"Audio payload size must be in (0, {MAX_AUDIO_BYTES}] bytes, got {len(raw_bytes)}"
        )
    try:
        data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
        mono = ensure_mono_float32(data)
    except Exception:
        suffix = Path(filename_hint).suffix or ".bin"
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(raw_bytes)
                tmp_path = Path(tmp.name)
            fallback_sr = target_sr or 44100
            mono, sr = _decode_with_ffmpeg(tmp_path, fallback_sr)
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

    if target_sr is not None and sr != target_sr:
        mono = resample_audio(mono, sr, target_sr)
        sr = target_sr
    return mono, int(sr)


def save_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> Path:
    """Write 1D float32 mono audio to a PCM16 WAV file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mono = np.clip(ensure_mono_float32(audio), -1.0, 1.0)
    sf.write(str(out_path), mono, sample_rate, subtype="PCM_16")
    return out_path


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    """Convert float32 [-1, 1] waveform to little-endian signed 16-bit PCM bytes."""
    mono = ensure_mono_float32(audio)
    if mono.size == 0:
        return b""
    clipped = np.clip(mono, -1.0, 1.0)
    pcm = np.round(clipped * 32767.0).astype("<i2")
    return pcm.tobytes()


def pcm16_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """Convert little-endian signed 16-bit PCM bytes to float32 [-1, 1] waveform."""
    if not pcm_bytes:
        return np.zeros(0, dtype=np.float32)
    usable = len(pcm_bytes) - (len(pcm_bytes) % 2)
    if usable <= 0:
        return np.zeros(0, dtype=np.float32)
    samples = np.frombuffer(pcm_bytes[:usable], dtype="<i2").astype(np.float32)
    return np.ascontiguousarray(samples / 32768.0, dtype=np.float32)


def crossfade_append(
    previous_tail: np.ndarray,
    new_chunk: np.ndarray,
    fade_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Crossfade the tail of a previous segment with the start of a new segment.

    Returns (emit_audio, next_tail) so callers can stream smoothly without clicks.
    """
    chunk = ensure_mono_float32(new_chunk)
    if fade_samples <= 0 or chunk.size == 0:
        if previous_tail.size > 0:
            return np.concatenate([previous_tail, chunk]), np.zeros(0, dtype=np.float32)
        return chunk, np.zeros(0, dtype=np.float32)

    if previous_tail.size == 0:
        if chunk.size <= fade_samples:
            return np.zeros(0, dtype=np.float32), chunk
        return chunk[:-fade_samples], chunk[-fade_samples:]

    overlap = min(fade_samples, previous_tail.size, chunk.size)
    t = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
    # Equal-power raised-cosine fade
    fade_out = 0.5 * (1.0 + np.cos(np.pi * t))
    fade_in = 1.0 - fade_out
    blended = previous_tail[-overlap:] * fade_out + chunk[:overlap] * fade_in

    prefix = previous_tail[:-overlap] if previous_tail.size > overlap else np.zeros(0, dtype=np.float32)
    remainder = chunk[overlap:]
    combined = np.concatenate([prefix, blended, remainder])
    if combined.size <= fade_samples:
        return np.zeros(0, dtype=np.float32), combined
    return combined[:-fade_samples], combined[-fade_samples:]
