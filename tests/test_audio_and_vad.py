"""Tests for audio I/O utilities, resampling, PCM16 conversion, and VAD segmentation."""

from __future__ import annotations

import numpy as np

from a8vc.audio.io import (
    crossfade_append,
    float32_to_pcm16,
    pad_to_multiple,
    pcm16_to_float32,
    resample_audio,
)
from a8vc.audio.vad import EnergyVAD, SpeechSegmenter


def test_resample_and_pad_alignment() -> None:
    sr_in = 16000
    t = np.linspace(0.0, 0.5, sr_in // 2, endpoint=False, dtype=np.float32)
    sine = 0.4 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)

    up = resample_audio(sine, 16000, 44100)
    assert abs(up.size - 22050) <= 2

    padded = pad_to_multiple(up, 2048)
    assert padded.size % 2048 == 0
    assert padded.size >= up.size


def test_pcm16_roundtrip() -> None:
    samples = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    raw = float32_to_pcm16(samples)
    restored = pcm16_to_float32(raw)
    assert restored.shape == samples.shape
    assert np.allclose(restored, samples, atol=1e-3)


def test_crossfade_append_preserves_total_length_smoothly() -> None:
    c1 = np.ones(1000, dtype=np.float32) * 0.5
    c2 = np.ones(1000, dtype=np.float32) * -0.5
    fade = 100

    out1, tail1 = crossfade_append(np.zeros(0, dtype=np.float32), c1, fade)
    out2, tail2 = crossfade_append(tail1, c2, fade)
    combined = np.concatenate([out1, out2, tail2])
    # Overlap reduces total length by fade samples
    assert combined.size == 1000 + 1000 - fade
    # Transition should be smooth, no sudden jump of 1.0 between adjacent samples
    max_step = float(np.max(np.abs(np.diff(combined))))
    assert max_step < 0.05


def test_energy_vad_segments_speech_and_silence() -> None:
    sr = 16000
    vad = EnergyVAD(threshold_db=-35.0)
    seg = SpeechSegmenter(
        vad=vad,
        sample_rate=sr,
        speech_start_ms=64,
        min_silence_ms=256,
        pre_speech_ms=64,
        min_segment_ms=160,
        max_segment_ms=2000,
    )

    silence = np.zeros(int(sr * 0.4), dtype=np.float32)
    t = np.linspace(0.0, 0.6, int(sr * 0.6), endpoint=False, dtype=np.float32)
    speech = (0.25 * np.sin(2.0 * np.pi * 300.0 * t)).astype(np.float32)

    stream = np.concatenate([silence, speech, silence])
    emitted = []
    # Feed in 20ms (320 sample) microphone-like blocks
    for i in range(0, stream.size, 320):
        emitted.extend(seg.feed(stream[i : i + 320]))

    assert len(emitted) == 1
    assert 400.0 <= emitted[0].duration_ms <= 900.0
    assert emitted[0].forced_cut is False


def test_vad_forced_split_on_continuous_speech() -> None:
    sr = 16000
    vad = EnergyVAD(threshold_db=-35.0)
    seg = SpeechSegmenter(
        vad=vad,
        sample_rate=sr,
        speech_start_ms=64,
        min_silence_ms=300,
        pre_speech_ms=0,
        min_segment_ms=160,
        max_segment_ms=600,
    )
    t = np.linspace(0.0, 1.5, int(sr * 1.5), endpoint=False, dtype=np.float32)
    speech = (0.3 * np.sin(2.0 * np.pi * 250.0 * t)).astype(np.float32)

    emitted = seg.feed(speech)
    assert len(emitted) >= 2
    assert emitted[0].forced_cut is True
