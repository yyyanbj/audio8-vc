"""Audio I/O, resampling, VAD segmentation, and local device streaming."""

from a8vc.audio.io import (
    crossfade_append,
    float32_to_pcm16,
    load_audio,
    load_audio_bytes,
    pad_to_multiple,
    pcm16_to_float32,
    resample_audio,
    save_wav,
)
from a8vc.audio.vad import EnergyVAD, SileroVADOnnx, SpeechSegment, SpeechSegmenter, create_vad

__all__ = [
    "EnergyVAD",
    "SileroVADOnnx",
    "SpeechSegment",
    "SpeechSegmenter",
    "create_vad",
    "crossfade_append",
    "float32_to_pcm16",
    "load_audio",
    "load_audio_bytes",
    "pad_to_multiple",
    "pcm16_to_float32",
    "resample_audio",
    "save_wav",
]
