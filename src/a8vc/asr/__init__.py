"""Pluggable lightweight ASR backends for Track 1 real-time VC and voice enrollment."""

from a8vc.asr.base import ASRBackend
from a8vc.asr.sherpa import MockASR, SherpaOnnxASR, create_asr

__all__ = ["ASRBackend", "MockASR", "SherpaOnnxASR", "create_asr"]
