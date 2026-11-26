"""Abstract ASR protocol for low-latency transcription."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ASRBackend(Protocol):
    """Minimal ASR interface used by both real-time VC segmentation and voice enrollment."""

    @property
    def name(self) -> str: ...

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe a mono float32 speech segment into clean text."""
        ...

    def close(self) -> None: ...
