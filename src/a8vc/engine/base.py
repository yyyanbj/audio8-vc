"""Abstract VoiceConversionEngine contract shared across Track 1 and Track 2 engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from a8vc.voices.format import VoicePack


@dataclass(slots=True)
class VCOutputEvent:
    """One converted audio chunk emitted by a `VoiceConversionEngine`."""

    audio: np.ndarray  # 1D float32 mono
    sample_rate: int
    voice_id: str
    transcript: str | None = None
    asr_ms: float = 0.0
    prefill_ms: float = 0.0
    ttfa_ms: float = 0.0
    cache_hit: bool = False
    is_final: bool = False


class VoiceConversionEngine(ABC):
    """Unified streaming interface for real-time voice conversion."""

    @property
    @abstractmethod
    def mode(self) -> str:
        """Engine identifier (e.g. 'cascaded' or 'codec_s2s')."""

    @property
    @abstractmethod
    def output_sample_rate(self) -> int:
        """Sample rate of emitted target-voice audio."""

    @property
    @abstractmethod
    def active_voice(self) -> VoicePack | None:
        """Currently selected target voice pack."""

    @abstractmethod
    def set_voice(self, voice: VoicePack, *, warm: bool = True) -> float:
        """Hot-switch target voice pack. Returns warm-up latency in ms."""

    @abstractmethod
    def process_audio(self, audio: np.ndarray, sample_rate: int) -> Iterator[VCOutputEvent]:
        """Feed an incoming mono float32 audio frame and yield zero or more output events."""

    @abstractmethod
    def flush(self) -> Iterator[VCOutputEvent]:
        """Flush any buffered incoming speech and emit remaining target audio."""

    @abstractmethod
    def cancel(self) -> None:
        """Cooperatively interrupt any in-flight conversion immediately."""

    @abstractmethod
    def reset(self) -> None:
        """Reset streaming state (VAD/crossfade buffers) between sessions."""

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        """Return runtime telemetry (voice, latency stats, cache stats, memory)."""

    @abstractmethod
    def close(self) -> None:
        """Release model sessions and resources."""
