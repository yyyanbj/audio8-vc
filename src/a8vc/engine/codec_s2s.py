"""Track 2 (Reserved): Native Codec-Space Speech-to-Speech Voice Conversion Engine.

Architecture Target
-------------------
Audio8-TTS-0.1B encodes 44.1 kHz audio into 10 discrete codebooks (`[10, T]` at 21.53 Hz),
where Codebook 0 carries high-level semantic/phonetic content and Codebooks 1..9 carry
fine acoustic/timbre residuals predicted by the 4-layer Fast-AR head conditioned on
Falcon-H1 Mamba-2 + Attention Slow-AR hidden states.

After fine-tuning `Edge0/Audio8-TTS-Preview-0.1b` on S2S VC objectives (see `training/`):
    Source Audio (44.1 kHz chunk)
        -> streaming codec_encoder (FP16 ONNX) -> Source Codebook 0 (semantic frames)
        -> S2S Slow/Fast AR conditioned on target `.a8voice` prefix state
        -> Target Codebooks [10, T]
        -> streaming codec_decoder (FP16 ONNX) -> Target Audio (44.1 kHz chunk)

This eliminates ASR entirely and reduces end-to-end latency to ~150-300 ms while
preserving source speaker prosody, pacing, and non-verbal expressions.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from a8vc.engine.base import VCOutputEvent, VoiceConversionEngine
from a8vc.voices.format import VoicePack


class CodecS2SEngine(VoiceConversionEngine):
    """Experimental scaffold for Track 2 fine-tuned Audio8-0.1B native S2S voice conversion."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        sample_rate: int = 44100,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self._sample_rate = sample_rate
        self._voice: VoicePack | None = None

    @property
    def mode(self) -> str:
        return "codec_s2s"

    @property
    def output_sample_rate(self) -> int:
        return self._sample_rate

    @property
    def active_voice(self) -> VoicePack | None:
        return self._voice

    def set_voice(self, voice: VoicePack, *, warm: bool = True) -> float:
        self._voice = voice
        return 0.0

    def process_audio(self, audio: np.ndarray, sample_rate: int) -> Iterator[VCOutputEvent]:
        raise NotImplementedError(
            "Track 2 CodecS2SEngine requires a fine-tuned S2S Audio8-0.1B ONNX checkpoint. "
            "Use engine.mode='cascaded' for zero-training real-time VC now, or see "
            "`training/README.md` to train a Track 2 S2S checkpoint on GPU."
        )

    def flush(self) -> Iterator[VCOutputEvent]:
        return iter(())

    def cancel(self) -> None:
        return None

    def reset(self) -> None:
        return None

    def stats(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "status": "scaffold_awaiting_s2s_checkpoint",
            "checkpoint_dir": str(self.checkpoint_dir),
            "active_voice": self._voice.voice_id if self._voice else None,
            "output_sample_rate": self._sample_rate,
        }

    def close(self) -> None:
        return None
