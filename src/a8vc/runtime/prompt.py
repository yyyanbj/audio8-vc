"""Audio8 ChatML prompt construction with split system-prefix / user-suffix tensors.

Splitting the prompt into:
1. `system_prefix`: `<|im_start|>system\n...<|speaker:0|>{ref_text}\n\nSpeech:\n{ref_codes}<|im_end|>\n`
2. `user_suffix`:   `<|im_start|>user\n{target_text}<|im_end|>\n<|im_start|>assistant\n<|voice|>`

preserves exact token-for-token equivalence with `Edge0-AI/Audio8_TTS` while enabling
in-memory Slow-AR KV/SSM prefix caching per `.a8voice` file.
"""

from __future__ import annotations

import re
from typing import Protocol

import numpy as np

_SPEAKER_TAG_RE = re.compile(r"<\|speaker:\d+\|>")
_CJK_NEWLINE_RE = re.compile(
    r"(?<=[一-鿿　-〿＀-￯])\s*\n+\s*"
    r"(?=[一-鿿　-〿＀-￯])"
)
_WHITESPACE_RUN_RE = re.compile(r"[\t\r\n ]+")


class TextTokenizer(Protocol):
    """Minimal tokenizer interface satisfied by HuggingFace `tokenizers.Tokenizer`."""

    def encode(self, sequence: str, add_special_tokens: bool = False) -> object: ...


def clean_text(text: str) -> str:
    """Clean text following Audio8's CJK-aware normalization rules."""
    cleaned = text.replace("\x00", "").strip()
    if not cleaned:
        return ""
    cleaned = _CJK_NEWLINE_RE.sub("", cleaned)
    cleaned = _WHITESPACE_RUN_RE.sub(" ", cleaned)
    return cleaned.strip()


def ensure_speaker_tag(text: str, default_speaker: int = 0) -> str:
    """Prefix text with `<|speaker:0|>` if no speaker tag is already present."""
    cleaned = clean_text(text)
    if _SPEAKER_TAG_RE.search(cleaned):
        return cleaned
    return f"<|speaker:{default_speaker}|>{cleaned}"


class PromptBuilder:
    """Builds `(1, num_codebooks + 1, L)` int64 prompt tensors for Audio8-0.1B."""

    def __init__(
        self,
        tokenizer: TextTokenizer,
        *,
        semantic_begin_id: int = 65537,
        num_codebooks: int = 10,
    ) -> None:
        self.tokenizer = tokenizer
        self.semantic_begin_id = int(semantic_begin_id)
        self.num_codebooks = int(num_codebooks)

    def _encode_ids(self, text: str) -> np.ndarray:
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        ids = getattr(encoded, "ids", encoded)
        return np.asarray(ids, dtype=np.int64)

    def build_system_prefix(
        self,
        reference_text: str,
        reference_codes: np.ndarray,
    ) -> np.ndarray:
        """Build the voice-only system turn tensor `(1, num_codebooks + 1, L_sys)`."""
        codes = np.asarray(reference_codes, dtype=np.int64)
        if codes.ndim != 2 or codes.shape[0] != self.num_codebooks or codes.shape[1] <= 0:
            raise ValueError(
                f"Expected reference_codes shape ({self.num_codebooks}, T>0), got {codes.shape}"
            )

        tagged_ref = ensure_speaker_tag(reference_text)
        head_text = (
            "<|im_start|>system\n"
            "convert the provided text to speech reference to the following:\n\n"
            f"Text:\n{tagged_ref}\n\n"
            "Speech:\n"
        )
        tail_text = "<|im_end|>\n"

        head_ids = self._encode_ids(head_text)
        tail_ids = self._encode_ids(tail_text)
        semantic_ids = codes[0] + self.semantic_begin_id

        num_frames = codes.shape[1]
        total_len = head_ids.size + num_frames + tail_ids.size
        tensor = np.zeros((1, self.num_codebooks + 1, total_len), dtype=np.int64)

        ref_start = head_ids.size
        ref_end = ref_start + num_frames
        tensor[0, 0, :ref_start] = head_ids
        tensor[0, 0, ref_start:ref_end] = semantic_ids
        tensor[0, 0, ref_end:] = tail_ids
        tensor[0, 1:, ref_start:ref_end] = codes
        return tensor

    def build_user_suffix(self, target_text: str) -> np.ndarray:
        """Build the utterance-specific user+assistant starter tensor `(1, num_codebooks + 1, L_usr)`."""
        cleaned = clean_text(target_text)
        if not cleaned:
            raise ValueError("target_text must be non-empty after cleaning")
        suffix_text = (
            f"<|im_start|>user\n{cleaned}<|im_end|>\n"
            "<|im_start|>assistant\n<|voice|>"
        )
        ids = self._encode_ids(suffix_text)
        tensor = np.zeros((1, self.num_codebooks + 1, ids.size), dtype=np.int64)
        tensor[0, 0, :] = ids
        return tensor

    def build_full(
        self,
        target_text: str,
        reference_text: str,
        reference_codes: np.ndarray,
    ) -> np.ndarray:
        """Build the complete Audio8 prompt tensor identical to `concat(system_prefix, user_suffix)`."""
        sys_prefix = self.build_system_prefix(reference_text, reference_codes)
        usr_suffix = self.build_user_suffix(target_text)
        return np.concatenate([sys_prefix, usr_suffix], axis=2)
