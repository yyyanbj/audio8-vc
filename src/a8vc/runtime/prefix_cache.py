"""Low-footprint in-memory LRU prefix cache for Audio8 Falcon-H1 Slow-AR states.

Instead of storing full `(24, 1, 2, 2048, 64)` KV buffers (~48 MB per voice), each snapshot
slices only `[:, :, :, :prefix_len, :]` (~6 MB for 256 tokens) plus Mamba `conv_states`
and `ssm_states` (~5 MB), totaling ~11 MB RAM per cached voice pack.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class PrefixStateSnapshot:
    """Compact snapshot of Slow-AR hybrid cache after prefilling a voice's system turn."""

    prefix_len: int
    cache_keys_prefix: np.ndarray  # [L, 1, H, prefix_len, D]
    cache_values_prefix: np.ndarray  # [L, 1, H, prefix_len, D]
    conv_states: np.ndarray  # [L, 1, C, d_conv]
    ssm_states: np.ndarray  # [L, 1, n_heads, d_head, d_state]

    @property
    def nbytes(self) -> int:
        return int(
            self.cache_keys_prefix.nbytes
            + self.cache_values_prefix.nbytes
            + self.conv_states.nbytes
            + self.ssm_states.nbytes
        )

    @classmethod
    def capture(cls, state: dict[str, np.ndarray], prefix_len: int) -> PrefixStateSnapshot:
        """Capture a compact defensive copy from a live Slow-AR workspace state."""
        if prefix_len <= 0:
            raise ValueError(f"prefix_len must be > 0, got {prefix_len}")
        max_len = state["cache_keys"].shape[3]
        if prefix_len > max_len:
            raise ValueError(f"prefix_len ({prefix_len}) exceeds cache max_seq_len ({max_len})")
        return cls(
            prefix_len=prefix_len,
            cache_keys_prefix=np.ascontiguousarray(state["cache_keys"][:, :, :, :prefix_len, :].copy()),
            cache_values_prefix=np.ascontiguousarray(state["cache_values"][:, :, :, :prefix_len, :].copy()),
            conv_states=np.ascontiguousarray(state["conv_states"].copy()),
            ssm_states=np.ascontiguousarray(state["ssm_states"].copy()),
        )

    def restore_into(self, workspace: dict[str, np.ndarray], dirty_len: int | None = None) -> None:
        """Copy this prefix snapshot into a pre-allocated Slow-AR workspace state in-place."""
        n = self.prefix_len
        workspace["cache_keys"][:, :, :, :n, :] = self.cache_keys_prefix
        workspace["cache_values"][:, :, :, :n, :] = self.cache_values_prefix
        clear_end = workspace["cache_keys"].shape[3] if dirty_len is None else min(
            workspace["cache_keys"].shape[3], max(n, dirty_len)
        )
        if clear_end > n:
            workspace["cache_keys"][:, :, :, n:clear_end, :] = 0
            workspace["cache_values"][:, :, :, n:clear_end, :] = 0
        np.copyto(workspace["conv_states"], self.conv_states)
        np.copyto(workspace["ssm_states"], self.ssm_states)


class PrefixStateCache:
    """Bounded LRU cache mapping voice pack cache_key -> `PrefixStateSnapshot`."""

    def __init__(self, max_entries: int = 2) -> None:
        self.max_entries = max(0, int(max_entries))
        self._store: OrderedDict[tuple[str, str, str], PrefixStateSnapshot] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: tuple[str, str, str]) -> PrefixStateSnapshot | None:
        if self.max_entries <= 0:
            self.misses += 1
            return None
        snap = self._store.get(key)
        if snap is None:
            self.misses += 1
            return None
        self._store.move_to_end(key)
        self.hits += 1
        return snap

    def put(self, key: tuple[str, str, str], snapshot: PrefixStateSnapshot) -> None:
        if self.max_entries <= 0:
            return
        self._store[key] = snapshot
        self._store.move_to_end(key)
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()

    @property
    def total_bytes(self) -> int:
        return sum(snap.nbytes for snap in self._store.values())

    def stats(self) -> dict[str, int | float]:
        return {
            "entries": len(self._store),
            "max_entries": self.max_entries,
            "hits": self.hits,
            "misses": self.misses,
            "memory_mb": round(self.total_bytes / (1024 * 1024), 2),
        }
