"""Tests for compact Slow-AR KV/SSM prefix state LRU cache."""

from __future__ import annotations

import numpy as np

from a8vc.runtime.prefix_cache import PrefixStateCache, PrefixStateSnapshot


def _make_state(max_seq_len: int = 128, fill_prefix: int = 20, value: float = 1.0) -> dict[str, np.ndarray]:
    state = {
        "cache_keys": np.zeros((2, 1, 2, max_seq_len, 8), dtype=np.float32),
        "cache_values": np.zeros((2, 1, 2, max_seq_len, 8), dtype=np.float32),
        "conv_states": np.full((2, 1, 16, 4), value, dtype=np.float32),
        "ssm_states": np.full((2, 1, 4, 4, 8), value * 2.0, dtype=np.float32),
    }
    state["cache_keys"][:, :, :, :fill_prefix, :] = value
    state["cache_values"][:, :, :, :fill_prefix, :] = value * 3.0
    return state


def test_snapshot_slices_kv_and_restores_cleanly() -> None:
    src = _make_state(max_seq_len=128, fill_prefix=24, value=2.5)
    snap = PrefixStateSnapshot.capture(src, prefix_len=24)

    # Snapshot only holds prefix_len=24, not full 128
    assert snap.cache_keys_prefix.shape == (2, 1, 2, 24, 8)
    full_bytes = src["cache_keys"].nbytes + src["cache_values"].nbytes
    assert snap.cache_keys_prefix.nbytes + snap.cache_values_prefix.nbytes < full_bytes / 4

    # Mutate source after capture to confirm defensive copy isolation
    src["cache_keys"].fill(99.0)
    src["ssm_states"].fill(99.0)

    # Restore into a dirty workspace buffer
    workspace = _make_state(max_seq_len=128, fill_prefix=80, value=7.0)
    snap.restore_into(workspace, dirty_len=80)

    assert np.all(workspace["cache_keys"][:, :, :, :24, :] == 2.5)
    assert np.all(workspace["cache_values"][:, :, :, :24, :] == 7.5)
    assert np.all(workspace["cache_keys"][:, :, :, 24:80, :] == 0.0)
    assert np.all(workspace["cache_values"][:, :, :, 24:80, :] == 0.0)
    assert np.all(workspace["conv_states"] == 2.5)
    assert np.all(workspace["ssm_states"] == 5.0)


def test_prefix_cache_lru_eviction_and_metrics() -> None:
    cache = PrefixStateCache(max_entries=2)
    k1 = ("v1", "h1", "t1")
    k2 = ("v2", "h2", "t2")
    k3 = ("v3", "h3", "t3")

    cache.put(k1, PrefixStateSnapshot.capture(_make_state(fill_prefix=10, value=1.0), 10))
    cache.put(k2, PrefixStateSnapshot.capture(_make_state(fill_prefix=12, value=2.0), 12))

    # Touch k1 so k2 becomes least-recently-used
    assert cache.get(k1) is not None
    cache.put(k3, PrefixStateSnapshot.capture(_make_state(fill_prefix=14, value=3.0), 14))

    assert cache.get(k2) is None  # evicted
    assert cache.get(k1) is not None
    assert cache.get(k3) is not None

    stats = cache.stats()
    assert stats["entries"] == 2
    assert stats["hits"] == 3
    assert stats["misses"] == 1
