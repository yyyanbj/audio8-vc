"""Gumbel-max top-k / top-p sampling and semantic repetition suppression for Audio8-0.1B."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def sample_codebook(
    logits: np.ndarray,
    *,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    rng: np.random.Generator | None = None,
) -> int:
    """Sample a discrete token index using top-k / top-p filtering + Gumbel-max noise."""
    values = np.asarray(logits, dtype=np.float64).reshape(-1).copy()
    vocab_size = values.size
    if vocab_size == 0:
        raise ValueError("Cannot sample from empty logits array")

    temp = max(float(temperature), 1e-5)
    p_limit = float(np.clip(top_p, 1e-5, 1.0))
    k_limit = int(np.clip(top_k, 1, vocab_size))

    # Sort descending to apply top-k and nucleus (top-p) masking
    order = np.argsort(values)[::-1]
    sorted_vals = values[order]

    # Softmax over sorted logits for cumulative probability
    max_val = sorted_vals[0]
    if not np.isfinite(max_val):
        return int(order[0])
    shifted = sorted_vals - max_val
    exp_vals = np.exp(np.clip(shifted, -700.0, 0.0))
    probs = exp_vals / np.sum(exp_vals)
    cumprobs = np.cumsum(probs)

    remove_mask = np.zeros(vocab_size, dtype=bool)
    if k_limit < vocab_size:
        remove_mask[k_limit:] = True
    # Keep first token that crosses top_p; mask strictly after it
    over_p = cumprobs > p_limit
    if np.any(over_p):
        remove_mask |= np.concatenate([[False], over_p[:-1]])
    remove_mask[0] = False  # Always retain top-1 candidate

    filtered = np.full(vocab_size, -np.inf, dtype=np.float64)
    keep_idx = order[~remove_mask]
    filtered[keep_idx] = values[keep_idx] / temp

    # Normalize retained logits to log-probabilities and add Gumbel noise
    max_kept = np.max(filtered[keep_idx])
    log_norm = max_kept + np.log(np.sum(np.exp(filtered[keep_idx] - max_kept)))
    log_probs = np.full(vocab_size, -np.inf, dtype=np.float64)
    log_probs[keep_idx] = filtered[keep_idx] - log_norm

    generator = rng if rng is not None else np.random.default_rng()
    u = np.clip(generator.random(vocab_size), 1e-12, 1.0 - 1e-12)
    gumbel = -np.log(-np.log(u))
    return int(np.argmax(log_probs + gumbel))


def sample_semantic(
    logits: np.ndarray,
    previous_tokens: Sequence[int],
    *,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_window: int = 10,
    rng: np.random.Generator | None = None,
) -> int:
    """Sample a semantic token (or EOS at last index) with step-0 EOS masking & repetition guard.

    Matches Audio8's `relative_semantic_then_eos` layout (`0..codebook_size-1` semantic,
    `codebook_size` EOS).
    """
    values = np.asarray(logits, dtype=np.float64).reshape(-1).copy()
    if not previous_tokens:
        # Step 0 must not immediately terminate generation
        values[-1] = -np.inf

    primary = sample_codebook(
        values,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        rng=rng,
    )
    fallback = sample_codebook(
        values,
        temperature=1.0,
        top_p=0.9,
        top_k=top_k,
        rng=rng,
    )

    if previous_tokens and repetition_window > 0:
        recent = previous_tokens[-repetition_window:]
        if primary in recent:
            return fallback
    return primary
