"""Tests for Audio8 prompt builder split-prefix equivalence and Gumbel-max sampling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from a8vc.runtime.manifest import load_runtime_manifest
from a8vc.runtime.prompt import PromptBuilder, clean_text, ensure_speaker_tag
from a8vc.runtime.sampling import sample_codebook, sample_semantic


@dataclass
class _Encoded:
    ids: list[int]


class _ByteTokenizer:
    """Deterministic lightweight tokenizer mapping UTF-8 bytes to token IDs."""

    def encode(self, sequence: str, add_special_tokens: bool = False) -> _Encoded:
        return _Encoded(ids=[b + 100 for b in sequence.encode("utf-8")])


def test_clean_text_cjk_and_latin() -> None:
    assert clean_text("\x00 你好\n世界 ") == "你好世界"
    assert clean_text("你好\nworld") == "你好 world"
    assert clean_text("  hello   \n\n  world  ") == "hello world"


def test_ensure_speaker_tag() -> None:
    assert ensure_speaker_tag("你好") == "<|speaker:0|>你好"
    assert ensure_speaker_tag("<|speaker:2|>hello") == "<|speaker:2|>hello"


def test_prompt_split_equals_full_prompt() -> None:
    builder = PromptBuilder(_ByteTokenizer(), semantic_begin_id=65537, num_codebooks=10)
    ref_codes = np.arange(10 * 12, dtype=np.int64).reshape(10, 12) % 4096

    sys_prefix = builder.build_system_prefix("这是参考音频", ref_codes)
    usr_suffix = builder.build_user_suffix("你好，世界！")
    full = builder.build_full("你好，世界！", "这是参考音频", ref_codes)

    assert sys_prefix.shape[0] == 1
    assert sys_prefix.shape[1] == 11
    assert usr_suffix.shape[1] == 11
    assert np.array_equal(np.concatenate([sys_prefix, usr_suffix], axis=2), full)

    # Verify reference codes occupy rows 1..10 and semantic offset occupies row 0
    nonzero_cols = np.where(sys_prefix[0, 1:, :].sum(axis=0) > 0)[0]
    assert nonzero_cols.size > 0
    ref_start = int(nonzero_cols[0])
    # Notice column 0 of ref_codes might have sum > 0 since arange starts at 0..108
    extracted_codes = sys_prefix[0, 1:, ref_start : ref_start + 12]
    assert np.array_equal(extracted_codes, ref_codes)
    assert np.array_equal(
        sys_prefix[0, 0, ref_start : ref_start + 12],
        ref_codes[0] + 65537,
    )
    # Text positions in user suffix have zeroed acoustic codebooks
    assert np.all(usr_suffix[0, 1:, :] == 0)

    # Extended prefix cache: `<|im_start|>user\n` is cached inside system_prefix, NOT user_suffix
    decoded_sys = bytes(int(x) - 100 for x in sys_prefix[0, 0, ref_start + 12 :]).decode("utf-8")
    decoded_usr = bytes(int(x) - 100 for x in usr_suffix[0, 0, :]).decode("utf-8")
    assert decoded_sys.endswith("<|im_start|>user\n")
    assert decoded_usr.startswith("你好，世界！")


def test_argpartition_sampling_matches_exact_topk_topp_support() -> None:
    rng_logits = np.random.default_rng(1234)
    logits = rng_logits.standard_normal(4097).astype(np.float32) * 3.0
    top_k = 12
    top_p = 0.75

    # Compute exact top-k / top-p support via brute-force full sort
    order = np.argsort(logits)[::-1]
    sorted_vals = logits[order].astype(np.float64)
    probs = np.exp(sorted_vals - sorted_vals[0])
    probs /= probs.sum()
    cum = np.cumsum(probs)
    over = np.nonzero(cum > top_p)[0]
    p_cut = int(over[0]) + 1 if over.size > 0 else logits.size
    expected_support = set(int(i) for i in order[: min(top_k, p_cut)])

    rng = np.random.default_rng(55)
    drawn = {
        sample_codebook(logits, temperature=0.8, top_p=top_p, top_k=top_k, rng=rng)
        for _ in range(300)
    }
    assert drawn.issubset(expected_support)
    assert int(order[0]) in drawn


def test_first_semantic_step_blocks_eos() -> None:
    logits = np.full(4097, -10.0, dtype=np.float32)
    logits[101] = 5.0
    logits[4096] = 50.0  # Huge EOS logit

    rng = np.random.default_rng(7)
    first = sample_semantic(logits, previous_tokens=[], temperature=0.5, top_k=10, rng=rng)
    assert first == 101

    later = sample_semantic(logits, previous_tokens=[101], temperature=0.5, top_k=10, rng=rng)
    assert later == 4096


def test_sampling_is_deterministic_with_seed_and_guards_repetition() -> None:
    logits = np.full(100, -5.0, dtype=np.float32)
    logits[10] = 3.0
    logits[20] = 2.9

    a = sample_codebook(logits, temperature=0.8, top_p=0.9, top_k=10, rng=np.random.default_rng(99))
    b = sample_codebook(logits, temperature=0.8, top_p=0.9, top_k=10, rng=np.random.default_rng(99))
    assert a == b
    assert a in {10, 20}


def test_load_runtime_manifest_matches_hf_schema(tmp_path: Path) -> None:
    payload = {
        "model_id": "Audio8/Audio8-TTS-Preview-0.1B-ONNX-INT8",
        "model_fingerprint": "audio8-tts-preview-0.1b-int8-v1",
        "default_precision": "int8",
        "default_codec_precision": "fp16",
        "slow_decode_model": "slow_ar_int8.onnx",
        "fast_model": "fast_ar_int8.onnx",
        "codec_models": "codec_decoder_fp16.onnx",
        "sample_rate": 44100,
        "codec_hop_length": 2048,
        "num_codebooks": 10,
        "codebook_size": 4096,
        "semantic_begin_id": 65537,
        "semantic_end_id": 69632,
        "im_end_id": 4096,
        "max_seq_len": 2048,
        "num_layers": 24,
        "n_local_heads": 2,
        "head_dim": 64,
        "num_fast_layers": 4,
        "fast_n_local_heads": 2,
        "fast_head_dim": 64,
        "mamba_d_conv": 4,
        "mamba_d_ssm": 768,
        "mamba_d_state": 64,
        "mamba_d_head": 32,
        "mamba_n_heads": 24,
        "mamba_n_groups": 1,
    }
    (tmp_path / "runtime_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    m = load_runtime_manifest(tmp_path)
    assert m.slow_model_path.name == "slow_ar_int8.onnx"
    assert m.fast_model_path.name == "fast_ar_int8.onnx"
    assert m.codec_decoder_path.name == "codec_decoder_fp16.onnx"
    assert m.conv_channels == 896
    assert m.im_end_id == 4096
    assert m.model_fingerprint == "audio8-tts-preview-0.1b-int8-v1"

