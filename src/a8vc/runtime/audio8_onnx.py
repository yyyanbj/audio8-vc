"""Audio8-TTS-0.1B ONNX INT8 streaming synthesis runtime with prefix state caching."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from a8vc.config import RuntimeConfig
from a8vc.runtime.manifest import RuntimeManifest, load_runtime_manifest
from a8vc.runtime.prefix_cache import PrefixStateCache, PrefixStateSnapshot
from a8vc.runtime.prompt import PromptBuilder
from a8vc.runtime.providers import create_onnx_session, resolve_providers
from a8vc.runtime.sampling import sample_codebook, sample_semantic
from a8vc.voices.format import VoicePack

_SLOW_INPUTS = {"codes", "position", "cache_keys", "cache_values", "conv_states", "ssm_states"}
_SLOW_OUTPUTS = {"logits", "hidden", "key_delta", "value_delta", "next_conv_states", "next_ssm_states"}
_FAST_BASE_INPUTS = {"slow_hidden", "token_id", "use_slow_hidden", "input_pos"}


@dataclass(slots=True)
class StreamAudioChunk:
    """One streamed audio chunk yielded by the Audio8 synthesizer."""

    audio: np.ndarray  # 1D float32 mono
    sample_rate: int
    frames_generated: int
    ttfa_ms: float
    prefill_ms: float
    cache_hit: bool
    is_final: bool = False


@dataclass(slots=True)
class SynthesisMetrics:
    """Performance metrics for a completed utterance synthesis."""

    voice_id: str
    num_frames: int
    audio_seconds: float
    prefill_ms: float
    ttfa_ms: float
    total_ms: float
    rtf: float
    cache_hit: bool


@runtime_checkable
class SynthesizerBackend(Protocol):
    """Protocol shared by Audio8OnnxRuntime and test/mock synthesizers."""

    @property
    def sample_rate(self) -> int: ...

    def warm_voice(self, voice: VoicePack) -> float: ...

    def stream_audio(
        self,
        text: str,
        voice: VoicePack,
        *,
        cancel_event: threading.Event | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_new_tokens: int | None = None,
        chunk_frames: int | None = None,
        seed: int | None = None,
    ) -> Iterator[StreamAudioChunk]: ...

    def decode_codes(self, codes: np.ndarray) -> np.ndarray: ...

    def cache_stats(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


def _ort_type_to_numpy(type_str: str) -> np.dtype[Any]:
    mapping = {
        "tensor(float)": np.dtype(np.float32),
        "tensor(float16)": np.dtype(np.float16),
        "tensor(int64)": np.dtype(np.int64),
        "tensor(int32)": np.dtype(np.int32),
        "tensor(bool)": np.dtype(np.bool_),
    }
    return mapping.get(type_str, np.dtype(np.float32))


def _resolve_providers(provider: str) -> list[str]:
    return resolve_providers(provider)


class Audio8OnnxRuntime:
    """High-efficiency ONNX INT8 runtime for Edge0-AI Audio8-TTS-0.1B."""

    def __init__(
        self,
        model_dir: str | Path,
        cfg: RuntimeConfig | None = None,
    ) -> None:
        from tokenizers import Tokenizer

        self.cfg = cfg or RuntimeConfig()
        self.manifest: RuntimeManifest = load_runtime_manifest(
            model_dir,
            precision=self.cfg.precision,
            codec_precision=self.cfg.codec_precision,
        )

        for label, path in (
            ("slow_ar", self.manifest.slow_model_path),
            ("fast_ar", self.manifest.fast_model_path),
            ("codec_decoder", self.manifest.codec_decoder_path),
            ("tokenizer", self.manifest.tokenizer_path),
        ):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing required Audio8 {label} file: {path}. "
                    "Run `python scripts/download_models.py --audio8-onnx`."
                )

        threads = self.cfg.threads
        if threads <= 0:
            cpu_count = os.cpu_count() or 4
            threads = max(1, min(8, cpu_count // 2))
        self.threads = threads

        self._slow, slow_ep, slow_warn = create_onnx_session(
            self.manifest.slow_model_path,
            provider=self.cfg.provider,
            threads=threads,
        )
        # If slow_ar already fell back to CPU, avoid redundant GPU error noise on fast/decoder
        effective_req = "cpu" if (slow_warn is not None and slow_ep == "CPUExecutionProvider") else self.cfg.provider
        self._fast, fast_ep, fast_warn = create_onnx_session(
            self.manifest.fast_model_path,
            provider=effective_req,
            threads=threads,
        )
        self._decoder, dec_ep, dec_warn = create_onnx_session(
            self.manifest.codec_decoder_path,
            provider=effective_req,
            threads=threads,
        )
        self.active_provider: str = slow_ep
        self.session_providers: dict[str, str] = {
            "slow_ar": slow_ep,
            "fast_ar": fast_ep,
            "codec_decoder": dec_ep,
        }
        self.provider_warning: str | None = slow_warn or fast_warn or dec_warn
        self._tokenizer = Tokenizer.from_file(str(self.manifest.tokenizer_path))

        self.prompt_builder = PromptBuilder(
            self._tokenizer,
            semantic_begin_id=self.manifest.semantic_begin_id,
            num_codebooks=self.manifest.num_codebooks,
        )
        self.prefix_cache = PrefixStateCache(max_entries=self.cfg.prefix_cache_size)
        self._lock = threading.Lock()

        self._validate_and_inspect_contract()
        # Reusable single workspace avoids repeated 48 MB allocations across utterances
        self._workspace = self._allocate_slow_state()
        self._dirty_seq_len = self.manifest.max_seq_len

    @property
    def sample_rate(self) -> int:
        return self.manifest.sample_rate

    def _validate_and_inspect_contract(self) -> None:
        slow_in = {inp.name: inp for inp in self._slow.get_inputs()}
        slow_out = {out.name: out for out in self._slow.get_outputs()}
        if not _SLOW_INPUTS.issubset(slow_in.keys()):
            raise RuntimeError(f"Unexpected slow_ar inputs: {sorted(slow_in.keys())}")
        if not _SLOW_OUTPUTS.issubset(slow_out.keys()):
            raise RuntimeError(f"Unexpected slow_ar outputs: {sorted(slow_out.keys())}")

        self._slow_dtypes = {name: _ort_type_to_numpy(inp.type) for name, inp in slow_in.items()}
        conv_shape = slow_in["conv_states"].shape
        if len(conv_shape) == 4 and isinstance(conv_shape[2], int):
            self._conv_channels = conv_shape[2]
        else:
            self._conv_channels = self.manifest.conv_channels

        fast_in = {inp.name: inp for inp in self._fast.get_inputs()}
        if not _FAST_BASE_INPUTS.issubset(fast_in.keys()):
            raise RuntimeError(f"Unexpected fast_ar inputs: {sorted(fast_in.keys())}")
        self._fast_dtypes = {name: _ort_type_to_numpy(inp.type) for name, inp in fast_in.items()}

        ck0_shape = fast_in["cache_key_0"].shape
        self._fast_kv_heads = int(ck0_shape[1]) if len(ck0_shape) == 4 and isinstance(ck0_shape[1], int) else self.manifest.n_local_heads
        self._fast_head_dim = int(ck0_shape[3]) if len(ck0_shape) == 4 and isinstance(ck0_shape[3], int) else self.manifest.fast_head_dim
        self._fast_output_names = [out.name for out in self._fast.get_outputs()]
        self._slow_output_names = [
            "logits",
            "hidden",
            "key_delta",
            "value_delta",
            "next_conv_states",
            "next_ssm_states",
        ]

        dec_in = self._decoder.get_inputs()[0]
        self._decoder_input_name = dec_in.name
        self._decoder_input_dtype = _ort_type_to_numpy(dec_in.type)

    def _allocate_slow_state(self) -> dict[str, np.ndarray]:
        m = self.manifest
        return {
            "cache_keys": np.zeros(
                (m.num_layers, 1, m.n_local_heads, m.max_seq_len, m.head_dim),
                dtype=self._slow_dtypes["cache_keys"],
            ),
            "cache_values": np.zeros(
                (m.num_layers, 1, m.n_local_heads, m.max_seq_len, m.head_dim),
                dtype=self._slow_dtypes["cache_values"],
            ),
            "conv_states": np.zeros(
                (m.num_layers, 1, self._conv_channels, m.mamba_d_conv),
                dtype=self._slow_dtypes["conv_states"],
            ),
            "ssm_states": np.zeros(
                (m.num_layers, 1, m.mamba_n_heads, m.mamba_d_head, m.mamba_d_state),
                dtype=self._slow_dtypes["ssm_states"],
            ),
        }

    def _zero_workspace(self) -> None:
        end = min(self.manifest.max_seq_len, self._dirty_seq_len)
        self._workspace["cache_keys"][:, :, :, :end, :] = 0
        self._workspace["cache_values"][:, :, :, :end, :] = 0
        self._workspace["conv_states"].fill(0)
        self._workspace["ssm_states"].fill(0)
        self._dirty_seq_len = 0

    def _allocate_fast_caches(self) -> dict[str, np.ndarray]:
        m = self.manifest
        caches: dict[str, np.ndarray] = {}
        for idx in range(m.num_fast_layers):
            k_name = f"cache_key_{idx}"
            v_name = f"cache_value_{idx}"
            caches[k_name] = np.zeros(
                (1, self._fast_kv_heads, m.num_codebooks, self._fast_head_dim),
                dtype=self._fast_dtypes[k_name],
            )
            caches[v_name] = np.zeros(
                (1, self._fast_kv_heads, m.num_codebooks, self._fast_head_dim),
                dtype=self._fast_dtypes[v_name],
            )
        return caches

    def _slow_step(
        self,
        column: np.ndarray,
        position: int,
        state: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        pos_arr = np.array([position], dtype=self._slow_dtypes["position"])
        feed = {
            "codes": np.ascontiguousarray(column, dtype=self._slow_dtypes["codes"]),
            "position": pos_arr,
            "cache_keys": state["cache_keys"],
            "cache_values": state["cache_values"],
            "conv_states": state["conv_states"],
            "ssm_states": state["ssm_states"],
        }
        logits, hidden, key_delta, value_delta, next_conv, next_ssm = self._slow.run(
            self._slow_output_names, feed
        )
        state["cache_keys"][:, :, :, position, :] = key_delta[:, :, :, 0, :] if key_delta.ndim == 5 else key_delta
        state["cache_values"][:, :, :, position, :] = value_delta[:, :, :, 0, :] if value_delta.ndim == 5 else value_delta
        np.copyto(state["conv_states"], next_conv)
        np.copyto(state["ssm_states"], next_ssm)
        return logits[0, -1], hidden[:, -1:, :]

    def _fast_step(
        self,
        slow_hidden: np.ndarray,
        token_id: int,
        use_slow_hidden: bool,
        input_pos: int,
        fast_caches: dict[str, np.ndarray],
    ) -> np.ndarray:
        feed: dict[str, np.ndarray] = {
            "slow_hidden": np.ascontiguousarray(slow_hidden, dtype=self._fast_dtypes["slow_hidden"]),
            "token_id": np.array([[token_id]], dtype=self._fast_dtypes["token_id"]),
            "use_slow_hidden": np.array([use_slow_hidden], dtype=self._fast_dtypes["use_slow_hidden"]),
            "input_pos": np.array([input_pos], dtype=self._fast_dtypes["input_pos"]),
        }
        feed.update(fast_caches)
        outputs = self._fast.run(self._fast_output_names, feed)
        logits = outputs[0][0, -1]

        out_idx = 1
        for layer_idx in range(self.manifest.num_fast_layers):
            k_delta = outputs[out_idx]
            v_delta = outputs[out_idx + 1]
            out_idx += 2
            k_len = k_delta.shape[2]
            v_len = v_delta.shape[2]
            fast_caches[f"cache_key_{layer_idx}"][:, :, input_pos : input_pos + k_len, :] = k_delta
            fast_caches[f"cache_value_{layer_idx}"][:, :, input_pos : input_pos + v_len, :] = v_delta
        return logits

    def _prepare_voice_prefix(self, voice: VoicePack) -> tuple[int, bool]:
        """Restore cached Slow-AR prefix state or prefill system turn and store snapshot."""
        key = voice.cache_key
        snap = self.prefix_cache.get(key)
        if snap is not None:
            snap.restore_into(self._workspace, dirty_len=self._dirty_seq_len)
            self._dirty_seq_len = snap.prefix_len
            return snap.prefix_len, True

        sys_prompt = self.prompt_builder.build_system_prefix(
            voice.manifest.reference_text, voice.codes
        )
        prefix_len = int(sys_prompt.shape[2])
        if prefix_len >= self.manifest.max_seq_len - 16:
            raise ValueError(
                f"Voice {voice.voice_id!r} reference prefix ({prefix_len} tokens) is too long "
                f"for max_seq_len={self.manifest.max_seq_len}; trim reference audio to <=15s."
            )

        self._zero_workspace()
        for pos in range(prefix_len):
            self._slow_step(sys_prompt[:, :, pos : pos + 1], pos, self._workspace)
        self._dirty_seq_len = prefix_len
        self.prefix_cache.put(key, PrefixStateSnapshot.capture(self._workspace, prefix_len))
        return prefix_len, False

    def warm_voice(self, voice: VoicePack) -> float:
        """Pre-compute and cache a voice's Slow-AR prefix state; returns elapsed ms."""
        t0 = time.perf_counter()
        with self._lock:
            self._prepare_voice_prefix(voice)
        return (time.perf_counter() - t0) * 1000.0

    def decode_codes(self, codes: np.ndarray) -> np.ndarray:
        """Decode `[10, T]` or `[1, 10, T]` codec codes into a 1D float32 waveform."""
        arr = np.asarray(codes, dtype=self._decoder_input_dtype)
        if arr.ndim == 2:
            arr = arr[None, :, :]
        if arr.ndim != 3 or arr.shape[1] != self.manifest.num_codebooks:
            raise ValueError(
                f"Expected codes shape [1, {self.manifest.num_codebooks}, T], got {arr.shape}"
            )
        audio = self._decoder.run(None, {self._decoder_input_name: np.ascontiguousarray(arr)})[0]
        return np.ascontiguousarray(np.asarray(audio, dtype=np.float32).reshape(-1))

    def stream_audio(
        self,
        text: str,
        voice: VoicePack,
        *,
        cancel_event: threading.Event | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_new_tokens: int | None = None,
        chunk_frames: int | None = None,
        seed: int | None = None,
    ) -> Iterator[StreamAudioChunk]:
        """Generate and stream target-voice audio chunks for `text` using `voice`."""
        temp = self.cfg.temperature if temperature is None else temperature
        p_val = self.cfg.top_p if top_p is None else top_p
        k_val = self.cfg.top_k if top_k is None else top_k
        max_steps = self.cfg.max_new_tokens if max_new_tokens is None else max_new_tokens
        emit_every = max(2, self.cfg.chunk_frames if chunk_frames is None else chunk_frames)
        context_frames = min(self.cfg.stream_context_frames, self.manifest.stream_context_frames)
        guard_frames = self.cfg.stream_guard_frames
        hop = self.manifest.codec_hop_length
        guard_samples = guard_frames * hop
        rng = np.random.default_rng(seed)

        user_prompt = self.prompt_builder.build_user_suffix(text)
        user_len = int(user_prompt.shape[2])

        with self._lock:
            t_start = time.perf_counter()
            prefix_len, cache_hit = self._prepare_voice_prefix(voice)

            max_avail = self.manifest.max_seq_len - (prefix_len + user_len)
            if max_avail <= 1:
                raise ValueError(
                    f"Prompt length ({prefix_len + user_len}) leaves no generation room "
                    f"in max_seq_len={self.manifest.max_seq_len}"
                )
            step_budget = min(int(max_steps), max_avail)

            logits = np.empty(0, dtype=np.float32)
            hidden = np.empty(0, dtype=np.float32)
            for offset in range(user_len):
                if cancel_event is not None and cancel_event.is_set():
                    return
                pos = prefix_len + offset
                logits, hidden = self._slow_step(
                    user_prompt[:, :, offset : offset + 1], pos, self._workspace
                )
                self._dirty_seq_len = max(self._dirty_seq_len, pos + 1)

            prefill_ms = (time.perf_counter() - t_start) * 1000.0
            prompt_len = prefix_len + user_len

            frames: list[np.ndarray] = []
            semantic_history: list[int] = []
            emitted_samples = 0
            ttfa_ms = 0.0
            eos_index = self.manifest.codebook_size

            for step in range(step_budget):
                if cancel_event is not None and cancel_event.is_set():
                    return

                rel_idx = sample_semantic(
                    logits,
                    semantic_history,
                    temperature=temp,
                    top_p=p_val,
                    top_k=k_val,
                    rng=rng,
                )
                if rel_idx == eos_index:
                    break

                semantic_token = self.manifest.semantic_begin_id + rel_idx
                semantic_history.append(rel_idx)

                frame = np.zeros(self.manifest.num_codebooks, dtype=np.int64)
                frame[0] = rel_idx

                fast_caches = self._allocate_fast_caches()
                # Fast step 0 ingests slow_hidden; steps 1..9 decode acoustic codebooks
                _ = self._fast_step(hidden, 0, True, 0, fast_caches)
                prev_code = int(frame[0])
                for cb_idx in range(1, self.manifest.num_codebooks):
                    cb_logits = self._fast_step(hidden, prev_code, False, cb_idx, fast_caches)
                    prev_code = sample_codebook(
                        cb_logits,
                        temperature=temp,
                        top_p=p_val,
                        top_k=k_val,
                        rng=rng,
                    )
                    frame[cb_idx] = prev_code

                frames.append(frame)

                # Feed completed frame back into Slow-AR for next frame prediction
                column = np.empty((1, self.manifest.num_codebooks + 1, 1), dtype=np.int64)
                column[0, 0, 0] = semantic_token
                column[0, 1:, 0] = frame
                next_pos = prompt_len + step
                if next_pos < self.manifest.max_seq_len:
                    logits, hidden = self._slow_step(column, next_pos, self._workspace)
                    self._dirty_seq_len = max(self._dirty_seq_len, next_pos + 1)

                if len(frames) % emit_every == 0:
                    start_frame = max(0, len(frames) - context_frames - emit_every)
                    window = np.stack(frames[start_frame:], axis=1)  # [10, W]
                    decoded = self.decode_codes(window)
                    abs_start = start_frame * hop
                    stable_end = abs_start + max(0, decoded.size - guard_samples)
                    begin = max(0, emitted_samples - abs_start)
                    end = max(begin, stable_end - abs_start)
                    if end > begin:
                        piece = decoded[begin:end].copy()
                        emitted_samples = abs_start + end
                        if ttfa_ms == 0.0:
                            ttfa_ms = (time.perf_counter() - t_start) * 1000.0
                        yield StreamAudioChunk(
                            audio=piece,
                            sample_rate=self.manifest.sample_rate,
                            frames_generated=len(frames),
                            ttfa_ms=ttfa_ms,
                            prefill_ms=prefill_ms,
                            cache_hit=cache_hit,
                            is_final=False,
                        )

            # Flush final tail samples
            if frames and not (cancel_event is not None and cancel_event.is_set()):
                start_frame = max(0, len(frames) - context_frames - emit_every)
                window = np.stack(frames[start_frame:], axis=1)
                decoded = self.decode_codes(window)
                abs_start = start_frame * hop
                begin = max(0, emitted_samples - abs_start)
                tail = decoded[begin:].copy() if begin < decoded.size else np.zeros(0, dtype=np.float32)
                if ttfa_ms == 0.0:
                    ttfa_ms = (time.perf_counter() - t_start) * 1000.0
                yield StreamAudioChunk(
                    audio=tail,
                    sample_rate=self.manifest.sample_rate,
                    frames_generated=len(frames),
                    ttfa_ms=ttfa_ms,
                    prefill_ms=prefill_ms,
                    cache_hit=cache_hit,
                    is_final=True,
                )

    def synthesize(
        self,
        text: str,
        voice: VoicePack,
        *,
        seed: int | None = None,
    ) -> tuple[np.ndarray, SynthesisMetrics]:
        """Synthesize full utterance and collect latency / RTF metrics."""
        t0 = time.perf_counter()
        pieces: list[np.ndarray] = []
        last_chunk: StreamAudioChunk | None = None
        for chunk in self.stream_audio(text, voice, seed=seed):
            last_chunk = chunk
            if chunk.audio.size > 0:
                pieces.append(chunk.audio)
        total_ms = (time.perf_counter() - t0) * 1000.0
        audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
        audio_sec = audio.size / float(self.manifest.sample_rate)
        rtf = (total_ms / 1000.0) / audio_sec if audio_sec > 0 else 0.0
        metrics = SynthesisMetrics(
            voice_id=voice.voice_id,
            num_frames=last_chunk.frames_generated if last_chunk else 0,
            audio_seconds=round(audio_sec, 3),
            prefill_ms=round(last_chunk.prefill_ms if last_chunk else 0.0, 2),
            ttfa_ms=round(last_chunk.ttfa_ms if last_chunk else total_ms, 2),
            total_ms=round(total_ms, 2),
            rtf=round(rtf, 3),
            cache_hit=last_chunk.cache_hit if last_chunk else False,
        )
        return audio, metrics

    def cache_stats(self) -> dict[str, Any]:
        return self.prefix_cache.stats()

    def close(self) -> None:
        self.prefix_cache.clear()
        self._slow = None
        self._fast = None
        self._decoder = None
