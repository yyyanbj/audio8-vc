"""Audio8-TTS-0.1B ONNX INT8 streaming synthesis runtime with prefix state caching."""

from __future__ import annotations

import logging
import queue
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
from a8vc.runtime.providers import (
    compute_global_pool_threads,
    compute_thread_plan,
    create_onnx_session,
    is_global_thread_pool_enabled,
    resolve_providers,
    select_model_provider,
)
from a8vc.runtime.sampling import sample_codebook, sample_semantic
from a8vc.runtime.slow_ar_kv import prepare_compact_slow_ar_model
from a8vc.voices.format import VoicePack

logger = logging.getLogger("a8vc.runtime.audio8_onnx")

_SLOW_INPUTS = {"codes", "position", "cache_keys", "cache_values", "conv_states", "ssm_states"}
_SLOW_OUTPUTS = {"logits", "hidden", "key_delta", "value_delta", "next_conv_states", "next_ssm_states"}
_FAST_BASE_INPUTS = {"slow_hidden", "token_id", "use_slow_hidden", "input_pos"}
_GUARD_FADE_SAMPLES = 256  # ~5.8 ms at 44.1 kHz; zero-latency overlap crossfade inside guard frame


def estimate_max_audio_frames(text: str, *, hard_cap: int = 512) -> int:
    """Compute a generous text-proportional frame ceiling (~21.53 fps) to prevent runaway AR decoding.

    When Audio8-0.1B stochastically misses `<|im_end|>` on a short phrase, bounding steps
    proportionally to CJK/non-CJK text length avoids 24-second babble stalls on CPU while
    leaving >=3x duration headroom for normal speech.
    """
    cap = max(32, int(hard_cap))
    stripped = text.strip()
    if not stripped:
        return min(cap, 96)
    cjk_chars = sum(
        1
        for ch in stripped
        if ("一" <= ch <= "鿿")
        or ("぀" <= ch <= "ヿ")
        or ("가" <= ch <= "힯")
    )
    other_chars = len(stripped) - cjk_chars
    est = int(cjk_chars * 10 + other_chars * 3.5 + 88)
    return min(cap, max(96, est))


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
    num_chunks: int = 0
    total_stall_ms: float = 0.0
    max_stall_ms: float = 0.0


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
        first_chunk_frames: int | None = None,
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

        slow_req, slow_note = select_model_provider(
            self.cfg.provider,
            model_role="slow_ar",
            precision=self.cfg.precision,
        )
        fast_req, fast_note = select_model_provider(
            self.cfg.provider,
            model_role="fast_ar",
            precision=self.cfg.precision,
        )
        dec_req, _ = select_model_provider(
            self.cfg.provider,
            model_role="codec_decoder",
            precision=self.cfg.codec_precision,
        )

        # Determine whether codec_decoder will execute on GPU so CPU AR & Codec
        # thread pools can be sized without oversubscribing physical cores.
        try:
            dec_resolved = resolve_providers(dec_req)
            codec_expected_gpu = bool(dec_resolved and dec_resolved[0] != "CPUExecutionProvider")
        except Exception:
            codec_expected_gpu = False

        want_global_pool = bool(
            (
                self.cfg.use_global_thread_pool
                and self.cfg.ar_threads == 0
                and self.cfg.codec_threads == 0
            )
            or is_global_thread_pool_enabled()
        )
        if want_global_pool:
            global_threads, self.threads = compute_global_pool_threads(
                threads=self.cfg.threads,
                codec_on_gpu=codec_expected_gpu,
                pipeline_decode=self.cfg.pipeline_decode,
            )
            self.use_global_thread_pool = True
            self.global_pool_threads: int | None = global_threads
            self.ar_threads = global_threads
            self.codec_threads = 2 if codec_expected_gpu else global_threads
            self.fast_threads = global_threads
            fast_spin = True
        else:
            self.ar_threads, self.codec_threads, self.threads = compute_thread_plan(
                threads=self.cfg.threads,
                ar_threads=self.cfg.ar_threads,
                codec_threads=self.cfg.codec_threads,
                codec_on_gpu=codec_expected_gpu,
                pipeline_decode=self.cfg.pipeline_decode,
            )
            self.use_global_thread_pool = False
            self.global_pool_threads = None
            # Per-session mode: Fast-AR executes tiny 1-token steps strictly sequentially
            # inside the AR worker. Running it with 1 thread and allow_spinning=False avoids
            # spawning an idle busy-spinning thread pool that starves Slow-AR and Codec.
            self.fast_threads = 1
            fast_spin = False

        # Bit-exact Slow-AR KV-cache compaction (2048 -> 512) cuts KV copy/mask overhead 4x
        # while seamlessly auto-expanding back to manifest.max_seq_len on long prompts.
        self._slow_model_path_active, self._active_max_seq_len = prepare_compact_slow_ar_model(
            self.manifest.slow_model_path,
            target_seq_len=self.cfg.slow_kv_max_seq_len,
            orig_seq_len=self.manifest.max_seq_len,
        )

        self._slow, slow_ep, slow_warn = create_onnx_session(
            self._slow_model_path_active,
            provider=slow_req,
            threads=self.ar_threads,
            inter_op_threads=1,
            allow_spinning=True,
            use_global_pool=self.use_global_thread_pool,
            global_pool_threads=self.global_pool_threads,
        )

        # If slow_ar attempted GPU and failed, avoid repeating identical GPU init failure on fast_ar
        if slow_warn is not None and slow_ep == "CPUExecutionProvider":
            fast_req = "cpu"
        self._fast, fast_ep, fast_warn = create_onnx_session(
            self.manifest.fast_model_path,
            provider=fast_req,
            threads=self.fast_threads,
            inter_op_threads=1,
            allow_spinning=fast_spin,
            use_global_pool=self.use_global_thread_pool,
        )

        if slow_warn is not None and slow_ep == "CPUExecutionProvider" and slow_note is None:
            dec_req = "cpu"
        self._decoder, dec_ep, dec_warn = create_onnx_session(
            self.manifest.codec_decoder_path,
            provider=dec_req,
            threads=self.codec_threads,
            inter_op_threads=1,
            allow_spinning=True,
            use_global_pool=self.use_global_thread_pool,
        )

        self.session_providers: dict[str, str] = {
            "slow_ar": slow_ep,
            "fast_ar": fast_ep,
            "codec_decoder": dec_ep,
        }
        self.active_provider: str = self._compute_active_provider()
        self.provider_note: str | None = slow_note or fast_note
        self.provider_warning: str | None = slow_warn or fast_warn or dec_warn
        if self.provider_note:
            logger.info("%s", self.provider_note)
        self._tokenizer = Tokenizer.from_file(str(self.manifest.tokenizer_path))

        self.prompt_builder = PromptBuilder(
            self._tokenizer,
            semantic_begin_id=self.manifest.semantic_begin_id,
            num_codebooks=self.manifest.num_codebooks,
        )
        self.prefix_cache = PrefixStateCache(max_entries=self.cfg.prefix_cache_size)
        self._lock = threading.Lock()
        self._decoder_lock = threading.Lock()
        self._decoder_warmed = False

        # Precompute raised-cosine weights for zero-latency guard-overlap crossfade
        t_fade = np.linspace(0.0, 1.0, _GUARD_FADE_SAMPLES, dtype=np.float32)
        self._guard_fade_out = 0.5 * (1.0 + np.cos(np.pi * t_fade))
        self._guard_fade_in = 1.0 - self._guard_fade_out

        self._validate_and_inspect_contract()
        # Reusable workspace + ping-pong Mamba state pair avoids per-step 5 MB copies
        self._workspace = self._allocate_slow_state()
        self._conv_pair = [
            self._workspace["conv_states"],
            np.zeros_like(self._workspace["conv_states"]),
        ]
        self._ssm_pair = [
            self._workspace["ssm_states"],
            np.zeros_like(self._workspace["ssm_states"]),
        ]
        self._slow_state_idx = 0
        self._dirty_seq_len = self._active_max_seq_len

        # Reusable Fast-AR buffers avoid per-frame allocations
        self._fast_caches = self._allocate_fast_caches()
        self._slow_iobs: list[Any] | None = None
        self._fast_iob: Any = None
        self._init_iobinding()

    @property
    def sample_rate(self) -> int:
        return self.manifest.sample_rate

    @property
    def active_max_seq_len(self) -> int:
        return self._active_max_seq_len

    def _validate_and_inspect_contract(self) -> None:
        slow_in = {inp.name: inp for inp in self._slow.get_inputs()}
        slow_out = {out.name: out for out in self._slow.get_outputs()}
        if not _SLOW_INPUTS.issubset(slow_in.keys()):
            raise RuntimeError(f"Unexpected slow_ar inputs: {sorted(slow_in.keys())}")
        if not _SLOW_OUTPUTS.issubset(slow_out.keys()):
            raise RuntimeError(f"Unexpected slow_ar outputs: {sorted(slow_out.keys())}")

        self._slow_dtypes = {name: _ort_type_to_numpy(inp.type) for name, inp in slow_in.items()}
        ck_shape = slow_in["cache_keys"].shape
        if len(ck_shape) == 5 and isinstance(ck_shape[3], int) and ck_shape[3] > 0:
            self._active_max_seq_len = int(ck_shape[3])
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
        seq_len = self._active_max_seq_len
        return {
            "cache_keys": np.zeros(
                (m.num_layers, 1, m.n_local_heads, seq_len, m.head_dim),
                dtype=self._slow_dtypes["cache_keys"],
            ),
            "cache_values": np.zeros(
                (m.num_layers, 1, m.n_local_heads, seq_len, m.head_dim),
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
        end = min(self._active_max_seq_len, self._dirty_seq_len)
        self._workspace["cache_keys"][:, :, :, :end, :] = 0
        self._workspace["cache_values"][:, :, :, :end, :] = 0
        self._workspace["conv_states"].fill(0)
        self._workspace["ssm_states"].fill(0)
        self._dirty_seq_len = 0

    def _expand_slow_ar_to_full_context(self) -> bool:
        """Seamlessly reload original full-context `slow_ar` if a long voice/prompt exceeds compact KV."""
        if (
            self._active_max_seq_len >= self.manifest.max_seq_len
            or self._slow_model_path_active == self.manifest.slow_model_path
        ):
            return False

        logger.info(
            "Expanding Slow-AR KV-cache from %d to full manifest max_seq_len=%d for long context",
            self._active_max_seq_len,
            self.manifest.max_seq_len,
        )
        slow_req = "cpu" if self.session_providers.get("slow_ar") == "CPUExecutionProvider" else self.cfg.provider
        full_sess, full_ep, _ = create_onnx_session(
            self.manifest.slow_model_path,
            provider=slow_req,
            threads=self.ar_threads,
            inter_op_threads=1,
            allow_spinning=True,
            use_global_pool=self.use_global_thread_pool,
        )
        self._slow = full_sess
        self._slow_model_path_active = self.manifest.slow_model_path
        self.session_providers["slow_ar"] = full_ep
        self._active_max_seq_len = self.manifest.max_seq_len
        self._validate_and_inspect_contract()
        self._workspace = self._allocate_slow_state()
        self._conv_pair = [
            self._workspace["conv_states"],
            np.zeros_like(self._workspace["conv_states"]),
        ]
        self._ssm_pair = [
            self._workspace["ssm_states"],
            np.zeros_like(self._workspace["ssm_states"]),
        ]
        self._slow_state_idx = 0
        self._dirty_seq_len = self._active_max_seq_len
        self._init_iobinding()
        return True

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

    def _compute_active_provider(self) -> str:
        for role in ("slow_ar", "codec_decoder", "fast_ar"):
            ep = self.session_providers.get(role, "CPUExecutionProvider")
            if ep != "CPUExecutionProvider":
                return ep
        return "CPUExecutionProvider"

    def _init_iobinding(self) -> None:
        """Bind pre-allocated CPU OrtValue buffers for zero-copy Slow-AR and Fast-AR execution."""
        self._slow_iobs = None
        self._fast_iob = None
        try:
            import onnxruntime as ort
        except Exception:
            return

        # Slow-AR CPU IOBinding with ping-pong Mamba conv/ssm buffers (eliminates 5 MB copy/step)
        if (
            self.session_providers.get("slow_ar") == "CPUExecutionProvider"
            and hasattr(self._slow, "io_binding")
        ):
            try:
                out_meta = {out.name: out for out in self._slow.get_outputs()}
                concrete_shapes: dict[str, tuple[int, ...]] = {}
                for name in ("logits", "hidden", "key_delta", "value_delta"):
                    raw_shape = out_meta[name].shape
                    if not raw_shape or not all(isinstance(d, int) and d > 0 for d in raw_shape):
                        concrete_shapes = {}
                        break
                    concrete_shapes[name] = tuple(int(d) for d in raw_shape)

                if concrete_shapes:
                    self._slow_col_buf = np.zeros(
                        (1, self.manifest.num_codebooks + 1, 1),
                        dtype=self._slow_dtypes["codes"],
                    )
                    self._slow_pos_buf = np.zeros((1,), dtype=self._slow_dtypes["position"])
                    self._slow_logits_buf = np.empty(
                        concrete_shapes["logits"],
                        dtype=_ort_type_to_numpy(out_meta["logits"].type),
                    )
                    self._slow_hidden_buf = np.empty(
                        concrete_shapes["hidden"],
                        dtype=_ort_type_to_numpy(out_meta["hidden"].type),
                    )
                    self._slow_kd_buf = np.empty(
                        concrete_shapes["key_delta"],
                        dtype=_ort_type_to_numpy(out_meta["key_delta"].type),
                    )
                    self._slow_vd_buf = np.empty(
                        concrete_shapes["value_delta"],
                        dtype=_ort_type_to_numpy(out_meta["value_delta"].type),
                    )

                    ov_col = ort.OrtValue.ortvalue_from_numpy(self._slow_col_buf)
                    ov_pos = ort.OrtValue.ortvalue_from_numpy(self._slow_pos_buf)
                    ov_ck = ort.OrtValue.ortvalue_from_numpy(self._workspace["cache_keys"])
                    ov_cv = ort.OrtValue.ortvalue_from_numpy(self._workspace["cache_values"])
                    ov_conv = [ort.OrtValue.ortvalue_from_numpy(x) for x in self._conv_pair]
                    ov_ssm = [ort.OrtValue.ortvalue_from_numpy(x) for x in self._ssm_pair]
                    ov_logits = ort.OrtValue.ortvalue_from_numpy(self._slow_logits_buf)
                    ov_hidden = ort.OrtValue.ortvalue_from_numpy(self._slow_hidden_buf)
                    ov_kd = ort.OrtValue.ortvalue_from_numpy(self._slow_kd_buf)
                    ov_vd = ort.OrtValue.ortvalue_from_numpy(self._slow_vd_buf)
                    # Hold strong references so OrtValue memory pointers remain alive
                    self._slow_ort_refs = (
                        ov_col,
                        ov_pos,
                        ov_ck,
                        ov_cv,
                        ov_conv,
                        ov_ssm,
                        ov_logits,
                        ov_hidden,
                        ov_kd,
                        ov_vd,
                    )

                    iobs: list[Any] = []
                    for cur in (0, 1):
                        nxt = 1 - cur
                        iob = self._slow.io_binding()
                        iob.bind_ortvalue_input("codes", ov_col)
                        iob.bind_ortvalue_input("position", ov_pos)
                        iob.bind_ortvalue_input("cache_keys", ov_ck)
                        iob.bind_ortvalue_input("cache_values", ov_cv)
                        iob.bind_ortvalue_input("conv_states", ov_conv[cur])
                        iob.bind_ortvalue_input("ssm_states", ov_ssm[cur])
                        iob.bind_ortvalue_output("logits", ov_logits)
                        iob.bind_ortvalue_output("hidden", ov_hidden)
                        iob.bind_ortvalue_output("key_delta", ov_kd)
                        iob.bind_ortvalue_output("value_delta", ov_vd)
                        iob.bind_ortvalue_output("next_conv_states", ov_conv[nxt])
                        iob.bind_ortvalue_output("next_ssm_states", ov_ssm[nxt])
                        iobs.append(iob)
                    self._slow_iobs = iobs
            except Exception as exc:
                logger.debug("Slow-AR IOBinding unavailable, using standard run(): %s", exc)
                self._slow_iobs = None

        # Fast-AR CPU IOBinding with reusable cache & output buffers
        if (
            self.session_providers.get("fast_ar") == "CPUExecutionProvider"
            and hasattr(self._fast, "io_binding")
        ):
            try:
                fast_in = {inp.name: inp for inp in self._fast.get_inputs()}
                fast_outs = list(self._fast.get_outputs())
                hidden_shape = fast_in["slow_hidden"].shape
                if (
                    all(isinstance(d, int) and d > 0 for d in hidden_shape)
                    and all(
                        out.shape and all(isinstance(d, int) and d > 0 for d in out.shape)
                        for out in fast_outs
                    )
                ):
                    self._fast_hidden_buf = np.zeros(
                        tuple(int(d) for d in hidden_shape),
                        dtype=self._fast_dtypes["slow_hidden"],
                    )
                    self._fast_tok_buf = np.zeros((1, 1), dtype=self._fast_dtypes["token_id"])
                    self._fast_use_buf = np.array([True], dtype=self._fast_dtypes["use_slow_hidden"])
                    self._fast_pos_buf = np.zeros((1,), dtype=self._fast_dtypes["input_pos"])
                    self._fast_out_bufs = [
                        np.empty(
                            tuple(int(d) for d in out.shape),
                            dtype=_ort_type_to_numpy(out.type),
                        )
                        for out in fast_outs
                    ]

                    refs: list[Any] = []
                    iob = self._fast.io_binding()
                    for name, arr in (
                        ("slow_hidden", self._fast_hidden_buf),
                        ("token_id", self._fast_tok_buf),
                        ("use_slow_hidden", self._fast_use_buf),
                        ("input_pos", self._fast_pos_buf),
                    ):
                        ov = ort.OrtValue.ortvalue_from_numpy(arr)
                        refs.append(ov)
                        iob.bind_ortvalue_input(name, ov)
                    for name, arr in self._fast_caches.items():
                        ov = ort.OrtValue.ortvalue_from_numpy(arr)
                        refs.append(ov)
                        iob.bind_ortvalue_input(name, ov)
                    for out, buf in zip(fast_outs, self._fast_out_bufs, strict=True):
                        ov = ort.OrtValue.ortvalue_from_numpy(buf)
                        refs.append(ov)
                        iob.bind_ortvalue_output(out.name, ov)
                    self._fast_ort_refs = refs
                    self._fast_iob = iob
            except Exception as exc:
                logger.debug("Fast-AR IOBinding unavailable, using standard run(): %s", exc)
                self._fast_iob = None

    def _fallback_session_to_cpu(self, role: str, exc: Exception) -> None:
        """Rebuild a failing GPU ONNX session on CPU so runtime kernel errors never crash synthesis."""
        prev_ep = self.session_providers.get(role, "CPUExecutionProvider")
        path_map = {
            "slow_ar": self._slow_model_path_active,
            "fast_ar": self.manifest.fast_model_path,
            "codec_decoder": self.manifest.codec_decoder_path,
        }
        attr_map = {
            "slow_ar": "_slow",
            "fast_ar": "_fast",
            "codec_decoder": "_decoder",
        }
        if role == "codec_decoder":
            role_threads = self.codec_threads
            role_spin = True
        elif role == "fast_ar":
            role_threads = self.fast_threads
            role_spin = self.use_global_thread_pool
        else:
            role_threads = self.ar_threads
            role_spin = True
        model_path = path_map[role]
        msg = (
            f"{role} 在 {prev_ep} 推理阶段执行失败，已自动回退到 CPUExecutionProvider: {exc}"
        )
        logger.warning("%s", msg)
        cpu_sess, _, _ = create_onnx_session(
            model_path,
            provider="cpu",
            threads=role_threads,
            inter_op_threads=1,
            allow_spinning=role_spin,
            use_global_pool=self.use_global_thread_pool,
        )
        setattr(self, attr_map[role], cpu_sess)
        self.session_providers[role] = "CPUExecutionProvider"
        self.active_provider = self._compute_active_provider()
        self.provider_warning = msg
        if role in {"slow_ar", "fast_ar"}:
            self._init_iobinding()

    def _slow_step(
        self,
        column: np.ndarray,
        position: int,
        state: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._slow_iobs is not None and state is self._workspace:
            np.copyto(self._slow_col_buf, column)
            self._slow_pos_buf[0] = position
            self._slow.run_with_iobinding(self._slow_iobs[self._slow_state_idx])
            self._slow_state_idx ^= 1
            self._workspace["conv_states"] = self._conv_pair[self._slow_state_idx]
            self._workspace["ssm_states"] = self._ssm_pair[self._slow_state_idx]
            kd = self._slow_kd_buf
            vd = self._slow_vd_buf
            state["cache_keys"][:, :, :, position, :] = kd[:, :, :, 0, :] if kd.ndim == 5 else kd
            state["cache_values"][:, :, :, position, :] = vd[:, :, :, 0, :] if vd.ndim == 5 else vd
            return self._slow_logits_buf[0, -1], self._slow_hidden_buf[:, -1:, :]

        pos_arr = np.array([position], dtype=self._slow_dtypes["position"])
        feed = {
            "codes": np.ascontiguousarray(column, dtype=self._slow_dtypes["codes"]),
            "position": pos_arr,
            "cache_keys": state["cache_keys"],
            "cache_values": state["cache_values"],
            "conv_states": state["conv_states"],
            "ssm_states": state["ssm_states"],
        }
        try:
            logits, hidden, key_delta, value_delta, next_conv, next_ssm = self._slow.run(
                self._slow_output_names, feed
            )
        except Exception as exc:
            if self.session_providers.get("slow_ar") == "CPUExecutionProvider":
                raise
            self._fallback_session_to_cpu("slow_ar", exc)
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
        try:
            outputs = self._fast.run(self._fast_output_names, feed)
        except Exception as exc:
            if self.session_providers.get("fast_ar") == "CPUExecutionProvider":
                raise
            self._fallback_session_to_cpu("fast_ar", exc)
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

    def _decode_fast_frame(
        self,
        slow_hidden: np.ndarray,
        semantic_rel_idx: int,
        *,
        temperature: float,
        top_p: float,
        top_k: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Decode all acoustic codebooks (1..num_codebooks-1) for one semantic frame."""
        frame = np.zeros(self.manifest.num_codebooks, dtype=np.int64)
        frame[0] = semantic_rel_idx

        for buf in self._fast_caches.values():
            buf.fill(0)

        if self._fast_iob is not None:
            np.copyto(self._fast_hidden_buf, slow_hidden)
            self._fast_tok_buf[0, 0] = 0
            self._fast_use_buf[0] = True
            self._fast_pos_buf[0] = 0
            self._fast.run_with_iobinding(self._fast_iob)
            for layer_idx in range(self.manifest.num_fast_layers):
                kd = self._fast_out_bufs[1 + 2 * layer_idx]
                vd = self._fast_out_bufs[2 + 2 * layer_idx]
                self._fast_caches[f"cache_key_{layer_idx}"][:, :, : kd.shape[2], :] = kd
                self._fast_caches[f"cache_value_{layer_idx}"][:, :, : vd.shape[2], :] = vd

            prev_code = int(semantic_rel_idx)
            self._fast_use_buf[0] = False
            for cb_idx in range(1, self.manifest.num_codebooks):
                self._fast_tok_buf[0, 0] = prev_code
                self._fast_pos_buf[0] = cb_idx
                self._fast.run_with_iobinding(self._fast_iob)
                for layer_idx in range(self.manifest.num_fast_layers):
                    kd = self._fast_out_bufs[1 + 2 * layer_idx]
                    vd = self._fast_out_bufs[2 + 2 * layer_idx]
                    self._fast_caches[f"cache_key_{layer_idx}"][
                        :, :, cb_idx : cb_idx + kd.shape[2], :
                    ] = kd
                    self._fast_caches[f"cache_value_{layer_idx}"][
                        :, :, cb_idx : cb_idx + vd.shape[2], :
                    ] = vd
                prev_code = sample_codebook(
                    self._fast_out_bufs[0][0, -1],
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    rng=rng,
                )
                frame[cb_idx] = prev_code
            return frame

        # Fallback non-IOBinding path (e.g. GPU Fast-AR or dynamic shapes)
        _ = self._fast_step(slow_hidden, 0, True, 0, self._fast_caches)
        prev_code = int(frame[0])
        for cb_idx in range(1, self.manifest.num_codebooks):
            cb_logits = self._fast_step(slow_hidden, prev_code, False, cb_idx, self._fast_caches)
            prev_code = sample_codebook(
                cb_logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                rng=rng,
            )
            frame[cb_idx] = prev_code
        return frame

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
        if prefix_len >= self._active_max_seq_len - 96 and self._active_max_seq_len < self.manifest.max_seq_len:
            self._expand_slow_ar_to_full_context()
        if prefix_len >= self._active_max_seq_len - 16:
            raise ValueError(
                f"Voice {voice.voice_id!r} reference prefix ({prefix_len} tokens) is too long "
                f"for max_seq_len={self._active_max_seq_len}; trim reference audio to <=15s."
            )

        self._zero_workspace()
        for pos in range(prefix_len):
            self._slow_step(sys_prompt[:, :, pos : pos + 1], pos, self._workspace)
        self._dirty_seq_len = prefix_len
        self.prefix_cache.put(key, PrefixStateSnapshot.capture(self._workspace, prefix_len))
        return prefix_len, False

    def warm_voice(self, voice: VoicePack) -> float:
        """Pre-compute and cache a voice's Slow-AR prefix state (and warm codec once); returns elapsed ms."""
        t0 = time.perf_counter()
        with self._lock:
            self._prepare_voice_prefix(voice)
            if not self._decoder_warmed:
                warm_frames = min(4, int(voice.codes.shape[1]))
                if warm_frames > 0:
                    try:
                        self.decode_codes(voice.codes[:, :warm_frames])
                        self._decoder_warmed = True
                    except Exception:
                        logger.debug("Optional codec warmup skipped", exc_info=True)
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
        feed = {self._decoder_input_name: np.ascontiguousarray(arr)}
        with self._decoder_lock:
            try:
                audio = self._decoder.run(None, feed)[0]
            except Exception as exc:
                if self.session_providers.get("codec_decoder") == "CPUExecutionProvider":
                    raise
                self._fallback_session_to_cpu("codec_decoder", exc)
                audio = self._decoder.run(None, feed)[0]
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
        first_chunk_frames: int | None = None,
        seed: int | None = None,
    ) -> Iterator[StreamAudioChunk]:
        """Generate and stream target-voice audio chunks for `text` using `voice`.

        Uses progressive first-chunk emission, zero-latency guard-overlap crossfading,
        and (when `cfg.pipeline_decode=True`) concurrent AR/Codec pipelining to minimize
        both TTFA and end-to-end streaming RTF.
        """
        temp = self.cfg.temperature if temperature is None else temperature
        p_val = self.cfg.top_p if top_p is None else top_p
        k_val = self.cfg.top_k if top_k is None else top_k
        max_steps = (
            estimate_max_audio_frames(text, hard_cap=self.cfg.max_new_tokens)
            if max_new_tokens is None
            else max_new_tokens
        )
        emit_every = max(2, self.cfg.chunk_frames if chunk_frames is None else int(chunk_frames))
        raw_first = (
            self.cfg.first_chunk_frames
            if first_chunk_frames is None
            else int(first_chunk_frames)
        )
        first_emit = max(2, min(emit_every, raw_first))
        context_frames = min(self.cfg.stream_context_frames, self.manifest.stream_context_frames)
        guard_frames = self.cfg.stream_guard_frames
        hop = self.manifest.codec_hop_length
        guard_samples = guard_frames * hop
        # Always hold back at least `_GUARD_FADE_SAMPLES` (5.8 ms) on non-final chunks so
        # even `stream_guard_frames=0` performs click-free raised-cosine boundary crossfading.
        holdback_samples = max(guard_samples, _GUARD_FADE_SAMPLES)
        rng = np.random.default_rng(seed)

        user_prompt = self.prompt_builder.build_user_suffix(text)
        user_len = int(user_prompt.shape[2])

        with self._lock:
            t_start = time.perf_counter()
            emitted_samples = 0
            ttfa_ms = 0.0
            prev_guard: np.ndarray | None = None

            def _decode_chunk(
                window: np.ndarray,
                start_frame: int,
                frames_generated: int,
                prefill_ms: float,
                cache_hit: bool,
                is_final: bool,
            ) -> StreamAudioChunk | None:
                nonlocal emitted_samples, ttfa_ms, prev_guard
                decoded = self.decode_codes(window)
                self._decoder_warmed = True
                abs_start = start_frame * hop
                if not is_final:
                    eff_holdback = (
                        holdback_samples
                        if decoded.size > holdback_samples
                        else min(holdback_samples, decoded.size // 4)
                    )
                    stable_end = abs_start + max(0, decoded.size - eff_holdback)
                    begin = max(0, emitted_samples - abs_start)
                    end = max(begin, stable_end - abs_start)
                else:
                    begin = max(0, emitted_samples - abs_start)
                    end = decoded.size

                if end > begin:
                    piece = decoded[begin:end].copy()
                    if prev_guard is not None and piece.size > 0:
                        ov = min(prev_guard.size, piece.size)
                        if ov == _GUARD_FADE_SAMPLES:
                            f_out = self._guard_fade_out
                            f_in = self._guard_fade_in
                        else:
                            t_ov = np.linspace(0.0, 1.0, ov, dtype=np.float32)
                            f_out = 0.5 * (1.0 + np.cos(np.pi * t_ov))
                            f_in = 1.0 - f_out
                        piece[:ov] = prev_guard[:ov] * f_out + piece[:ov] * f_in
                    emitted_samples = abs_start + end
                    if not is_final and end < decoded.size:
                        ov_next = min(_GUARD_FADE_SAMPLES, decoded.size - end)
                        prev_guard = decoded[end : end + ov_next].copy()
                    else:
                        prev_guard = None
                    if ttfa_ms == 0.0:
                        ttfa_ms = (time.perf_counter() - t_start) * 1000.0
                    return StreamAudioChunk(
                        audio=piece,
                        sample_rate=self.manifest.sample_rate,
                        frames_generated=frames_generated,
                        ttfa_ms=ttfa_ms,
                        prefill_ms=prefill_ms,
                        cache_hit=cache_hit,
                        is_final=is_final,
                    )

                if is_final:
                    prev_guard = None
                    if ttfa_ms == 0.0:
                        ttfa_ms = (time.perf_counter() - t_start) * 1000.0
                    return StreamAudioChunk(
                        audio=np.zeros(0, dtype=np.float32),
                        sample_rate=self.manifest.sample_rate,
                        frames_generated=frames_generated,
                        ttfa_ms=ttfa_ms,
                        prefill_ms=prefill_ms,
                        cache_hit=cache_hit,
                        is_final=True,
                    )
                return None

            def _is_cancelled(local_cancel: threading.Event | None = None) -> bool:
                return bool(
                    (cancel_event is not None and cancel_event.is_set())
                    or (local_cancel is not None and local_cancel.is_set())
                )

            def _iter_windows(
                local_cancel: threading.Event | None = None,
            ) -> Iterator[tuple[np.ndarray, int, int, float, bool, bool]]:
                prefix_len, cache_hit = self._prepare_voice_prefix(voice)
                min_gen_headroom = min(int(max_steps), 96)
                if (
                    prefix_len + user_len + min_gen_headroom > self._active_max_seq_len
                    and self._active_max_seq_len < self.manifest.max_seq_len
                ):
                    if self._expand_slow_ar_to_full_context():
                        prefix_len, cache_hit = self._prepare_voice_prefix(voice)

                max_avail = self._active_max_seq_len - (prefix_len + user_len)
                if max_avail <= 1:
                    raise ValueError(
                        f"Prompt length ({prefix_len + user_len}) leaves no generation room "
                        f"in max_seq_len={self._active_max_seq_len}"
                    )
                step_budget = min(int(max_steps), max_avail)

                logits = np.empty(0, dtype=np.float32)
                hidden = np.empty(0, dtype=np.float32)
                for offset in range(user_len):
                    if _is_cancelled(local_cancel):
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
                eos_index = self.manifest.codebook_size
                cur_step = first_emit
                next_emit = cur_step
                last_emit_nf = 0
                column = np.empty((1, self.manifest.num_codebooks + 1, 1), dtype=np.int64)

                for step in range(step_budget):
                    if _is_cancelled(local_cancel):
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

                    frame = self._decode_fast_frame(
                        hidden,
                        rel_idx,
                        temperature=temp,
                        top_p=p_val,
                        top_k=k_val,
                        rng=rng,
                    )
                    frames.append(frame)

                    # Dispatch completed chunk to Codec BEFORE next Slow-AR step (~20 ms head-start),
                    # progressively doubling chunk size (`2 -> 4 -> 8 -> 12`) to minimize both TTFA and stalls.
                    if len(frames) == next_emit:
                        step_sz = len(frames) - last_emit_nf
                        start_frame = max(0, len(frames) - context_frames - step_sz)
                        window = np.stack(frames[start_frame:], axis=1)
                        yield (window, start_frame, len(frames), prefill_ms, cache_hit, False)
                        last_emit_nf = len(frames)
                        cur_step = min(emit_every, cur_step * 2)
                        next_emit += cur_step
                        if _is_cancelled(local_cancel):
                            return

                    # Feed completed frame back into Slow-AR for next frame prediction
                    next_pos = prompt_len + step
                    if step + 1 < step_budget and next_pos < self._active_max_seq_len:
                        column[0, 0, 0] = semantic_token
                        column[0, 1:, 0] = frame
                        logits, hidden = self._slow_step(column, next_pos, self._workspace)
                        self._dirty_seq_len = max(self._dirty_seq_len, next_pos + 1)

                if frames and not _is_cancelled(local_cancel):
                    step_sz = max(guard_frames, 1, len(frames) - last_emit_nf)
                    start_frame = max(0, len(frames) - context_frames - step_sz)
                    window = np.stack(frames[start_frame:], axis=1)
                    yield (window, start_frame, len(frames), prefill_ms, cache_hit, True)

            if not self.cfg.pipeline_decode:
                for item in _iter_windows():
                    chunk = _decode_chunk(*item)
                    if chunk is not None:
                        yield chunk
                return

            # Concurrent pipelined mode: AR runs in dedicated producer thread while caller
            # thread decodes codec windows in parallel and yields chunks immediately.
            decode_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=4)
            local_cancel = threading.Event()

            def _put_item(tag: str, payload: Any) -> None:
                while True:
                    if _is_cancelled(local_cancel):
                        return
                    try:
                        decode_queue.put((tag, payload), timeout=0.05)
                        return
                    except queue.Full:
                        if _is_cancelled(local_cancel):
                            return

            def _ar_worker() -> None:
                try:
                    for item in _iter_windows(local_cancel=local_cancel):
                        _put_item("chunk", item)
                    _put_item("done", None)
                except Exception as exc:
                    _put_item("error", exc)

            ar_thread = threading.Thread(
                target=_ar_worker,
                name="a8vc-ar-producer",
                daemon=True,
            )
            ar_thread.start()
            try:
                while True:
                    if cancel_event is not None and cancel_event.is_set():
                        local_cancel.set()
                        break
                    try:
                        tag, payload = decode_queue.get(timeout=0.05)
                    except queue.Empty:
                        if not ar_thread.is_alive():
                            try:
                                tag, payload = decode_queue.get_nowait()
                            except queue.Empty:
                                break
                        else:
                            continue
                    if tag == "done":
                        break
                    if tag == "error":
                        raise payload
                    if _is_cancelled(local_cancel):
                        break
                    chunk = _decode_chunk(*payload)
                    if chunk is not None:
                        yield chunk
            finally:
                local_cancel.set()
                while True:
                    try:
                        decode_queue.get_nowait()
                    except queue.Empty:
                        break
                ar_thread.join()

    def synthesize(
        self,
        text: str,
        voice: VoicePack,
        *,
        seed: int | None = None,
    ) -> tuple[np.ndarray, SynthesisMetrics]:
        """Synthesize full utterance and collect latency, stall, and RTF metrics."""
        t0 = time.perf_counter()
        pieces: list[np.ndarray] = []
        emit_times_ms: list[float] = []
        chunk_durs_ms: list[float] = []
        last_chunk: StreamAudioChunk | None = None
        for chunk in self.stream_audio(text, voice, seed=seed):
            last_chunk = chunk
            if chunk.audio.size > 0:
                now_ms = (time.perf_counter() - t0) * 1000.0
                pieces.append(chunk.audio)
                emit_times_ms.append(now_ms)
                chunk_durs_ms.append(chunk.audio.size / float(self.manifest.sample_rate) * 1000.0)
        total_ms = (time.perf_counter() - t0) * 1000.0
        audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
        audio_sec = audio.size / float(self.manifest.sample_rate)
        rtf = (total_ms / 1000.0) / audio_sec if audio_sec > 0 else 0.0

        total_stall_ms = 0.0
        max_stall_ms = 0.0
        if emit_times_ms:
            play_cursor = emit_times_ms[0]
            for arrival_ms, dur_ms in zip(emit_times_ms, chunk_durs_ms, strict=True):
                if arrival_ms > play_cursor:
                    stall = arrival_ms - play_cursor
                    total_stall_ms += stall
                    if stall > max_stall_ms:
                        max_stall_ms = stall
                    play_cursor = arrival_ms + dur_ms
                else:
                    play_cursor += dur_ms

        metrics = SynthesisMetrics(
            voice_id=voice.voice_id,
            num_frames=last_chunk.frames_generated if last_chunk else 0,
            audio_seconds=round(audio_sec, 3),
            prefill_ms=round(last_chunk.prefill_ms if last_chunk else 0.0, 2),
            ttfa_ms=round(last_chunk.ttfa_ms if last_chunk else total_ms, 2),
            total_ms=round(total_ms, 2),
            rtf=round(rtf, 3),
            cache_hit=last_chunk.cache_hit if last_chunk else False,
            num_chunks=len(pieces),
            total_stall_ms=round(total_stall_ms, 2),
            max_stall_ms=round(max_stall_ms, 2),
        )
        return audio, metrics

    def cache_stats(self) -> dict[str, Any]:
        return self.prefix_cache.stats()

    def close(self) -> None:
        self.prefix_cache.clear()
        self._slow_iobs = None
        self._fast_iob = None
        self._slow = None
        self._fast = None
        self._decoder = None
