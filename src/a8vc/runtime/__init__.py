"""Audio8-TTS-0.1B ONNX INT8 inference runtime, prompt builder, sampler, and prefix cache."""

from a8vc.runtime.audio8_onnx import (
    Audio8OnnxRuntime,
    StreamAudioChunk,
    SynthesisMetrics,
    SynthesizerBackend,
)
from a8vc.runtime.manifest import RuntimeManifest, load_runtime_manifest
from a8vc.runtime.prefix_cache import PrefixStateCache, PrefixStateSnapshot
from a8vc.runtime.prompt import PromptBuilder, clean_text, ensure_speaker_tag
from a8vc.runtime.providers import (
    create_onnx_session,
    probe_provider_info,
    resolve_providers,
)
from a8vc.runtime.sampling import sample_codebook, sample_semantic

__all__ = [
    "Audio8OnnxRuntime",
    "PrefixStateCache",
    "PrefixStateSnapshot",
    "PromptBuilder",
    "RuntimeManifest",
    "StreamAudioChunk",
    "SynthesisMetrics",
    "SynthesizerBackend",
    "clean_text",
    "create_onnx_session",
    "ensure_speaker_tag",
    "load_runtime_manifest",
    "probe_provider_info",
    "resolve_providers",
    "sample_codebook",
    "sample_semantic",
]
