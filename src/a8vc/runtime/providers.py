"""ONNX Runtime execution provider selection, Windows GPU/CUDA setup, and safe CPU fallback."""

from __future__ import annotations

import logging
import os
import sys
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from a8vc.paths import get_bundle_dir, get_executable_dir, get_user_data_dir

logger = logging.getLogger("a8vc.runtime.providers")

_GPU_ENV_LOCK = threading.Lock()
_GPU_ENV_PREPARED = False
_PRELOAD_ATTEMPTED = False
_REGISTERED_DLL_DIRS: set[str] = set()
_DLL_DIR_HANDLES: list[Any] = []

_PROVIDER_TO_EP = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "dml": "DmlExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
}

_AUTO_GPU_PRIORITY = (
    "CUDAExecutionProvider",
    "CoreMLExecutionProvider",
    "DmlExecutionProvider",
)


def _version_sort_key(path: Path) -> tuple[int, ...]:
    """Extract numeric version segments from directory names like `v12.6` or `v9.8`."""
    parts: list[int] = []
    for token in path.name.lstrip("vV").replace("-", ".").split("."):
        if token.isdigit():
            parts.append(int(token))
    return tuple(parts)


def discover_windows_cuda_dirs() -> list[Path]:
    """Locate existing CUDA 12 / cuDNN 9 DLL directories on Windows.

    Checks app-local `cuda_libs`, bundled PyInstaller `nvidia/*/bin`, `CUDA_PATH*` env vars,
    and standard NVIDIA CUDA 12 / cuDNN 9 installation folders under Program Files.
    """
    if sys.platform != "win32" and os.name != "nt":
        return []

    candidates: list[Path] = []
    seen: set[str] = set()

    def _add(p: Path) -> None:
        try:
            if not p.is_dir():
                return
            key = str(p.resolve()).lower()
            if key not in seen:
                seen.add(key)
                candidates.append(p.resolve())
        except OSError:
            return

    # 1. App-local / user-local CUDA DLL drop-in directories
    exe_dir = get_executable_dir()
    bundle_dir = get_bundle_dir()
    _add(exe_dir / "cuda_libs")
    _add(get_user_data_dir() / "cuda_libs")

    # 2. Bundled nvidia-*-cu12 pip wheel bin directories (optional offline CUDA build)
    for root in (bundle_dir / "nvidia", exe_dir / "_internal" / "nvidia", exe_dir / "nvidia"):
        if root.is_dir():
            for child in sorted(root.iterdir()):
                _add(child / "bin")

    # 3. CUDA_PATH and CUDA_PATH_V12_* environment variables
    cuda_path = os.environ.get("CUDA_PATH", "").strip()
    if cuda_path:
        _add(Path(cuda_path) / "bin")
    v12_env_keys = sorted(
        (k for k in os.environ if k.upper().startswith("CUDA_PATH_V12")),
        reverse=True,
    )
    for key in v12_env_keys:
        val = os.environ.get(key, "").strip()
        if val:
            _add(Path(val) / "bin")

    # 4. Standard NVIDIA Windows install locations (CUDA 12.x & cuDNN 9.x)
    prog_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    cuda_root = prog_files / "NVIDIA GPU Computing Toolkit" / "CUDA"
    if cuda_root.is_dir():
        v12_dirs = sorted(
            (d for d in cuda_root.iterdir() if d.is_dir() and d.name.lower().startswith("v12")),
            key=_version_sort_key,
            reverse=True,
        )
        for d in v12_dirs:
            _add(d / "bin")

    cudnn_root = prog_files / "NVIDIA" / "CUDNN"
    if cudnn_root.is_dir():
        v9_dirs = sorted(
            (d for d in cudnn_root.iterdir() if d.is_dir() and d.name.lower().startswith("v9")),
            key=_version_sort_key,
            reverse=True,
        )
        for d in v9_dirs:
            bin_dir = d / "bin"
            if bin_dir.is_dir():
                sub_12 = sorted(
                    (s for s in bin_dir.iterdir() if s.is_dir() and s.name.startswith("12")),
                    key=_version_sort_key,
                    reverse=True,
                )
                for sub in sub_12:
                    _add(sub)
                _add(bin_dir)

    return candidates


def prepare_gpu_environment(provider: str = "auto") -> list[Path]:
    """Register Windows CUDA/cuDNN DLL directories and preload ORT CUDA DLLs when needed."""
    global _GPU_ENV_PREPARED, _PRELOAD_ATTEMPTED

    if provider not in {"cuda", "auto"}:
        return []

    found_dirs: list[Path] = []
    with _GPU_ENV_LOCK:
        if sys.platform == "win32" or os.name == "nt":
            found_dirs = discover_windows_cuda_dirs()
            add_dll_dir = getattr(os, "add_dll_directory", None)
            current_path = os.environ.get("PATH", "")
            path_entries = [p for p in current_path.split(os.pathsep) if p]
            path_lower = {p.lower() for p in path_entries}
            new_path_prefix: list[str] = []

            for directory in found_dirs:
                dir_str = str(directory)
                key = dir_str.lower()
                if key not in _REGISTERED_DLL_DIRS:
                    if add_dll_dir is not None:
                        try:
                            _DLL_DIR_HANDLES.append(add_dll_dir(dir_str))
                        except OSError as exc:
                            logger.debug("os.add_dll_directory(%s) failed: %s", dir_str, exc)
                    _REGISTERED_DLL_DIRS.add(key)
                if key not in path_lower:
                    new_path_prefix.append(dir_str)
                    path_lower.add(key)

            if new_path_prefix:
                os.environ["PATH"] = os.pathsep.join(new_path_prefix + path_entries)

            _GPU_ENV_PREPARED = True

        if not _PRELOAD_ATTEMPTED:
            _PRELOAD_ATTEMPTED = True
            try:
                import onnxruntime as ort

                preload = getattr(ort, "preload_dlls", None)
                if callable(preload):
                    preload()
            except Exception as exc:
                logger.debug("onnxruntime.preload_dlls() skipped: %s", exc)

    return found_dirs


def resolve_providers(
    provider: str,
    available: Iterable[str] | None = None,
) -> list[str]:
    """Resolve configured provider name into an ordered ONNX Runtime provider list."""
    if available is None:
        import onnxruntime as ort

        avail_set = set(ort.get_available_providers())
    else:
        avail_set = set(available)

    normalized = (provider or "cpu").strip().lower()
    if normalized == "cpu":
        return ["CPUExecutionProvider"]
    if normalized in _PROVIDER_TO_EP:
        ep = _PROVIDER_TO_EP[normalized]
        if ep in avail_set:
            return [ep, "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]
    # "auto"
    for ep in _AUTO_GPU_PRIORITY:
        if ep in avail_set:
            return [ep, "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def build_session_options(
    providers: Sequence[str],
    *,
    threads: int = 2,
    ort_module: Any = None,
) -> Any:
    """Create tuned ONNX Runtime `SessionOptions` for the selected providers.

    Applies DirectML's required sequential execution + disabled memory-pattern settings
    when `DmlExecutionProvider` is active.
    """
    ort = ort_module
    if ort is None:
        import onnxruntime as ort

    safe_threads = max(1, int(threads))
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = safe_threads
    opts.inter_op_num_threads = max(1, safe_threads // 2)
    opts.log_severity_level = 3

    if "DmlExecutionProvider" in providers:
        # Required by ONNX Runtime DirectML Execution Provider
        opts.enable_mem_pattern = False
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    return opts


def _fallback_hint(requested_ep: str, error: Exception | None = None) -> str:
    detail = f": {error}" if error is not None else ""
    if requested_ep == "CUDAExecutionProvider":
        return (
            f"CUDAExecutionProvider 初始化失败，已自动回退到 CPU{detail}。"
            "请确认已安装 NVIDIA 显卡驱动与 CUDA 12 + cuDNN 9（或改用免装 CUDA 的 DirectML GPU 版本）。"
        )
    if requested_ep == "DmlExecutionProvider":
        return (
            f"DmlExecutionProvider 初始化失败，已自动回退到 CPU{detail}。"
            "请确认系统支持 DirectX 12 且显卡驱动已更新。"
        )
    return f"{requested_ep} 不可用，已自动回退到 CPU{detail}。"


def create_onnx_session(
    model_path: str | Path,
    *,
    provider: str = "cpu",
    threads: int = 2,
    ort_module: Any = None,
) -> tuple[Any, str, str | None]:
    """Create an ONNX Runtime `InferenceSession` with automatic safe CPU fallback.

    Returns `(session, active_provider, warning_message)`.
    """
    ort = ort_module
    if ort is None:
        import onnxruntime as ort

    normalized = (provider or "cpu").strip().lower()
    if normalized in {"cuda", "auto"}:
        prepare_gpu_environment(normalized)

    available = list(ort.get_available_providers())
    providers = resolve_providers(normalized, available)

    warning: str | None = None
    requested_ep = _PROVIDER_TO_EP.get(normalized)
    if (
        requested_ep is not None
        and requested_ep != "CPUExecutionProvider"
        and requested_ep not in available
    ):
        warning = (
            f"当前 ONNX Runtime 未内置 {requested_ep}（可用: {', '.join(available)}），"
            "已自动使用 CPUExecutionProvider。如需 GPU 加速，请下载对应的 DirectML 或 CUDA GPU 安装包。"
        )
        logger.warning(warning)

    opts = build_session_options(providers, threads=threads, ort_module=ort)
    try:
        session = ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)
    except Exception as exc:
        if providers == ["CPUExecutionProvider"]:
            raise
        primary = providers[0]
        warning = _fallback_hint(primary, exc)
        logger.warning("%s", warning)
        cpu_opts = build_session_options(
            ["CPUExecutionProvider"], threads=threads, ort_module=ort
        )
        session = ort.InferenceSession(
            str(model_path),
            sess_options=cpu_opts,
            providers=["CPUExecutionProvider"],
        )

    session_providers = list(session.get_providers()) if hasattr(session, "get_providers") else providers
    active_provider = session_providers[0] if session_providers else "CPUExecutionProvider"

    if (
        warning is None
        and requested_ep is not None
        and requested_ep != "CPUExecutionProvider"
        and active_provider != requested_ep
    ):
        warning = _fallback_hint(requested_ep)
        logger.warning("%s", warning)

    return session, active_provider, warning


def detect_ort_flavor(available_providers: Sequence[str]) -> str:
    """Classify installed ONNX Runtime wheel flavor from its available providers."""
    avail = set(available_providers)
    if "CUDAExecutionProvider" in avail:
        return "cuda"
    if "DmlExecutionProvider" in avail:
        return "dml"
    if "CoreMLExecutionProvider" in avail:
        return "coreml"
    return "cpu"


def probe_provider_info(configured_provider: str = "cpu") -> dict[str, Any]:
    """Return diagnostic details about ONNX Runtime providers and GPU readiness."""
    try:
        import onnxruntime as ort

        available = list(ort.get_available_providers())
        ort_version = getattr(ort, "__version__", "unknown")
    except Exception as exc:
        return {
            "configured_provider": configured_provider,
            "available_providers": [],
            "resolved_providers": ["CPUExecutionProvider"],
            "ort_flavor": "unavailable",
            "ort_version": "",
            "gpu_available": False,
            "cuda_dll_dirs": [],
            "error": str(exc),
        }

    flavor = detect_ort_flavor(available)
    resolved = resolve_providers(configured_provider, available)
    cuda_dirs = [str(p) for p in discover_windows_cuda_dirs()]
    return {
        "configured_provider": configured_provider,
        "available_providers": available,
        "resolved_providers": resolved,
        "ort_flavor": flavor,
        "ort_version": ort_version,
        "gpu_available": any(ep in available for ep in _AUTO_GPU_PRIORITY),
        "cuda_dll_dirs": cuda_dirs,
    }
