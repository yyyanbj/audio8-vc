"""Model inspection and download management for CLI and Windows Desktop UI."""

from __future__ import annotations

import shutil
import tarfile
import tempfile
import threading
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from a8vc.config import AppConfig
from a8vc.runtime.manifest import load_runtime_manifest
from a8vc.voices.enroll import VoiceEnroller

AUDIO8_ONNX_REPO = "Edge0/audio8-TTS-0.1B-ONNX-INT8"
AUDIO8_PT_REPO = "Edge0/Audio8-TTS-Preview-0.1b"
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"

SENSEVOICE_INT8_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09.tar.bz2"
)
SILERO_VAD_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
)


def safe_extract_tar_bz2(archive_path: Path, dest_dir: Path) -> None:
    """Safely extract a `.tar.bz2` archive, blocking path traversal and symlinks."""
    dest_resolved = dest_dir.resolve()
    with tarfile.open(archive_path, "r:bz2") as tf:
        for member in tf.getmembers():
            target = (dest_resolved / member.name).resolve()
            if not target.is_relative_to(dest_resolved):
                raise RuntimeError(f"Refusing to extract unsafe archive path: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"Refusing to extract archive symlink: {member.name}")
        if hasattr(tarfile, "data_filter"):
            tf.extractall(dest_resolved, filter="data")
        else:
            tf.extractall(dest_resolved)


def download_file(
    url: str,
    dest: Path,
    *,
    timeout: int = 60,
    on_progress: Callable[[int, int | None], None] | None = None,
) -> Path:
    """Atomically download a file via a `.part` temporary file with optional progress callback."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Audio8-VC/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as out:
            content_len = resp.headers.get("Content-Length")
            total = int(content_len) if content_len and content_len.isdigit() else None
            downloaded = 0
            while True:
                chunk = resp.read(256 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if on_progress is not None:
                    on_progress(downloaded, total)
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return dest


def download_hf_snapshot(
    repo_id: str,
    local_dir: Path,
    *,
    ignore_patterns: list[str] | None = None,
    hf_endpoint: str | None = None,
) -> Path:
    """Download a Hugging Face repository snapshot into `local_dir`."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download models. "
            "Install with: uv sync --extra download"
        ) from exc

    local_dir.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "local_dir": str(local_dir),
        "ignore_patterns": ignore_patterns,
    }
    if hf_endpoint and hf_endpoint.strip():
        kwargs["endpoint"] = hf_endpoint.strip().rstrip("/")
    snapshot_download(**kwargs)
    return local_dir


def download_sensevoice_asr(
    asr_dir: Path,
    *,
    url: str = SENSEVOICE_INT8_URL,
    on_progress: Callable[[int, int | None], None] | None = None,
) -> Path:
    """Download and extract the sherpa-onnx SenseVoice INT8 multilingual ASR model."""
    asr_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        archive = Path(tmpdir) / "sensevoice.tar.bz2"
        download_file(url, archive, on_progress=on_progress)
        safe_extract_tar_bz2(archive, asr_dir)
    return asr_dir


def download_silero_vad(
    vad_path: Path,
    *,
    url: str = SILERO_VAD_URL,
    on_progress: Callable[[int, int | None], None] | None = None,
) -> Path:
    """Download the Silero VAD ONNX model (~2 MB)."""
    return download_file(url, vad_path, on_progress=on_progress)


def inspect_models(cfg: AppConfig) -> dict[str, Any]:
    """Inspect local availability of Audio8 ONNX, registration encoder, ASR, and Silero VAD."""
    model_dir = Path(cfg.paths.model_dir)
    asr_dir = Path(cfg.paths.asr_model_dir)
    vad_path = Path(cfg.paths.vad_model_path)

    audio8_ready = False
    audio8_missing: list[str] = []
    manifest_path = model_dir / "runtime_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = load_runtime_manifest(
                model_dir,
                precision=cfg.runtime.precision,
                codec_precision=cfg.runtime.codec_precision,
            )
            required = [
                ("slow_model", manifest.slow_model_path),
                ("fast_model", manifest.fast_model_path),
                ("codec_decoder", manifest.codec_decoder_path),
                ("tokenizer", manifest.tokenizer_path),
            ]
            for label, path in required:
                if not path.is_file():
                    audio8_missing.append(f"{label}: {path.name}")
            audio8_ready = len(audio8_missing) == 0
        except Exception as exc:
            audio8_missing.append(str(exc))
    else:
        audio8_missing.append("runtime_manifest.json")

    enroller_status = VoiceEnroller(model_dir).status()
    registration_ready = bool(enroller_status["available"])

    asr_ready = False
    if cfg.asr.backend == "mock":
        asr_ready = True
    elif asr_dir.is_dir():
        has_tokens = any(asr_dir.rglob("tokens.txt"))
        has_onnx = any(asr_dir.rglob("*.onnx"))
        asr_ready = has_tokens and has_onnx

    silero_ready = vad_path.is_file()

    return {
        "audio8_onnx": {
            "ready": audio8_ready,
            "path": str(model_dir),
            "missing": audio8_missing,
        },
        "registration": {
            "ready": registration_ready,
            "path": enroller_status["encoder_path"],
        },
        "asr": {
            "ready": asr_ready,
            "backend": cfg.asr.backend,
            "model_type": cfg.asr.model_type,
            "path": str(asr_dir),
        },
        "silero_vad": {
            "ready": silero_ready,
            "path": str(vad_path),
            "optional": True,
        },
        "ready_for_tts": audio8_ready,
        "ready_for_enroll": registration_ready,
        "ready_for_vc": audio8_ready and asr_ready,
    }


@dataclass
class DownloadJobState:
    """Thread-safe snapshot of an asynchronous model download job."""

    status: str = "idle"  # idle | running | completed | error
    current_step: str = ""
    completed_steps: int = 0
    total_steps: int = 0
    percent: float = 0.0
    message: str = ""
    error: str = ""
    started_at: float | None = None
    finished_at: float | None = None
    steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "current_step": self.current_step,
            "completed_steps": self.completed_steps,
            "total_steps": self.total_steps,
            "percent": round(self.percent, 1),
            "message": self.message,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "steps": list(self.steps),
        }


class ModelDownloadManager:
    """Runs model downloads in a background thread with observable progress."""

    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg
        self._lock = threading.Lock()
        self._state = DownloadJobState()
        self._thread: threading.Thread | None = None

    def status(self) -> dict[str, Any]:
        with self._lock:
            job = self._state.to_dict()
        return {
            "models": inspect_models(self._cfg),
            "download": job,
        }

    def start_download(
        self,
        *,
        audio8_onnx: bool = True,
        include_registration: bool = True,
        asr_sensevoice: bool = True,
        silero_vad: bool = True,
        hf_endpoint: str | None = None,
        onnx_repo: str = AUDIO8_ONNX_REPO,
    ) -> dict[str, Any]:
        steps: list[str] = []
        if audio8_onnx:
            steps.append("audio8_onnx")
        if asr_sensevoice:
            steps.append("asr_sensevoice")
        if silero_vad:
            steps.append("silero_vad")
        if not steps:
            raise ValueError("Select at least one model component to download.")

        with self._lock:
            if self._state.status == "running":
                raise RuntimeError("A model download is already in progress.")
            self._state = DownloadJobState(
                status="running",
                current_step=steps[0],
                completed_steps=0,
                total_steps=len(steps),
                percent=0.0,
                message="准备开始下载模型...",
                error="",
                started_at=time.time(),
                finished_at=None,
                steps=steps,
            )

        def _update(**kwargs: Any) -> None:
            with self._lock:
                for key, val in kwargs.items():
                    setattr(self._state, key, val)

        def _worker() -> None:
            total = len(steps)
            try:
                for idx, step in enumerate(steps):
                    base_pct = (idx / total) * 100.0
                    span_pct = 100.0 / total
                    if step == "audio8_onnx":
                        _update(
                            current_step=step,
                            percent=base_pct + 2.0,
                            message=(
                                f"正在下载 Audio8-0.1B ONNX INT8"
                                f"{'（含音色编码器）' if include_registration else ''}..."
                            ),
                        )
                        ignore = None if include_registration else ["registration/*"]
                        download_hf_snapshot(
                            onnx_repo,
                            Path(self._cfg.paths.model_dir),
                            ignore_patterns=ignore,
                            hf_endpoint=hf_endpoint,
                        )
                    elif step == "asr_sensevoice":
                        _update(
                            current_step=step,
                            percent=base_pct + 2.0,
                            message="正在下载 SenseVoice INT8 多语种 ASR 模型...",
                        )

                        def _asr_prog(
                            done: int,
                            size: int | None,
                            _base: float = base_pct,
                            _span: float = span_pct,
                        ) -> None:
                            frac = (done / size) if size else 0.5
                            mb = done / (1024 * 1024)
                            total_str = f"/{size / (1024 * 1024):.1f} MB" if size else ""
                            _update(
                                percent=_base + frac * _span * 0.85,
                                message=f"正在下载 SenseVoice ASR ({mb:.1f}{total_str})...",
                            )

                        download_sensevoice_asr(
                            Path(self._cfg.paths.asr_model_dir),
                            on_progress=_asr_prog,
                        )
                    elif step == "silero_vad":
                        _update(
                            current_step=step,
                            percent=base_pct + 2.0,
                            message="正在下载 Silero VAD ONNX (~2 MB)...",
                        )
                        download_silero_vad(Path(self._cfg.paths.vad_model_path))

                    _update(
                        completed_steps=idx + 1,
                        percent=((idx + 1) / total) * 100.0,
                    )

                _update(
                    status="completed",
                    current_step="",
                    percent=100.0,
                    message="模型下载完成，已就绪！",
                    finished_at=time.time(),
                )
            except Exception as exc:
                _update(
                    status="error",
                    error=str(exc),
                    message=f"模型下载失败: {exc}",
                    finished_at=time.time(),
                )

        self._thread = threading.Thread(
            target=_worker, name="a8vc-model-downloader", daemon=True
        )
        self._thread.start()
        return self.status()


def clear_directory_contents(path: Path) -> None:
    """Helper used in tests or cleanup to safely wipe a directory if it exists."""
    if path.is_dir():
        shutil.rmtree(path)
