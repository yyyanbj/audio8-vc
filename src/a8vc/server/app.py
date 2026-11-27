"""FastAPI HTTP + WebSocket streaming service for Audio8-VC."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import subprocess
import sys
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from a8vc import __version__
from a8vc.asr.base import ASRBackend
from a8vc.asr.sherpa import create_asr
from a8vc.audio.device import LocalAudioBridge, list_audio_devices
from a8vc.audio.io import float32_to_pcm16, pcm16_to_float32
from a8vc.audio.vad import SpeechSegmenter, create_vad
from a8vc.config import AppConfig
from a8vc.engine.base import VCOutputEvent
from a8vc.engine.cascaded import CascadedVCEngine, RealtimeVCWorker, get_process_rss_mb
from a8vc.engine.live_session import LocalLiveSessionManager
from a8vc.models_manager import HF_MIRROR_ENDPOINT, ModelDownloadManager
from a8vc.paths import (
    get_data_dir,
    get_icon_path,
    get_logs_dir,
    get_static_dir,
    is_frozen,
    is_portable_mode,
)
from a8vc.runtime.audio8_onnx import Audio8OnnxRuntime, SynthesizerBackend
from a8vc.runtime.providers import probe_provider_info
from a8vc.voices.catalog import (
    ensure_builtins_seeded,
    install_builtin_voices,
    list_builtin_voices,
)
from a8vc.voices.enroll import VoiceEnroller
from a8vc.voices.store import VoiceLibrary

logger = logging.getLogger("a8vc.server")


class TTSStreamRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)
    voice_id: str = Field(..., min_length=1, max_length=64)
    temperature: float | None = Field(default=None, gt=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1, le=4096)
    max_new_tokens: int | None = Field(default=None, ge=16, le=2048)
    chunk_frames: int | None = Field(default=None, ge=2, le=64)
    seed: int | None = None


class ActiveVoiceRequest(BaseModel):
    voice_id: str = Field(..., min_length=1, max_length=64)
    warm: bool = True


class LiveStartRequest(BaseModel):
    voice_id: str = Field(..., min_length=1, max_length=64)
    input_device: str | int | None = ""
    output_device: str | int | None = ""


class LiveSwitchVoiceRequest(BaseModel):
    voice_id: str = Field(..., min_length=1, max_length=64)


class ModelDownloadRequest(BaseModel):
    audio8_onnx: bool = True
    include_registration: bool = True
    asr_sensevoice: bool = True
    silero_vad: bool = True
    use_mirror: bool = False
    hf_endpoint: str | None = None


class OpenDirRequest(BaseModel):
    target: str = Field(default="data", pattern="^(data|voices|models|logs)$")


class InstallBuiltinsRequest(BaseModel):
    voice_ids: list[str] | None = None
    overwrite: bool = False


def _open_directory_in_os(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


class ServerState:
    """Holds shared voice library, synthesizer, ASR backend, model manager, and live session."""

    def __init__(
        self,
        cfg: AppConfig,
        *,
        library: VoiceLibrary | None = None,
        synthesizer: SynthesizerBackend | None = None,
        asr: ASRBackend | None = None,
        enroller: VoiceEnroller | None = None,
        desktop_mode: bool = False,
        bridge_factory: Callable[..., Any] = LocalAudioBridge,
    ) -> None:
        self.cfg = cfg
        self.desktop_mode = desktop_mode
        if library is not None:
            self.library = library
        else:
            self.library = VoiceLibrary(cfg.paths.voices_dir)
            ensure_builtins_seeded(self.library)
        self._synthesizer = synthesizer
        self._asr = asr
        self.enroller = enroller or VoiceEnroller(
            cfg.paths.model_dir,
            provider=cfg.runtime.provider,
            threads=max(1, cfg.runtime.threads or 2),
        )
        self.model_manager = ModelDownloadManager(cfg)
        self._active_voice_id: str | None = cfg.server.default_voice or None
        self._init_lock = threading.Lock()
        self.live_session = LocalLiveSessionManager(
            cfg,
            create_engine=lambda _vid: self.create_vc_engine(load_voice=False),
            load_voice=self.library.load,
            on_voice_active=self._set_active_voice_internal,
            bridge_factory=bridge_factory,
        )

    def _set_active_voice_internal(self, voice_id: str) -> None:
        self._active_voice_id = voice_id

    def get_synthesizer(self) -> SynthesizerBackend:
        if self._synthesizer is None:
            with self._init_lock:
                if self._synthesizer is None:
                    self._synthesizer = Audio8OnnxRuntime(
                        self.cfg.paths.model_dir, self.cfg.runtime
                    )
        return self._synthesizer

    def get_asr(self) -> ASRBackend:
        if self._asr is None:
            with self._init_lock:
                if self._asr is None:
                    self._asr = create_asr(self.cfg.asr, self.cfg.paths.asr_model_dir)
        return self._asr

    def create_vc_engine(
        self, voice_id: str | None = None, *, load_voice: bool = True
    ) -> CascadedVCEngine:
        vad = create_vad(self.cfg.vad, self.cfg.paths.vad_model_path)
        segmenter = SpeechSegmenter(
            vad=vad,
            sample_rate=self.cfg.audio.input_sample_rate,
            speech_start_ms=self.cfg.vad.speech_start_ms,
            min_silence_ms=self.cfg.vad.min_silence_ms,
            pre_speech_ms=self.cfg.vad.pre_speech_ms,
            min_segment_ms=self.cfg.vad.min_segment_ms,
            max_segment_ms=self.cfg.vad.max_segment_ms,
        )
        engine = CascadedVCEngine(
            synthesizer=self.get_synthesizer(),
            asr=self.get_asr(),
            segmenter=segmenter,
            crossfade_ms=self.cfg.audio.crossfade_ms,
        )
        if load_voice:
            target_vid = voice_id or self._active_voice_id
            if target_vid and self.library.exists(target_vid):
                pack = self.library.load(target_vid)
                engine.set_voice(pack, warm=True)
        return engine

    def close(self) -> None:
        """Stop active local audio sessions and release loaded runtimes."""
        try:
            self.live_session.stop()
        except Exception:
            pass
        if self._synthesizer is not None:
            try:
                self._synthesizer.close()
            except Exception:
                pass
            self._synthesizer = None
        if self._asr is not None:
            try:
                self._asr.close()
            except Exception:
                pass
            self._asr = None

    @property
    def active_voice_id(self) -> str | None:
        return self._active_voice_id

    @active_voice_id.setter
    def active_voice_id(self, value: str | None) -> None:
        self._active_voice_id = value


def create_app(
    cfg: AppConfig | None = None,
    *,
    state: ServerState | None = None,
    desktop_mode: bool = False,
) -> FastAPI:
    """Create the Audio8-VC FastAPI application."""
    app_cfg = cfg or AppConfig()
    srv_state = state or ServerState(app_cfg, desktop_mode=desktop_mode)

    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            srv_state.close()

    app = FastAPI(
        title="Audio8-VC Real-Time Voice Changer",
        version=__version__,
        description="Low-footprint real-time voice changer built on Edge0-AI Audio8-TTS-0.1B.",
        lifespan=_lifespan,
    )
    app.state.a8vc = srv_state

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "detail": f"{type(exc).__name__}: {exc}",
                "error_type": type(exc).__name__,
            },
        )

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        html_path = get_static_dir() / "index.html"
        if html_path.is_file():
            return HTMLResponse(html_path.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>Audio8-VC Server</h1><p>Visit <a href='/docs'>/docs</a>.</p>")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Any:
        icon = get_icon_path("ico")
        if icon is not None and icon.is_file():
            return FileResponse(str(icon), media_type="image/x-icon")
        raise HTTPException(status_code=404, detail="Icon not found")

    @app.get("/icon.svg", include_in_schema=False)
    async def icon_svg() -> Any:
        icon = get_icon_path("svg")
        if icon is not None and icon.is_file():
            return FileResponse(str(icon), media_type="image/svg+xml")
        raise HTTPException(status_code=404, detail="Icon not found")

    @app.get("/icon.png", include_in_schema=False)
    async def icon_png() -> Any:
        icon = get_icon_path("png")
        if icon is not None and icon.is_file():
            return FileResponse(str(icon), media_type="image/png")
        raise HTTPException(status_code=404, detail="Icon not found")

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        synth_loaded = srv_state._synthesizer is not None
        asr_loaded = srv_state._asr is not None
        models_snapshot = srv_state.model_manager.status()
        live_status = srv_state.live_session.status()
        active_provider = (
            getattr(srv_state._synthesizer, "active_provider", None)
            if synth_loaded
            else None
        )
        provider_warning = (
            getattr(srv_state._synthesizer, "provider_warning", None)
            if synth_loaded
            else None
        )
        provider_note = (
            getattr(srv_state._synthesizer, "provider_note", None)
            if synth_loaded
            else None
        )
        session_providers = (
            getattr(srv_state._synthesizer, "session_providers", None)
            if synth_loaded
            else None
        )
        return {
            "status": "ok",
            "version": __version__,
            "engine_mode": srv_state.cfg.engine.mode,
            "desktop_mode": srv_state.desktop_mode,
            "synthesizer_loaded": synth_loaded,
            "asr_loaded": asr_loaded,
            "configured_provider": srv_state.cfg.runtime.provider,
            "active_provider": active_provider,
            "provider_warning": provider_warning,
            "provider_note": provider_note,
            "session_providers": session_providers,
            "active_voice_id": srv_state.active_voice_id,
            "voice_count": len(srv_state.library.list_voices()),
            "registration": srv_state.enroller.status(),
            "prefix_cache": srv_state._synthesizer.cache_stats() if synth_loaded else None,
            "models": models_snapshot["models"],
            "live_running": live_status["running"],
            "rss_mb": get_process_rss_mb(),
        }

    @app.get("/api/system/info")
    async def system_info() -> dict[str, Any]:
        data_dir = get_data_dir(desktop=srv_state.desktop_mode)
        provider_info = probe_provider_info(srv_state.cfg.runtime.provider)
        synth_loaded = srv_state._synthesizer is not None
        return {
            "version": __version__,
            "platform": sys.platform,
            "platform_release": platform.platform(),
            "is_frozen": is_frozen(),
            "is_portable": is_portable_mode(),
            "desktop_mode": srv_state.desktop_mode,
            "data_dir": str(data_dir),
            "voices_dir": str(srv_state.library.root),
            "model_dir": str(srv_state.cfg.paths.model_dir),
            "asr_model_dir": str(srv_state.cfg.paths.asr_model_dir),
            "vad_model_path": str(srv_state.cfg.paths.vad_model_path),
            "logs_dir": str(get_logs_dir(desktop=srv_state.desktop_mode)),
            "provider": srv_state.cfg.runtime.provider,
            "asr_provider": srv_state.cfg.asr.provider,
            "active_provider": (
                getattr(srv_state._synthesizer, "active_provider", None)
                if synth_loaded
                else None
            ),
            "provider_warning": (
                getattr(srv_state._synthesizer, "provider_warning", None)
                if synth_loaded
                else None
            ),
            **provider_info,
        }

    @app.post("/api/system/open-dir")
    async def open_dir(req: OpenDirRequest) -> dict[str, Any]:
        data_dir = get_data_dir(desktop=srv_state.desktop_mode)
        mapping = {
            "data": data_dir,
            "voices": srv_state.library.root,
            "models": Path(srv_state.cfg.paths.model_dir).parent,
            "logs": get_logs_dir(desktop=srv_state.desktop_mode),
        }
        target_path = mapping[req.target]
        try:
            await asyncio.to_thread(_open_directory_in_os, target_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to open directory: {exc}") from exc
        return {"status": "opened", "path": str(target_path)}

    @app.get("/api/models/status")
    async def models_status() -> dict[str, Any]:
        return srv_state.model_manager.status()

    @app.post("/api/models/download")
    async def models_download(req: ModelDownloadRequest) -> dict[str, Any]:
        endpoint = req.hf_endpoint
        if req.use_mirror and not endpoint:
            endpoint = HF_MIRROR_ENDPOINT
        try:
            return srv_state.model_manager.start_download(
                audio8_onnx=req.audio8_onnx,
                include_registration=req.include_registration,
                asr_sensevoice=req.asr_sensevoice,
                silero_vad=req.silero_vad,
                hf_endpoint=endpoint,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/devices")
    async def get_devices() -> dict[str, Any]:
        try:
            devices = await asyncio.to_thread(list_audio_devices)
            inputs = [d for d in devices if d["max_input_channels"] > 0]
            outputs = [d for d in devices if d["max_output_channels"] > 0]
            return {
                "available": True,
                "devices": devices,
                "inputs": inputs,
                "outputs": outputs,
                "virtual_outputs": [d for d in outputs if d.get("is_virtual_cable")],
            }
        except Exception as exc:
            return {
                "available": False,
                "error": str(exc),
                "devices": [],
                "inputs": [],
                "outputs": [],
                "virtual_outputs": [],
            }

    @app.get("/api/live/status")
    async def live_status() -> dict[str, Any]:
        return srv_state.live_session.status()

    @app.post("/api/live/start")
    async def live_start(req: LiveStartRequest) -> dict[str, Any]:
        if not srv_state.library.exists(req.voice_id):
            raise HTTPException(
                status_code=404,
                detail=f"音色 {req.voice_id!r} 不存在，请先选择或安装音色。",
            )
        try:
            return await asyncio.to_thread(
                srv_state.live_session.start,
                voice_id=req.voice_id,
                input_device=req.input_device,
                output_device=req.output_device,
            )
        except FileNotFoundError as exc:
            logger.warning("live_start missing required file: %s", exc)
            raise HTTPException(
                status_code=503,
                detail=f"所需模型文件缺失，请先点击「一键下载所需模型」: {exc}",
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("live_start failed for voice_id=%r", req.voice_id)
            raise HTTPException(
                status_code=500, detail=f"{type(exc).__name__}: {exc}"
            ) from exc

    @app.post("/api/live/stop")
    async def live_stop() -> dict[str, Any]:
        return await asyncio.to_thread(srv_state.live_session.stop)

    @app.post("/api/live/voice")
    async def live_switch_voice(req: LiveSwitchVoiceRequest) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(srv_state.live_session.switch_voice, req.voice_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("live_switch_voice failed for voice_id=%r", req.voice_id)
            raise HTTPException(
                status_code=500, detail=f"{type(exc).__name__}: {exc}"
            ) from exc

    @app.post("/api/live/flush")
    async def live_flush() -> dict[str, Any]:
        flushed = srv_state.live_session.flush()
        return {"flushed": flushed}

    @app.get("/api/voices")
    async def list_voices() -> dict[str, Any]:
        return {
            "active_voice_id": srv_state.active_voice_id,
            "voices": srv_state.library.list_voices(),
        }

    @app.get("/api/voices/builtins")
    async def list_builtin_voice_catalog() -> dict[str, Any]:
        installed_ids = {
            v["voice_id"]
            for v in srv_state.library.list_voices()
            if "error" not in v
        }
        return {
            "builtins": list_builtin_voices(installed_ids=installed_ids),
        }

    @app.post("/api/voices/builtins/install")
    async def install_builtins(req: InstallBuiltinsRequest) -> dict[str, Any]:
        try:
            res = await asyncio.to_thread(
                install_builtin_voices,
                srv_state.library,
                voice_ids=req.voice_ids,
                overwrite=req.overwrite,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        installed_ids = {
            v["voice_id"]
            for v in srv_state.library.list_voices()
            if "error" not in v
        }
        return {
            **res,
            "builtins": list_builtin_voices(installed_ids=installed_ids),
            "voices": srv_state.library.list_voices(),
        }

    @app.get("/api/voices/{voice_id}")
    async def get_voice(voice_id: str) -> dict[str, Any]:
        try:
            pack = srv_state.library.load(voice_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        data = pack.manifest.model_dump(mode="json")
        data["size_bytes"] = pack.path.stat().st_size if pack.path and pack.path.is_file() else 0
        return data

    @app.get("/api/voices/{voice_id}/download")
    async def download_voice(voice_id: str) -> FileResponse:
        try:
            path = srv_state.library.path_for(voice_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"Voice {voice_id!r} not found")
        return FileResponse(
            path=str(path),
            media_type="application/octet-stream",
            filename=path.name,
        )

    @app.post("/api/voices/active")
    async def set_active_voice(req: ActiveVoiceRequest) -> dict[str, Any]:
        try:
            pack = srv_state.library.load(req.voice_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        warm_ms = 0.0
        if req.warm:
            try:
                synth = srv_state.get_synthesizer()
                warm_ms = await asyncio.to_thread(synth.warm_voice, pack)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                logger.exception("set_active_voice warm failed for voice_id=%r", req.voice_id)
                raise HTTPException(
                    status_code=500, detail=f"{type(exc).__name__}: {exc}"
                ) from exc

        srv_state.active_voice_id = pack.voice_id
        return {
            "active_voice_id": pack.voice_id,
            "warm_ms": round(warm_ms, 2),
        }

    @app.post("/api/voices/import")
    async def import_voice_pack(
        file: Annotated[UploadFile, File(...)],
        overwrite: Annotated[bool, Form()] = False,
    ) -> dict[str, Any]:
        raw = await file.read()
        try:
            pack = srv_state.library.import_bytes(raw, overwrite=overwrite)
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "status": "imported",
            "voice": pack.manifest.model_dump(mode="json"),
        }

    @app.post("/api/voices/enroll")
    async def enroll_voice(
        audio: Annotated[UploadFile, File(...)],
        voice_id: Annotated[str, Form(...)],
        reference_text: Annotated[str, Form()] = "",
        name: Annotated[str, Form()] = "",
        description: Annotated[str, Form()] = "",
        language: Annotated[str, Form()] = "zh",
        tags: Annotated[str, Form()] = "",
        overwrite: Annotated[bool, Form()] = False,
    ) -> dict[str, Any]:
        raw = await audio.read()
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        asr = None
        if not reference_text.strip():
            try:
                asr = srv_state.get_asr()
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"reference_text omitted and ASR auto-transcription unavailable: {exc}",
                ) from exc

        try:
            pack = await asyncio.to_thread(
                srv_state.enroller.enroll_bytes,
                raw,
                filename=audio.filename or "reference.wav",
                voice_id=voice_id,
                reference_text=reference_text or None,
                asr=asr,
                name=name or None,
                description=description,
                language=language,
                tags=tag_list,
            )
            srv_state.library.save(pack, overwrite=overwrite)
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("enroll_voice failed for voice_id=%r", voice_id)
            raise HTTPException(
                status_code=500, detail=f"{type(exc).__name__}: {exc}"
            ) from exc

        return {
            "status": "enrolled",
            "voice": pack.manifest.model_dump(mode="json"),
        }

    @app.delete("/api/voices/{voice_id}")
    async def delete_voice(voice_id: str) -> JSONResponse:
        try:
            deleted = srv_state.library.delete(voice_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Voice {voice_id!r} not found")
        if srv_state.active_voice_id == voice_id:
            srv_state.active_voice_id = None
        return JSONResponse({"status": "deleted", "voice_id": voice_id})

    @app.post("/api/tts/stream")
    async def tts_stream(req: TTSStreamRequest) -> StreamingResponse:
        try:
            pack = srv_state.library.load(req.voice_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            synth = await asyncio.to_thread(srv_state.get_synthesizer)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Audio8 推理模型尚未下载，请先点击「一键下载所需模型」: {exc}",
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("tts_stream synthesizer init failed")
            raise HTTPException(
                status_code=500, detail=f"{type(exc).__name__}: {exc}"
            ) from exc

        cancel = threading.Event()
        stream_iter = synth.stream_audio(
            req.text,
            pack,
            cancel_event=cancel,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            max_new_tokens=req.max_new_tokens,
            chunk_frames=req.chunk_frames,
            seed=req.seed,
        )

        # Prime the first chunk before committing HTTP 200 headers so prefill/runtime errors
        # return structured JSON errors instead of mid-stream Internal Server Error resets.
        def _next_pcm() -> bytes | None:
            for chunk in stream_iter:
                if chunk.audio.size > 0:
                    return float32_to_pcm16(chunk.audio)
            return None

        try:
            first_pcm = await asyncio.to_thread(_next_pcm)
        except ValueError as exc:
            cancel.set()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            cancel.set()
            logger.exception("tts_stream synthesis failed for voice_id=%r", req.voice_id)
            raise HTTPException(
                status_code=500, detail=f"{type(exc).__name__}: {exc}"
            ) from exc

        def _generate() -> Any:
            try:
                if first_pcm:
                    yield first_pcm
                for chunk in stream_iter:
                    if chunk.audio.size > 0:
                        yield float32_to_pcm16(chunk.audio)
            except Exception:
                logger.exception("tts_stream mid-stream error for voice_id=%r", req.voice_id)
            finally:
                cancel.set()

        headers = {
            "X-Sample-Rate": str(synth.sample_rate),
            "X-Channels": "1",
            "X-Sample-Format": "pcm_s16le",
            "X-Voice-Id": pack.voice_id,
        }
        return StreamingResponse(_generate(), media_type="audio/pcm", headers=headers)

    @app.websocket("/ws/vc")
    async def ws_voice_changer(websocket: WebSocket) -> None:
        await websocket.accept()
        loop = asyncio.get_running_loop()
        out_queue: asyncio.Queue[bytes | dict[str, Any] | None] = asyncio.Queue()

        def _enqueue(item: bytes | dict[str, Any]) -> None:
            if not loop.is_closed():
                loop.call_soon_threadsafe(out_queue.put_nowait, item)

        try:
            engine = await asyncio.to_thread(srv_state.create_vc_engine)
        except Exception as exc:
            logger.exception("WebSocket /ws/vc engine initialization failed")
            try:
                await websocket.send_json(
                    {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
                )
                await websocket.close(code=1011)
            except Exception:
                pass
            return

        input_sr = srv_state.cfg.audio.input_sample_rate

        def _on_event(event: VCOutputEvent) -> None:
            if event.transcript:
                _enqueue(
                    {
                        "type": "transcript",
                        "text": event.transcript,
                        "voice_id": event.voice_id,
                        "asr_ms": event.asr_ms,
                    }
                )
            if event.audio.size > 0:
                _enqueue(float32_to_pcm16(event.audio))
            if event.is_final:
                _enqueue(
                    {
                        "type": "metrics",
                        "voice_id": event.voice_id,
                        "asr_ms": event.asr_ms,
                        "prefill_ms": event.prefill_ms,
                        "ttfa_ms": event.ttfa_ms,
                        "cache_hit": event.cache_hit,
                        "dropped_segments": worker.dropped_segments,
                    }
                )

        def _on_error(exc: Exception) -> None:
            _enqueue({"type": "error", "message": f"{type(exc).__name__}: {exc}"})

        worker = RealtimeVCWorker(
            engine,
            _on_event,
            max_pending_segments=srv_state.cfg.engine.max_pending_segments,
            on_error=_on_error,
        )
        worker.start()

        async def _sender() -> None:
            while True:
                item = await out_queue.get()
                if item is None:
                    break
                try:
                    if isinstance(item, bytes):
                        await websocket.send_bytes(item)
                    else:
                        await websocket.send_json(item)
                except Exception:
                    break

        sender_task = asyncio.create_task(_sender())
        try:
            await websocket.send_json(
                {
                    "type": "ready",
                    "mode": engine.mode,
                    "input_sample_rate": input_sr,
                    "output_sample_rate": engine.output_sample_rate,
                    "voice_id": engine.active_voice.voice_id if engine.active_voice else None,
                }
            )
            while True:
                message = await websocket.receive()
                msg_type = message.get("type")
                if msg_type == "websocket.disconnect":
                    break
                raw_bytes = message.get("bytes")
                if raw_bytes is not None:
                    if engine.active_voice is None:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": "No voice selected. Send {'type':'set_voice','voice_id':...} first.",
                            }
                        )
                        continue
                    try:
                        audio_f32 = pcm16_to_float32(raw_bytes)
                        worker.submit_audio(audio_f32, input_sr)
                    except Exception as exc:
                        logger.exception("WebSocket /ws/vc audio submission failed")
                        await websocket.send_json(
                            {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
                        )
                    continue

                raw_text = message.get("text")
                if raw_text is None:
                    continue
                try:
                    cmd = json.loads(raw_text)
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "message": "Invalid JSON command"})
                    continue

                action = cmd.get("type")
                if action in {"config", "set_voice"}:
                    if "input_sample_rate" in cmd:
                        try:
                            input_sr = max(8000, min(192000, int(cmd["input_sample_rate"])))
                        except (TypeError, ValueError):
                            await websocket.send_json(
                                {"type": "error", "message": "Invalid input_sample_rate"}
                            )
                            continue
                    vid = cmd.get("voice_id")
                    if vid:
                        try:
                            pack = srv_state.library.load(str(vid))
                            engine.cancel()
                            warm_ms = await asyncio.to_thread(engine.set_voice, pack, warm=True)
                            srv_state.active_voice_id = pack.voice_id
                            await websocket.send_json(
                                {
                                    "type": "voice_changed",
                                    "voice_id": pack.voice_id,
                                    "warm_ms": round(warm_ms, 2),
                                }
                            )
                        except Exception as exc:
                            logger.exception("WebSocket /ws/vc set_voice failed for %r", vid)
                            await websocket.send_json(
                                {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
                            )
                elif action == "flush":
                    try:
                        worker.flush()
                    except Exception as exc:
                        logger.exception("WebSocket /ws/vc flush failed")
                        await websocket.send_json(
                            {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
                        )
                elif action == "cancel":
                    engine.cancel()
                    engine.reset()
                    await websocket.send_json({"type": "cancelled"})
                elif action == "ping":
                    await websocket.send_json({"type": "pong", "stats": engine.stats()})
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.exception("WebSocket /ws/vc unexpected error")
        finally:
            worker.stop()
            await out_queue.put(None)
            await sender_task

    return app
