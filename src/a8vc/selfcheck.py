"""Runtime self-check for Audio8-VC (`a8vc self-check`).

Designed to run offline (without requiring downloaded model weights) in both
development environments and frozen Windows PyInstaller distributions. Catches
packaging, ABI, DLL, WebSocket, VAD, and server regressions before release.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from a8vc.config import AppConfig


@dataclass(slots=True)
class SelfCheckItem:
    """Result of a single runtime self-check probe."""

    name: str
    ok: bool
    detail: str
    elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _run_probe(name: str, fn: Any) -> SelfCheckItem:
    t0 = time.perf_counter()
    try:
        detail = str(fn())
        return SelfCheckItem(
            name=name,
            ok=True,
            detail=detail,
            elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        )
    except Exception as exc:
        return SelfCheckItem(
            name=name,
            ok=False,
            detail=f"{type(exc).__name__}: {exc}",
            elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        )


def _check_onnxruntime() -> str:
    import onnxruntime as ort

    providers = list(ort.get_available_providers())
    if "CPUExecutionProvider" not in providers:
        raise RuntimeError(f"CPUExecutionProvider missing from ONNX Runtime ({providers})")
    return f"onnxruntime {ort.__version__}, providers={providers}"


def _check_sherpa_coimport() -> str:
    # Always verify onnxruntime + sherpa_onnx co-import cleanly in the same process
    import onnxruntime as ort  # noqa: F401
    import sherpa_onnx

    if not hasattr(sherpa_onnx, "OfflineRecognizer"):
        raise RuntimeError("sherpa_onnx.OfflineRecognizer is unavailable")
    ver = getattr(sherpa_onnx, "__version__", "unknown")
    return f"sherpa_onnx {ver} loaded cleanly alongside onnxruntime {ort.__version__}"


def _check_core_imports() -> str:
    required = [
        "fastapi",
        "starlette",
        "uvicorn",
        "websockets",
        "soundfile",
        "scipy.signal",
        "tokenizers",
        "pydantic",
        "python_multipart",
    ]
    loaded: list[str] = []
    for mod in required:
        importlib.import_module(mod)
        loaded.append(mod)
    return f"imported {len(loaded)} core modules"


def _check_uvicorn_websocket_backend() -> str:
    from uvicorn.protocols.websockets import auto as ws_auto

    proto = getattr(ws_auto, "AutoWebSocketsProtocol", None)
    if proto is None:
        raise RuntimeError(
            "uvicorn AutoWebSocketsProtocol is None (missing websockets_sansio_impl / wsproto_impl)"
        )
    return f"protocol={proto.__module__}.{proto.__name__}"


def _check_provider_policy() -> str:
    from a8vc.runtime.providers import select_model_provider

    dml_avail = ["DmlExecutionProvider", "CPUExecutionProvider"]
    slow_ep, slow_note = select_model_provider(
        "dml", model_role="slow_ar", precision="int8", available=dml_avail
    )
    fast_ep, _ = select_model_provider(
        "dml", model_role="fast_ar", precision="int8", available=dml_avail
    )
    dec_ep, _ = select_model_provider(
        "dml", model_role="codec_decoder", precision="fp16", available=dml_avail
    )
    if slow_ep != "cpu" or fast_ep != "cpu" or not slow_note:
        raise RuntimeError(
            f"Expected DirectML INT8 AR roles to route to cpu, got slow={slow_ep}, fast={fast_ep}"
        )
    if dec_ep != "dml":
        raise RuntimeError(f"Expected DirectML FP16 codec_decoder to stay dml, got {dec_ep}")
    return "DirectML INT8 AR -> CPU guard and FP16 codec -> DML verified"


class _FakeTensorMeta:
    def __init__(self, name: str, shape: list[Any]) -> None:
        self.name = name
        self.shape = shape


class _FakeSileroSession:
    """In-memory mock ONNX session supporting sherpa ('x','h','c'), v5, and v4 schemas."""

    def __init__(self, schema: str) -> None:
        self.schema = schema
        if schema == "sherpa":
            self._inputs = [
                _FakeTensorMeta("x", [1, 512]),
                _FakeTensorMeta("h", [2, 1, 64]),
                _FakeTensorMeta("c", [2, 1, 64]),
            ]
        elif schema == "v5":
            self._inputs = [
                _FakeTensorMeta("input", [1, None]),
                _FakeTensorMeta("state", [2, 1, 128]),
                _FakeTensorMeta("sr", []),
            ]
        else:
            self._inputs = [
                _FakeTensorMeta("input", [1, 512]),
                _FakeTensorMeta("sr", []),
                _FakeTensorMeta("h", [2, 1, 64]),
                _FakeTensorMeta("c", [2, 1, 64]),
            ]

    def get_inputs(self) -> list[_FakeTensorMeta]:
        return self._inputs

    def run(self, _output_names: Any, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        expected = {inp.name for inp in self._inputs}
        if set(feed.keys()) != expected:
            raise RuntimeError(f"Feed mismatch for {self.schema}: got {set(feed)}, want {expected}")
        prob = np.array([[0.42]], dtype=np.float32)
        if self.schema == "v5":
            return [prob, np.zeros((2, 1, 128), dtype=np.float32)]
        return [prob, np.zeros((2, 1, 64), dtype=np.float32), np.zeros((2, 1, 64), dtype=np.float32)]


def _check_vad_signatures(cfg: AppConfig) -> str:
    from a8vc.audio.vad import SileroVADOnnx

    frame = np.zeros(512, dtype=np.float32)
    tested: list[str] = []
    for schema in ("sherpa", "v5", "v4"):
        vad = SileroVADOnnx("dummy.onnx", session=_FakeSileroSession(schema))
        p = vad.probability(frame)
        if not (0.0 <= p <= 1.0):
            raise RuntimeError(f"Invalid probability {p} for schema={schema}")
        tested.append(schema)

    vad_path = Path(cfg.paths.vad_model_path)
    if vad_path.is_file():
        real_vad = SileroVADOnnx(vad_path)
        p_real = real_vad.probability(frame)
        if not (0.0 <= p_real <= 1.0):
            raise RuntimeError(f"Invalid probability {p_real} from real {vad_path}")
        tested.append(f"real({vad_path.name}:{real_vad._schema})")

    return f"verified VAD schemas: {', '.join(tested)}"


def _check_builtin_voices() -> str:
    from a8vc.paths import get_builtin_voices_dir
    from a8vc.voices.catalog import load_builtin_catalog
    from a8vc.voices.format import A8VOICE_EXT, verify_voice_pack

    builtins_dir = get_builtin_voices_dir()
    catalog = load_builtin_catalog(builtins_dir)
    if len(catalog) < 15:
        raise RuntimeError(f"Expected >= 15 built-in voices, found {len(catalog)}")

    for entry in catalog:
        fname = str(entry.get("filename") or f"{entry['voice_id']}{A8VOICE_EXT}")
        pack_path = builtins_dir / fname
        if not pack_path.is_file():
            raise FileNotFoundError(f"Bundled voice pack missing: {pack_path}")
        verify_voice_pack(pack_path)

    return f"verified {len(catalog)} bundled .a8voice packs"


def _check_embedded_http_and_websocket(cfg: AppConfig) -> str:
    """Boot a real embedded uvicorn server and verify HTTP JSON + TCP WebSocket `/ws/vc`."""
    from a8vc.audio.io import float32_to_pcm16
    from a8vc.desktop import _start_embedded_server, find_available_port, wait_for_server_ready
    from a8vc.runtime.audio8_onnx import StreamAudioChunk
    from a8vc.server.app import ServerState, create_app
    from a8vc.voices.format import VoicePack

    class _StubASR:
        def transcribe(self, audio_16k: np.ndarray, sample_rate: int = 16000) -> str:
            return "selfcheck"

    class _StubSynth:
        sample_rate = 44100
        active_provider = "CPUExecutionProvider"
        provider_warning = None
        provider_note = None
        session_providers = {"slow_ar": "CPUExecutionProvider"}

        def warm_voice(self, voice: VoicePack) -> float:
            return 0.5

        def cache_stats(self) -> dict[str, Any]:
            return {"size": 0, "capacity": 4, "hits": 1, "misses": 0}

        def stream_audio(self, text: str, voice: VoicePack, **kwargs: Any) -> Any:
            yield StreamAudioChunk(
                audio=np.zeros(4410, dtype=np.float32),
                sample_rate=44100,
                frames_generated=4,
                ttfa_ms=1.0,
                prefill_ms=0.5,
                cache_hit=True,
                is_final=True,
            )

    with tempfile.TemporaryDirectory(prefix="a8vc_selfcheck_") as tmpdir:
        test_cfg = cfg.model_copy(deep=True)
        test_cfg.paths.voices_dir = Path(tmpdir) / "voices"
        test_cfg.vad.backend = "energy"
        state = ServerState(
            test_cfg,
            synthesizer=_StubSynth(),
            asr=_StubASR(),
            desktop_mode=False,
        )
        app = create_app(test_cfg, state=state)
        port = find_available_port("127.0.0.1", 0)
        server, thread = _start_embedded_server(app, "127.0.0.1", port)
        url = f"http://127.0.0.1:{port}"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        try:
            if not wait_for_server_ready(url, timeout_sec=10.0, server_thread=thread):
                raise RuntimeError(f"Embedded uvicorn server failed to start on {url}")

            # 1. Verify /api/health returns valid JSON
            with opener.open(f"{url}/api/health", timeout=5.0) as resp:
                health = json.loads(resp.read().decode("utf-8"))
                if health.get("status") != "ok":
                    raise RuntimeError(f"Unexpected /api/health payload: {health}")

            # 2. Verify / returns index.html with safe response parser
            with opener.open(f"{url}/", timeout=5.0) as resp:
                html = resp.read().decode("utf-8")
                if "parseApiResponse" not in html:
                    raise RuntimeError("index.html missing parseApiResponse helper")

            # 3. Verify structured JSON error on missing voice (never plain-text 500)
            req = urllib.request.Request(
                f"{url}/api/tts/stream",
                data=json.dumps({"voice_id": "nonexistent_selfcheck_voice", "text": "hi"}).encode(
                    "utf-8"
                ),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                opener.open(req, timeout=5.0)
                raise RuntimeError("Expected HTTP 404 for nonexistent voice on /api/tts/stream")
            except urllib.error.HTTPError as err:
                if err.code != 404:
                    raise RuntimeError(f"Expected HTTP 404, got {err.code}") from err
                err_body = json.loads(err.read().decode("utf-8"))
                if "detail" not in err_body:
                    raise RuntimeError(f"Expected JSON error detail, got {err_body}") from err

            # 4. Verify real TCP WebSocket upgrade + config handshake on /ws/vc
            voices = state.library.list_voices()
            if not voices:
                raise RuntimeError("No voices seeded into temporary self-check library")
            first_voice = voices[0]["voice_id"]

            async def _probe_ws() -> tuple[dict[str, Any], dict[str, Any], int]:
                import websockets

                async with websockets.connect(f"ws://127.0.0.1:{port}/ws/vc") as ws:
                    ready = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
                    await ws.send(
                        json.dumps(
                            {
                                "type": "config",
                                "voice_id": first_voice,
                                "input_sample_rate": 48000,
                            }
                        )
                    )
                    changed = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
                    # Stream a 0.5s synthetic tone at 48kHz + flush and verify PCM output arrives
                    t = np.linspace(0.0, 0.5, 24000, endpoint=False, dtype=np.float32)
                    tone = 0.25 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
                    await ws.send(float32_to_pcm16(tone))
                    await ws.send(json.dumps({"type": "flush"}))

                    pcm_bytes = 0
                    while True:
                        msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        if isinstance(msg, bytes):
                            pcm_bytes += len(msg)
                        else:
                            ev = json.loads(msg)
                            if ev.get("type") == "metrics":
                                break
                            if ev.get("type") == "error":
                                raise RuntimeError(f"/ws/vc returned error: {ev.get('message')}")
                    return ready, changed, pcm_bytes

            ready_msg, changed_msg, pcm_bytes = asyncio.run(_probe_ws())
            if ready_msg.get("type") != "ready":
                raise RuntimeError(f"Unexpected /ws/vc ready message: {ready_msg}")
            if changed_msg.get("type") != "voice_changed":
                raise RuntimeError(f"Unexpected /ws/vc voice_changed message: {changed_msg}")
            if pcm_bytes <= 0:
                raise RuntimeError("Expected non-empty PCM16 audio output from /ws/vc")

            return (
                f"HTTP + real TCP WebSocket /ws/vc verified on 127.0.0.1:{port} "
                f"(voice={first_voice}, pcm_out={pcm_bytes}B)"
            )
        finally:
            state.close()
            server.should_exit = True
            thread.join(timeout=5.0)


def run_self_check(cfg: AppConfig) -> dict[str, Any]:
    """Run all runtime self-checks and return a JSON-serializable report."""
    checks = [
        _run_probe("onnxruntime", _check_onnxruntime),
        _run_probe("sherpa_coimport", _check_sherpa_coimport),
        _run_probe("core_imports", _check_core_imports),
        _run_probe("uvicorn_websocket_backend", _check_uvicorn_websocket_backend),
        _run_probe("provider_policy", _check_provider_policy),
        _run_probe("vad_signatures", lambda: _check_vad_signatures(cfg)),
        _run_probe("builtin_voices", _check_builtin_voices),
        _run_probe("embedded_http_and_websocket", lambda: _check_embedded_http_and_websocket(cfg)),
    ]
    passed = all(c.ok for c in checks)
    return {
        "ok": passed,
        "passed": sum(1 for c in checks if c.ok),
        "failed": sum(1 for c in checks if not c.ok),
        "total": len(checks),
        "checks": [c.to_dict() for c in checks],
    }
