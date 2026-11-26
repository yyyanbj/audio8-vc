"""Tests for Windows desktop mode, path resolution, model manager, local live API, and packaging."""

from __future__ import annotations

import ast
import importlib.util
import io
import json
import struct
import sys
import tarfile
import threading
import time
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from a8vc import desktop, models_manager, paths
from a8vc.asr.sherpa import MockASR
from a8vc.audio.device import AudioPlaybackQueue
from a8vc.config import AppConfig, load_config
from a8vc.engine.live_session import LocalLiveSessionManager
from a8vc.runtime.audio8_onnx import StreamAudioChunk
from a8vc.server.app import ServerState, create_app
from a8vc.voices.format import VoicePack, build_voice_pack, save_voice_pack
from a8vc.voices.store import VoiceLibrary

REPO_ROOT = Path(__file__).resolve().parents[1]


class _StubSynth:
    def __init__(self, sample_rate: int = 44100) -> None:
        self._sr = sample_rate
        self._warmed: set[str] = set()

    @property
    def sample_rate(self) -> int:
        return self._sr

    def warm_voice(self, voice: VoicePack) -> float:
        self._warmed.add(voice.voice_id)
        return 1.25

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
        audio = np.full(1024, 0.1, dtype=np.float32)
        yield StreamAudioChunk(
            audio=audio,
            sample_rate=self._sr,
            frames_generated=4,
            ttfa_ms=10.0,
            prefill_ms=2.0,
            cache_hit=voice.voice_id in self._warmed,
            is_final=True,
        )

    def decode_codes(self, codes: np.ndarray) -> np.ndarray:
        return np.zeros(codes.shape[-1] * 2048, dtype=np.float32)

    def cache_stats(self) -> dict[str, Any]:
        return {
            "entries": len(self._warmed),
            "max_entries": 2,
            "hits": 1,
            "misses": 1,
            "memory_mb": 0.1,
        }

    def close(self) -> None:
        return None


class _FakeAudioBridge:
    """In-memory replacement for `LocalAudioBridge` so tests run without audio hardware."""

    instances: list[_FakeAudioBridge] = []

    def __init__(
        self,
        on_input_chunk: Callable[[np.ndarray, int], None],
        input_sample_rate: int = 16000,
        output_sample_rate: int = 44100,
        input_block_ms: int = 20,
        input_device: str | int | None = None,
        output_device: str | int | None = None,
    ) -> None:
        self.on_input_chunk = on_input_chunk
        self.active_input_sample_rate = input_sample_rate
        self.active_output_sample_rate = output_sample_rate
        self.input_device = input_device
        self.output_device = output_device
        self.playback = AudioPlaybackQueue(sample_rate=output_sample_rate)
        self.started = False
        self.stopped = False
        _FakeAudioBridge.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True
        self.playback.clear()


def _seed_voices(root: Path) -> VoiceLibrary:
    lib = VoiceLibrary(root)
    rng = np.random.default_rng(11)
    for vid, name in (("hero", "Hero"), ("mage", "Mage")):
        pack = build_voice_pack(
            voice_id=vid,
            codes=rng.integers(0, 4096, size=(10, 16), dtype=np.int64),
            reference_text=f"Hello from {name}.",
            name=name,
            language="en",
        )
        lib.save(pack)
    return lib


def test_paths_portable_frozen_and_readonly_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exe_dir = tmp_path / "install_dir"
    bundle_dir = exe_dir / "_internal"
    bundled_model = exe_dir / "models" / "audio8-tts-0.1b-onnx-int8"
    bundled_model.mkdir(parents=True)
    (bundle_dir / "configs").mkdir(parents=True)
    (bundle_dir / "configs" / "default.toml").write_text(
        '[runtime]\nprovider = "dml"\n', encoding="utf-8"
    )

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle_dir), raising=False)
    monkeypatch.setattr(sys, "executable", str(exe_dir / "Audio8-VC.exe"))
    monkeypatch.delenv("A8VC_DATA_DIR", raising=False)
    monkeypatch.delenv("A8VC_PORTABLE", raising=False)

    # Portable marker next to executable puts writable data under <exe_dir>/data
    (exe_dir / "portable.flag").write_text("portable", encoding="utf-8")
    assert paths.is_frozen()
    assert paths.is_portable_mode()
    data_dir = paths.get_data_dir(desktop=True)
    assert data_dir == (exe_dir / "data").resolve()

    # Readonly fallback finds bundled model in <exe_dir>/models while voices stay writable
    cfg = load_config(desktop=True)
    assert cfg.runtime.provider == "dml"
    assert cfg.paths.model_dir == bundled_model.resolve()
    assert cfg.paths.voices_dir == (data_dir / "voices").resolve()

    # Non-portable Windows mode resolves to %LOCALAPPDATA%\Audio8-VC
    (exe_dir / "portable.flag").unlink()
    local_app = tmp_path / "LocalAppData"
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(local_app))
    assert not paths.is_portable_mode()
    assert paths.get_user_data_dir() == local_app / "Audio8-VC"


def test_safe_extract_tar_bz2_blocks_traversal_and_symlinks(tmp_path: Path) -> None:
    good_tar = tmp_path / "good.tar.bz2"
    with tarfile.open(good_tar, "w:bz2") as tf:
        payload = b"hello"
        info = tarfile.TarInfo("sub/tokens.txt")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))

    out_good = tmp_path / "out_good"
    out_good.mkdir()
    models_manager.safe_extract_tar_bz2(good_tar, out_good)
    assert (out_good / "sub" / "tokens.txt").read_bytes() == b"hello"

    bad_tar = tmp_path / "bad.tar.bz2"
    with tarfile.open(bad_tar, "w:bz2") as tf:
        payload = b"evil"
        info = tarfile.TarInfo("../escape.txt")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))

    with pytest.raises(RuntimeError, match="unsafe archive path"):
        models_manager.safe_extract_tar_bz2(bad_tar, tmp_path / "out_bad")

    sym_tar = tmp_path / "sym.tar.bz2"
    with tarfile.open(sym_tar, "w:bz2") as tf:
        info = tarfile.TarInfo("link.txt")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tf.addfile(info)

    with pytest.raises(RuntimeError, match="archive symlink"):
        models_manager.safe_extract_tar_bz2(sym_tar, tmp_path / "out_sym")


def test_model_download_manager_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = AppConfig()
    cfg.paths.model_dir = tmp_path / "models" / "audio8"
    cfg.paths.asr_model_dir = tmp_path / "models" / "asr"
    cfg.paths.vad_model_path = tmp_path / "models" / "vad" / "silero_vad.onnx"

    def _fake_hf(
        repo_id: str,
        local_dir: Path,
        *,
        ignore_patterns: list[str] | None = None,
        hf_endpoint: str | None = None,
    ) -> Path:
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "slow_int8.onnx").write_bytes(b"s")
        (local_dir / "fast_int8.onnx").write_bytes(b"f")
        (local_dir / "codec_fp16.onnx").write_bytes(b"c")
        (local_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
        (local_dir / "runtime_manifest.json").write_text(
            json.dumps(
                {
                    "slow_decode_models": {"int8": "slow_int8.onnx"},
                    "fast_models": {"int8": "fast_int8.onnx"},
                    "codec_models": {"fp16": "codec_fp16.onnx"},
                    "tokenizer_file": "tokenizer.json",
                }
            ),
            encoding="utf-8",
        )
        if ignore_patterns is None:
            reg = local_dir / "registration"
            reg.mkdir(parents=True, exist_ok=True)
            (reg / "codec_encoder_fp16.onnx").write_bytes(b"r")
        return local_dir

    def _fake_asr(asr_dir: Path, **kwargs: Any) -> Path:
        cb = kwargs.get("on_progress")
        if cb:
            cb(512, 1024)
            cb(1024, 1024)
        (asr_dir / "sv").mkdir(parents=True, exist_ok=True)
        (asr_dir / "sv" / "tokens.txt").write_text("a\n", encoding="utf-8")
        (asr_dir / "sv" / "model.int8.onnx").write_bytes(b"m")
        return asr_dir

    def _fake_vad(vad_path: Path, **kwargs: Any) -> Path:
        vad_path.parent.mkdir(parents=True, exist_ok=True)
        vad_path.write_bytes(b"vad")
        return vad_path

    monkeypatch.setattr(models_manager, "download_hf_snapshot", _fake_hf)
    monkeypatch.setattr(models_manager, "download_sensevoice_asr", _fake_asr)
    monkeypatch.setattr(models_manager, "download_silero_vad", _fake_vad)

    mgr = models_manager.ModelDownloadManager(cfg)
    assert mgr.status()["models"]["ready_for_vc"] is False

    mgr.start_download(audio8_onnx=True, include_registration=True, asr_sensevoice=True, silero_vad=True)
    for _ in range(50):
        st = mgr.status()
        if st["download"]["status"] in {"completed", "error"}:
            break
        time.sleep(0.02)

    final = mgr.status()
    assert final["download"]["status"] == "completed"
    assert final["download"]["percent"] == 100.0
    assert final["models"]["ready_for_vc"] is True
    assert final["models"]["ready_for_enroll"] is True
    assert final["models"]["silero_vad"]["ready"] is True


def test_desktop_api_and_local_live_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _FakeAudioBridge.instances.clear()
    lib = _seed_voices(tmp_path / "voices")
    cfg = AppConfig()
    cfg.paths.voices_dir = lib.root
    cfg.paths.model_dir = tmp_path / "models" / "audio8"
    cfg.paths.asr_model_dir = tmp_path / "models" / "asr"
    cfg.paths.vad_model_path = tmp_path / "models" / "vad" / "silero_vad.onnx"
    cfg.vad.backend = "energy"
    cfg.vad.speech_start_ms = 64
    cfg.vad.min_silence_ms = 160
    cfg.vad.min_segment_ms = 120

    state = ServerState(
        cfg,
        library=lib,
        synthesizer=_StubSynth(),
        asr=MockASR("测试本机变声"),
        desktop_mode=True,
        bridge_factory=_FakeAudioBridge,
    )
    app = create_app(cfg, state=state, desktop_mode=True)

    with TestClient(app) as client:
        info = client.get("/api/system/info")
        assert info.status_code == 200
        assert info.json()["desktop_mode"] is True

        m_res = client.get("/api/models/status")
        assert m_res.status_code == 200
        assert "models" in m_res.json() and "download" in m_res.json()

        fav = client.get("/favicon.ico")
        assert fav.status_code == 200
        assert fav.content[:4] == b"\x00\x00\x01\x00"

        start_res = client.post(
            "/api/live/start",
            json={"voice_id": "hero", "input_device": "", "output_device": ""},
        )
        assert start_res.status_code == 200
        body = start_res.json()
        assert body["running"] is True
        assert body["voice_id"] == "hero"
        assert len(_FakeAudioBridge.instances) == 1
        bridge = _FakeAudioBridge.instances[0]
        assert bridge.started is True

        # Feed loud speech + silence through fake mic callback and confirm converted audio arrives
        t = np.linspace(0.0, 0.3, int(16000 * 0.3), endpoint=False, dtype=np.float32)
        speech = (0.3 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
        silence = np.zeros(int(16000 * 0.3), dtype=np.float32)
        bridge.on_input_chunk(speech, 16000)
        bridge.on_input_chunk(silence, 16000)

        for _ in range(50):
            st = client.get("/api/live/status").json()
            if st["last_transcript"]:
                break
            time.sleep(0.03)
        st = client.get("/api/live/status").json()
        assert st["last_transcript"] == "测试本机变声"
        assert bridge.playback.buffered_ms > 0

        # Hot-switch voice while running
        sw = client.post("/api/live/voice", json={"voice_id": "mage"})
        assert sw.status_code == 200
        assert sw.json()["voice_id"] == "mage"

        fl = client.post("/api/live/flush")
        assert fl.status_code == 200
        assert fl.json()["flushed"] is True

        stop_res = client.post("/api/live/stop")
        assert stop_res.status_code == 200
        assert stop_res.json()["running"] is False
        assert bridge.stopped is True


def test_desktop_helpers_stdio_port_association_and_health(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Frozen console=False Windows app starts with sys.stdout / sys.stderr = None
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    log_path = tmp_path / "logs" / "desktop.log"
    desktop.ensure_safe_stdio(log_path)
    assert sys.stdout is not None
    assert sys.stderr is not None
    print("hello-frozen-stdout")
    sys.stdout.flush()
    assert "hello-frozen-stdout" in log_path.read_text(encoding="utf-8")

    port = desktop.find_available_port("127.0.0.1", preferred=0)
    assert port > 0

    # Fast-fail when server thread is already dead
    dead_thread = threading.Thread(target=lambda: None)
    dead_thread.start()
    dead_thread.join()
    t0 = time.monotonic()
    assert desktop.wait_for_server_ready(
        f"http://127.0.0.1:{port}", timeout_sec=5.0, server_thread=dead_thread
    ) is False
    assert (time.monotonic() - t0) < 1.0

    # Real local HTTP server readiness check
    class _HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

        def log_message(self, format: str, *args: Any) -> None:
            return None

    httpd = HTTPServer(("127.0.0.1", 0), _HealthHandler)
    srv_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    srv_thread.start()
    try:
        assert desktop.wait_for_server_ready(
            f"http://127.0.0.1:{httpd.server_port}",
            timeout_sec=3.0,
            server_thread=srv_thread,
        ) is True
    finally:
        httpd.shutdown()
        httpd.server_close()

    # Double-click .a8voice association import
    cfg = AppConfig()
    cfg.paths.voices_dir = tmp_path / "voices"
    state = ServerState(cfg, synthesizer=_StubSynth(), asr=MockASR(), desktop_mode=True)
    rng = np.random.default_rng(99)
    ext_pack = build_voice_pack(
        voice_id="associated_v1",
        codes=rng.integers(0, 4096, size=(10, 12), dtype=np.int64),
        reference_text="Hi there",
    )
    ext_file = tmp_path / "associated_v1.a8voice"
    save_voice_pack(ext_pack, ext_file)
    imported_id = desktop._import_associated_voice(state, str(ext_file))
    assert imported_id == "associated_v1"
    assert state.active_voice_id == "associated_v1"
    assert state.library.exists("associated_v1")


def test_windows_packaging_assets_and_spec_integrity(tmp_path: Path) -> None:
    # 1. Icon generator produces valid 7-icon ICO and 256x256 PNG
    gen_script = REPO_ROOT / "packaging" / "windows" / "generate_icon.py"
    spec_mod = importlib.util.spec_from_file_location("generate_icon", gen_script)
    assert spec_mod is not None and spec_mod.loader is not None
    mod = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(mod)

    ico_path, png_path = mod.generate_icons(tmp_path)
    ico_bytes = ico_path.read_bytes()
    reserved, img_type, count = struct.unpack("<HHH", ico_bytes[:6])
    assert (reserved, img_type, count) == (0, 1, 7)
    png_bytes = png_path.read_bytes()
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    w, h = struct.unpack(">II", png_bytes[16:24])
    assert (w, h) == (256, 256)

    # 2. PyInstaller 6 spec parses cleanly and avoids removed PyInstaller 5 arguments
    spec_text = (REPO_ROOT / "packaging" / "windows" / "audio8_vc.spec").read_text(encoding="utf-8")
    ast.parse(spec_text)
    assert "cipher=" not in spec_text
    assert ".zipfiles" not in spec_text
    assert ".zipped_data" not in spec_text
    assert 'name="Audio8-VC"' in spec_text
    assert 'name="a8vc"' in spec_text

    # 3. Inno Setup installer script includes expected metadata, clean upgrade, and .a8voice association
    iss_text = (REPO_ROOT / "packaging" / "windows" / "installer.iss").read_text(encoding="utf-8")
    assert "Audio8-VC.exe" in iss_text
    assert "a8vc.exe" in iss_text
    assert ".a8voice" in iss_text
    assert "Audio8VC.VoicePack" in iss_text
    assert "IsValidWebView2Version" in iss_text
    assert "[InstallDelete]" in iss_text

    # 4. Workflow uses hashtable splatting (not array splatting) and builds CPU + GPU variants
    wf_text = (REPO_ROOT / ".github" / "workflows" / "build-windows.yml").read_text(encoding="utf-8")
    assert '@("-Provider"' not in wf_text
    assert 'Provider = "${{ matrix.provider }}"' in wf_text
    assert '["cpu","dml","cuda"]' in wf_text

    # 5. LocalLiveSessionManager rejects voice switch when not running
    cfg = AppConfig()
    lib = _seed_voices(tmp_path / "voices_mgr")
    mgr = LocalLiveSessionManager(
        cfg,
        create_engine=lambda _v: None,  # type: ignore[arg-type,return-value]
        load_voice=lib.load,
    )
    with pytest.raises(RuntimeError, match="not running"):
        mgr.switch_voice("hero")


def test_onnx_providers_directml_options_and_gpu_fallback(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from a8vc.runtime.providers import (
        build_session_options,
        create_onnx_session,
        detect_ort_flavor,
        probe_provider_info,
        resolve_providers,
    )

    assert resolve_providers("cpu", ["CUDAExecutionProvider", "CPUExecutionProvider"]) == [
        "CPUExecutionProvider"
    ]
    assert resolve_providers("dml", ["DmlExecutionProvider", "CPUExecutionProvider"]) == [
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert resolve_providers("cuda", ["CUDAExecutionProvider", "CPUExecutionProvider"]) == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert resolve_providers("auto", ["DmlExecutionProvider", "CPUExecutionProvider"]) == [
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert resolve_providers("cuda", ["CPUExecutionProvider"]) == ["CPUExecutionProvider"]

    assert detect_ort_flavor(["CUDAExecutionProvider", "CPUExecutionProvider"]) == "cuda"
    assert detect_ort_flavor(["DmlExecutionProvider", "CPUExecutionProvider"]) == "dml"
    assert detect_ort_flavor(["CPUExecutionProvider"]) == "cpu"

    class _FakeSessionOptions:
        def __init__(self) -> None:
            self.graph_optimization_level = None
            self.intra_op_num_threads = 0
            self.inter_op_num_threads = 0
            self.log_severity_level = 0
            self.enable_mem_pattern = True
            self.execution_mode = "parallel"

    class _FakeSession:
        def __init__(self, providers: list[str]) -> None:
            self._providers = providers

        def get_providers(self) -> list[str]:
            return list(self._providers)

    attempts: list[list[str]] = []

    def _failing_cuda_session(path: str, sess_options: Any, providers: list[str]) -> _FakeSession:
        attempts.append(list(providers))
        if "CUDAExecutionProvider" in providers:
            raise RuntimeError("cublasLt64_12.dll not found")
        return _FakeSession(providers)

    fake_ort = SimpleNamespace(
        SessionOptions=_FakeSessionOptions,
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
        get_available_providers=lambda: ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"],
        InferenceSession=_failing_cuda_session,
    )

    # DirectML must disable memory pattern & use sequential execution
    dml_opts = build_session_options(["DmlExecutionProvider", "CPUExecutionProvider"], threads=4, ort_module=fake_ort)
    assert dml_opts.enable_mem_pattern is False
    assert dml_opts.execution_mode == "sequential"

    cpu_opts = build_session_options(["CPUExecutionProvider"], threads=4, ort_module=fake_ort)
    assert cpu_opts.enable_mem_pattern is True

    # Failed CUDA init cleanly falls back to CPU with a diagnostic warning
    session, active_ep, warning = create_onnx_session(
        tmp_path / "dummy.onnx",
        provider="cuda",
        threads=2,
        ort_module=fake_ort,
    )
    assert isinstance(session, _FakeSession)
    assert active_ep == "CPUExecutionProvider"
    assert warning is not None and "CUDA 12" in warning
    assert attempts == [
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ["CPUExecutionProvider"],
    ]

    # Real ORT probe always returns structured diagnostics
    info = probe_provider_info("cpu")
    assert info["configured_provider"] == "cpu"
    assert "CPUExecutionProvider" in info["available_providers"]
