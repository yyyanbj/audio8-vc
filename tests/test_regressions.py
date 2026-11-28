"""Regression tests for v0.2.0 voice changer crashes, WebSocket inference, VAD, and self-check."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from a8vc.audio.vad import EnergyVAD, SileroVADOnnx, create_vad
from a8vc.cli import main as cli_main
from a8vc.config import AppConfig, VADConfig
from a8vc.runtime.providers import select_model_provider
from a8vc.selfcheck import run_self_check
from a8vc.server.app import ServerState, create_app
from a8vc.voices.format import build_voice_pack
from a8vc.voices.store import VoiceLibrary


class _Meta:
    def __init__(self, name: str, shape: list[Any]) -> None:
        self.name = name
        self.shape = shape


class _MockSileroSession:
    def __init__(self, inputs: list[_Meta], expected_prob: float = 0.77) -> None:
        self._inputs = inputs
        self._expected_prob = expected_prob
        self.last_feed: dict[str, np.ndarray] = {}

    def get_inputs(self) -> list[_Meta]:
        return self._inputs

    def run(self, _output_names: Any, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.last_feed = feed
        assert set(feed.keys()) == {inp.name for inp in self._inputs}
        prob = np.array([[self._expected_prob]], dtype=np.float32)
        if "state" in feed:
            return [prob, np.zeros_like(feed["state"])]
        return [prob, np.zeros_like(feed["h"]), np.zeros_like(feed["c"])]


def test_silero_vad_supports_all_three_onnx_signatures() -> None:
    frame = np.zeros(512, dtype=np.float32)

    # 1. sherpa-onnx silero_vad.onnx export: ['x', 'h', 'c']
    sherpa_sess = _MockSileroSession(
        [_Meta("x", [1, 512]), _Meta("h", [2, 1, 64]), _Meta("c", [2, 1, 64])],
        expected_prob=0.81,
    )
    vad_sherpa = SileroVADOnnx("dummy.onnx", session=sherpa_sess)
    assert abs(vad_sherpa.probability(frame) - 0.81) < 1e-5
    assert "x" in sherpa_sess.last_feed and "sr" not in sherpa_sess.last_feed

    # 2. Official Silero v5 export: ['input', 'state', 'sr']
    v5_sess = _MockSileroSession(
        [_Meta("input", [1, None]), _Meta("state", [2, 1, 128]), _Meta("sr", [])],
        expected_prob=0.63,
    )
    vad_v5 = SileroVADOnnx("dummy.onnx", session=v5_sess)
    assert abs(vad_v5.probability(frame) - 0.63) < 1e-5
    assert v5_sess.last_feed["input"].shape == (1, 576)

    # 3. Official Silero v4 export: ['input', 'sr', 'h', 'c']
    v4_sess = _MockSileroSession(
        [_Meta("input", [1, 512]), _Meta("sr", []), _Meta("h", [2, 1, 64]), _Meta("c", [2, 1, 64])],
        expected_prob=0.52,
    )
    vad_v4 = SileroVADOnnx("dummy.onnx", session=v4_sess)
    assert abs(vad_v4.probability(frame) - 0.52) < 1e-5


def test_create_vad_accepts_str_path_and_falls_back_on_corrupted_onnx(tmp_path: Path) -> None:
    bad_onnx = tmp_path / "broken_silero.onnx"
    bad_onnx.write_bytes(b"not-a-valid-onnx-model")

    vad = create_vad(VADConfig(backend="auto"), str(bad_onnx))
    assert isinstance(vad, EnergyVAD)


def test_select_model_provider_guards_directml_int8_ar_models() -> None:
    dml_avail = ["DmlExecutionProvider", "CPUExecutionProvider"]
    cuda_avail = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    coreml_avail = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

    slow_ep, slow_note = select_model_provider(
        "dml", model_role="slow_ar", precision="int8", available=dml_avail
    )
    fast_ep, fast_note = select_model_provider(
        "dml", model_role="fast_ar", precision="int8", available=dml_avail
    )
    dec_ep, dec_note = select_model_provider(
        "dml", model_role="codec_decoder", precision="fp16", available=dml_avail
    )
    cuda_int8_ep, cuda_int8_note = select_model_provider(
        "cuda", model_role="slow_ar", precision="int8", available=cuda_avail
    )
    cuda_dec_ep, cuda_dec_note = select_model_provider(
        "cuda", model_role="codec_decoder", precision="fp16", available=cuda_avail
    )
    cuda_fp16_ar_ep, cuda_fp16_ar_note = select_model_provider(
        "cuda", model_role="slow_ar", precision="fp16", available=cuda_avail
    )
    coreml_int8_ep, coreml_int8_note = select_model_provider(
        "coreml", model_role="fast_ar", precision="int8", available=coreml_avail
    )
    coreml_dec_ep, coreml_dec_note = select_model_provider(
        "coreml", model_role="codec_decoder", precision="fp16", available=coreml_avail
    )

    assert slow_ep == "cpu" and slow_note is not None
    assert fast_ep == "cpu" and fast_note is not None
    assert dec_ep == "dml" and dec_note is None
    assert cuda_int8_ep == "cpu" and cuda_int8_note is not None
    assert cuda_dec_ep == "cuda" and cuda_dec_note is None
    assert cuda_fp16_ar_ep == "cuda" and cuda_fp16_ar_note is None
    assert coreml_int8_ep == "cpu" and coreml_int8_note is not None
    assert coreml_dec_ep == "coreml" and coreml_dec_note is None


def test_tts_stream_primes_first_chunk_and_returns_json_500_on_synthesis_error(
    tmp_path: Path,
) -> None:
    lib = VoiceLibrary(tmp_path / "voices")
    pack = build_voice_pack(
        voice_id="demo",
        codes=np.zeros((10, 8), dtype=np.int64),
        reference_text="hello",
        name="Demo",
        language="en",
    )
    lib.save(pack)

    class _ExplodingSynth:
        sample_rate = 44100

        def warm_voice(self, voice: Any) -> float:
            raise RuntimeError("DirectML 80070057 boom")

        def stream_audio(self, text: str, voice: Any, **kwargs: Any) -> Any:
            raise RuntimeError("DirectML 80070057 boom")
            yield  # pragma: no cover

        def cache_stats(self) -> dict[str, Any]:
            return {}

    cfg = AppConfig()
    cfg.paths.voices_dir = tmp_path / "voices"
    state = ServerState(cfg, synthesizer=_ExplodingSynth(), desktop_mode=False)
    app = create_app(cfg, state=state)

    with TestClient(app) as client:
        resp = client.post("/api/tts/stream", json={"voice_id": "demo", "text": "hello"})
        assert resp.status_code == 500
        body = resp.json()
        assert "RuntimeError" in body["detail"]
        assert "DirectML 80070057 boom" in body["detail"]


def test_ws_vc_reports_engine_init_failure_as_json_error_frame(tmp_path: Path) -> None:
    cfg = AppConfig()
    cfg.paths.voices_dir = tmp_path / "voices"
    state = ServerState(cfg, desktop_mode=False)

    def _boom() -> Any:
        raise RuntimeError("Simulated missing sherpa-onnx-core DLL")

    state.create_vc_engine = _boom  # type: ignore[method-assign]
    app = create_app(cfg, state=state)

    with TestClient(app) as client:
        with client.websocket_connect("/ws/vc") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "Simulated missing sherpa-onnx-core DLL" in msg["message"]


def test_self_check_passes_and_cli_json_works(capsys: object) -> None:
    report = run_self_check(AppConfig())
    assert report["ok"] is True, report
    assert report["failed"] == 0
    assert report["total"] >= 8

    rc = cli_main(["self-check", "--json"])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert parsed["ok"] is True


def test_icon_endpoints_serve_valid_svg_png_and_ico(tmp_path: Path) -> None:
    cfg = AppConfig()
    cfg.paths.voices_dir = tmp_path / "voices"
    app = create_app(cfg)

    with TestClient(app) as client:
        r_svg = client.get("/icon.svg")
        assert r_svg.status_code == 200
        assert "<svg" in r_svg.text

        r_png = client.get("/icon.png")
        assert r_png.status_code == 200
        assert r_png.content.startswith(b"\x89PNG\r\n\x1a\n")

        r_ico = client.get("/favicon.ico")
        assert r_ico.status_code == 200
        assert r_ico.content[:4] == b"\x00\x00\x01\x00"

        r_html = client.get("/")
        assert r_html.status_code == 200
        assert 'src="/icon.svg"' in r_html.text


def test_select_webview_icon_never_passes_png_on_windows() -> None:
    from a8vc.desktop import select_webview_icon

    assert select_webview_icon(platform="win32", frozen=True) is None
    win_dev_icon = select_webview_icon(platform="win32", frozen=False)
    assert win_dev_icon is not None
    assert win_dev_icon.suffix.lower() == ".ico"

    linux_icon = select_webview_icon(platform="linux", frozen=False)
    assert linux_icon is not None
    assert linux_icon.suffix.lower() == ".png"


def test_unblock_windows_bundle_dlls_strips_zone_identifier(tmp_path: Path) -> None:
    from a8vc.desktop import unblock_windows_bundle_dlls

    dll1 = tmp_path / "_internal" / "pythonnet" / "runtime" / "Python.Runtime.dll"
    dll2 = tmp_path / "_internal" / "webview" / "lib" / "Microsoft.Web.WebView2.Core.dll"
    dll1.parent.mkdir(parents=True)
    dll2.parent.mkdir(parents=True)
    dll1.write_bytes(b"mz")
    dll2.write_bytes(b"mz")

    seen: list[str] = []

    def _fake_delete(ads_path: str) -> bool:
        seen.append(ads_path)
        return True

    count = unblock_windows_bundle_dlls([tmp_path], delete_ads=_fake_delete)
    assert count == 2
    assert f"{dll1}:Zone.Identifier" in seen
    assert f"{dll2}:Zone.Identifier" in seen


def test_desktop_smoke_test_and_unshown_browser_fallback(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import sys
    import threading
    import types

    from a8vc import desktop

    class _FakeEvent:
        def __init__(self) -> None:
            self._ev = threading.Event()
            self._cbs: list[Any] = []

        def __iadd__(self, fn: Any) -> _FakeEvent:
            self._cbs.append(fn)
            return self

        def set(self) -> None:
            self._ev.set()
            for cb in self._cbs:
                cb()

        def is_set(self) -> bool:
            return self._ev.is_set()

    class _FakeWindow:
        def __init__(self) -> None:
            self.events = types.SimpleNamespace(shown=_FakeEvent(), loaded=_FakeEvent())
            self.destroyed = False

        def evaluate_js(self, _script: str) -> str:
            return json.dumps(
                {
                    "title": "Audio8-VC 实时变声器",
                    "iconOk": True,
                    "hasStatus": True,
                }
            )

        def destroy(self) -> None:
            self.destroyed = True

    def _make_fake_webview(*, fire_shown: bool) -> Any:
        mod = types.ModuleType("webview")
        mod.renderer = "edgechromium"  # type: ignore[attr-defined]
        holder: dict[str, Any] = {}

        def _create_window(**_kwargs: Any) -> _FakeWindow:
            w = _FakeWindow()
            holder["win"] = w
            return w

        def _start(func: Any = None, gui: Any = None, icon: Any = None) -> None:
            holder["start_kwargs"] = {"gui": gui, "icon": icon}
            if fire_shown:
                holder["win"].events.shown.set()
                holder["win"].events.loaded.set()
                if func is not None:
                    func()

        mod.create_window = _create_window  # type: ignore[attr-defined]
        mod.start = _start  # type: ignore[attr-defined]
        mod._holder = holder  # type: ignore[attr-defined]
        return mod

    cfg = AppConfig()
    cfg.paths.voices_dir = tmp_path / "voices"
    monkeypatch.setenv("A8VC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("A8VC_NO_DIALOG", "1")
    monkeypatch.setattr(desktop, "has_windows_webview2", lambda: True)

    # 1. Healthy GUI smoke-test passes and writes JSON report
    good_wv = _make_fake_webview(fire_shown=True)
    monkeypatch.setitem(sys.modules, "webview", good_wv)
    report_ok = tmp_path / "smoke_ok.json"
    rc_ok = desktop.launch_desktop(
        cfg,
        port=0,
        smoke_test=True,
        smoke_timeout=5.0,
        smoke_report=report_ok,
    )
    assert rc_ok == 0
    rep = json.loads(report_ok.read_text(encoding="utf-8"))
    assert rep["ok"] is True
    assert rep["shown"] is True
    assert rep["loaded"] is True
    assert rep["icon_ok"] is True
    assert good_wv._holder["win"].destroyed is True

    # 2. If webview.start() returns before `shown` fires, smoke-test FAILS (exit 1)
    bad_wv = _make_fake_webview(fire_shown=False)
    monkeypatch.setitem(sys.modules, "webview", bad_wv)
    report_bad = tmp_path / "smoke_bad.json"
    rc_bad = desktop.launch_desktop(
        cfg,
        port=0,
        smoke_test=True,
        smoke_timeout=5.0,
        smoke_report=report_bad,
    )
    assert rc_bad == 1
    rep_bad = json.loads(report_bad.read_text(encoding="utf-8"))
    assert rep_bad["ok"] is False
    assert rep_bad["shown"] is False

    # 3. In normal mode, unshown webview MUST fall back to webbrowser.open()
    opened_urls: list[str] = []

    def _fake_open(url: str) -> bool:
        opened_urls.append(url)
        raise KeyboardInterrupt()

    monkeypatch.setattr(desktop.webbrowser, "open", _fake_open)
    rc_fb = desktop.launch_desktop(cfg, port=0, smoke_test=False)
    assert rc_fb == 0
    assert len(opened_urls) == 1
    assert opened_urls[0].startswith("http://127.0.0.1:")


def test_estimate_max_audio_frames_bounds_short_utterances() -> None:
    from a8vc.runtime.audio8_onnx import estimate_max_audio_frames

    short_en = estimate_max_audio_frames("Hello browser test.", hard_cap=512)
    short_zh = estimate_max_audio_frames("你好世界", hard_cap=512)
    long_text = estimate_max_audio_frames("今天天气不错，" * 20, hard_cap=512)

    assert 120 <= short_en <= 180
    assert 100 <= short_zh <= 160
    assert long_text == 512


def test_compute_thread_plan_and_session_options_for_min_latency() -> None:
    from a8vc.runtime.providers import build_session_options, compute_thread_plan

    # Pipelined CPU mode partitions budget without oversubscription
    ar16, dec16, tot16 = compute_thread_plan(cpu_count=16, codec_on_gpu=False, pipeline_decode=True)
    assert tot16 == 8
    assert ar16 == 3 and dec16 == 5
    assert ar16 + dec16 <= tot16

    ar8, dec8, tot8 = compute_thread_plan(cpu_count=8, codec_on_gpu=False, pipeline_decode=True)
    assert tot8 == 4
    assert ar8 == 2 and dec8 == 2
    assert ar8 + dec8 <= tot8

    # GPU codec mode keeps AR in its 3-thread sweet spot
    ar_gpu, dec_gpu, _ = compute_thread_plan(cpu_count=16, codec_on_gpu=True, pipeline_decode=True)
    assert ar_gpu == 3 and dec_gpu == 2

    # Serial mode allows Codec to use full thread budget
    ar_ser, dec_ser, tot_ser = compute_thread_plan(
        cpu_count=16, codec_on_gpu=False, pipeline_decode=False
    )
    assert ar_ser == 3 and dec_ser == tot_ser == 8

    # Explicit overrides always win
    assert compute_thread_plan(threads=10, ar_threads=4, codec_threads=6) == (4, 6, 10)

    # Session options enable low-latency sequential execution, spinning & denormal flush
    class _FakeSessOpts:
        def __init__(self) -> None:
            self.intra_op_num_threads = 0
            self.inter_op_num_threads = 0
            self.execution_mode = None
            self.graph_optimization_level = None
            self.enable_mem_pattern = True
            self.entries: dict[str, str] = {}

        def add_session_config_entry(self, key: str, value: str) -> None:
            self.entries[key] = value

    class _FakeOrt:
        SessionOptions = _FakeSessOpts

        class ExecutionMode:
            ORT_SEQUENTIAL = "SEQ"

        class GraphOptimizationLevel:
            ORT_ENABLE_ALL = "ALL"

    opts = build_session_options(
        ["CPUExecutionProvider"],
        threads=3,
        inter_op_threads=1,
        allow_spinning=True,
        ort_module=_FakeOrt,
    )
    assert opts.intra_op_num_threads == 3
    assert opts.inter_op_num_threads == 1
    assert opts.execution_mode == "SEQ"
    assert opts.entries["session.intra_op.allow_spinning"] == "1"
    assert opts.entries["session.set_denormal_as_zero"] == "1"


def test_stream_audio_pipelined_matches_serial_and_guard_crossfade(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import types

    from a8vc.config import RuntimeConfig
    from a8vc.runtime import audio8_onnx

    # Minimal dummy model directory + manifest
    manifest = {
        "model_id": "test-mini",
        "model_fingerprint": "test-mini-v1",
        "slow_decode_model": "slow.onnx",
        "fast_model": "fast.onnx",
        "codec_models": "dec.onnx",
        "sample_rate": 44100,
        "codec_hop_length": 512,
        "num_codebooks": 10,
        "codebook_size": 32,
        "semantic_begin_id": 1000,
        "semantic_end_id": 1031,
        "im_end_id": 32,
        "max_seq_len": 128,
        "num_layers": 2,
        "n_local_heads": 2,
        "head_dim": 8,
        "num_fast_layers": 1,
        "fast_n_local_heads": 2,
        "fast_head_dim": 8,
        "mamba_d_conv": 2,
        "mamba_d_ssm": 8,
        "mamba_d_state": 4,
        "mamba_d_head": 4,
        "mamba_n_heads": 2,
        "mamba_n_groups": 1,
    }
    (tmp_path / "runtime_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for fname in ("slow.onnx", "fast.onnx", "dec.onnx", "tokenizer.json"):
        (tmp_path / fname).write_bytes(b"stub")

    class _IO:
        def __init__(self, name: str, shape: list[int], typ: str) -> None:
            self.name = name
            self.shape = shape
            self.type = typ

    class _FakeSlow:
        def __init__(self) -> None:
            self._gen_steps = 0

        def get_inputs(self) -> list[_IO]:
            return [
                _IO("codes", [1, 11, 1], "tensor(int64)"),
                _IO("position", [1], "tensor(int64)"),
                _IO("cache_keys", [2, 1, 2, 128, 8], "tensor(float)"),
                _IO("cache_values", [2, 1, 2, 128, 8], "tensor(float)"),
                _IO("conv_states", [2, 1, 16, 2], "tensor(float)"),
                _IO("ssm_states", [2, 1, 2, 4, 4], "tensor(float)"),
            ]

        def get_outputs(self) -> list[_IO]:
            return [
                _IO("logits", [1, 1, 33], "tensor(float)"),
                _IO("hidden", [1, 1, 16], "tensor(float)"),
                _IO("key_delta", [2, 1, 2, 8], "tensor(float)"),
                _IO("value_delta", [2, 1, 2, 8], "tensor(float)"),
                _IO("next_conv_states", [2, 1, 16, 2], "tensor(float)"),
                _IO("next_ssm_states", [2, 1, 2, 4, 4], "tensor(float)"),
            ]

        def run(self, _outs: Any, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
            pos = int(feed["position"][0])
            logits = np.full((1, 1, 33), -20.0, dtype=np.float32)
            # Emit 14 deterministic audio frames then EOS (index 32)
            tok = 32 if pos >= 65 else (pos % 30)
            logits[0, 0, tok] = 10.0
            hidden = np.full((1, 1, 16), float(pos % 5) * 0.1, dtype=np.float32)
            kd = np.zeros((2, 1, 2, 8), dtype=np.float32)
            vd = np.zeros((2, 1, 2, 8), dtype=np.float32)
            return [logits, hidden, kd, vd, feed["conv_states"].copy(), feed["ssm_states"].copy()]

    class _FakeFast:
        def get_inputs(self) -> list[_IO]:
            return [
                _IO("slow_hidden", [1, 1, 16], "tensor(float)"),
                _IO("token_id", [1, 1], "tensor(int64)"),
                _IO("use_slow_hidden", [1], "tensor(bool)"),
                _IO("input_pos", [1], "tensor(int64)"),
                _IO("cache_key_0", [1, 2, 10, 8], "tensor(float)"),
                _IO("cache_value_0", [1, 2, 10, 8], "tensor(float)"),
            ]

        def get_outputs(self) -> list[_IO]:
            return [
                _IO("logits", [1, 1, 32], "tensor(float)"),
                _IO("key_delta_0", [1, 2, 1, 8], "tensor(float)"),
                _IO("value_delta_0", [1, 2, 1, 8], "tensor(float)"),
            ]

        def run(self, _outs: Any, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
            idx = int(feed["input_pos"][0])
            logits = np.full((1, 1, 32), -20.0, dtype=np.float32)
            logits[0, 0, (idx * 3) % 32] = 10.0
            return [
                logits,
                np.zeros((1, 2, 1, 8), dtype=np.float32),
                np.zeros((1, 2, 1, 8), dtype=np.float32),
            ]

    class _FakeDecoder:
        def get_inputs(self) -> list[_IO]:
            return [_IO("codes", [1, 10, 1], "tensor(int64)")]

        def run(self, _outs: Any, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
            codes = feed["codes"]
            n_frames = codes.shape[2]
            # Deliberately include a tiny window-length-dependent DC offset to verify guard crossfade
            base = np.linspace(-0.3, 0.3, n_frames * 512, dtype=np.float32)
            return [(base + 0.02 * (n_frames % 3)).reshape(1, 1, -1)]

    class _FakeTokenizer:
        @staticmethod
        def from_file(_path: str) -> Any:
            class _Tok:
                def encode(self, seq: str, add_special_tokens: bool = False) -> Any:
                    return types.SimpleNamespace(ids=[1 + (i % 50) for i in range(max(1, len(seq) // 8))])

            return _Tok()

    create_calls: dict[str, dict[str, Any]] = {}

    def _fake_create_sess(path: Path, **kwargs: Any) -> tuple[Any, str, None]:
        create_calls[path.name] = dict(kwargs)
        if path.name == "slow.onnx":
            return _FakeSlow(), "CPUExecutionProvider", None
        if path.name == "fast.onnx":
            return _FakeFast(), "CPUExecutionProvider", None
        return _FakeDecoder(), "CPUExecutionProvider", None

    monkeypatch.setattr(audio8_onnx, "create_onnx_session", _fake_create_sess)
    monkeypatch.setitem(
        __import__("sys").modules,
        "tokenizers",
        types.SimpleNamespace(Tokenizer=_FakeTokenizer),
    )

    pack = build_voice_pack(
        voice_id="mini",
        codes=np.zeros((10, 8), dtype=np.int64),
        reference_text="hi",
        name="Mini",
        language="en",
    )

    monkeypatch.setattr(audio8_onnx, "is_global_thread_pool_enabled", lambda: False)

    rt_pipe = audio8_onnx.Audio8OnnxRuntime(
        tmp_path,
        RuntimeConfig(
            first_chunk_frames=2,
            chunk_frames=6,
            stream_context_frames=4,
            stream_guard_frames=0,
            pipeline_decode=True,
            use_global_thread_pool=True,
        ),
    )
    # Default global thread pool mode shares one spinning pool across Slow-AR, Fast-AR, and Codec
    assert create_calls["slow.onnx"]["use_global_pool"] is True
    assert create_calls["fast.onnx"]["use_global_pool"] is True
    assert create_calls["fast.onnx"]["allow_spinning"] is True
    assert create_calls["dec.onnx"]["use_global_pool"] is True
    assert create_calls["slow.onnx"]["allow_spinning"] is True
    assert create_calls["dec.onnx"]["allow_spinning"] is True

    # Per-session fallback mode: Fast-AR must not spawn a busy-spinning pool that starves Slow-AR/Codec
    create_calls.clear()
    rt_serial = audio8_onnx.Audio8OnnxRuntime(
        tmp_path,
        RuntimeConfig(
            first_chunk_frames=2,
            chunk_frames=6,
            stream_context_frames=4,
            stream_guard_frames=0,
            pipeline_decode=False,
            use_global_thread_pool=False,
        ),
    )
    assert create_calls["fast.onnx"]["use_global_pool"] is False
    assert create_calls["fast.onnx"]["threads"] == 1
    assert create_calls["fast.onnx"]["allow_spinning"] is False
    assert create_calls["slow.onnx"]["allow_spinning"] is True
    assert create_calls["dec.onnx"]["allow_spinning"] is True

    pipe_chunks = list(rt_pipe.stream_audio("hello test", pack, seed=123))
    serial_chunks = list(rt_serial.stream_audio("hello test", pack, seed=123))

    assert len(pipe_chunks) >= 3
    # Progressive doubling ramp: first_chunk=2 -> +4 (cumulative 6) -> +6 (cumulative 12) -> final
    assert pipe_chunks[0].frames_generated == 2
    assert pipe_chunks[1].frames_generated == 6
    assert pipe_chunks[-1].is_final is True
    pipe_wav = np.concatenate([c.audio for c in pipe_chunks])
    serial_wav = np.concatenate([c.audio for c in serial_chunks])
    assert np.array_equal(pipe_wav, serial_wav)

    wav2, metrics = rt_pipe.synthesize("hello test", pack, seed=123)
    assert metrics.cache_hit is True
    assert metrics.num_chunks == len(pipe_chunks)
    assert metrics.total_stall_ms >= 0.0
    assert metrics.max_stall_ms >= 0.0
    assert np.array_equal(wav2, pipe_wav)
    rt_pipe.close()
    rt_serial.close()


def test_tts_stream_closes_iterator_on_error_and_completion(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from a8vc.config import AppConfig
    from a8vc.runtime.audio8_onnx import StreamAudioChunk
    from a8vc.server.app import ServerState, create_app

    cfg = AppConfig()
    cfg.paths.voices_dir = tmp_path / "voices"
    pack = build_voice_pack(
        voice_id="close_test",
        codes=np.zeros((10, 4), dtype=np.int64),
        reference_text="hi",
        name="Close Test",
        language="en",
    )
    state = ServerState(cfg)
    state.library.save(pack, overwrite=True)

    closed_events: list[str] = []

    class _FakeSynth:
        sample_rate = 44100
        active_provider = "CPUExecutionProvider"
        provider_warning = None
        provider_note = None
        session_providers = {"slow_ar": "CPUExecutionProvider"}
        ar_threads = 2
        fast_threads = 1
        codec_threads = 2

        def stream_audio(self, text: str, _pack: Any, **_kwargs: Any) -> Any:
            def _gen() -> Any:
                try:
                    if text == "boom":
                        raise ValueError("bad prompt")
                    yield StreamAudioChunk(
                        audio=np.full(512, 0.1, dtype=np.float32),
                        sample_rate=44100,
                        frames_generated=4,
                        ttfa_ms=10.0,
                        prefill_ms=5.0,
                        cache_hit=True,
                        is_final=True,
                    )
                finally:
                    closed_events.append(text)

            return _gen()

        def close(self) -> None:
            return None

    state._synthesizer = _FakeSynth()  # type: ignore[assignment]
    app = create_app(cfg, state=state)
    client = TestClient(app)

    r_ok = client.post("/api/tts/stream", json={"voice_id": "close_test", "text": "ok"})
    assert r_ok.status_code == 200
    assert len(r_ok.content) == 512 * 2

    r_err = client.post("/api/tts/stream", json={"voice_id": "close_test", "text": "boom"})
    assert r_err.status_code == 400
    assert closed_events == ["ok", "boom"]


def test_slow_ar_kv_compaction_patches_caches_and_falls_back(tmp_path: Path) -> None:
    from a8vc.runtime import slow_ar_kv

    def _make_value_info(name: str, dims: list[int]) -> bytes:
        shape_bytes = b"".join(
            slow_ar_kv._encode_ld(1, slow_ar_kv._encode_varint_field(1, d)) for d in dims
        )
        tensor_type = slow_ar_kv._encode_varint_field(1, 1) + slow_ar_kv._encode_ld(2, shape_bytes)
        type_proto = slow_ar_kv._encode_ld(1, tensor_type)
        return slow_ar_kv._encode_ld(1, name.encode("utf-8")) + slow_ar_kv._encode_ld(2, type_proto)

    def _make_const_node(name: str, arr: np.ndarray) -> bytes:
        tensor_bytes = slow_ar_kv._encode_tensor_proto(arr, name=f"{name}_t")
        attr_bytes = (
            slow_ar_kv._encode_ld(1, b"value")
            + slow_ar_kv._encode_ld(5, tensor_bytes)
            + slow_ar_kv._encode_varint_field(20, 4)
        )
        return (
            slow_ar_kv._encode_ld(3, name.encode("utf-8"))
            + slow_ar_kv._encode_ld(4, b"Constant")
            + slow_ar_kv._encode_ld(5, attr_bytes)
        )

    orig_len = 2048
    graph_parts = [
        slow_ar_kv._encode_ld(11, _make_value_info("cache_keys", [24, 1, 2, orig_len, 64])),
        slow_ar_kv._encode_ld(11, _make_value_info("cache_values", [24, 1, 2, orig_len, 64])),
        slow_ar_kv._encode_ld(11, _make_value_info("codes", [1, 11, 1])),
        slow_ar_kv._encode_ld(13, _make_value_info("mask_inter", [1, 1, 1, orig_len])),
        slow_ar_kv._encode_ld(1, _make_const_node("c_arange", np.arange(orig_len, dtype=np.int64))),
        slow_ar_kv._encode_ld(1, _make_const_node("c_mask2d", np.zeros((1, orig_len), dtype=np.float32))),
        slow_ar_kv._encode_ld(1, _make_const_node("c_mask4d", np.ones((1, 1, 1, orig_len), dtype=np.float32))),
        slow_ar_kv._encode_ld(1, _make_const_node("c_len1", np.array([orig_len], dtype=np.int64))),
        slow_ar_kv._encode_ld(1, _make_const_node("c_shape4", np.array([1, 1, 1, orig_len], dtype=np.int64))),
        slow_ar_kv._encode_ld(1, _make_const_node("c_other", np.array([42], dtype=np.int64))),
    ]
    model_bytes = slow_ar_kv._encode_varint_field(1, 9) + slow_ar_kv._encode_ld(7, b"".join(graph_parts))

    src = tmp_path / "slow_ar_int8.onnx"
    src.write_bytes(model_bytes)

    compact_path, eff_len = slow_ar_kv.prepare_compact_slow_ar_model(
        src, target_seq_len=512, orig_seq_len=2048
    )
    assert eff_len == 512
    assert compact_path.name == "slow_ar_int8.kv512.onnx"
    assert compact_path.is_file()
    meta_path = tmp_path / "slow_ar_int8.kv512.onnx.meta.json"
    assert meta_path.is_file()

    # Re-patching 512 -> 256 proves every expected input, value_info, and mask constant is now 512
    _, stats_round2 = slow_ar_kv.patch_slow_ar_kv_bytes(
        compact_path.read_bytes(), target_seq_len=256, orig_seq_len=512
    )
    assert set(stats_round2["inputs"]) == {"cache_keys", "cache_values"}
    assert len(stats_round2["consts"]) == 5
    assert stats_round2["value_infos"] == 1

    # Second call hits cache without rewriting
    mtime_before = compact_path.stat().st_mtime_ns
    compact_path_2, eff_len_2 = slow_ar_kv.prepare_compact_slow_ar_model(
        src, target_seq_len=512, orig_seq_len=2048
    )
    assert compact_path_2 == compact_path
    assert eff_len_2 == 512
    assert compact_path_2.stat().st_mtime_ns == mtime_before

    # Disabled or non-matching graph falls back cleanly
    same_path, full_len = slow_ar_kv.prepare_compact_slow_ar_model(
        src, target_seq_len=0, orig_seq_len=2048
    )
    assert same_path == src and full_len == 2048

    stub = tmp_path / "stub.onnx"
    stub.write_bytes(b"not-a-model")
    fb_path, fb_len = slow_ar_kv.prepare_compact_slow_ar_model(
        stub, target_seq_len=512, orig_seq_len=2048
    )
    assert fb_path == stub and fb_len == 2048


def test_global_thread_pool_sizing_and_session_options(monkeypatch: Any) -> None:
    import types

    from a8vc.runtime import providers

    # Pipelined CPU sizing leaves room for 2 active caller threads alongside G-1 workers
    assert providers.compute_global_pool_threads(threads=8, pipeline_decode=True) == (6, 8)
    assert providers.compute_global_pool_threads(threads=4, pipeline_decode=True) == (3, 4)
    assert providers.compute_global_pool_threads(threads=2, pipeline_decode=True) == (2, 2)
    # GPU codec / serial mode uses full pool up to cap
    assert providers.compute_global_pool_threads(threads=0, codec_on_gpu=True, cpu_count=16) == (6, 8)
    assert providers.compute_global_pool_threads(threads=8, codec_on_gpu=True) == (8, 8)
    assert providers.compute_global_pool_threads(threads=6, pipeline_decode=False) == (6, 6)

    pool_calls: list[tuple[int, int]] = []
    created_opts: list[Any] = []

    class _FakeOpts:
        def __init__(self) -> None:
            self.graph_optimization_level = None
            self.intra_op_num_threads = 0
            self.inter_op_num_threads = 0
            self.use_per_session_threads = True
            self.configs: dict[str, str] = {}

        def add_session_config_entry(self, key: str, value: str) -> None:
            self.configs[key] = value

    class _FakeSession:
        def __init__(self, _path: str, sess_options: Any, providers: list[str]) -> None:
            created_opts.append(sess_options)
            self._providers = providers

        def get_providers(self) -> list[str]:
            return self._providers

    fake_ort = types.SimpleNamespace(
        GraphOptimizationLevel=types.SimpleNamespace(ORT_ENABLE_ALL=99),
        ExecutionMode=types.SimpleNamespace(ORT_SEQUENTIAL=0),
        SessionOptions=_FakeOpts,
        InferenceSession=_FakeSession,
        get_available_providers=lambda: ["CPUExecutionProvider"],
        set_global_thread_pool_sizes=lambda intra, inter: pool_calls.append((intra, inter)),
    )

    _, active_ep, warn = providers.create_onnx_session(
        "dummy.onnx",
        provider="cpu",
        threads=6,
        inter_op_threads=1,
        allow_spinning=True,
        use_global_pool=True,
        global_pool_threads=6,
        ort_module=fake_ort,
    )
    assert active_ep == "CPUExecutionProvider"
    assert warn is None
    assert pool_calls == [(6, 1)]
    assert created_opts[0].use_per_session_threads is False
    assert created_opts[0].intra_op_num_threads == 0
    assert created_opts[0].configs["session.intra_op.allow_spinning"] == "1"

    # Secondary sessions (e.g. codec_encoder during voice enrollment) that omit use_global_pool
    # must inherit use_per_session_threads=False once the process-global pool is active.
    monkeypatch.setattr(providers, "_GLOBAL_POOL_THREADS", 6)
    fake_real_ort = types.SimpleNamespace(
        __name__="onnxruntime",
        GraphOptimizationLevel=types.SimpleNamespace(ORT_ENABLE_ALL=99),
        ExecutionMode=types.SimpleNamespace(ORT_SEQUENTIAL=0),
        SessionOptions=_FakeOpts,
        InferenceSession=_FakeSession,
        get_available_providers=lambda: ["CPUExecutionProvider"],
    )
    providers.create_onnx_session("encoder.onnx", provider="cpu", ort_module=fake_real_ort)
    assert created_opts[-1].use_per_session_threads is False
    assert created_opts[-1].intra_op_num_threads == 0

    # Already-created global pool error is handled idempotently
    def _already_created(_intra: int, _inter: int) -> None:
        raise RuntimeError("Global thread pools have already been created, cannot replace them")

    fake_ort_existing = types.SimpleNamespace(set_global_thread_pool_sizes=_already_created)
    assert providers.ensure_global_thread_pool(6, 1, ort_module=fake_ort_existing) == 6


def test_slow_ar_auto_expands_to_full_context_on_long_voice(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import types

    from a8vc.config import RuntimeConfig
    from a8vc.runtime import audio8_onnx

    manifest = {
        "model_id": "test-expand",
        "model_fingerprint": "test-expand-v1",
        "slow_decode_model": "slow.onnx",
        "fast_model": "fast.onnx",
        "codec_models": "dec.onnx",
        "sample_rate": 44100,
        "codec_hop_length": 512,
        "num_codebooks": 10,
        "codebook_size": 32,
        "semantic_begin_id": 1000,
        "semantic_end_id": 1031,
        "im_end_id": 32,
        "max_seq_len": 256,
        "num_layers": 2,
        "n_local_heads": 2,
        "head_dim": 8,
        "num_fast_layers": 1,
        "fast_n_local_heads": 2,
        "fast_head_dim": 8,
        "mamba_d_conv": 2,
        "mamba_d_ssm": 8,
        "mamba_d_state": 4,
        "mamba_d_head": 4,
        "mamba_n_heads": 2,
        "mamba_n_groups": 1,
    }
    (tmp_path / "runtime_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for fname in ("slow.onnx", "slow.kv128.onnx", "fast.onnx", "dec.onnx", "tokenizer.json"):
        (tmp_path / fname).write_bytes(b"stub")

    class _IO:
        def __init__(self, name: str, shape: list[int], typ: str) -> None:
            self.name = name
            self.shape = shape
            self.type = typ

    class _FakeSlow:
        def __init__(self, seq_len: int) -> None:
            self.seq_len = seq_len

        def get_inputs(self) -> list[_IO]:
            return [
                _IO("codes", [1, 11, 1], "tensor(int64)"),
                _IO("position", [1], "tensor(int64)"),
                _IO("cache_keys", [2, 1, 2, self.seq_len, 8], "tensor(float)"),
                _IO("cache_values", [2, 1, 2, self.seq_len, 8], "tensor(float)"),
                _IO("conv_states", [2, 1, 16, 2], "tensor(float)"),
                _IO("ssm_states", [2, 1, 2, 4, 4], "tensor(float)"),
            ]

        def get_outputs(self) -> list[_IO]:
            return [
                _IO("logits", [1, 1, 33], "tensor(float)"),
                _IO("hidden", [1, 1, 16], "tensor(float)"),
                _IO("key_delta", [2, 1, 2, 8], "tensor(float)"),
                _IO("value_delta", [2, 1, 2, 8], "tensor(float)"),
                _IO("next_conv_states", [2, 1, 16, 2], "tensor(float)"),
                _IO("next_ssm_states", [2, 1, 2, 4, 4], "tensor(float)"),
            ]

        def run(self, _outs: Any, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
            assert feed["cache_keys"].shape[3] == self.seq_len
            logits = np.full((1, 1, 33), -20.0, dtype=np.float32)
            logits[0, 0, 32] = 10.0
            return [
                logits,
                np.zeros((1, 1, 16), dtype=np.float32),
                np.zeros((2, 1, 2, 8), dtype=np.float32),
                np.zeros((2, 1, 2, 8), dtype=np.float32),
                feed["conv_states"].copy(),
                feed["ssm_states"].copy(),
            ]

    class _FakeFast:
        def get_inputs(self) -> list[_IO]:
            return [
                _IO("slow_hidden", [1, 1, 16], "tensor(float)"),
                _IO("token_id", [1, 1], "tensor(int64)"),
                _IO("use_slow_hidden", [1], "tensor(bool)"),
                _IO("input_pos", [1], "tensor(int64)"),
                _IO("cache_key_0", [1, 2, 10, 8], "tensor(float)"),
                _IO("cache_value_0", [1, 2, 10, 8], "tensor(float)"),
            ]

        def get_outputs(self) -> list[_IO]:
            return [
                _IO("logits", [1, 1, 32], "tensor(float)"),
                _IO("key_delta_0", [1, 2, 1, 8], "tensor(float)"),
                _IO("value_delta_0", [1, 2, 1, 8], "tensor(float)"),
            ]

    class _FakeDecoder:
        def get_inputs(self) -> list[_IO]:
            return [_IO("codes", [1, 10, 1], "tensor(int64)")]

    class _FakeTokenizer:
        @staticmethod
        def from_file(_path: str) -> Any:
            class _Tok:
                def encode(self, seq: str, add_special_tokens: bool = False) -> Any:
                    return types.SimpleNamespace(ids=[1 + (i % 50) for i in range(max(1, len(seq) // 6))])

            return _Tok()

    loaded_slow_paths: list[str] = []

    def _fake_create_sess(path: Path, **_kwargs: Any) -> tuple[Any, str, None]:
        if path.name == "slow.kv128.onnx":
            loaded_slow_paths.append(path.name)
            return _FakeSlow(128), "CPUExecutionProvider", None
        if path.name == "slow.onnx":
            loaded_slow_paths.append(path.name)
            return _FakeSlow(256), "CPUExecutionProvider", None
        if path.name == "fast.onnx":
            return _FakeFast(), "CPUExecutionProvider", None
        return _FakeDecoder(), "CPUExecutionProvider", None

    monkeypatch.setattr(audio8_onnx, "is_global_thread_pool_enabled", lambda: False)
    monkeypatch.setattr(
        audio8_onnx,
        "prepare_compact_slow_ar_model",
        lambda _p, **_kw: (tmp_path / "slow.kv128.onnx", 128),
    )
    monkeypatch.setattr(audio8_onnx, "create_onnx_session", _fake_create_sess)
    monkeypatch.setitem(
        __import__("sys").modules,
        "tokenizers",
        types.SimpleNamespace(Tokenizer=_FakeTokenizer),
    )

    rt = audio8_onnx.Audio8OnnxRuntime(tmp_path, RuntimeConfig(slow_kv_max_seq_len=128))
    assert rt.active_max_seq_len == 128
    assert loaded_slow_paths == ["slow.kv128.onnx"]

    # Short voice stays in compact 128-slot KV cache
    short_pack = build_voice_pack(
        voice_id="short",
        codes=np.zeros((10, 6), dtype=np.int64),
        reference_text="hi",
        name="Short",
        language="en",
    )
    rt.warm_voice(short_pack)
    assert rt.active_max_seq_len == 128

    # Long voice exceeds 128 - 96 headroom -> auto-expands seamlessly to full 256-slot KV cache
    long_pack = build_voice_pack(
        voice_id="long",
        codes=np.zeros((10, 60), dtype=np.int64),
        reference_text="a moderately long reference voice prompt",
        name="Long",
        language="en",
    )
    rt.warm_voice(long_pack)
    assert rt.active_max_seq_len == 256
    assert rt._workspace["cache_keys"].shape[3] == 256
    assert loaded_slow_paths == ["slow.kv128.onnx", "slow.onnx"]
    rt.close()


def test_speculative_asr_hits_when_idle_and_misses_when_speech_resumes() -> None:
    import threading
    import time

    from a8vc.audio.vad import SpeechSegmenter
    from a8vc.engine.cascaded import CascadedVCEngine, RealtimeVCWorker
    from a8vc.runtime.audio8_onnx import StreamAudioChunk

    class _ScriptedVAD:
        frame_samples = 512

        def __init__(self) -> None:
            self.probs: list[float] = []

        def probability(self, _frame: np.ndarray) -> float:
            return self.probs.pop(0) if self.probs else 0.0

        def reset(self) -> None:
            return None

    class _CountingASR:
        name = "counting-asr"

        def __init__(self) -> None:
            self.calls = 0

        def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
            self.calls += 1
            return f"len={audio.size}"

        def close(self) -> None:
            return None

    class _FakeSynth:
        sample_rate = 44100

        def warm_voice(self, _voice: Any) -> float:
            return 1.0

        def stream_audio(self, _text: str, _voice: Any, **_kwargs: Any) -> Any:
            yield StreamAudioChunk(
                audio=np.full(512, 0.05, dtype=np.float32),
                sample_rate=44100,
                frames_generated=2,
                ttfa_ms=5.0,
                prefill_ms=1.0,
                cache_hit=True,
                is_final=True,
            )

        def cache_stats(self) -> dict[str, Any]:
            return {}

        def close(self) -> None:
            return None

    vad = _ScriptedVAD()
    seg = SpeechSegmenter(
        vad,
        sample_rate=16000,
        speech_start_ms=32,   # 1 frame
        min_silence_ms=192,   # 6 frames
        pre_speech_ms=0,
        min_segment_ms=96,    # 3 frames
        max_segment_ms=5000,
    )

    # First verify segmenter speculative peek matches finalized waveform bit-for-bit
    frame = np.full(512, 0.25, dtype=np.float32)
    vad.probs = [0.9, 0.9, 0.9, 0.9, 0.1, 0.1]
    assert seg.feed(np.tile(frame, 6)) == []
    cand = seg.peek_speculative_segment()
    assert cand is not None and cand.spec_key is not None
    # Deduplicated until audio progresses
    assert seg.peek_speculative_segment() is None

    vad.probs = [0.1, 0.1, 0.1, 0.1]
    finalized = seg.feed(np.tile(frame, 4))
    assert len(finalized) == 1
    assert finalized[0].spec_key == cand.spec_key
    assert np.array_equal(finalized[0].audio, cand.audio)

    # Now test RealtimeVCWorker hit on idle and safe miss when speech resumes
    seg.reset()
    asr = _CountingASR()
    engine = CascadedVCEngine(synthesizer=_FakeSynth(), asr=asr, segmenter=seg)
    pack = build_voice_pack(
        voice_id="spec_voice",
        codes=np.zeros((10, 4), dtype=np.int64),
        reference_text="hi",
        name="Spec",
        language="en",
    )
    engine.set_voice(pack, warm=False)

    events: list[Any] = []
    got_event = threading.Event()

    def _on_evt(evt: Any) -> None:
        events.append(evt)
        got_event.set()

    worker = RealtimeVCWorker(engine, _on_evt)
    worker.start()
    try:
        # Utterance 1: 4 speech + 2 silence -> triggers idle speculative ASR; wait briefly -> 4 silence finalizes
        vad.probs = [0.9, 0.9, 0.9, 0.9, 0.1, 0.1]
        worker.submit_audio(np.tile(frame, 6), 16000)
        deadline = time.monotonic() + 2.0
        while asr.calls < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert asr.calls == 1

        vad.probs = [0.1, 0.1, 0.1, 0.1]
        worker.submit_audio(np.tile(frame, 4), 16000)
        assert got_event.wait(timeout=2.0)
        assert worker.speculative_asr_hits == 1
        assert asr.calls == 1  # Speculative transcript reused with zero duplicate ASR
        assert events[-1].transcript == f"len={6 * 512}"

        # Utterance 2: 4 speech + 2 silence (speculatively transcribed) -> speech resumes (+2 speech) -> 6 silence
        got_event.clear()
        vad.probs = [0.9, 0.9, 0.9, 0.9, 0.1, 0.1]
        worker.submit_audio(np.tile(frame, 6), 16000)
        deadline = time.monotonic() + 2.0
        while asr.calls < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert asr.calls == 2

        # Speech resumes -> spec_key changes -> finalized utterance must re-run ASR on complete audio
        vad.probs = [0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        worker.submit_audio(np.tile(frame, 8), 16000)
        assert got_event.wait(timeout=2.0)
        assert worker.speculative_asr_hits == 1  # Still 1; stale speculation was NOT reused
        assert asr.calls == 3
        assert events[-1].transcript == f"len={10 * 512}"
    finally:
        worker.stop()





