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

    slow_ep, slow_note = select_model_provider(
        "dml", model_role="slow_ar", precision="int8", available=dml_avail
    )
    fast_ep, fast_note = select_model_provider(
        "dml", model_role="fast_ar", precision="int8", available=dml_avail
    )
    dec_ep, dec_note = select_model_provider(
        "dml", model_role="codec_decoder", precision="fp16", available=dml_avail
    )
    cuda_ep, cuda_note = select_model_provider(
        "cuda",
        model_role="slow_ar",
        precision="int8",
        available=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    assert slow_ep == "cpu" and slow_note is not None
    assert fast_ep == "cpu" and fast_note is not None
    assert dec_ep == "dml" and dec_note is None
    assert cuda_ep == "cuda" and cuda_note is None


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
