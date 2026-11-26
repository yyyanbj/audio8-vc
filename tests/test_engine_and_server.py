"""End-to-end tests for CascadedVCEngine, CodecS2SEngine scaffold, FastAPI HTTP & WS /ws/vc."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from a8vc.asr.sherpa import MockASR
from a8vc.audio.io import float32_to_pcm16
from a8vc.audio.vad import EnergyVAD, SpeechSegmenter
from a8vc.config import AppConfig
from a8vc.engine.cascaded import CascadedVCEngine
from a8vc.engine.codec_s2s import CodecS2SEngine
from a8vc.runtime.audio8_onnx import StreamAudioChunk
from a8vc.server.app import ServerState, create_app
from a8vc.voices.format import VoicePack, build_voice_pack
from a8vc.voices.store import VoiceLibrary


class _FakeSynthesizer:
    """Fast deterministic synthesizer stub implementing `SynthesizerBackend`."""

    def __init__(self, sample_rate: int = 44100) -> None:
        self._sr = sample_rate
        self._warmed: set[tuple[str, str, str]] = set()
        self.hits = 0
        self.misses = 0

    @property
    def sample_rate(self) -> int:
        return self._sr

    def warm_voice(self, voice: VoicePack) -> float:
        self._warmed.add(voice.cache_key)
        return 1.5

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
        hit = voice.cache_key in self._warmed
        if hit:
            self.hits += 1
        else:
            self.misses += 1
            self._warmed.add(voice.cache_key)

        for idx in range(2):
            if cancel_event is not None and cancel_event.is_set():
                return
            t = np.linspace(0.0, 0.05, int(self._sr * 0.05), endpoint=False, dtype=np.float32)
            freq = 300.0 if voice.voice_id == "alice" else 520.0
            audio = (0.2 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
            yield StreamAudioChunk(
                audio=audio,
                sample_rate=self._sr,
                frames_generated=(idx + 1) * 4,
                ttfa_ms=12.0,
                prefill_ms=2.0 if hit else 40.0,
                cache_hit=hit,
                is_final=(idx == 1),
            )

    def decode_codes(self, codes: np.ndarray) -> np.ndarray:
        return np.zeros(codes.shape[-1] * 2048, dtype=np.float32)

    def cache_stats(self) -> dict[str, Any]:
        return {
            "entries": len(self._warmed),
            "max_entries": 2,
            "hits": self.hits,
            "misses": self.misses,
            "memory_mb": 0.1,
        }

    def close(self) -> None:
        return None


def _seed_library(tmp_path: Path) -> VoiceLibrary:
    lib = VoiceLibrary(tmp_path / "voices")
    rng = np.random.default_rng(7)
    for vid, name in (("alice", "Alice"), ("bob", "Bob")):
        pack = build_voice_pack(
            voice_id=vid,
            codes=rng.integers(0, 4096, size=(10, 24), dtype=np.int64),
            reference_text=f"Hello, my name is {name}.",
            name=name,
            language="en",
        )
        lib.save(pack)
    return lib


def test_cascaded_vc_engine_streaming_and_voice_switch(tmp_path: Path) -> None:
    lib = _seed_library(tmp_path)
    synth = _FakeSynthesizer()
    asr = MockASR("你好变声器")
    seg = SpeechSegmenter(
        EnergyVAD(threshold_db=-35.0),
        sample_rate=16000,
        speech_start_ms=64,
        min_silence_ms=200,
        min_segment_ms=160,
    )
    engine = CascadedVCEngine(synthesizer=synth, asr=asr, segmenter=seg, crossfade_ms=5)
    engine.set_voice(lib.load("alice"), warm=True)

    sr = 16000
    t = np.linspace(0.0, 0.4, int(sr * 0.4), endpoint=False, dtype=np.float32)
    speech = (0.3 * np.sin(2.0 * np.pi * 320.0 * t)).astype(np.float32)
    silence = np.zeros(int(sr * 0.35), dtype=np.float32)

    events = list(engine.process_audio(np.concatenate([speech, silence]), sample_rate=sr))
    assert len(events) >= 1
    assert events[0].voice_id == "alice"
    assert events[0].transcript == "你好变声器"
    assert events[0].cache_hit is True
    assert events[-1].is_final is True

    # Hot-switch voice to bob
    engine.set_voice(lib.load("bob"), warm=True)
    events2 = list(engine.process_audio(np.concatenate([speech, silence]), sample_rate=sr))
    assert events2[0].voice_id == "bob"
    assert engine.stats()["segments_processed"] == 2
    engine.close()


def test_codec_s2s_scaffold_contract(tmp_path: Path) -> None:
    engine = CodecS2SEngine(tmp_path / "s2s_ckpt")
    assert engine.mode == "codec_s2s"
    assert engine.output_sample_rate == 44100
    with pytest.raises(NotImplementedError, match="Track 2"):
        list(engine.process_audio(np.zeros(1600, dtype=np.float32), 16000))


def test_http_and_websocket_server(tmp_path: Path) -> None:
    lib = _seed_library(tmp_path)
    cfg = AppConfig()
    cfg.paths.voices_dir = lib.root
    cfg.vad.energy_threshold_db = -35.0
    cfg.vad.speech_start_ms = 64
    cfg.vad.min_silence_ms = 200
    cfg.vad.min_segment_ms = 160

    state = ServerState(
        cfg,
        library=lib,
        synthesizer=_FakeSynthesizer(),
        asr=MockASR("实时变声测试"),
    )
    app = create_app(cfg, state=state)
    client = TestClient(app)

    # Health & Web UI
    assert client.get("/").status_code == 200
    h = client.get("/api/health").json()
    assert h["status"] == "ok"
    assert h["voice_count"] == 2

    # List & inspect & download voice
    voices = client.get("/api/voices").json()["voices"]
    assert {v["voice_id"] for v in voices} == {"alice", "bob"}
    assert client.get("/api/voices/alice").json()["name"] == "Alice"
    dl = client.get("/api/voices/alice/download")
    assert dl.status_code == 200
    assert len(dl.content) > 100

    # Set active voice & stream TTS preview
    act = client.post("/api/voices/active", json={"voice_id": "alice", "warm": True})
    assert act.status_code == 200
    assert act.json()["active_voice_id"] == "alice"

    tts = client.post("/api/tts/stream", json={"voice_id": "alice", "text": "hello"})
    assert tts.status_code == 200
    assert tts.headers["X-Sample-Rate"] == "44100"
    assert len(tts.content) > 0

    # Bidirectional WebSocket voice conversion test
    with client.websocket_connect("/ws/vc") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        assert ready["voice_id"] == "alice"

        ws.send_json({"type": "set_voice", "voice_id": "bob"})
        changed = ws.receive_json()
        assert changed["type"] == "voice_changed"
        assert changed["voice_id"] == "bob"

        # Send 0.4s speech + 0.35s silence as PCM16 16kHz
        sr = 16000
        t = np.linspace(0.0, 0.4, int(sr * 0.4), endpoint=False, dtype=np.float32)
        speech = (0.3 * np.sin(2.0 * np.pi * 300.0 * t)).astype(np.float32)
        silence = np.zeros(int(sr * 0.35), dtype=np.float32)
        ws.send_bytes(float32_to_pcm16(np.concatenate([speech, silence])))

        got_transcript = False
        got_audio = 0
        got_metrics = False
        for _ in range(10):
            msg = ws.receive()
            if "bytes" in msg and msg["bytes"] is not None:
                got_audio += len(msg["bytes"])
            elif "text" in msg and msg["text"] is not None:
                import json

                payload = json.loads(msg["text"])
                if payload["type"] == "transcript":
                    got_transcript = True
                    assert payload["voice_id"] == "bob"
                elif payload["type"] == "metrics":
                    got_metrics = True
                    break

        assert got_transcript is True
        assert got_audio > 0
        assert got_metrics is True
