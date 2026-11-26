"""Controller for local audio-device real-time voice conversion (mic -> speaker / virtual cable)."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from a8vc.audio.device import LocalAudioBridge
from a8vc.config import AppConfig
from a8vc.engine.base import VCOutputEvent
from a8vc.engine.cascaded import CascadedVCEngine, RealtimeVCWorker
from a8vc.voices.format import VoicePack


def _normalize_device_id(value: str | int | None) -> str | int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    cleaned = value.strip()
    if not cleaned:
        return None
    try:
        return int(cleaned)
    except ValueError:
        return cleaned


class LocalLiveSessionManager:
    """Manages an active local-device real-time VC session for desktop and API control."""

    def __init__(
        self,
        cfg: AppConfig,
        *,
        create_engine: Callable[[str | None], CascadedVCEngine],
        load_voice: Callable[[str], VoicePack],
        on_voice_active: Callable[[str], None] | None = None,
        bridge_factory: Callable[..., Any] = LocalAudioBridge,
    ) -> None:
        self._cfg = cfg
        self._create_engine = create_engine
        self._load_voice = load_voice
        self._on_voice_active = on_voice_active
        self._bridge_factory = bridge_factory

        self._lock = threading.RLock()
        self._running = False
        self._engine: CascadedVCEngine | None = None
        self._worker: RealtimeVCWorker | None = None
        self._bridge: Any = None

        self._voice_id: str | None = None
        self._input_device: str | int | None = None
        self._output_device: str | int | None = None
        self._started_at: float | None = None
        self._last_transcript: str = ""
        self._last_metrics: dict[str, Any] | None = None
        self._last_error: str = ""
        self._recent_events: deque[dict[str, Any]] = deque(maxlen=25)

    def _record_event(self, kind: str, payload: dict[str, Any]) -> None:
        entry = {"ts": round(time.time(), 3), "type": kind, **payload}
        with self._lock:
            self._recent_events.append(entry)

    def start(
        self,
        *,
        voice_id: str,
        input_device: str | int | None = None,
        output_device: str | int | None = None,
    ) -> dict[str, Any]:
        """Start local microphone -> voice conversion -> speaker/virtual-cable playback."""
        in_dev = _normalize_device_id(input_device)
        out_dev = _normalize_device_id(output_device)
        pack = self._load_voice(voice_id)

        with self._lock:
            if self._running:
                self._stop_locked()

            engine: CascadedVCEngine | None = None
            worker: RealtimeVCWorker | None = None
            bridge: Any = None
            try:
                engine = self._create_engine(pack.voice_id)
                warm_ms = engine.set_voice(pack, warm=True)

                def _on_event(event: VCOutputEvent) -> None:
                    with self._lock:
                        active_bridge = self._bridge
                    if event.transcript:
                        with self._lock:
                            self._last_transcript = event.transcript
                        self._record_event(
                            "transcript",
                            {
                                "text": event.transcript,
                                "voice_id": event.voice_id,
                                "asr_ms": event.asr_ms,
                            },
                        )
                    if event.audio.size > 0 and active_bridge is not None:
                        active_bridge.playback.push(event.audio, source_sr=event.sample_rate)
                    if event.is_final:
                        metrics = {
                            "voice_id": event.voice_id,
                            "asr_ms": event.asr_ms,
                            "prefill_ms": event.prefill_ms,
                            "ttfa_ms": event.ttfa_ms,
                            "cache_hit": event.cache_hit,
                            "dropped_segments": worker.dropped_segments if worker else 0,
                        }
                        with self._lock:
                            self._last_metrics = metrics
                        self._record_event("metrics", metrics)

                worker = RealtimeVCWorker(
                    engine,
                    _on_event,
                    max_pending_segments=self._cfg.engine.max_pending_segments,
                )
                bridge = self._bridge_factory(
                    on_input_chunk=worker.submit_audio,
                    input_sample_rate=self._cfg.audio.input_sample_rate,
                    output_sample_rate=self._cfg.audio.output_sample_rate,
                    input_block_ms=self._cfg.audio.input_block_ms,
                    input_device=in_dev,
                    output_device=out_dev,
                )
                self._bridge = bridge
                worker.start()
                bridge.start()

                self._engine = engine
                self._worker = worker
                self._running = True
                self._voice_id = pack.voice_id
                self._input_device = in_dev
                self._output_device = out_dev
                self._started_at = time.time()
                self._last_error = ""
                self._record_event(
                    "started",
                    {
                        "voice_id": pack.voice_id,
                        "warm_ms": round(warm_ms, 2),
                        "input_sr": getattr(
                            bridge, "active_input_sample_rate", self._cfg.audio.input_sample_rate
                        ),
                        "output_sr": getattr(
                            bridge, "active_output_sample_rate", self._cfg.audio.output_sample_rate
                        ),
                    },
                )
                if self._on_voice_active is not None:
                    self._on_voice_active(pack.voice_id)
                return self.status()
            except Exception as exc:
                self._last_error = str(exc)
                if bridge is not None:
                    try:
                        bridge.stop()
                    except Exception:
                        pass
                if worker is not None:
                    try:
                        worker.stop()
                    except Exception:
                        pass
                self._bridge = None
                self._worker = None
                self._engine = None
                self._running = False
                raise

    def switch_voice(self, voice_id: str) -> dict[str, Any]:
        """Hot-switch target voice while keeping local audio devices running."""
        pack = self._load_voice(voice_id)
        with self._lock:
            if not self._running or self._engine is None:
                raise RuntimeError("Local live voice conversion is not running.")
            self._engine.cancel()
            if self._bridge is not None:
                self._bridge.playback.clear()
            warm_ms = self._engine.set_voice(pack, warm=True)
            self._voice_id = pack.voice_id
            self._record_event(
                "voice_changed",
                {"voice_id": pack.voice_id, "warm_ms": round(warm_ms, 2)},
            )
            if self._on_voice_active is not None:
                self._on_voice_active(pack.voice_id)
            return {
                "voice_id": pack.voice_id,
                "warm_ms": round(warm_ms, 2),
                "running": True,
            }

    def flush(self) -> bool:
        """Force-flush the current buffered speech segment immediately."""
        with self._lock:
            if not self._running or self._worker is None:
                return False
            self._worker.flush()
            return True

    def _stop_locked(self) -> None:
        bridge = self._bridge
        worker = self._worker
        engine = self._engine
        self._bridge = None
        self._worker = None
        self._engine = None
        self._running = False
        self._started_at = None

        if engine is not None:
            try:
                engine.cancel()
            except Exception:
                pass
        if bridge is not None:
            try:
                bridge.stop()
            except Exception:
                pass
        if worker is not None:
            try:
                worker.stop()
            except Exception:
                pass
        self._record_event("stopped", {})

    def stop(self) -> dict[str, Any]:
        """Stop active local device voice conversion and release audio streams."""
        with self._lock:
            if self._running:
                self._stop_locked()
            return self.status()

    def status(self) -> dict[str, Any]:
        """Return current session state, device rates, and recent metrics."""
        with self._lock:
            buffered_ms = 0.0
            active_in_sr = self._cfg.audio.input_sample_rate
            active_out_sr = self._cfg.audio.output_sample_rate
            if self._bridge is not None:
                buffered_ms = round(float(self._bridge.playback.buffered_ms), 1)
                active_in_sr = int(
                    getattr(self._bridge, "active_input_sample_rate", active_in_sr)
                )
                active_out_sr = int(
                    getattr(self._bridge, "active_output_sample_rate", active_out_sr)
                )
            return {
                "running": self._running,
                "voice_id": self._voice_id,
                "input_device": self._input_device,
                "output_device": self._output_device,
                "active_input_sample_rate": active_in_sr,
                "active_output_sample_rate": active_out_sr,
                "buffered_ms": buffered_ms,
                "uptime_sec": round(time.time() - self._started_at, 1)
                if self._running and self._started_at
                else 0.0,
                "last_transcript": self._last_transcript,
                "last_metrics": self._last_metrics,
                "last_error": self._last_error,
                "events": list(self._recent_events)[-10:],
            }
