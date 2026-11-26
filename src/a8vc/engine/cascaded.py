"""Track 1 real-time Voice Conversion Engine: VAD -> Lightweight ASR -> Audio8-0.1B ONNX INT8."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

from a8vc.asr.base import ASRBackend
from a8vc.audio.io import crossfade_append, ensure_mono_float32, resample_audio
from a8vc.audio.vad import SpeechSegment, SpeechSegmenter
from a8vc.engine.base import VCOutputEvent, VoiceConversionEngine
from a8vc.runtime.audio8_onnx import SynthesizerBackend
from a8vc.voices.format import VoicePack


def get_process_rss_mb() -> float:
    """Read current process resident memory (RSS) in MiB across Windows, Linux, and macOS."""
    import os
    import sys

    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class _ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            get_mem_info = getattr(ctypes.windll.kernel32, "K32GetProcessMemoryInfo", None)
            if get_mem_info is None:
                get_mem_info = ctypes.windll.psapi.GetProcessMemoryInfo
            if get_mem_info(handle, ctypes.byref(counters), counters.cb):
                return round(counters.WorkingSetSize / (1024 * 1024), 2)
        except Exception:
            return 0.0

    try:
        with open("/proc/self/statm", encoding="utf-8") as fh:
            fields = fh.read().split()
            rss_pages = int(fields[1])
            page_size = os.sysconf("SC_PAGE_SIZE")
            return round((rss_pages * page_size) / (1024 * 1024), 2)
    except Exception:
        pass

    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS returns bytes; Linux/BSD returns KiB
        divisor = (1024 * 1024) if sys.platform == "darwin" else 1024
        return round(rss / divisor, 2)
    except Exception:
        return 0.0


class CascadedVCEngine(VoiceConversionEngine):
    """Zero-training real-time VC engine combining VAD, lightweight ASR, and Audio8-0.1B ONNX INT8."""

    def __init__(
        self,
        *,
        synthesizer: SynthesizerBackend,
        asr: ASRBackend,
        segmenter: SpeechSegmenter,
        crossfade_ms: int = 12,
    ) -> None:
        self._synth = synthesizer
        self._asr = asr
        self._segmenter = segmenter
        self._crossfade_samples = int(round((crossfade_ms / 1000.0) * synthesizer.sample_rate))
        self._voice: VoicePack | None = None
        self._voice_lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._crossfade_tail = np.zeros(0, dtype=np.float32)

        self._segments_processed = 0
        self._last_asr_ms = 0.0
        self._last_ttfa_ms = 0.0
        self._last_prefill_ms = 0.0

    @property
    def mode(self) -> str:
        return "cascaded"

    @property
    def output_sample_rate(self) -> int:
        return self._synth.sample_rate

    @property
    def active_voice(self) -> VoicePack | None:
        with self._voice_lock:
            return self._voice

    def set_voice(self, voice: VoicePack, *, warm: bool = True) -> float:
        with self._voice_lock:
            self._voice = voice
        if warm:
            return self._synth.warm_voice(voice)
        return 0.0

    def cancel(self) -> None:
        self._cancel_event.set()

    def reset(self) -> None:
        self._cancel_event.clear()
        self._segmenter.reset()
        self._crossfade_tail = np.zeros(0, dtype=np.float32)

    def _convert_segment(self, segment: SpeechSegment) -> Iterator[VCOutputEvent]:
        voice = self.active_voice
        if voice is None:
            raise RuntimeError("No active voice selected. Call `set_voice(voice_pack)` first.")

        self._cancel_event.clear()
        t_asr = time.perf_counter()
        transcript = self._asr.transcribe(segment.audio, sample_rate=segment.sample_rate).strip()
        asr_ms = (time.perf_counter() - t_asr) * 1000.0
        self._last_asr_ms = asr_ms
        if not transcript:
            return

        self._segments_processed += 1
        first_chunk = True
        for chunk in self._synth.stream_audio(
            transcript,
            voice,
            cancel_event=self._cancel_event,
        ):
            if self._cancel_event.is_set():
                return
            self._last_prefill_ms = chunk.prefill_ms
            self._last_ttfa_ms = chunk.ttfa_ms

            emit_audio = chunk.audio
            if self._crossfade_samples > 0 and emit_audio.size > 0:
                emit_audio, self._crossfade_tail = crossfade_append(
                    self._crossfade_tail,
                    emit_audio,
                    self._crossfade_samples,
                )

            if emit_audio.size > 0 or chunk.is_final:
                if chunk.is_final and self._crossfade_tail.size > 0:
                    emit_audio = (
                        np.concatenate([emit_audio, self._crossfade_tail])
                        if emit_audio.size > 0
                        else self._crossfade_tail.copy()
                    )
                    self._crossfade_tail = np.zeros(0, dtype=np.float32)
                yield VCOutputEvent(
                    audio=emit_audio,
                    sample_rate=chunk.sample_rate,
                    voice_id=voice.voice_id,
                    transcript=transcript if first_chunk else None,
                    asr_ms=round(asr_ms, 2),
                    prefill_ms=round(chunk.prefill_ms, 2),
                    ttfa_ms=round(chunk.ttfa_ms, 2),
                    cache_hit=chunk.cache_hit,
                    is_final=chunk.is_final,
                )
                first_chunk = False

    def process_audio(self, audio: np.ndarray, sample_rate: int) -> Iterator[VCOutputEvent]:
        mono = ensure_mono_float32(audio)
        if mono.size == 0:
            return
        if sample_rate != self._segmenter.sample_rate:
            mono = resample_audio(mono, sample_rate, self._segmenter.sample_rate)
        segments = self._segmenter.feed(mono)
        for seg in segments:
            yield from self._convert_segment(seg)

    def flush(self) -> Iterator[VCOutputEvent]:
        seg = self._segmenter.flush()
        if seg is not None:
            yield from self._convert_segment(seg)

    def stats(self) -> dict[str, Any]:
        voice = self.active_voice
        return {
            "mode": self.mode,
            "asr_backend": self._asr.name,
            "active_voice": voice.voice_id if voice else None,
            "output_sample_rate": self.output_sample_rate,
            "segments_processed": self._segments_processed,
            "last_asr_ms": round(self._last_asr_ms, 2),
            "last_prefill_ms": round(self._last_prefill_ms, 2),
            "last_ttfa_ms": round(self._last_ttfa_ms, 2),
            "prefix_cache": self._synth.cache_stats(),
            "rss_mb": get_process_rss_mb(),
        }

    def close(self) -> None:
        self.cancel()
        self._asr.close()
        self._synth.close()


class RealtimeVCWorker:
    """Non-blocking background worker bridging live audio capture to `CascadedVCEngine`.

    Runs VAD on the lightweight producer side and pushes completed segments into a bounded
    queue so live audio callbacks never stall and stale backlogs drop cleanly.
    """

    def __init__(
        self,
        engine: CascadedVCEngine,
        on_event: Callable[[VCOutputEvent], None],
        *,
        max_pending_segments: int = 2,
    ) -> None:
        self.engine = engine
        self._on_event = on_event
        self._max_pending = max(1, int(max_pending_segments))
        self._queue: queue.Queue[SpeechSegment | None] = queue.Queue()
        self._stop_event = threading.Event()
        self._dropped_segments = 0
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="a8vc-realtime-worker",
            daemon=True,
        )
        self._thread.start()

    def submit_audio(self, audio: np.ndarray, sample_rate: int) -> None:
        mono = ensure_mono_float32(audio)
        if mono.size == 0:
            return
        if sample_rate != self.engine._segmenter.sample_rate:
            mono = resample_audio(mono, sample_rate, self.engine._segmenter.sample_rate)
        for seg in self.engine._segmenter.feed(mono):
            self._enqueue_segment(seg)

    def flush(self) -> None:
        seg = self.engine._segmenter.flush()
        if seg is not None:
            self._enqueue_segment(seg)

    def _enqueue_segment(self, seg: SpeechSegment) -> None:
        while self._queue.qsize() >= self._max_pending:
            try:
                old = self._queue.get_nowait()
                if old is not None:
                    self._dropped_segments += 1
            except queue.Empty:
                break
        self._queue.put(seg)

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                seg = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if seg is None:
                break
            try:
                for event in self.engine._convert_segment(seg):
                    if self._stop_event.is_set():
                        break
                    self._on_event(event)
            except Exception:
                # Keep worker resilient during transient live audio/runtime errors
                continue

    @property
    def dropped_segments(self) -> int:
        return self._dropped_segments

    def stop(self) -> None:
        self._stop_event.set()
        self.engine.cancel()
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
