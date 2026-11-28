"""Track 1 real-time Voice Conversion Engine: VAD -> Lightweight ASR -> Audio8-0.1B ONNX INT8."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from a8vc.asr.base import ASRBackend
from a8vc.audio.io import ensure_mono_float32, resample_audio
from a8vc.audio.vad import SpeechSegment, SpeechSegmenter
from a8vc.engine.base import VCOutputEvent, VoiceConversionEngine
from a8vc.runtime.audio8_onnx import SynthesizerBackend
from a8vc.voices.format import VoicePack

logger = logging.getLogger("a8vc.engine.cascaded")


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
        crossfade_ms: int = 6,
    ) -> None:
        self._synth = synthesizer
        self._asr = asr
        self._segmenter = segmenter
        self._edge_fade_samples = int(round((crossfade_ms / 1000.0) * synthesizer.sample_rate))
        self._voice: VoicePack | None = None
        self._voice_lock = threading.Lock()
        self._asr_lock = threading.Lock()
        self._cancel_event = threading.Event()

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

    def transcribe_segment(self, segment: SpeechSegment) -> tuple[str, float]:
        """Thread-safely transcribe a speech segment and return `(text, elapsed_ms)`."""
        t_asr = time.perf_counter()
        with self._asr_lock:
            transcript = self._asr.transcribe(
                segment.audio, sample_rate=segment.sample_rate
            ).strip()
        return transcript, (time.perf_counter() - t_asr) * 1000.0

    def _convert_segment(
        self,
        segment: SpeechSegment,
        *,
        precomputed_asr: tuple[str, float] | None = None,
    ) -> Iterator[VCOutputEvent]:
        voice = self.active_voice
        if voice is None:
            raise RuntimeError("No active voice selected. Call `set_voice(voice_pack)` first.")

        self._cancel_event.clear()
        if precomputed_asr is not None:
            transcript, asr_ms = precomputed_asr
        else:
            transcript, asr_ms = self.transcribe_segment(segment)
        self._last_asr_ms = asr_ms
        if not transcript:
            return

        self._segments_processed += 1
        seg_sec = segment.audio.size / max(1, segment.sample_rate)
        cfg_max_tokens = getattr(getattr(self._synth, "cfg", None), "max_new_tokens", 512)
        vc_max_tokens = min(int(cfg_max_tokens), max(96, int((seg_sec * 2.5 + 2.5) * 22)))
        first_chunk = True
        for chunk in self._synth.stream_audio(
            transcript,
            voice,
            cancel_event=self._cancel_event,
            max_new_tokens=vc_max_tokens,
        ):
            if self._cancel_event.is_set():
                return
            self._last_prefill_ms = chunk.prefill_ms
            self._last_ttfa_ms = chunk.ttfa_ms

            # Intra-utterance chunks are already seamlessly blended by Audio8OnnxRuntime's
            # zero-latency guard-overlap crossfade; only apply zero-delay boundary ramps on
            # utterance start/end to prevent speaker clicks without holding back samples.
            emit_audio = chunk.audio
            if self._edge_fade_samples > 0 and emit_audio.size > 0 and (first_chunk or chunk.is_final):
                emit_audio = emit_audio.copy()
                fade_n = min(self._edge_fade_samples, emit_audio.size)
                t_fade = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
                ramp = 0.5 * (1.0 - np.cos(np.pi * t_fade))
                if first_chunk:
                    emit_audio[:fade_n] *= ramp
                if chunk.is_final:
                    emit_audio[-fade_n:] *= ramp[::-1]

            if emit_audio.size > 0 or chunk.is_final:
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
        max_pending_segments: int = 1,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        self.engine = engine
        self._on_event = on_event
        self._on_error = on_error
        self._max_pending = max(1, int(max_pending_segments))
        self._queue: queue.Queue[SpeechSegment | None] = queue.Queue()
        self._stop_event = threading.Event()
        self._dropped_segments = 0
        self._error_count = 0
        self._last_error = ""
        self._last_error_ts = 0.0
        self._error_lock = threading.Lock()
        self._thread: threading.Thread | None = None

        # Speculative ASR state: runs SenseVoice during VAD trailing silence ONLY while the
        # synthesis worker is idle, hiding ~40-70ms of ASR latency behind VAD endpointing
        # with zero CPU contention during active Audio8 inference.
        self._synth_busy = threading.Event()
        self._spec_lock = threading.Lock()
        self._spec_key: tuple[int, int] | None = None
        self._spec_done = threading.Event()
        self._spec_transcript: str | None = None
        self._spec_executor: ThreadPoolExecutor | None = None
        self._spec_hits = 0

    def _report_error(self, exc: Exception, *, context: str) -> None:
        msg = f"{type(exc).__name__}: {exc}"
        now = time.monotonic()
        should_notify = False
        with self._error_lock:
            self._error_count += 1
            if msg != self._last_error or (now - self._last_error_ts) >= 3.0:
                self._last_error = msg
                self._last_error_ts = now
                should_notify = True
        if should_notify:
            logger.exception("RealtimeVCWorker error during %s: %s", context, msg)
            if self._on_error is not None:
                try:
                    self._on_error(exc)
                except Exception:
                    logger.debug("on_error callback raised", exc_info=True)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        if self._spec_executor is None:
            self._spec_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="a8vc-spec-asr",
            )
        self._thread = threading.Thread(
            target=self._run_loop,
            name="a8vc-realtime-worker",
            daemon=True,
        )
        self._thread.start()

    def _run_speculative_asr(self, candidate: SpeechSegment, done_evt: threading.Event) -> None:
        try:
            if self._stop_event.is_set() or self._synth_busy.is_set():
                return
            transcript, _ = self.engine.transcribe_segment(candidate)
            with self._spec_lock:
                if self._spec_key == candidate.spec_key:
                    self._spec_transcript = transcript
        except Exception:
            logger.debug("Speculative ASR skipped due to exception", exc_info=True)
        finally:
            done_evt.set()

    def _maybe_schedule_speculative_asr(self) -> None:
        if (
            self._stop_event.is_set()
            or self._synth_busy.is_set()
            or self._spec_executor is None
            or self.engine.active_voice is None
        ):
            return
        peek_fn = getattr(self.engine._segmenter, "peek_speculative_segment", None)
        if not callable(peek_fn):
            return
        candidate = peek_fn()
        if candidate is None or candidate.spec_key is None:
            return
        with self._spec_lock:
            if self._spec_key == candidate.spec_key:
                return
            done_evt = threading.Event()
            self._spec_key = candidate.spec_key
            self._spec_transcript = None
            self._spec_done = done_evt
        try:
            self._spec_executor.submit(self._run_speculative_asr, candidate, done_evt)
        except RuntimeError:
            done_evt.set()

    def _consume_speculative_asr(self, seg: SpeechSegment) -> tuple[str, float] | None:
        if seg.spec_key is None:
            return None
        with self._spec_lock:
            if self._spec_key != seg.spec_key:
                return None
            done_evt = self._spec_done
        t_wait = time.perf_counter()
        if not done_evt.wait(timeout=2.0):
            return None
        wait_ms = (time.perf_counter() - t_wait) * 1000.0
        with self._spec_lock:
            if self._spec_key != seg.spec_key or self._spec_transcript is None:
                return None
            transcript = self._spec_transcript
            self._spec_key = None
            self._spec_transcript = None
        self._spec_hits += 1
        return transcript, wait_ms

    def submit_audio(self, audio: np.ndarray, sample_rate: int) -> None:
        try:
            mono = ensure_mono_float32(audio)
            if mono.size == 0:
                return
            if sample_rate != self.engine._segmenter.sample_rate:
                mono = resample_audio(mono, sample_rate, self.engine._segmenter.sample_rate)
            segments = self.engine._segmenter.feed(mono)
            for seg in segments:
                self._enqueue_segment(seg)
            if not segments:
                self._maybe_schedule_speculative_asr()
        except Exception as exc:
            self._report_error(exc, context="submit_audio")

    def flush(self) -> None:
        try:
            seg = self.engine._segmenter.flush()
            if seg is not None:
                self._enqueue_segment(seg)
        except Exception as exc:
            self._report_error(exc, context="flush")

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
                precomputed = self._consume_speculative_asr(seg)
                self._synth_busy.set()
                for event in self.engine._convert_segment(seg, precomputed_asr=precomputed):
                    if self._stop_event.is_set():
                        break
                    self._on_event(event)
            except Exception as exc:
                self._report_error(exc, context="convert_segment")
                continue
            finally:
                self._synth_busy.clear()

    @property
    def speculative_asr_hits(self) -> int:
        return self._spec_hits

    @property
    def dropped_segments(self) -> int:
        return self._dropped_segments

    @property
    def error_count(self) -> int:
        with self._error_lock:
            return self._error_count

    @property
    def last_error(self) -> str:
        with self._error_lock:
            return self._last_error

    def stop(self) -> None:
        self._stop_event.set()
        self.engine.cancel()
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._spec_executor is not None:
            self._spec_executor.shutdown(wait=False, cancel_futures=True)
            self._spec_executor = None
