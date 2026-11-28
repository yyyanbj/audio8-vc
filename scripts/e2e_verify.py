#!/usr/bin/env python3
"""End-to-end verification harness for Audio8-VC (CLI, Server HTTP/WS, Enrollment, & Headless Browser).

Exercises the real `a8vc` binary (frozen Windows `a8vc.exe` or source CLI) with real
Audio8-0.1B ONNX INT8, SenseVoice ASR, and Silero VAD models across:
  1. CLI verification (`--version`, `info --json`, `self-check --json`, `voice builtins --json`)
  2. Live server HTTP/API verification (`/`, icons, `/api/health`, auto-seeded voices,
     `/api/tts/stream`, error JSON formatting, `/api/voices/enroll`)
  3. Direct WebSocket `/ws/vc` real-speech conversion (48 kHz -> Silero VAD -> SenseVoice ASR -> Audio8)
  4. Real headless Chromium/Edge browser E2E via Chrome DevTools Protocol (CDP) with virtual mic:
     `--use-fake-ui-for-media-stream --use-fake-device-for-media-stream --use-file-for-fake-audio-capture`
  5. Standalone CLI synthesis & `.a8voice` voice packing/verification (`a8vc synth`, `voice pack`, `verify`)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from websockets.sync.client import connect as ws_connect

# Avoid UnicodeEncodeError on legacy Windows console/pipe codepages (e.g. cp1252)
# when printing CJK ASR transcripts or error diagnostics.
for _stdio_stream in (sys.stdout, sys.stderr):
    if _stdio_stream is not None and hasattr(_stdio_stream, "reconfigure"):
        try:
            _stdio_stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            try:
                _stdio_stream.reconfigure(errors="replace")
            except Exception:
                pass

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _run_cmd(
    cmd: list[str],
    *,
    env: dict[str, str],
    cwd: Path | None = None,
    timeout: float = 180.0,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        env=env,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed (rc={proc.returncode}): {' '.join(cmd)}\n"
            f"--- STDOUT ---\n{proc.stdout[-2000:]}\n"
            f"--- STDERR ---\n{proc.stderr[-2000:]}"
        )
    return proc


def _pcm16_to_float(raw: bytes) -> np.ndarray:
    even = len(raw) - (len(raw) % 2)
    if even <= 0:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(raw[:even], dtype=np.int16).astype(np.float32) / 32768.0


def _float_to_pcm16(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def _resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    dst_len = max(1, int(round(audio.size * (dst_sr / float(src_sr)))))
    x_old = np.linspace(0.0, 1.0, num=audio.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=dst_len, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def _write_pcm16_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write standard 44-byte RIFF PCM16 mono WAV (compatible with Chromium fake audio capture)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = _float_to_pcm16(audio)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


def _prepare_fake_mic_wavs(
    speech_44k: np.ndarray,
    speech_sr: int,
    work_dir: Path,
) -> tuple[Path, Path, np.ndarray]:
    """Build 44.1kHz enrollment WAV + 16kHz Chromium fake-mic WAV + 48kHz WS test signal."""
    # Trim extreme silence tails slightly while keeping clean speech, then normalize.
    # Cap clip length to <=4.0s so enrollment, /ws/vc, and fake-mic stay deterministic on CPU.
    max_samples = int(4.0 * speech_sr)
    if speech_44k.size > max_samples:
        speech_44k = speech_44k[:max_samples]
    peak = float(np.max(np.abs(speech_44k))) if speech_44k.size else 0.0
    if peak < 1e-4:
        raise RuntimeError("Synthesized speech audio is silent; cannot prepare E2E speech WAV")
    norm_44k = (speech_44k / peak * 0.78).astype(np.float32)

    enroll_wav = work_dir / "speech_44k.wav"
    _write_pcm16_wav(enroll_wav, norm_44k, speech_sr)

    # Fake mic WAV: 0.6s silence + speech + 1.0s silence at 16 kHz mono PCM16
    # Gives Silero VAD clean speech start + end boundaries every loop cycle.
    speech_16k = _resample_linear(norm_44k, speech_sr, 16000)
    lead_16k = np.zeros(int(0.6 * 16000), dtype=np.float32)
    tail_16k = np.zeros(int(1.0 * 16000), dtype=np.float32)
    fake_mic_16k = np.concatenate([lead_16k, speech_16k, tail_16k])
    fake_mic_wav = work_dir / "fake_mic_16k.wav"
    _write_pcm16_wav(fake_mic_wav, fake_mic_16k, 16000)

    # 48 kHz test signal for direct WS test (matching typical browser AudioContext sampleRate)
    speech_48k = _resample_linear(norm_44k, speech_sr, 48000)
    lead_48k = np.zeros(int(0.4 * 48000), dtype=np.float32)
    tail_48k = np.zeros(int(0.8 * 48000), dtype=np.float32)
    ws_signal_48k = np.concatenate([lead_48k, speech_48k, tail_48k])
    return enroll_wav, fake_mic_wav, ws_signal_48k


def _discover_browser(explicit: str | None = None) -> Path | None:
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
        found = shutil.which(explicit)
        return Path(found) if found else None

    candidates: list[str] = []
    if sys.platform == "win32":
        program_files = [
            os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
            os.environ.get("PROGRAMFILES", r"C:\Program Files"),
            os.environ.get("LOCALAPPDATA", ""),
        ]
        for base in program_files:
            if not base:
                continue
            candidates.extend(
                [
                    str(Path(base) / "Microsoft" / "Edge" / "Application" / "msedge.exe"),
                    str(Path(base) / "Google" / "Chrome" / "Application" / "chrome.exe"),
                ]
            )
        for name in ("msedge", "chrome", "chromium"):
            hit = shutil.which(name)
            if hit:
                candidates.append(hit)
    else:
        for name in (
            "google-chrome",
            "google-chrome-stable",
            "microsoft-edge",
            "chromium",
            "chromium-browser",
        ):
            hit = shutil.which(name)
            if hit:
                candidates.append(hit)

    for cand in candidates:
        if cand and Path(cand).is_file():
            return Path(cand)
    return None


# ---------------------------------------------------------------------------
# Minimal Chrome DevTools Protocol (CDP) Client (stdlib + websockets only)
# ---------------------------------------------------------------------------


_E2E_INIT_SCRIPT = r"""
(() => {
  window.__e2e = {
    wsOpened: 0,
    wsClosed: 0,
    wsBinarySentBytes: 0,
    wsSentMaxAbs: 0,
    wsBinaryRecvBytes: 0,
    wsMessages: [],
    playedBuffers: 0,
    playedSamples: 0,
    consoleErrors: [],
    pageErrors: [],
  };

  const OrigWS = window.WebSocket;
  window.WebSocket = function(url, protocols) {
    const sock = protocols !== undefined ? new OrigWS(url, protocols) : new OrigWS(url);
    const origSend = sock.send.bind(sock);
    sock.send = function(data) {
      let buf = null;
      if (data instanceof ArrayBuffer) {
        window.__e2e.wsBinarySentBytes += data.byteLength;
        buf = new Int16Array(data, 0, Math.floor(data.byteLength / 2));
      } else if (ArrayBuffer.isView(data)) {
        window.__e2e.wsBinarySentBytes += data.byteLength;
        buf = new Int16Array(data.buffer, data.byteOffset, Math.floor(data.byteLength / 2));
      }
      if (buf) {
        let peak = window.__e2e.wsSentMaxAbs;
        for (let i = 0; i < buf.length; i += 8) {
          const a = Math.abs(buf[i]);
          if (a > peak) peak = a;
        }
        window.__e2e.wsSentMaxAbs = peak;
      }
      return origSend(data);
    };
    sock.addEventListener("open", () => { window.__e2e.wsOpened += 1; });
    sock.addEventListener("close", () => { window.__e2e.wsClosed += 1; });
    sock.addEventListener("message", (ev) => {
      if (ev.data instanceof ArrayBuffer) {
        window.__e2e.wsBinaryRecvBytes += ev.data.byteLength;
      } else if (typeof Blob !== "undefined" && ev.data instanceof Blob) {
        window.__e2e.wsBinaryRecvBytes += ev.data.size;
      } else if (typeof ev.data === "string") {
        try {
          window.__e2e.wsMessages.push(JSON.parse(ev.data));
        } catch (_err) {}
      }
    });
    return sock;
  };
  window.WebSocket.prototype = OrigWS.prototype;
  Object.assign(window.WebSocket, {
    CONNECTING: OrigWS.CONNECTING,
    OPEN: OrigWS.OPEN,
    CLOSING: OrigWS.CLOSING,
    CLOSED: OrigWS.CLOSED,
  });

  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  if (AudioCtx && AudioCtx.prototype && AudioCtx.prototype.createBufferSource) {
    const origCreateBufSrc = AudioCtx.prototype.createBufferSource;
    AudioCtx.prototype.createBufferSource = function(...args) {
      const src = origCreateBufSrc.apply(this, args);
      const origStart = src.start.bind(src);
      src.start = function(...sargs) {
        window.__e2e.playedBuffers += 1;
        if (src.buffer) {
          window.__e2e.playedSamples += src.buffer.length;
        }
        return origStart(...sargs);
      };
      return src;
    };
  }

  window.addEventListener("error", (ev) => {
    window.__e2e.pageErrors.push(String(ev.message || ev.error || "window error"));
  });
  window.addEventListener("unhandledrejection", (ev) => {
    window.__e2e.pageErrors.push(String(ev.reason || "unhandledrejection"));
  });
})();
"""


class CDPClient:
    """Tiny synchronous CDP client built on `websockets.sync.client`."""

    def __init__(self, ws_url: str) -> None:
        self._cm = ws_connect(ws_url, open_timeout=15, close_timeout=5, max_size=16 * 1024 * 1024)
        self._ws = self._cm.__enter__()
        self._next_id = 0
        self.console_errors: list[str] = []
        self.exceptions: list[str] = []
        self.http_errors: list[dict[str, Any]] = []
        self.loading_failures: list[dict[str, Any]] = []

    def close(self) -> None:
        try:
            self._cm.__exit__(None, None, None)
        except Exception:
            pass

    def _record_event(self, msg: dict[str, Any]) -> None:
        method = msg.get("method", "")
        params = msg.get("params") or {}
        if method == "Runtime.exceptionThrown":
            details = params.get("exceptionDetails") or {}
            text = details.get("text") or ""
            exc_obj = details.get("exception") or {}
            desc = exc_obj.get("description") or exc_obj.get("value") or ""
            self.exceptions.append(f"{text}: {desc}".strip(": "))
        elif method == "Runtime.consoleAPICalled":
            if params.get("type") == "error":
                args = params.get("args") or []
                parts = [str(a.get("value", a.get("description", ""))) for a in args]
                self.console_errors.append(" ".join(parts))
        elif method == "Log.entryAdded":
            entry = params.get("entry") or {}
            if entry.get("level") == "error":
                self.console_errors.append(
                    f"[{entry.get('source', 'log')}] {entry.get('text', '')}"
                )
        elif method == "Network.responseReceived":
            resp = params.get("response") or {}
            status = int(resp.get("status", 0))
            url = str(resp.get("url", ""))
            if status >= 400 and url.startswith("http://127.0.0.1"):
                self.http_errors.append({"status": status, "url": url})
        elif method == "Network.loadingFailed":
            if not params.get("canceled"):
                self.loading_failures.append(
                    {
                        "type": params.get("type", ""),
                        "errorText": params.get("errorText", ""),
                    }
                )

    def send(self, method: str, params: dict[str, Any] | None = None, timeout: float = 20.0) -> Any:
        self._next_id += 1
        req_id = self._next_id
        payload: dict[str, Any] = {"id": req_id, "method": method}
        if params:
            payload["params"] = params
        self._ws.send(json.dumps(payload))

        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"CDP command timed out: {method}")
            raw = self._ws.recv(timeout=remaining)
            msg = json.loads(raw)
            if msg.get("id") == req_id:
                if "error" in msg:
                    raise RuntimeError(f"CDP error in {method}: {msg['error']}")
                return msg.get("result", {})
            self._record_event(msg)

    def drain_events(self, duration: float = 0.05) -> None:
        deadline = time.monotonic() + duration
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                raw = self._ws.recv(timeout=min(0.05, remaining))
            except TimeoutError:
                break
            msg = json.loads(raw)
            if "method" in msg:
                self._record_event(msg)

    def evaluate(
        self,
        expression: str,
        *,
        await_promise: bool = False,
        user_gesture: bool = False,
        timeout: float = 20.0,
    ) -> Any:
        res = self.send(
            "Runtime.evaluate",
            {
                "expression": expression,
                "returnByValue": True,
                "awaitPromise": await_promise,
                "userGesture": user_gesture,
            },
            timeout=timeout,
        )
        if "exceptionDetails" in res:
            details = res["exceptionDetails"]
            raise RuntimeError(f"JS evaluation error: {details}")
        return (res.get("result") or {}).get("value")


def _wait_cdp_page_ws(cdp_port: int, timeout: float = 25.0) -> str:
    deadline = time.monotonic() + timeout
    last_err = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{cdp_port}/json/list", timeout=2.0) as resp:
                targets = json.loads(resp.read().decode("utf-8"))
            for item in targets:
                if item.get("type") == "page" and item.get("webSocketDebuggerUrl"):
                    return str(item["webSocketDebuggerUrl"])
        except Exception as exc:
            last_err = str(exc)
        time.sleep(0.25)
    raise RuntimeError(f"Timed out waiting for CDP page target on port {cdp_port}: {last_err}")


# ---------------------------------------------------------------------------
# E2E Test Phases
# ---------------------------------------------------------------------------


def _phase_cli_checks(
    cli_cmd: list[str],
    env: dict[str, str],
    *,
    expected_provider: str,
    expected_ort_version: str,
    expect_portable: bool,
) -> dict[str, Any]:
    print("[Phase 1] Running CLI verification (--version, info, self-check, builtins)...")
    ver = _run_cmd([*cli_cmd, "--version"], env=env, timeout=30.0)
    version_str = ver.stdout.strip()
    print(f"  --version -> {version_str}")

    info_proc = _run_cmd([*cli_cmd, "info", "--json"], env=env, timeout=60.0)
    info = json.loads(info_proc.stdout)
    if info.get("configured_provider") != expected_provider:
        raise AssertionError(
            f"configured_provider mismatch: expected {expected_provider!r}, got {info.get('configured_provider')!r}"
        )
    if info.get("asr_provider") != "cpu":
        raise AssertionError(f"asr_provider mismatch: expected 'cpu', got {info.get('asr_provider')!r}")
    if expected_ort_version and info.get("ort_version") != expected_ort_version:
        raise AssertionError(
            f"ort_version mismatch: expected {expected_ort_version!r}, got {info.get('ort_version')!r}"
        )
    if bool(info.get("is_portable")) != expect_portable:
        raise AssertionError(
            f"is_portable mismatch: expected {expect_portable}, got {info.get('is_portable')}"
        )
    models = info.get("models") or {}
    if not info.get("models_ready_for_vc"):
        raise AssertionError(f"Expected models_ready_for_vc=True, got models={models}")
    if not (models.get("silero_vad") or {}).get("ready"):
        raise AssertionError(f"Expected silero_vad ready=True, got models={models}")
    print(
        f"  info --json -> provider={info['configured_provider']}, ort={info['ort_version']}, "
        f"portable={info['is_portable']}, data_dir={info['data_dir']}"
    )

    sc_proc = _run_cmd([*cli_cmd, "self-check", "--json"], env=env, timeout=90.0)
    sc = json.loads(sc_proc.stdout)
    if not sc.get("ok") or sc.get("failed", 1) != 0 or sc.get("total", 0) < 8:
        raise AssertionError(f"self-check failed: {sc}")
    print(f"  self-check -> {sc['passed']}/{sc['total']} passed")

    # Also verify non-JSON and legacy CP1252 pipe output (catches Windows codepage CJK crashes)
    _run_cmd([*cli_cmd, "voice", "builtins"], env=env, timeout=30.0)
    cp1252_env = {**env, "PYTHONIOENCODING": "cp1252"}
    _run_cmd([*cli_cmd, "voice", "builtins"], env=cp1252_env, timeout=30.0)
    bi_cp1252 = _run_cmd([*cli_cmd, "voice", "builtins", "--json"], env=cp1252_env, timeout=30.0)
    builtins_cp1252 = json.loads(bi_cp1252.stdout)
    if len(builtins_cp1252) < 15:
        raise AssertionError(f"CP1252 builtins JSON mismatch: got {len(builtins_cp1252)}")

    bi_proc = _run_cmd([*cli_cmd, "voice", "builtins", "--json"], env=env, timeout=30.0)
    builtins = json.loads(bi_proc.stdout)
    if len(builtins) < 15:
        raise AssertionError(f"Expected >= 15 bundled built-in voices, got {len(builtins)}")
    print(f"  voice builtins -> {len(builtins)} presets verified (UTF-8 & CP1252 pipes OK)")
    return {
        "version": version_str,
        "ort_version": info.get("ort_version"),
        "is_portable": info.get("is_portable"),
        "data_dir": info.get("data_dir"),
        "self_check_passed": sc["passed"],
        "builtins_count": len(builtins),
        "registration_ready": bool((models.get("registration") or {}).get("ready")),
    }


def _wait_server_ready(base_url: str, proc: subprocess.Popen[str], timeout: float = 60.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_err = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"a8vc serve exited early with code {proc.returncode}")
        try:
            r = httpx.get(f"{base_url}/api/health", timeout=2.0)
            if r.status_code == 200:
                return r.json()
            last_err = f"HTTP {r.status_code}"
        except Exception as exc:
            last_err = str(exc)
        time.sleep(0.3)
    raise RuntimeError(f"Timed out waiting for server at {base_url}: {last_err}")


def _phase_server_api_and_ws(
    base_url: str,
    ws_url: str,
    work_dir: Path,
    *,
    expected_provider: str,
    registration_ready: bool,
) -> tuple[dict[str, Any], Path, Path]:
    print("[Phase 2] Verifying live server HTTP API, icons, streaming TTS, enrollment, and /ws/vc...")
    with httpx.Client(base_url=base_url, timeout=180.0) as client:
        # 1. Static HTML & Brand Icons
        r_html = client.get("/")
        assert r_html.status_code == 200, f"GET / -> {r_html.status_code}"
        assert 'src="/icon.svg"' in r_html.text and 'href="/favicon.ico"' in r_html.text

        r_svg = client.get("/icon.svg")
        assert r_svg.status_code == 200 and "<svg" in r_svg.text, "GET /icon.svg invalid"

        r_png = client.get("/icon.png")
        assert r_png.status_code == 200 and r_png.content.startswith(b"\x89PNG\r\n\x1a\n"), (
            "GET /icon.png invalid"
        )

        r_ico = client.get("/favicon.ico")
        assert r_ico.status_code == 200 and r_ico.content[:4] == b"\x00\x00\x01\x00", (
            "GET /favicon.ico invalid"
        )
        print("  static UI & brand icons (/icon.svg, /icon.png, /favicon.ico) -> 200 OK")

        # 2. Health, System Info, Auto-seeded Voices
        health = client.get("/api/health").json()
        assert health["status"] == "ok", health
        assert health["configured_provider"] == expected_provider, health
        assert health["voice_count"] >= 15, f"Expected >=15 auto-seeded voices, got {health}"
        assert health["models"]["ready_for_vc"] is True, health
        assert health["models"]["silero_vad"]["ready"] is True, health

        voices_resp = client.get("/api/voices").json()
        voice_ids = [v["voice_id"] for v in voices_resp["voices"] if "error" not in v]
        assert len(voice_ids) >= 15, f"Auto-seeding failed: {voice_ids}"
        tts_voice = "kokoro_heart_en" if "kokoro_heart_en" in voice_ids else voice_ids[0]
        vc_voice = "edge0_default" if "edge0_default" in voice_ids else voice_ids[0]
        print(f"  auto-seeded voices -> {len(voice_ids)} voices (tts={tts_voice}, vc={vc_voice})")

        # 3. Regression check: missing voice returns JSON 404 (never raw text 500)
        r_bad = client.post(
            "/api/tts/stream", json={"voice_id": "nonexistent_voice_id", "text": "test"}
        )
        assert r_bad.status_code == 404, f"Expected 404, got {r_bad.status_code}: {r_bad.text}"
        bad_json = r_bad.json()
        assert "detail" in bad_json, f"Expected JSON detail on 404, got {bad_json}"

        # 4. Real Audio8-0.1B Streaming TTS
        t0 = time.perf_counter()
        r_tts = client.post(
            "/api/tts/stream",
            json={
                "voice_id": tts_voice,
                "text": "Hello world, voice changer test.",
                "temperature": 0.4,
                "top_p": 0.85,
                "max_new_tokens": 120,
                "seed": 42,
            },
        )
        tts_ms = (time.perf_counter() - t0) * 1000.0
        assert r_tts.status_code == 200, f"/api/tts/stream failed: {r_tts.status_code} {r_tts.text}"
        tts_sr = int(r_tts.headers.get("X-Sample-Rate", "44100"))
        tts_pcm = r_tts.content
        assert len(tts_pcm) >= 24000 and len(tts_pcm) % 2 == 0, (
            f"Unexpected TTS PCM length: {len(tts_pcm)} bytes"
        )
        speech_44k = _pcm16_to_float(tts_pcm)
        tts_rms = float(np.sqrt(np.mean(speech_44k**2)))
        assert tts_rms > 0.005, f"TTS output appears silent (rms={tts_rms:.6f})"
        print(
            f"  /api/tts/stream -> {len(tts_pcm)} bytes ({speech_44k.size / tts_sr:.2f}s @ {tts_sr}Hz, "
            f"rms={tts_rms:.3f}) in {tts_ms:.0f}ms"
        )

        # Prepare speech WAVs for enrollment, direct WS, and Chromium fake mic
        enroll_wav, fake_mic_wav, ws_signal_48k = _prepare_fake_mic_wavs(speech_44k, tts_sr, work_dir)

        # 5. Real Voice Enrollment via HTTP multipart upload (if registration encoder exists)
        enrolled_via_api = False
        if registration_ready:
            with enroll_wav.open("rb") as fh:
                r_enr = client.post(
                    "/api/voices/enroll",
                    files={"audio": ("speech_44k.wav", fh, "audio/wav")},
                    data={
                        "voice_id": "e2e_api_voice",
                        "name": "E2E API Voice",
                        "reference_text": "Hello world, voice changer test.",
                        "language": "en",
                        "overwrite": "true",
                    },
                )
            assert r_enr.status_code == 200, (
                f"/api/voices/enroll failed: {r_enr.status_code} {r_enr.text}"
            )
            enr_body = r_enr.json()
            assert enr_body["voice"]["voice_id"] == "e2e_api_voice", enr_body
            enrolled_via_api = True
            print(f"  /api/voices/enroll -> created e2e_api_voice ({enr_body['voice']['num_frames']} frames)")

    # 6. Direct WebSocket /ws/vc E2E (48 kHz input -> Silero VAD -> SenseVoice ASR -> Audio8)
    print("  Testing direct WebSocket /ws/vc at 48 kHz input...")
    ws_audio_bytes = 0
    ws_transcripts: list[str] = []
    ws_metrics: list[dict[str, Any]] = []
    ws_errors: list[str] = []

    with ws_connect(ws_url, open_timeout=15, close_timeout=5, max_size=16 * 1024 * 1024) as ws:
        ready = json.loads(ws.recv(timeout=30.0))
        assert ready.get("type") == "ready", f"Expected ready frame, got {ready}"

        ws.send(
            json.dumps(
                {
                    "type": "config",
                    "voice_id": vc_voice,
                    "input_sample_rate": 48000,
                }
            )
        )
        vc_changed = json.loads(ws.recv(timeout=60.0))
        assert vc_changed.get("type") == "voice_changed", f"Expected voice_changed, got {vc_changed}"

        # Stream 48 kHz PCM16 speech + silence in 4096-sample chunks, then flush
        pcm_48k = _float_to_pcm16(ws_signal_48k)
        chunk_bytes = 4096 * 2
        for off in range(0, len(pcm_48k), chunk_bytes):
            ws.send(pcm_48k[off : off + chunk_bytes])
        ws.send(json.dumps({"type": "flush"}))

        deadline = time.monotonic() + 150.0
        while time.monotonic() < deadline:
            try:
                msg = ws.recv(timeout=1.0)
            except TimeoutError:
                if ws_metrics and ws_audio_bytes >= 16000:
                    break
                continue
            if isinstance(msg, bytes):
                ws_audio_bytes += len(msg)
            else:
                data = json.loads(msg)
                mtype = data.get("type")
                if mtype == "transcript":
                    ws_transcripts.append(str(data.get("text", "")))
                elif mtype == "metrics":
                    ws_metrics.append(data)
                elif mtype == "error":
                    ws_errors.append(str(data.get("message", data)))
            if ws_metrics and ws_audio_bytes >= 16000:
                # Drain any immediate trailing chunks for up to 1.5s
                drain_end = time.monotonic() + 1.5
                while time.monotonic() < drain_end:
                    try:
                        extra = ws.recv(timeout=0.25)
                        if isinstance(extra, bytes):
                            ws_audio_bytes += len(extra)
                    except TimeoutError:
                        break
                break

    if ws_errors:
        raise AssertionError(f"Direct /ws/vc received error frame(s): {ws_errors}")
    if not ws_transcripts or not any(t.strip() for t in ws_transcripts):
        raise AssertionError(f"Direct /ws/vc produced no ASR transcript (metrics={ws_metrics})")
    if not ws_metrics or ws_audio_bytes < 16000:
        raise AssertionError(
            f"Direct /ws/vc insufficient output: metrics={ws_metrics}, audio_bytes={ws_audio_bytes}"
        )
    print(
        f"  direct /ws/vc -> transcript={ws_transcripts[0]!r}, "
        f"audio={ws_audio_bytes} bytes, metrics={ws_metrics[0]}"
    )

    return (
        {
            "tts_bytes": len(tts_pcm),
            "tts_ms": round(tts_ms, 1),
            "enrolled_via_api": enrolled_via_api,
            "ws_transcript": ws_transcripts[0],
            "ws_audio_bytes": ws_audio_bytes,
            "ws_metrics": ws_metrics[0],
        },
        enroll_wav,
        fake_mic_wav,
    )


def _phase_browser_e2e(
    base_url: str,
    fake_mic_wav: Path,
    work_dir: Path,
    *,
    browser_path: Path,
) -> dict[str, Any]:
    print(f"[Phase 3] Running real headless browser E2E via CDP ({browser_path})...")
    cdp_port = _find_free_port()
    profile_dir = work_dir / "browser_profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    browser_log = work_dir / "browser_stderr.log"

    # Snap-packaged Chromium on Ubuntu has a private /tmp namespace and cannot read /tmp/* WAVs.
    staged_mic_dir: Path | None = None
    effective_mic_wav = fake_mic_wav
    if sys.platform.startswith("linux") and (
        "/snap/" in str(browser_path) or "/snap/" in str(browser_path.resolve())
    ):
        staged_mic_dir = Path(tempfile.mkdtemp(prefix="a8vc_e2e_mic_", dir=str(Path.home())))
        effective_mic_wav = staged_mic_dir / fake_mic_wav.name
        shutil.copy2(fake_mic_wav, effective_mic_wav)

    args = [
        str(browser_path),
        "--headless=new",
        "--no-first-run",
        "--no-default-browser-check",
        "--no-sandbox",
        "--disable-features=AudioServiceSandbox",
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--autoplay-policy=no-user-gesture-required",
        "--use-fake-ui-for-media-stream",
        "--use-fake-device-for-media-stream",
        f"--use-file-for-fake-audio-capture={effective_mic_wav}",
        f"--remote-debugging-port={cdp_port}",
        f"--user-data-dir={profile_dir}",
        "about:blank",
    ]

    log_fh = browser_log.open("w", encoding="utf-8", errors="replace")
    browser_proc = subprocess.Popen(
        args,
        stdout=subprocess.DEVNULL,
        stderr=log_fh,
    )
    cdp: CDPClient | None = None
    try:
        ws_debug_url = _wait_cdp_page_ws(cdp_port, timeout=25.0)
        cdp = CDPClient(ws_debug_url)
        cdp.send("Page.enable")
        cdp.send("Runtime.enable")
        cdp.send("Network.enable")
        cdp.send("Log.enable")
        cdp.send("Page.addScriptToEvaluateOnNewDocument", {"source": _E2E_INIT_SCRIPT})

        # Navigate to live Audio8-VC WebUI
        cdp.send("Page.navigate", {"url": f"{base_url}/"})

        # Wait for UI bootstrap: voices populated, model pills ready, brand icon loaded
        deadline = time.monotonic() + 30.0
        ui_ready: dict[str, Any] = {}
        while time.monotonic() < deadline:
            cdp.drain_events(0.15)
            ui_ready = cdp.evaluate(
                """(() => {
                  const sel = document.getElementById("vc-voice");
                  const img = document.querySelector("img.brand-icon");
                  const pillA8 = document.getElementById("m-audio8");
                  const pillAsr = document.getElementById("m-asr");
                  const pillVad = document.getElementById("m-vad");
                  return {
                    readyState: document.readyState,
                    voiceCount: sel ? sel.options.length : 0,
                    selectedVoice: sel ? sel.value : "",
                    iconOk: Boolean(img && img.complete && img.naturalWidth > 0),
                    a8Ok: Boolean(pillA8 && pillA8.classList.contains("ok")),
                    asrOk: Boolean(pillAsr && pillAsr.classList.contains("ok")),
                    vadOk: Boolean(pillVad && pillVad.classList.contains("ok")),
                  };
                })()"""
            )
            if (
                ui_ready
                and ui_ready["readyState"] == "complete"
                and ui_ready["voiceCount"] >= 15
                and ui_ready["iconOk"]
                and ui_ready["a8Ok"]
                and ui_ready["asrOk"]
                and ui_ready["vadOk"]
            ):
                break
            time.sleep(0.25)
        else:
            raise AssertionError(f"Browser UI did not finish bootstrapping cleanly: {ui_ready}")
        print(
            f"  browser page loaded -> voices={ui_ready['voiceCount']}, iconOk={ui_ready['iconOk']}, "
            f"pills=(a8={ui_ready['a8Ok']}, asr={ui_ready['asrOk']}, vad={ui_ready['vadOk']})"
        )

        # Step 3a: Browser Streaming TTS button test (#btn-tts)
        cdp.evaluate(
            """(() => {
              const sel = document.getElementById("vc-voice");
              for (const opt of sel.options) {
                if (opt.value === "kokoro_heart_en") { sel.value = opt.value; break; }
              }
              document.getElementById("tts-text").value = "Hello browser test.";
              document.getElementById("btn-tts").click();
              return sel.value;
            })()""",
            user_gesture=True,
        )
        tts_deadline = time.monotonic() + 150.0
        tts_state: dict[str, Any] = {}
        while time.monotonic() < tts_deadline:
            cdp.drain_events(0.2)
            tts_state = cdp.evaluate(
                """(() => ({
                  disabled: document.getElementById("btn-tts").disabled,
                  playedBuffers: window.__e2e.playedBuffers,
                  playedSamples: window.__e2e.playedSamples,
                  logText: document.getElementById("log").textContent || "",
                }))()"""
            )
            if not tts_state["disabled"] and tts_state["playedBuffers"] >= 1:
                break
            time.sleep(0.3)
        else:
            raise AssertionError(f"Browser #btn-tts did not play audio within timeout: {tts_state}")

        if "试听失败" in tts_state["logText"] or "试听异常" in tts_state["logText"]:
            raise AssertionError(f"Browser TTS logged an error:\n{tts_state['logText']}")
        if tts_state["playedSamples"] < 8000:
            raise AssertionError(f"Browser TTS played too few samples: {tts_state}")
        print(
            f"  browser #btn-tts -> playedBuffers={tts_state['playedBuffers']}, "
            f"playedSamples={tts_state['playedSamples']}"
        )

        buffers_before_vc = int(tts_state["playedBuffers"])

        # Step 3b: Browser Real-Time WebSocket Voice Changer (#tab-web -> #btn-start)
        cdp.evaluate(
            """(() => {
              document.getElementById("tab-web").click();
              const sel = document.getElementById("vc-voice");
              for (const opt of sel.options) {
                if (opt.value === "edge0_default") { sel.value = opt.value; break; }
              }
              document.getElementById("btn-start").click();
              return true;
            })()""",
            user_gesture=True,
        )

        vc_deadline = time.monotonic() + 180.0
        sent_started_at: float | None = None
        used_flush = False
        vc_state: dict[str, Any] = {}
        while time.monotonic() < vc_deadline:
            cdp.drain_events(0.2)
            vc_state = cdp.evaluate(
                """(() => ({
                  wsOpened: window.__e2e.wsOpened,
                  wsClosed: window.__e2e.wsClosed,
                  wsBinarySentBytes: window.__e2e.wsBinarySentBytes,
                  wsSentMaxAbs: window.__e2e.wsSentMaxAbs,
                  wsBinaryRecvBytes: window.__e2e.wsBinaryRecvBytes,
                  wsMessages: window.__e2e.wsMessages,
                  playedBuffers: window.__e2e.playedBuffers,
                  playedSamples: window.__e2e.playedSamples,
                  pageErrors: window.__e2e.pageErrors,
                  metricsText: document.getElementById("vc-metrics").textContent || "",
                  logText: document.getElementById("log").textContent || "",
                  flushDisabled: document.getElementById("btn-flush").disabled,
                }))()"""
            )

            error_msgs = [m for m in vc_state["wsMessages"] if m.get("type") == "error"]
            if error_msgs:
                raise AssertionError(f"Browser /ws/vc received error frame(s): {error_msgs}")

            if vc_state["wsBinarySentBytes"] > 0 and sent_started_at is None:
                sent_started_at = time.monotonic()

            if (
                sent_started_at is not None
                and (time.monotonic() - sent_started_at) > 8.0
                and vc_state["wsSentMaxAbs"] == 0
            ):
                raise AssertionError(
                    "Browser fake microphone is streaming all-zero silence "
                    f"(sent={vc_state['wsBinarySentBytes']}B, maxAbs=0). "
                    "Check --use-file-for-fake-audio-capture path and audio sandbox flags."
                )

            has_transcript = any(m.get("type") == "transcript" for m in vc_state["wsMessages"])
            has_metrics = any(m.get("type") == "metrics" for m in vc_state["wsMessages"])
            vc_played = vc_state["playedBuffers"] > buffers_before_vc

            if has_transcript and has_metrics and vc_state["wsBinaryRecvBytes"] >= 8000 and vc_played:
                break

            # Fallback flush only if mic audio has been streaming >18s without natural VAD end
            if (
                not used_flush
                and sent_started_at is not None
                and (time.monotonic() - sent_started_at) > 18.0
                and not vc_state["flushDisabled"]
                and not has_transcript
            ):
                used_flush = True
                cdp.evaluate('document.getElementById("btn-flush").click()', user_gesture=True)
            time.sleep(0.35)
        else:
            raise AssertionError(
                f"Browser /ws/vc E2E timed out!\n"
                f"state={json.dumps(vc_state, ensure_ascii=False, indent=2)}\n"
                f"console_errors={cdp.console_errors}\n"
                f"exceptions={cdp.exceptions}\n"
                f"http_errors={cdp.http_errors}"
            )

        # Click Stop and verify clean teardown
        cdp.evaluate('document.getElementById("btn-stop").click()', user_gesture=True)
        time.sleep(0.4)
        cdp.drain_events(0.2)
        after_stop = cdp.evaluate(
            """(() => ({
              startDisabled: document.getElementById("btn-start").disabled,
              stopDisabled: document.getElementById("btn-stop").disabled,
              metricsText: document.getElementById("vc-metrics").textContent || "",
            }))()"""
        )
        assert not after_stop["startDisabled"] and after_stop["stopDisabled"], after_stop
        assert "已停止" in after_stop["metricsText"], after_stop

        # Final zero-error checks across CDP & JS probe
        if cdp.http_errors:
            raise AssertionError(f"Browser saw HTTP >= 400 response(s): {cdp.http_errors}")
        if cdp.exceptions or vc_state["pageErrors"]:
            raise AssertionError(
                f"Browser JS exception(s): cdp={cdp.exceptions}, page={vc_state['pageErrors']}"
            )
        if cdp.console_errors:
            raise AssertionError(f"Browser console.error(s): {cdp.console_errors}")
        if "WebSocket 错误" in vc_state["logText"] or "Internal Server Error" in vc_state["logText"]:
            raise AssertionError(f"Browser UI log contained error:\n{vc_state['logText']}")

        transcripts = [
            m.get("text", "") for m in vc_state["wsMessages"] if m.get("type") == "transcript"
        ]
        metrics_msgs = [m for m in vc_state["wsMessages"] if m.get("type") == "metrics"]
        print(
            f"  browser /ws/vc -> sent={vc_state['wsBinarySentBytes']}B, "
            f"recv={vc_state['wsBinaryRecvBytes']}B, transcripts={transcripts}, "
            f"used_flush={used_flush}, playedBuffers={vc_state['playedBuffers']}"
        )
        return {
            "browser": str(browser_path),
            "tts_played_samples": tts_state["playedSamples"],
            "ws_binary_sent_bytes": vc_state["wsBinarySentBytes"],
            "ws_sent_max_abs": vc_state["wsSentMaxAbs"],
            "ws_binary_recv_bytes": vc_state["wsBinaryRecvBytes"],
            "ws_transcripts": transcripts,
            "ws_metrics": metrics_msgs[0] if metrics_msgs else None,
            "used_flush": used_flush,
            "http_errors": len(cdp.http_errors),
            "console_errors": len(cdp.console_errors),
        }
    except Exception:
        log_fh.flush()
        if browser_log.is_file():
            tail = browser_log.read_text(encoding="utf-8", errors="replace")[-3000:]
            if tail.strip():
                print("\n--- Browser Stderr Tail ---\n" + tail, file=sys.stderr)
        raise
    finally:
        if cdp is not None:
            cdp.close()
        try:
            browser_proc.terminate()
            browser_proc.wait(timeout=8)
        except Exception:
            try:
                browser_proc.kill()
                browser_proc.wait(timeout=5)
            except Exception:
                pass
        log_fh.close()
        if staged_mic_dir is not None:
            shutil.rmtree(staged_mic_dir, ignore_errors=True)


def _phase_cli_synth_and_pack(
    cli_cmd: list[str],
    env: dict[str, str],
    enroll_wav: Path,
    work_dir: Path,
    *,
    registration_ready: bool,
) -> dict[str, Any]:
    print("[Phase 4] Running standalone CLI voice list, synth, and .a8voice pack/verify...")
    _run_cmd([*cli_cmd, "voice", "list"], env=env, timeout=30.0)

    cli_wav = work_dir / "cli_synth.wav"
    _run_cmd(
        [*cli_cmd, "synth", "--voice", "edge0_default", "--text", "Hello CLI.", "--out", str(cli_wav)],
        env=env,
        timeout=150.0,
    )
    assert cli_wav.is_file() and cli_wav.stat().st_size > 12000, f"CLI synth WAV invalid: {cli_wav}"
    print(f"  a8vc synth -> {cli_wav.name} ({cli_wav.stat().st_size} bytes)")

    packed = False
    if registration_ready:
        pack_out = work_dir / "cli_packed.a8voice"
        _run_cmd(
            [
                *cli_cmd,
                "voice",
                "pack",
                "--audio",
                str(enroll_wav),
                "--id",
                "e2e_cli_voice",
                "--name",
                "E2E CLI Voice",
                "--text",
                "Hello world, voice changer test.",
                "--lang",
                "en",
                "--out",
                str(pack_out),
                "--overwrite",
            ],
            env=env,
            timeout=150.0,
        )
        ver_proc = _run_cmd([*cli_cmd, "voice", "verify", str(pack_out)], env=env, timeout=30.0)
        ver_json = json.loads(ver_proc.stdout)
        assert ver_json.get("valid") is True, f"voice verify failed: {ver_json}"
        packed = True
        print(f"  a8vc voice pack & verify -> {pack_out.name} ({pack_out.stat().st_size} bytes, valid=True)")

    return {"cli_synth_bytes": cli_wav.stat().st_size, "cli_voice_packed": packed}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audio8-VC full real-model & headless browser E2E verifier")
    parser.add_argument(
        "--cli",
        default="",
        help="Path to a8vc executable (e.g. a8vc.exe). Defaults to `python -m a8vc.cli`.",
    )
    parser.add_argument(
        "--data-dir",
        default="",
        help="Writable A8VC_DATA_DIR (ignored when --portable is set).",
    )
    parser.add_argument(
        "--portable",
        action="store_true",
        help="Verify portable mode (expects portable.flag next to CLI and does not set A8VC_DATA_DIR).",
    )
    parser.add_argument("--model-dir", default="", help="Optional A8VC_MODEL_DIR override.")
    parser.add_argument("--asr-dir", default="", help="Optional A8VC_ASR_MODEL_DIR override.")
    parser.add_argument("--vad-path", default="", help="Optional A8VC_VAD_MODEL_PATH override.")
    parser.add_argument("--expected-provider", default="cpu", help="Expected configured_provider (default: cpu).")
    parser.add_argument(
        "--expected-ort-version",
        default="1.24.4",
        help="Expected ONNX Runtime ABI version (pass empty string to skip check).",
    )
    parser.add_argument("--browser-path", default="", help="Optional explicit path to msedge/chrome/chromium.")
    parser.add_argument("--skip-browser", action="store_true", help="Skip headless browser CDP phase.")
    parser.add_argument("--skip-cli-synth", action="store_true", help="Skip Phase 4 standalone CLI synth/pack.")
    parser.add_argument("--work-dir", default="", help="Optional working directory for temp WAVs & logs.")
    args = parser.parse_args(argv)

    cli_cmd = [str(Path(args.cli).resolve())] if args.cli else [sys.executable, "-m", "a8vc.cli"]

    env = dict(os.environ)
    # Remove inherited overrides so test isolation is deterministic
    for k in (
        "A8VC_DATA_DIR",
        "A8VC_PORTABLE",
        "A8VC_MODEL_DIR",
        "A8VC_ASR_MODEL_DIR",
        "A8VC_VAD_MODEL_PATH",
        "A8VC_PROVIDER",
    ):
        env.pop(k, None)

    if not args.portable and args.data_dir:
        env["A8VC_DATA_DIR"] = str(Path(args.data_dir).resolve())
    if args.model_dir:
        env["A8VC_MODEL_DIR"] = str(Path(args.model_dir).resolve())
    if args.asr_dir:
        env["A8VC_ASR_MODEL_DIR"] = str(Path(args.asr_dir).resolve())
    if args.vad_path:
        env["A8VC_VAD_MODEL_PATH"] = str(Path(args.vad_path).resolve())

    browser_exe: Path | None = None
    if not args.skip_browser:
        browser_exe = _discover_browser(args.browser_path or None)
        if browser_exe is None:
            raise RuntimeError(
                "No Chromium/Edge browser found for E2E verification. "
                "Install Microsoft Edge/Chromium or pass --browser-path."
            )

    created_temp = False
    if args.work_dir:
        work_dir = Path(args.work_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="a8vc_e2e_"))
        created_temp = True

    server_log_path = work_dir / "server.log"
    server_proc: subprocess.Popen[str] | None = None
    server_log_fh = None
    t_start = time.perf_counter()

    try:
        print(f"=== Audio8-VC Full E2E Verification ({platform.platform()}) ===")
        print(f"CLI      : {' '.join(cli_cmd)}")
        print(f"Portable : {args.portable}")
        if browser_exe:
            print(f"Browser  : {browser_exe}")

        cli_summary = _phase_cli_checks(
            cli_cmd,
            env,
            expected_provider=args.expected_provider,
            expected_ort_version=args.expected_ort_version,
            expect_portable=args.portable,
        )

        # Start live server
        port = _find_free_port()
        base_url = f"http://127.0.0.1:{port}"
        ws_url = f"ws://127.0.0.1:{port}/ws/vc"
        server_log_fh = server_log_path.open("w", encoding="utf-8", errors="replace")
        server_proc = subprocess.Popen(
            [*cli_cmd, "serve", "--host", "127.0.0.1", "--port", str(port)],
            env=env,
            stdout=server_log_fh,
            stderr=subprocess.STDOUT,
            text=True,
        )
        _wait_server_ready(base_url, server_proc, timeout=60.0)
        print(f"  live server ready at {base_url}")

        api_summary, enroll_wav, fake_mic_wav = _phase_server_api_and_ws(
            base_url,
            ws_url,
            work_dir,
            expected_provider=args.expected_provider,
            registration_ready=cli_summary["registration_ready"],
        )

        browser_summary: dict[str, Any] | None = None
        if not args.skip_browser and browser_exe is not None:
            browser_summary = _phase_browser_e2e(
                base_url,
                fake_mic_wav,
                work_dir,
                browser_path=browser_exe,
            )

        # Stop server cleanly before Phase 4
        server_proc.terminate()
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_proc.wait(timeout=5)
        server_proc = None

        cli_synth_summary: dict[str, Any] | None = None
        if not args.skip_cli_synth:
            cli_synth_summary = _phase_cli_synth_and_pack(
                cli_cmd,
                env,
                enroll_wav,
                work_dir,
                registration_ready=cli_summary["registration_ready"],
            )

        elapsed = round(time.perf_counter() - t_start, 1)
        summary = {
            "ok": True,
            "platform": platform.platform(),
            "elapsed_sec": elapsed,
            "cli": cli_summary,
            "api_and_ws": api_summary,
            "browser": browser_summary,
            "cli_synth": cli_synth_summary,
        }
        print("\n=== Audio8-VC E2E Verification PASSED ===")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        if server_log_fh is not None:
            try:
                server_log_fh.flush()
            except Exception:
                pass
        server_tail = ""
        if server_log_path.is_file():
            server_tail = server_log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            print("\n--- Server Log Tail ---\n" + server_tail, file=sys.stderr)
        err_summary = f"{type(exc).__name__}: {exc}"
        try:
            (work_dir / "e2e_failure.txt").write_text(
                err_summary + "\n\n--- Server Log Tail ---\n" + server_tail,
                encoding="utf-8",
            )
        except Exception:
            pass
        if os.environ.get("GITHUB_ACTIONS", "").lower() == "true":
            one_line = " ".join(err_summary.split())[:800]
            print(f"::error title=Audio8-VC E2E Failure::{one_line}", flush=True)
        raise
    finally:
        if server_proc is not None:
            try:
                server_proc.terminate()
                server_proc.wait(timeout=8)
            except Exception:
                try:
                    server_proc.kill()
                    server_proc.wait(timeout=5)
                except Exception:
                    pass
        if server_log_fh is not None:
            try:
                server_log_fh.close()
            except Exception:
                pass
        if created_temp:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
