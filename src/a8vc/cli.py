"""Command-line interface for Audio8-VC (`a8vc`)."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from a8vc import __version__
from a8vc.asr.sherpa import create_asr
from a8vc.audio.device import LocalAudioBridge, list_audio_devices
from a8vc.audio.io import save_wav
from a8vc.audio.vad import SpeechSegmenter, create_vad
from a8vc.config import AppConfig, load_config
from a8vc.engine.base import VCOutputEvent
from a8vc.engine.cascaded import CascadedVCEngine, RealtimeVCWorker, get_process_rss_mb
from a8vc.runtime.audio8_onnx import Audio8OnnxRuntime
from a8vc.voices.catalog import install_builtin_voices, list_builtin_voices
from a8vc.voices.enroll import VoiceEnroller
from a8vc.voices.format import verify_voice_pack
from a8vc.voices.store import VoiceLibrary


def _is_utf8_stream(stream: Any) -> bool:
    enc = getattr(stream, "encoding", None) or ""
    return enc.lower().replace("_", "-") in {"utf-8", "utf8"}


def _configure_stdio() -> None:
    """Avoid UnicodeEncodeError on legacy Windows console/pipe codepages (e.g. cp1252)."""
    for stream in (sys.stdout, sys.stderr):
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        if not _is_utf8_stream(stream):
            try:
                stream.reconfigure(errors="replace")
            except Exception:
                pass


def _print_json(payload: Any) -> None:
    """Emit valid JSON without UnicodeEncodeError or codepage mojibake on Windows pipes."""
    print(json.dumps(payload, ensure_ascii=not _is_utf8_stream(sys.stdout), indent=2))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="a8vc",
        description="Audio8-VC: Low-footprint real-time voice changer powered by Edge0-AI Audio8-0.1B",
    )
    parser.add_argument("--version", action="version", version=f"a8vc {__version__}")
    parser.add_argument("--config", type=str, default=None, help="Path to TOML config file")

    sub = parser.add_subparsers(dest="command", required=True)

    # --- voice subcommands ---
    voice_p = sub.add_parser("voice", help="Manage portable .a8voice voice packs")
    voice_sub = voice_p.add_subparsers(dest="voice_cmd", required=True)

    voice_sub.add_parser("list", help="List installed .a8voice voice packs")

    inspect_p = voice_sub.add_parser("inspect", help="Inspect a voice pack's metadata")
    inspect_p.add_argument("target", help="voice_id in library or path to .a8voice file")

    verify_p = voice_sub.add_parser("verify", help="Verify .a8voice integrity & SHA-256")
    verify_p.add_argument("path", help="Path to .a8voice file")

    pack_p = voice_sub.add_parser("pack", help="Encode reference audio into a .a8voice file")
    pack_p.add_argument("--audio", required=True, help="Path to 0.5-30s clean speech audio")
    pack_p.add_argument("--id", required=True, dest="voice_id", help="Voice identifier (e.g. aria)")
    pack_p.add_argument("--text", default="", help="Exact transcript of the reference audio")
    pack_p.add_argument(
        "--auto-asr",
        action="store_true",
        help="Automatically transcribe reference audio using configured ASR if --text is omitted",
    )
    pack_p.add_argument("--name", default="", help="Display name")
    pack_p.add_argument("--description", default="", help="Optional voice description")
    pack_p.add_argument("--lang", default="zh", help="Language code (e.g. zh, en)")
    pack_p.add_argument("--tags", default="", help="Comma-separated tags")
    pack_p.add_argument("--out", default="", help="Optional output .a8voice path (default: voices_dir)")
    pack_p.add_argument("--overwrite", action="store_true", help="Overwrite existing voice pack")

    imp_p = voice_sub.add_parser(
        "import-official", help="Convert official Audio8 ONNX voice folder into .a8voice"
    )
    imp_p.add_argument("official_dir", help="Directory containing meta.json and codes.npy")
    imp_p.add_argument("--id", default="", dest="voice_id", help="Target voice_id")
    imp_p.add_argument("--name", default="", help="Display name")
    imp_p.add_argument("--lang", default="zh", help="Language code")
    imp_p.add_argument("--overwrite", action="store_true", help="Overwrite if voice exists")

    exp_p = voice_sub.add_parser(
        "export-official", help="Export a .a8voice pack back to official Audio8 ONNX folder format"
    )
    exp_p.add_argument("voice_id", help="Voice ID to export")
    exp_p.add_argument("output_dir", help="Parent directory for official voice folder")

    builtins_p = voice_sub.add_parser(
        "builtins", help="List curated built-in open-source .a8voice presets"
    )
    builtins_p.add_argument("--json", action="store_true", dest="as_json", help="Output JSON")

    inst_b_p = voice_sub.add_parser(
        "install-builtins", help="Install bundled open-source built-in .a8voice presets into voices_dir"
    )
    inst_b_p.add_argument(
        "--id",
        action="append",
        dest="voice_ids",
        default=[],
        help="Specific built-in voice_id to install (repeatable; omit to install all)",
    )
    inst_b_p.add_argument("--overwrite", action="store_true", help="Overwrite existing installed voices")

    # --- devices ---
    sub.add_parser("devices", help="List available local audio input/output devices")

    # --- info ---
    info_p = sub.add_parser(
        "info", help="Show runtime configuration, GPU/ONNX providers, paths, and model status"
    )
    info_p.add_argument("--json", action="store_true", dest="as_json", help="Output machine-readable JSON")

    # --- synth ---
    synth_p = sub.add_parser("synth", help="Synthesize a test WAV using a .a8voice pack")
    synth_p.add_argument("--voice", required=True, help="Voice ID or path to .a8voice")
    synth_p.add_argument("--text", required=True, help="Target text to synthesize")
    synth_p.add_argument("--out", required=True, help="Output WAV path")
    synth_p.add_argument("--seed", type=int, default=None, help="Optional RNG seed")

    # --- benchmark ---
    bench_p = sub.add_parser("benchmark", help="Benchmark cold vs prefix-cached TTFA, RTF & RAM")
    bench_p.add_argument("--voice", required=True, help="Voice ID or path to .a8voice")
    bench_p.add_argument(
        "--text",
        default="你好，这是一个高效率低占用的实时变声性能基准测试。",
        help="Test utterance text",
    )
    bench_p.add_argument("--runs", type=int, default=3, help="Number of runs (run 1 is cold)")

    # --- live ---
    live_p = sub.add_parser("live", help="Run local real-time microphone -> target voice passthrough")
    live_p.add_argument("--voice", required=True, help="Initial target voice ID or .a8voice path")
    live_p.add_argument("--input-device", default="", help="Input device index or name")
    live_p.add_argument("--output-device", default="", help="Output device index or name")

    # --- serve ---
    serve_p = sub.add_parser("serve", help="Start HTTP + WebSocket streaming server and Web UI")
    serve_p.add_argument("--host", default=None, help="Bind host (default from config)")
    serve_p.add_argument("--port", type=int, default=None, help="Bind port (default from config)")
    serve_p.add_argument("--voice", default=None, help="Default active voice ID")

    # --- desktop ---
    desk_p = sub.add_parser(
        "desktop", help="Launch the Windows/Cross-Platform Desktop GUI application"
    )
    desk_p.add_argument("--host", default="127.0.0.1", help="Local bind host")
    desk_p.add_argument("--port", type=int, default=None, help="Preferred local port")
    desk_p.add_argument("--voice", default=None, help="Default active voice ID")
    desk_p.add_argument(
        "--portable",
        action="store_true",
        help="Store models, voices, and logs inside ./data next to the executable",
    )
    desk_p.add_argument(
        "--browser",
        action="store_true",
        help="Open in default browser instead of native WebView window",
    )
    desk_p.add_argument(
        "voice_file",
        nargs="?",
        default=None,
        help="Optional .a8voice file to auto-import on launch",
    )

    # --- download ---
    dl_p = sub.add_parser(
        "download", help="Download Audio8-0.1B ONNX INT8, SenseVoice ASR, and Silero VAD models"
    )
    dl_p.add_argument("--audio8-onnx", action="store_true", help="Download Audio8-0.1B ONNX INT8")
    dl_p.add_argument(
        "--no-registration",
        action="store_true",
        help="Skip optional registration/codec_encoder_fp16.onnx",
    )
    dl_p.add_argument("--asr-sensevoice", action="store_true", help="Download SenseVoice INT8 ASR")
    dl_p.add_argument("--silero-vad", action="store_true", help="Download Silero VAD ONNX (~2 MB)")
    dl_p.add_argument(
        "--all",
        action="store_true",
        dest="download_all",
        help="Download full Track 1 stack (Audio8 ONNX + Registration + SenseVoice ASR + Silero VAD)",
    )
    dl_p.add_argument(
        "--mirror",
        action="store_true",
        help="Use Hugging Face mirror (https://hf-mirror.com) for faster downloads in China",
    )
    dl_p.add_argument("--hf-endpoint", default="", help="Custom Hugging Face endpoint URL")

    return parser


def _parse_device_arg(value: str) -> str | int:
    if not value:
        return ""
    try:
        return int(value)
    except ValueError:
        return value


def _cmd_voice(args: argparse.Namespace, cfg: AppConfig) -> int:
    lib = VoiceLibrary(cfg.paths.voices_dir)

    if args.voice_cmd == "list":
        voices = lib.list_voices()
        if not voices:
            print(f"No .a8voice packs found in {lib.root}")
            print("Tip: run `a8vc voice install-builtins` to install bundled open-source preset voices.")
            return 0
        for item in voices:
            if "error" in item:
                print(f"- {item['voice_id']} [INVALID: {item['error']}]")
            else:
                kb = item["size_bytes"] / 1024.0
                print(
                    f"- {item['voice_id']:<20} name={item['name']}  lang={item['language']}  "
                    f"dur={item['duration_sec']}s  frames={item['num_frames']}  size={kb:.1f}KB"
                )
        return 0

    if args.voice_cmd == "builtins":
        installed_ids = {v["voice_id"] for v in lib.list_voices() if "error" not in v}
        catalog = list_builtin_voices(installed_ids=installed_ids)
        if args.as_json:
            _print_json(catalog)
            return 0
        if not catalog:
            print("No bundled built-in voices found.")
            return 0
        for item in catalog:
            mark = "installed" if item["installed"] else "available"
            kb = item["size_bytes"] / 1024.0
            print(
                f"- {item['voice_id']:<22} [{item['category_label']}] {item['name']}  "
                f"lang={item['language']}  dur={item['duration_sec']}s  size={kb:.1f}KB  "
                f"({item['license']}, {mark}) - {item['source_project']}"
            )
        return 0

    if args.voice_cmd == "install-builtins":
        res = install_builtin_voices(
            lib,
            voice_ids=args.voice_ids or None,
            overwrite=args.overwrite,
        )
        print(
            f"Installed {len(res['installed'])} built-in voice(s) into {lib.root} "
            f"(skipped {len(res['skipped'])} already present)."
        )
        for vid in res["installed"]:
            print(f"  + {vid}")
        return 0

    if args.voice_cmd == "inspect":
        pack = lib.load(args.target, verify_checksum=True)
        _print_json(pack.manifest.model_dump(mode="json"))
        return 0

    if args.voice_cmd == "verify":
        report = verify_voice_pack(args.path)
        _print_json(report)
        return 0

    if args.voice_cmd == "pack":
        asr = None
        if not args.text.strip():
            if not args.auto_asr:
                raise SystemExit("Error: pass --text <transcript> or --auto-asr for auto-transcription.")
            asr = create_asr(cfg.asr, cfg.paths.asr_model_dir)
        enroller = VoiceEnroller(
            cfg.paths.model_dir,
            provider=cfg.runtime.provider,
            threads=max(1, cfg.runtime.threads or 2),
        )
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
        pack = enroller.enroll_file(
            args.audio,
            voice_id=args.voice_id,
            reference_text=args.text or None,
            asr=asr,
            name=args.name or None,
            description=args.description,
            language=args.lang,
            tags=tags,
        )
        if args.out:
            from a8vc.voices.format import save_voice_pack

            out_path = save_voice_pack(pack, args.out)
        else:
            out_path = lib.save(pack, overwrite=args.overwrite)
        print(
            f"Saved .a8voice pack: {out_path} "
            f"({pack.manifest.num_frames} frames, {pack.manifest.duration_sec}s, "
            f"{out_path.stat().st_size / 1024.0:.1f} KB)"
        )
        return 0

    if args.voice_cmd == "import-official":
        pack = lib.import_official(
            args.official_dir,
            voice_id=args.voice_id or None,
            name=args.name or None,
            language=args.lang,
            overwrite=args.overwrite,
        )
        print(f"Imported official voice into {pack.path}")
        return 0

    if args.voice_cmd == "export-official":
        dest = lib.export_official(args.voice_id, args.output_dir)
        print(f"Exported official voice directory to {dest}")
        return 0

    return 1


def _cmd_devices() -> int:
    for dev in list_audio_devices():
        flags: list[str] = []
        if dev["is_default_input"]:
            flags.append("DEFAULT_IN")
        if dev["is_default_output"]:
            flags.append("DEFAULT_OUT")
        if dev.get("is_virtual_cable"):
            flags.append("VIRTUAL")
        tag = f" [{' '.join(flags)}]" if flags else ""
        hostapi = f" [{dev['hostapi']}]" if dev.get("hostapi") else ""
        print(
            f"[{dev['index']:>2}] {dev['name']}{hostapi} "
            f"(in={dev['max_input_channels']}, out={dev['max_output_channels']}, "
            f"sr={int(dev['default_samplerate'])}){tag}"
        )
    return 0


def _cmd_synth(args: argparse.Namespace, cfg: AppConfig) -> int:
    lib = VoiceLibrary(cfg.paths.voices_dir)
    pack = lib.load(args.voice)
    rt = Audio8OnnxRuntime(cfg.paths.model_dir, cfg.runtime)
    audio, metrics = rt.synthesize(args.text, pack, seed=args.seed)
    out = save_wav(args.out, audio, rt.sample_rate)
    print(f"Wrote {out} ({metrics.audio_seconds}s @ {rt.sample_rate} Hz)")
    print(
        f"Metrics: prefill={metrics.prefill_ms}ms  ttfa={metrics.ttfa_ms}ms  "
        f"total={metrics.total_ms}ms  RTF={metrics.rtf}  cache_hit={metrics.cache_hit}  "
        f"RSS={get_process_rss_mb()}MB"
    )
    rt.close()
    return 0


def _cmd_benchmark(args: argparse.Namespace, cfg: AppConfig) -> int:
    lib = VoiceLibrary(cfg.paths.voices_dir)
    pack = lib.load(args.voice)
    rt = Audio8OnnxRuntime(cfg.paths.model_dir, cfg.runtime)
    print(f"Benchmarking voice={pack.voice_id} ({pack.manifest.num_frames} ref frames) runs={args.runs}")
    for i in range(1, max(1, args.runs) + 1):
        _, m = rt.synthesize(args.text, pack, seed=42 + i)
        label = "cold" if not m.cache_hit else "cached"
        print(
            f"  Run {i} [{label:<6}]: prefill={m.prefill_ms:>7.1f}ms  ttfa={m.ttfa_ms:>7.1f}ms  "
            f"total={m.total_ms:>7.1f}ms  audio={m.audio_seconds:>5.2f}s  RTF={m.rtf:.3f}  "
            f"RSS={get_process_rss_mb():.1f}MB"
        )
    print(f"Prefix cache: {rt.cache_stats()}")
    rt.close()
    return 0


def _cmd_live(args: argparse.Namespace, cfg: AppConfig) -> int:
    lib = VoiceLibrary(cfg.paths.voices_dir)
    initial_pack = lib.load(args.voice)
    rt = Audio8OnnxRuntime(cfg.paths.model_dir, cfg.runtime)
    asr = create_asr(cfg.asr, cfg.paths.asr_model_dir)
    vad = create_vad(cfg.vad, cfg.paths.vad_model_path)
    segmenter = SpeechSegmenter(
        vad=vad,
        sample_rate=cfg.audio.input_sample_rate,
        speech_start_ms=cfg.vad.speech_start_ms,
        min_silence_ms=cfg.vad.min_silence_ms,
        pre_speech_ms=cfg.vad.pre_speech_ms,
        min_segment_ms=cfg.vad.min_segment_ms,
        max_segment_ms=cfg.vad.max_segment_ms,
    )
    engine = CascadedVCEngine(
        synthesizer=rt,
        asr=asr,
        segmenter=segmenter,
        crossfade_ms=cfg.audio.crossfade_ms,
    )

    warm_ms = engine.set_voice(initial_pack, warm=True)
    print(
        f"Active voice: {initial_pack.voice_id} (warmed prefix cache in {warm_ms:.1f} ms, "
        f"RSS={get_process_rss_mb():.1f} MB)"
    )

    in_dev = _parse_device_arg(args.input_device) or cfg.audio.input_device
    out_dev = _parse_device_arg(args.output_device) or cfg.audio.output_device

    bridge: LocalAudioBridge | None = None

    def _on_vc_event(event: VCOutputEvent) -> None:
        if event.transcript:
            print(f"[ASR {event.asr_ms:.0f}ms] {event.transcript}", flush=True)
        if event.audio.size > 0 and bridge is not None:
            bridge.playback.push(event.audio, source_sr=event.sample_rate)
        if event.is_final:
            print(
                f"[VC {event.voice_id}] prefill={event.prefill_ms:.0f}ms "
                f"({'cached' if event.cache_hit else 'cold'}) ttfa={event.ttfa_ms:.0f}ms",
                flush=True,
            )

    worker = RealtimeVCWorker(
        engine,
        _on_vc_event,
        max_pending_segments=cfg.engine.max_pending_segments,
    )
    bridge = LocalAudioBridge(
        on_input_chunk=worker.submit_audio,
        input_sample_rate=cfg.audio.input_sample_rate,
        output_sample_rate=cfg.audio.output_sample_rate,
        input_block_ms=cfg.audio.input_block_ms,
        input_device=in_dev,
        output_device=out_dev,
    )

    worker.start()
    bridge.start()
    print("Live voice changer started! Type a voice_id to hot-switch, 'list' to view voices, or 'q' to quit.")
    try:
        while True:
            cmd = input("a8vc> ").strip()
            if cmd in {"q", "quit", "exit"}:
                break
            if cmd == "list":
                for item in lib.list_voices():
                    print(f"  - {item['voice_id']} ({item.get('name', '')})")
                continue
            if cmd == "stats":
                _print_json(engine.stats())
                continue
            if cmd:
                try:
                    new_pack = lib.load(cmd)
                    engine.cancel()
                    bridge.playback.clear()
                    ms = engine.set_voice(new_pack, warm=True)
                    print(f"Switched voice -> {new_pack.voice_id} (warm={ms:.1f}ms)")
                except Exception as exc:
                    print(f"Failed to switch voice: {exc}")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        bridge.stop()
        worker.stop()
        engine.close()
    return 0


def _cmd_serve(args: argparse.Namespace, cfg: AppConfig) -> int:
    import uvicorn

    from a8vc.server.app import create_app

    if args.host is not None:
        cfg.server.host = args.host
    if args.port is not None:
        cfg.server.port = args.port
    if args.voice is not None:
        cfg.server.default_voice = args.voice

    app = create_app(cfg)
    print(f"Starting Audio8-VC server at http://{cfg.server.host}:{cfg.server.port}")
    uvicorn.run(app, host=cfg.server.host, port=cfg.server.port, log_level="info")
    return 0


def _cmd_desktop(args: argparse.Namespace) -> int:
    import os

    from a8vc.desktop import launch_desktop

    if args.portable:
        os.environ["A8VC_PORTABLE"] = "1"
    cfg = load_config(args.config, desktop=True)
    if args.voice:
        cfg.server.default_voice = args.voice
    return launch_desktop(
        cfg,
        host=args.host,
        port=args.port,
        use_browser=args.browser,
        import_voice_file=args.voice_file,
    )


def _cmd_download(args: argparse.Namespace, cfg: AppConfig) -> int:
    from pathlib import Path

    from a8vc.models_manager import (
        AUDIO8_ONNX_REPO,
        HF_MIRROR_ENDPOINT,
        download_hf_snapshot,
        download_sensevoice_asr,
        download_silero_vad,
        inspect_models,
    )

    want_audio8 = bool(args.audio8_onnx or args.download_all)
    want_asr = bool(args.asr_sensevoice or args.download_all)
    want_vad = bool(args.silero_vad or args.download_all)
    if not any([want_audio8, want_asr, want_vad]):
        _print_json(inspect_models(cfg))
        print("\nTip: run `a8vc download --all` (add `--mirror` in mainland China) to download models.")
        return 0

    endpoint = args.hf_endpoint.strip() or (HF_MIRROR_ENDPOINT if args.mirror else None)
    if want_audio8:
        print(f"Downloading {AUDIO8_ONNX_REPO} -> {cfg.paths.model_dir} ...")
        ignore = ["registration/*"] if args.no_registration else None
        download_hf_snapshot(
            AUDIO8_ONNX_REPO,
            Path(cfg.paths.model_dir),
            ignore_patterns=ignore,
            hf_endpoint=endpoint,
        )
    if want_asr:
        print(f"Downloading SenseVoice INT8 ASR -> {cfg.paths.asr_model_dir} ...")
        download_sensevoice_asr(Path(cfg.paths.asr_model_dir))
    if want_vad:
        print(f"Downloading Silero VAD -> {cfg.paths.vad_model_path} ...")
        download_silero_vad(Path(cfg.paths.vad_model_path))

    print("Download complete! Model status:")
    _print_json(inspect_models(cfg))
    return 0


def _cmd_info(args: argparse.Namespace, cfg: AppConfig) -> int:
    import platform

    from a8vc.models_manager import inspect_models
    from a8vc.paths import get_data_dir, is_frozen, is_portable_mode
    from a8vc.runtime.providers import probe_provider_info

    provider_info = probe_provider_info(cfg.runtime.provider)
    models = inspect_models(cfg)
    payload = {
        "version": __version__,
        "platform": sys.platform,
        "platform_release": platform.platform(),
        "is_frozen": is_frozen(),
        "is_portable": is_portable_mode(),
        "data_dir": str(get_data_dir()),
        "configured_provider": cfg.runtime.provider,
        "asr_provider": cfg.asr.provider,
        **provider_info,
        "models_ready_for_vc": models["ready_for_vc"],
        "models": models,
    }
    if args.as_json:
        _print_json(payload)
        return 0

    print(f"Audio8-VC v{payload['version']} ({payload['platform_release']})")
    print(f"  Frozen / Portable   : frozen={payload['is_frozen']} portable={payload['is_portable']}")
    print(f"  Data dir            : {payload['data_dir']}")
    print(f"  Configured provider : runtime={payload['configured_provider']} asr={payload['asr_provider']}")
    print(f"  ORT version / flavor: {payload['ort_version']} ({payload['ort_flavor']})")
    print(f"  Available providers : {', '.join(payload['available_providers']) or '(none)'}")
    print(f"  Resolved providers  : {', '.join(payload['resolved_providers'])}")
    if payload["cuda_dll_dirs"]:
        print(f"  Detected CUDA dirs  : {', '.join(payload['cuda_dll_dirs'])}")
    print(f"  Models ready for VC : {payload['models_ready_for_vc']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "desktop":
        return _cmd_desktop(args)

    cfg = load_config(args.config)

    if args.command == "voice":
        return _cmd_voice(args, cfg)
    if args.command == "devices":
        return _cmd_devices()
    if args.command == "info":
        return _cmd_info(args, cfg)
    if args.command == "synth":
        return _cmd_synth(args, cfg)
    if args.command == "benchmark":
        return _cmd_benchmark(args, cfg)
    if args.command == "live":
        return _cmd_live(args, cfg)
    if args.command == "serve":
        return _cmd_serve(args, cfg)
    if args.command == "download":
        return _cmd_download(args, cfg)
    return 1


if __name__ == "__main__":
    sys.exit(main())
