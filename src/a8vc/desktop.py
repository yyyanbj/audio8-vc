"""Windows & cross-platform Desktop GUI launcher for Audio8-VC.

Starts the local FastAPI + WebSocket backend on an available localhost port and hosts the
control panel inside a lightweight native WebView2 window (`pywebview`), with automatic
Mark-of-the-Web unblocking on Windows portable ZIPs, verified window-shown detection,
native error reporting, and graceful fallback to the system browser.
"""

from __future__ import annotations

import argparse
import ctypes
import inspect
import io
import json
import logging
import multiprocessing
import os
import socket
import sys
import threading
import time
import traceback
import urllib.request
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import Any

from a8vc import __version__
from a8vc.config import AppConfig, load_config
from a8vc.paths import (
    get_bundle_dir,
    get_data_dir,
    get_executable_dir,
    get_icon_path,
    get_logs_dir,
    is_frozen,
)
from a8vc.server.app import ServerState, create_app
from a8vc.voices.format import A8VOICE_EXT

logger = logging.getLogger("a8vc.desktop")

APP_USER_MODEL_ID = "Edge0.Audio8VC.Desktop"

_WEBVIEW2_CLIENT_GUIDS = (
    "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}",  # Evergreen WebView2 Runtime
    "{2CD8A007-E189-409D-A2C8-9AF4EF3C72AA}",  # Beta
    "{0D50BFEC-CD6A-4F9A-964C-C7416E3ACB10}",  # Dev
    "{65C35B14-6C1D-4122-AC46-7148CC9D6497}",  # Canary
)

_MANAGED_BUNDLE_SUBDIRS = ("pythonnet", "webview", "clr_loader")


class _NullStream(io.TextIOBase):
    """Fallback stream when running frozen on Windows with `console=False`."""

    def write(self, s: str) -> int:
        return len(s)

    def flush(self) -> None:
        return None

    def isatty(self) -> bool:
        return False


def ensure_safe_stdio(log_file: Path | None = None) -> None:
    """Guarantee `sys.stdout` and `sys.stderr` are valid writable text streams."""
    fallback: Any = None
    if log_file is not None:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fallback = log_file.open("a", encoding="utf-8", buffering=1)
        except Exception:
            fallback = None
    if fallback is None:
        fallback = _NullStream()

    if sys.stdout is None:
        sys.stdout = fallback
    if sys.stderr is None:
        sys.stderr = fallback


def configure_desktop_logging(log_file: Path) -> None:
    """Configure UTF-8 file logging (and console logging when attached)."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    if sys.stderr is not None and not isinstance(sys.stderr, _NullStream):
        try:
            if getattr(sys.stderr, "fileno", None) and sys.stderr.fileno() >= 0:
                handlers.append(logging.StreamHandler(sys.stderr))
        except Exception:
            pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def set_windows_app_id(app_id: str = APP_USER_MODEL_ID) -> None:
    """Set explicit Windows taskbar AppUserModelID for proper icon grouping."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        pass


def show_windows_error_dialog(title: str, message: str) -> None:
    """Show a native Windows error dialog when `Audio8-VC.exe` (`console=False`) fails to start."""
    if sys.platform != "win32":
        return
    if os.environ.get("A8VC_NO_DIALOG", "").strip().lower() in {"1", "true", "yes"}:
        return
    try:
        # MB_OK (0x0) | MB_ICONERROR (0x10) | MB_SETFOREGROUND (0x10000)
        ctypes.windll.user32.MessageBoxW(0, str(message), str(title), 0x00010010)
    except Exception:
        pass


def select_webview_icon(
    *,
    platform: str | None = None,
    frozen: bool | None = None,
) -> Path | None:
    """Choose a platform-safe icon argument for `webview.start(icon=...)`.

    On Windows, `pywebview`'s WinForms backend constructs `System.Drawing.Icon(path)`,
    which ONLY accepts `.ico` files and raises `System.ArgumentException` inside the .NET
    STA thread if passed a `.png` file (causing `Audio8-VC.exe` to fail to open).
    When frozen on Windows, `Audio8-VC.exe` already embeds `audio8_vc.ico` in its PE
    resources, which `pywebview` extracts natively via `ExtractIconW`; omitting `icon=`
    there is safest across all .NET Framework versions.
    """
    plat = sys.platform if platform is None else platform
    is_frz = is_frozen() if frozen is None else frozen

    if plat == "win32":
        if is_frz:
            return None
        ico = get_icon_path("ico")
        if ico is not None and ico.suffix.lower() == ".ico" and ico.is_file():
            return ico
        return None

    return get_icon_path("png")


def unblock_windows_bundle_dlls(
    roots: list[Path] | None = None,
    *,
    delete_ads: Callable[[str], bool] | None = None,
) -> int:
    """Strip Windows Mark-of-the-Web (`:Zone.Identifier`) ADS from bundled managed DLLs.

    When users download a Portable ZIP in Edge/Chrome and extract it with Windows Explorer,
    Windows propagates `:Zone.Identifier` (`ZoneId=3`) onto extracted files. `.NET Framework`
    refuses to load managed assemblies (`Python.Runtime.dll`, `Microsoft.Web.WebView2.*.dll`)
    carrying MOTW unless unblocked first.
    """
    if delete_ads is None:
        if sys.platform != "win32":
            return 0
        kernel32 = getattr(getattr(ctypes, "windll", None), "kernel32", None)
        if kernel32 is None:
            return 0

        def _win32_delete_ads(ads_path: str) -> bool:
            try:
                return bool(kernel32.DeleteFileW(ads_path))
            except Exception:
                return False

        delete_ads = _win32_delete_ads

    scan_roots: list[Path] = []
    if roots is not None:
        scan_roots.extend(roots)
    else:
        for candidate in (get_bundle_dir(), get_executable_dir()):
            if candidate not in scan_roots:
                scan_roots.append(candidate)

    targets: list[Path] = []
    seen: set[str] = set()

    def _add_file(path: Path) -> None:
        key = str(path).lower()
        if key not in seen and path.is_file():
            seen.add(key)
            targets.append(path)

    for root in scan_roots:
        if not root.is_dir():
            continue
        for sub in _MANAGED_BUNDLE_SUBDIRS:
            for base in (root / sub, root / "_internal" / sub):
                if base.is_dir():
                    for file_path in base.rglob("*"):
                        if file_path.suffix.lower() in {".dll", ".exe", ".pyd"}:
                            _add_file(file_path)
        for top_item in root.iterdir():
            if top_item.is_file() and top_item.suffix.lower() in {".dll", ".exe"}:
                _add_file(top_item)

    unblocked = 0
    for file_path in targets:
        try:
            if delete_ads(f"{file_path}:Zone.Identifier"):
                unblocked += 1
        except Exception:
            continue
    return unblocked


def has_windows_webview2() -> bool:
    """Return True if Microsoft Edge WebView2 Runtime is registered on Windows."""
    if sys.platform != "win32":
        return True
    try:
        import winreg
    except Exception:
        return True

    roots_and_paths = (
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\EdgeUpdate\Clients"),
        (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\EdgeUpdate\Clients"),
    )
    for root, base in roots_and_paths:
        for guid in _WEBVIEW2_CLIENT_GUIDS:
            try:
                with winreg.OpenKey(root, f"{base}\\{guid}") as key:
                    ver, _ = winreg.QueryValueEx(key, "pv")
                    ver_s = str(ver or "").strip()
                    if ver_s and ver_s != "0.0.0.0":
                        return True
            except OSError:
                continue
    return False


def find_available_port(host: str = "127.0.0.1", preferred: int = 8025) -> int:
    """Return `preferred` if bindable on `host`, else an OS-assigned free ephemeral port."""
    for candidate in (preferred, 0):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((host, candidate))
                return int(sock.getsockname()[1])
        except OSError:
            continue
    raise RuntimeError(f"Unable to bind any local TCP port on {host}")


def wait_for_server_ready(
    url: str,
    timeout_sec: float = 15.0,
    *,
    server_thread: threading.Thread | None = None,
) -> bool:
    """Poll the local `/api/health` endpoint bypassing any system HTTP proxy."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout_sec
    health_url = f"{url.rstrip('/')}/api/health"
    while time.monotonic() < deadline:
        if server_thread is not None and not server_thread.is_alive():
            return False
        try:
            with opener.open(health_url, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.1)
    return False


def _import_associated_voice(state: ServerState, file_arg: str | None) -> str | None:
    """Import a `.a8voice` file passed via Windows double-click file association."""
    if not file_arg:
        return None
    candidate = Path(file_arg)
    if not candidate.is_file() or candidate.suffix.lower() != A8VOICE_EXT:
        return None
    try:
        pack = state.library.import_bytes(candidate.read_bytes(), overwrite=True)
        state.active_voice_id = pack.voice_id
        logger.info("Auto-imported associated voice pack: %s (%s)", pack.voice_id, candidate)
        return pack.voice_id
    except Exception as exc:
        logger.warning("Failed to auto-import %s: %s", candidate, exc)
        return None


def _start_embedded_server(
    app: Any, host: str, port: int
) -> tuple[Any, threading.Thread]:
    import uvicorn

    class _ThreadedServer(uvicorn.Server):
        startup_errors: list[BaseException]

        def install_signal_handlers(self) -> None:
            return None

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_config=None,
        access_log=False,
    )
    server = _ThreadedServer(config=config)
    errors: list[BaseException] = []
    server.startup_errors = errors

    def _run_server() -> None:
        try:
            server.run()
        except BaseException as exc:
            errors.append(exc)
            logger.exception("Embedded uvicorn server crashed")

    thread = threading.Thread(target=_run_server, name="a8vc-uvicorn", daemon=True)
    thread.start()
    return server, thread


def _write_smoke_report(report_path: Path | None, payload: dict[str, Any]) -> None:
    if report_path is None:
        return
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("Failed to write smoke report %s: %s", report_path, exc)


def _attach_window_event(events_obj: Any, attr: str, flag: threading.Event) -> None:
    """Subscribe `flag.set` to a `pywebview` window event if available."""
    ev = getattr(events_obj, attr, None)
    if ev is None:
        return
    try:
        ev += lambda *_args, **_kwargs: flag.set()
    except Exception:
        pass


def _is_window_event_set(events_obj: Any, attr: str, flag: threading.Event) -> bool:
    if flag.is_set():
        return True
    ev = getattr(events_obj, attr, None)
    if ev is not None and hasattr(ev, "is_set"):
        try:
            return bool(ev.is_set())
        except Exception:
            return False
    return False


def launch_desktop(
    cfg: AppConfig,
    *,
    host: str = "127.0.0.1",
    port: int | None = None,
    use_browser: bool = False,
    import_voice_file: str | None = None,
    smoke_test: bool = False,
    smoke_timeout: float = 30.0,
    smoke_report: str | Path | None = None,
) -> int:
    """Start the local Audio8-VC server and open the desktop GUI window."""
    t0 = time.perf_counter()
    log_file = get_logs_dir(desktop=True) / "a8vc-desktop.log"
    ensure_safe_stdio(log_file)
    configure_desktop_logging(log_file)
    set_windows_app_id()

    report_path = Path(smoke_report) if smoke_report else None
    unblocked_dlls = unblock_windows_bundle_dlls()
    if unblocked_dlls > 0:
        logger.info("Removed Mark-of-the-Web (Zone.Identifier) from %d bundled files", unblocked_dlls)

    preferred_port = port if port is not None else cfg.server.port
    bound_port = find_available_port(host, preferred_port)
    url = f"http://{host}:{bound_port}"

    state = ServerState(cfg, desktop_mode=True)
    _import_associated_voice(state, import_voice_file)
    app = create_app(cfg, state=state, desktop_mode=True)

    logger.info(
        "Starting Audio8-VC Desktop v%s at %s (frozen=%s, smoke_test=%s)",
        __version__,
        url,
        is_frozen(),
        smoke_test,
    )
    server, server_thread = _start_embedded_server(app, host, bound_port)
    server_errors: list[BaseException] = list(getattr(server, "startup_errors", []))

    if not wait_for_server_ready(url, timeout_sec=15.0, server_thread=server_thread):
        server_errors = list(getattr(server, "startup_errors", []))
        err_detail = (
            f"{type(server_errors[0]).__name__}: {server_errors[0]}"
            if server_errors
            else f"Timed out waiting for {url}/api/health"
        )
        logger.error("Local backend failed to become ready at %s (%s)", url, err_detail)
        server.should_exit = True
        state.close()
        _write_smoke_report(
            report_path,
            {
                "ok": False,
                "error": f"Backend failed to become ready: {err_detail}",
                "url": url,
                "unblocked_dlls": unblocked_dlls,
                "log_file": str(log_file),
                "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            },
        )
        if not smoke_test:
            show_windows_error_dialog(
                "Audio8-VC 启动失败",
                f"本地语音服务启动失败：{err_detail}\n\n日志文件：{log_file}",
            )
        return 1

    smoke_state: dict[str, Any] = {
        "ok": False,
        "shown": False,
        "loaded": False,
        "renderer": None,
        "url": url,
        "js_title": "",
        "icon_ok": False,
        "has_status_pill": False,
        "unblocked_dlls": unblocked_dlls,
        "icon_passed": None,
        "error": None,
        "log_file": str(log_file),
    }

    try:
        launched_webview = False
        if not use_browser:
            if sys.platform == "win32" and not has_windows_webview2():
                msg = (
                    "Microsoft Edge WebView2 Runtime not detected; "
                    "skipping legacy MSHTML and falling back to system browser."
                )
                logger.warning(msg)
                smoke_state["error"] = msg
            else:
                try:
                    if smoke_test and sys.platform == "win32":
                        # Avoid GPU process hangs on headless Windows VMs without a physical GPU
                        os.environ.setdefault(
                            "WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS",
                            "--disable-gpu --no-first-run",
                        )

                    import webview

                    window = webview.create_window(
                        title=f"Audio8-VC 实时变声器 v{__version__}",
                        url=url,
                        width=1180,
                        height=820,
                        min_size=(880, 620),
                        text_select=True,
                    )
                    shown_event = threading.Event()
                    loaded_event = threading.Event()
                    events_obj = getattr(window, "events", None)
                    _attach_window_event(events_obj, "shown", shown_event)
                    _attach_window_event(events_obj, "loaded", loaded_event)

                    start_kwargs: dict[str, Any] = {}
                    try:
                        start_params = set(inspect.signature(webview.start).parameters.keys())
                    except (TypeError, ValueError):
                        start_params = set()

                    icon_file = select_webview_icon()
                    if icon_file is not None and "icon" in start_params:
                        start_kwargs["icon"] = str(icon_file)
                        smoke_state["icon_passed"] = str(icon_file)

                    if "storage_path" in start_params and "private_mode" in start_params:
                        wv_dir_name = f"webview2-smoke-{os.getpid()}" if smoke_test else "webview2"
                        wv_storage = get_data_dir(desktop=True) / wv_dir_name
                        try:
                            wv_storage.mkdir(parents=True, exist_ok=True)
                            start_kwargs["storage_path"] = str(wv_storage)
                            start_kwargs["private_mode"] = False
                        except Exception:
                            pass

                    if sys.platform == "win32":
                        start_kwargs["gui"] = "edgechromium"

                    stop_probe = threading.Event()
                    probe_done = threading.Event()

                    if smoke_test:
                        def _smoke_probe() -> None:
                            try:
                                deadline = time.monotonic() + max(5.0, float(smoke_timeout))
                                while time.monotonic() < deadline and not stop_probe.is_set():
                                    if _is_window_event_set(events_obj, "shown", shown_event):
                                        smoke_state["shown"] = True
                                        break
                                    time.sleep(0.1)

                                while time.monotonic() < deadline and not stop_probe.is_set():
                                    if _is_window_event_set(events_obj, "loaded", loaded_event):
                                        smoke_state["loaded"] = True
                                        break
                                    time.sleep(0.1)

                                smoke_state["renderer"] = getattr(webview, "renderer", None)

                                if stop_probe.is_set():
                                    return

                                if smoke_state["shown"] and smoke_state["loaded"]:
                                    js_deadline = min(deadline, time.monotonic() + 8.0)
                                    probe_js = (
                                        "JSON.stringify({"
                                        "title: document.title || '',"
                                        "iconOk: !!(document.querySelector('.brand-icon') && "
                                        "document.querySelector('.brand-icon').complete && "
                                        "document.querySelector('.brand-icon').naturalWidth > 0),"
                                        "hasStatus: !!document.getElementById('pill-status')"
                                        "})"
                                    )
                                    last_probe: dict[str, Any] = {}
                                    while time.monotonic() < js_deadline and not stop_probe.is_set():
                                        try:
                                            raw = window.evaluate_js(probe_js)
                                            parsed = (
                                                json.loads(raw)
                                                if isinstance(raw, str)
                                                else (raw if isinstance(raw, dict) else {})
                                            )
                                            if isinstance(parsed, dict):
                                                last_probe = parsed
                                                if (
                                                    "Audio8-VC" in str(parsed.get("title", ""))
                                                    and parsed.get("iconOk")
                                                    and parsed.get("hasStatus")
                                                ):
                                                    break
                                        except Exception as js_exc:
                                            smoke_state["error"] = f"evaluate_js failed: {js_exc}"
                                        time.sleep(0.25)

                                    smoke_state["js_title"] = str(last_probe.get("title", ""))
                                    smoke_state["icon_ok"] = bool(last_probe.get("iconOk", False))
                                    smoke_state["has_status_pill"] = bool(
                                        last_probe.get("hasStatus", False)
                                    )
                                    smoke_state["ok"] = (
                                        "Audio8-VC" in smoke_state["js_title"]
                                        and smoke_state["icon_ok"]
                                        and smoke_state["has_status_pill"]
                                    )
                                    if not smoke_state["ok"] and not smoke_state["error"]:
                                        smoke_state["error"] = (
                                            f"Unexpected WebView DOM state: {last_probe}"
                                        )
                                else:
                                    smoke_state["error"] = (
                                        f"Timed out waiting for WebView window "
                                        f"(shown={smoke_state['shown']}, loaded={smoke_state['loaded']})"
                                    )
                            except Exception as exc:
                                logger.exception("Desktop smoke probe failed")
                                smoke_state["error"] = f"{type(exc).__name__}: {exc}"
                            finally:
                                try:
                                    window.destroy()
                                except Exception:
                                    pass
                                probe_done.set()

                        start_kwargs["func"] = _smoke_probe

                    webview.start(**start_kwargs)
                    stop_probe.set()
                    if smoke_test:
                        probe_done.wait(timeout=3.0)
                    window_was_shown = _is_window_event_set(events_obj, "shown", shown_event)
                    smoke_state["shown"] = smoke_state["shown"] or window_was_shown
                    smoke_state["loaded"] = smoke_state["loaded"] or _is_window_event_set(
                        events_obj, "loaded", loaded_event
                    )
                    if not window_was_shown:
                        logger.error(
                            "pywebview exited without showing a window (renderer=%s); "
                            "treating native GUI launch as failed.",
                            getattr(webview, "renderer", None),
                        )
                        if not smoke_state["error"]:
                            smoke_state["error"] = (
                                "pywebview.start() returned before window.events.shown fired"
                            )
                        launched_webview = False
                    else:
                        launched_webview = True
                except Exception as exc:
                    logger.warning("pywebview unavailable or failed (%s); falling back to browser", exc)
                    smoke_state["error"] = f"{type(exc).__name__}: {exc}"
                    launched_webview = False

        if smoke_test:
            smoke_state["elapsed_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
            _write_smoke_report(report_path, smoke_state)
            logger.info("Desktop smoke-test result: %s", json.dumps(smoke_state, ensure_ascii=False))
            return 0 if smoke_state["ok"] else 1

        if not launched_webview:
            logger.info("Opening Audio8-VC control panel in default browser: %s", url)
            try:
                opened = webbrowser.open(url)
                if not opened:
                    logger.warning("webbrowser.open(%s) returned False", url)
                print(f"Audio8-VC is running at {url}. Press Ctrl+C to quit.")
                while server_thread.is_alive():
                    time.sleep(0.5)
            except KeyboardInterrupt:
                pass
    finally:
        logger.info("Shutting down Audio8-VC Desktop...")
        state.close()
        server.should_exit = True
        server_thread.join(timeout=5.0)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="a8vc-desktop",
        description="Launch the Audio8-VC Windows/Cross-Platform Desktop Voice Changer",
    )
    parser.add_argument("--version", action="version", version=f"Audio8-VC {__version__}")
    parser.add_argument("--config", type=str, default=None, help="Optional TOML config file")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Local bind host")
    parser.add_argument("--port", type=int, default=None, help="Preferred local port")
    parser.add_argument("--voice", type=str, default=None, help="Default active voice_id")
    parser.add_argument(
        "--portable",
        action="store_true",
        help="Store models, voices, and logs inside ./data next to the executable",
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Open in the default web browser instead of a native WebView window",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Launch the real desktop GUI window, verify UI load, then auto-close and exit",
    )
    parser.add_argument(
        "--smoke-timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds for --smoke-test window load",
    )
    parser.add_argument(
        "--smoke-report",
        type=str,
        default=None,
        help="Optional JSON output file path for --smoke-test results",
    )
    parser.add_argument(
        "voice_file",
        nargs="?",
        default=None,
        help="Optional .a8voice file to auto-import and activate on startup",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    multiprocessing.freeze_support()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    is_smoke = "--smoke-test" in raw_argv
    if is_smoke:
        os.environ["A8VC_NO_DIALOG"] = "1"

    try:
        early_log = get_logs_dir(desktop=True) / "a8vc-desktop.log"
    except Exception:
        early_log = None
    ensure_safe_stdio(early_log)

    parser = _build_parser()
    args = parser.parse_args(raw_argv)

    if args.portable:
        os.environ["A8VC_PORTABLE"] = "1"

    try:
        cfg = load_config(args.config, desktop=True)
        if args.voice:
            cfg.server.default_voice = args.voice

        return launch_desktop(
            cfg,
            host=args.host,
            port=args.port,
            use_browser=args.browser,
            import_voice_file=args.voice_file,
            smoke_test=args.smoke_test,
            smoke_timeout=args.smoke_timeout,
            smoke_report=args.smoke_report,
        )
    except Exception as exc:
        tb = traceback.format_exc()
        log_path = early_log or (Path.home() / "a8vc-desktop-crash.log")
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(f"\n[FATAL] Audio8-VC Desktop startup error:\n{tb}\n")
        except Exception:
            pass
        if args.smoke_report:
            _write_smoke_report(
                Path(args.smoke_report),
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "log_file": str(log_path),
                },
            )
        if not args.smoke_test:
            show_windows_error_dialog(
                "Audio8-VC 启动失败",
                f"启动过程发生未预期错误：{type(exc).__name__}: {exc}\n\n日志文件：{log_path}",
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
