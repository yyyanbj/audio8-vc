"""Windows & cross-platform Desktop GUI launcher for Audio8-VC.

Starts the local FastAPI + WebSocket backend on an available localhost port and hosts the
control panel inside a lightweight native WebView2 window (`pywebview`), with graceful
fallback to the system browser.
"""

from __future__ import annotations

import argparse
import ctypes
import io
import logging
import multiprocessing
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any

from a8vc import __version__
from a8vc.config import AppConfig, load_config
from a8vc.paths import get_icon_path, get_logs_dir, is_frozen
from a8vc.server.app import ServerState, create_app
from a8vc.voices.format import A8VOICE_EXT

logger = logging.getLogger("a8vc.desktop")

APP_USER_MODEL_ID = "Edge0.Audio8VC.Desktop"


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
    thread = threading.Thread(target=server.run, name="a8vc-uvicorn", daemon=True)
    thread.start()
    return server, thread


def launch_desktop(
    cfg: AppConfig,
    *,
    host: str = "127.0.0.1",
    port: int | None = None,
    use_browser: bool = False,
    import_voice_file: str | None = None,
) -> int:
    """Start the local Audio8-VC server and open the desktop GUI window."""
    log_file = get_logs_dir(desktop=True) / "a8vc-desktop.log"
    ensure_safe_stdio(log_file)
    configure_desktop_logging(log_file)
    set_windows_app_id()

    preferred_port = port if port is not None else cfg.server.port
    bound_port = find_available_port(host, preferred_port)
    url = f"http://{host}:{bound_port}"

    state = ServerState(cfg, desktop_mode=True)
    _import_associated_voice(state, import_voice_file)
    app = create_app(cfg, state=state, desktop_mode=True)

    logger.info("Starting Audio8-VC Desktop v%s at %s (frozen=%s)", __version__, url, is_frozen())
    server, server_thread = _start_embedded_server(app, host, bound_port)

    if not wait_for_server_ready(url, timeout_sec=15.0, server_thread=server_thread):
        logger.error("Local backend failed to become ready at %s", url)
        server.should_exit = True
        state.close()
        return 1

    try:
        launched_webview = False
        if not use_browser:
            try:
                import webview

                webview.create_window(
                    title=f"Audio8-VC 实时变声器 v{__version__}",
                    url=url,
                    width=1180,
                    height=820,
                    min_size=(880, 620),
                    text_select=True,
                )
                launched_webview = True
                start_kwargs: dict[str, Any] = {}
                icon_file = get_icon_path("png") or get_icon_path("ico")
                if icon_file is not None:
                    import inspect

                    if "icon" in inspect.signature(webview.start).parameters:
                        start_kwargs["icon"] = str(icon_file)
                if sys.platform == "win32":
                    try:
                        webview.start(gui="edgechromium", **start_kwargs)
                    except Exception as exc:
                        logger.warning("WebView2 (edgechromium) start failed, retrying default: %s", exc)
                        webview.start(**start_kwargs)
                else:
                    webview.start(**start_kwargs)
            except Exception as exc:
                logger.warning("pywebview unavailable or failed (%s); falling back to browser", exc)
                launched_webview = False

        if not launched_webview:
            logger.info("Opening Audio8-VC control panel in default browser: %s", url)
            webbrowser.open(url)
            print(f"Audio8-VC is running at {url}. Press Ctrl+C to quit.")
            try:
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
        "voice_file",
        nargs="?",
        default=None,
        help="Optional .a8voice file to auto-import and activate on startup",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    multiprocessing.freeze_support()
    parser = _build_parser()
    args = parser.parse_args(argv)

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


if __name__ == "__main__":
    raise SystemExit(main())
