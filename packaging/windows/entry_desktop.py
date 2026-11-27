"""PyInstaller windowed entry point for `Audio8-VC.exe`."""

from __future__ import annotations

import ctypes
import multiprocessing
import os
import sys
import tempfile
import traceback
from pathlib import Path


def _report_fatal_startup_error(exc: BaseException) -> None:
    tb = traceback.format_exc()
    log_path = Path(tempfile.gettempdir()) / "a8vc-desktop-crash.log"
    try:
        local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
        if local_app_data:
            log_path = Path(local_app_data) / "Audio8-VC" / "logs" / "a8vc-desktop.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"\n[FATAL] Audio8-VC.exe early startup failure:\n{tb}\n")
    except Exception:
        pass

    no_dialog = (
        "--smoke-test" in sys.argv
        or os.environ.get("A8VC_NO_DIALOG", "").strip().lower() in {"1", "true", "yes"}
    )
    if sys.platform == "win32" and not no_dialog:
        try:
            ctypes.windll.user32.MessageBoxW(
                0,
                f"Audio8-VC 启动失败：{type(exc).__name__}: {exc}\n\n日志：{log_path}",
                "Audio8-VC 启动失败",
                0x00010010,
            )
        except Exception:
            pass


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        from a8vc.desktop import main

        raise SystemExit(main(sys.argv[1:]))
    except SystemExit:
        raise
    except BaseException as exc:
        _report_fatal_startup_error(exc)
        raise SystemExit(1) from exc
