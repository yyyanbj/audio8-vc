"""PyInstaller windowed entry point for `Audio8-VC.exe`."""

from __future__ import annotations

import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from a8vc.desktop import main

    raise SystemExit(main(sys.argv[1:]))
