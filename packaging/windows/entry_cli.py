"""PyInstaller console CLI entry point for `a8vc.exe`."""

from __future__ import annotations

import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from a8vc.cli import main

    raise SystemExit(main(sys.argv[1:]))
