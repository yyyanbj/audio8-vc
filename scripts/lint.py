#!/usr/bin/env python3
"""Convenience entrypoint for the Audio8-VC extensible lint suite.

Examples:
    uv run python scripts/lint.py
    uv run python scripts/lint.py --list-rules
    uv run python scripts/lint.py --category ci,packaging
    uv run python scripts/lint.py --select DEP,RT001 --json
"""

from __future__ import annotations

import sys

from a8vc.lint.runner import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
