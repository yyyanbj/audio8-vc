"""CLI runner for the Audio8-VC extensible lint suite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from a8vc.lint.core import default_registry, discover_repo_root, format_report_text


def _is_utf8_stream(stream: Any) -> bool:
    enc = getattr(stream, "encoding", None) or ""
    return enc.lower().replace("_", "-") in {"utf-8", "utf8"}


def _emit_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=not _is_utf8_stream(sys.stdout), indent=2))


def build_lint_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="a8vc-lint",
        description="Run Audio8-VC engineering, packaging, CI, runtime, and web lint rules.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Repository root directory (auto-detected if omitted)",
    )
    parser.add_argument(
        "--select",
        type=str,
        default="",
        help="Comma-separated rule IDs or prefixes to run (e.g. CI,PKG003,DEP)",
    )
    parser.add_argument(
        "--ignore",
        type=str,
        default="",
        help="Comma-separated rule IDs or prefixes to skip",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="",
        help="Comma-separated rule categories (ci,packaging,deps,runtime,server,web,cli,voices)",
    )
    parser.add_argument(
        "--list-rules",
        action="store_true",
        help="List all registered lint rules and exit",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as failures",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output machine-readable JSON",
    )
    return parser


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def run_lint_cli(args: argparse.Namespace) -> int:
    select = _split_csv(args.select)
    ignore = _split_csv(args.ignore)
    categories = _split_csv(args.category)

    if args.list_rules:
        rules = default_registry.list_rules(
            select=select or None,
            ignore=ignore or None,
            categories=categories or None,
        )
        if args.as_json:
            _emit_json([r.to_metadata_dict() for r in rules])
        else:
            print(f"Registered Audio8-VC lint rules ({len(rules)}):")
            for r in rules:
                print(f"  {r.rule_id:<7} [{r.category:<9}] {r.title}")
                print(f"          {r.description}")
                print(f"          history: {r.history}")
        return 0

    root = Path(args.root).resolve() if args.root else discover_repo_root()
    report = default_registry.run(
        root,
        select=select or None,
        ignore=ignore or None,
        categories=categories or None,
    )
    if args.as_json:
        _emit_json(report.to_dict(strict=args.strict))
    else:
        print(format_report_text(report, strict=args.strict))
    return 0 if report.ok(strict=args.strict) else 1


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure") and not _is_utf8_stream(stream):
            try:
                stream.reconfigure(errors="replace")
            except Exception:
                pass
    parser = build_lint_parser()
    args = parser.parse_args(argv)
    return run_lint_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
