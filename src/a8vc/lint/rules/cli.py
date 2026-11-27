"""CLI and Windows console compatibility lint rules (CLIxxx)."""

from __future__ import annotations

import ast

from a8vc.lint.core import LintContext, LintFinding, register_rule


@register_rule(
    "CLI001",
    title="CLI must configure non-UTF-8 Windows stdio safely and avoid raw ensure_ascii=False JSON prints",
    category="cli",
    description=(
        "Windows PowerShell / piped stdout defaults to cp1252/gbk codepages; printing CJK voice names "
        "or `json.dumps(..., ensure_ascii=False)` directly raises UnicodeEncodeError."
    ),
    history="v0.2.0 Windows CI failed on `a8vc.exe voice list` / `voice builtins --json` (fixed in commit 96b9cf7).",
)
def check_windows_stdio_safety(ctx: LintContext) -> list[LintFinding]:
    rel = "src/a8vc/cli.py"
    tree = ctx.parse_python(rel)
    if tree is None:
        return [LintFinding(rule_id="CLI001", path=rel, message="Missing src/a8vc/cli.py.")]

    findings: list[LintFinding] = []
    fn_map = {
        n.name: n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
    }

    if "_configure_stdio" not in fn_map:
        findings.append(
            LintFinding(
                rule_id="CLI001",
                path=rel,
                message="src/a8vc/cli.py must define `_configure_stdio()` to harden non-UTF-8 stdio streams.",
                hint="Reconfigure non-UTF-8 stdout/stderr with `errors='replace'` at CLI startup.",
            )
        )

    main_fn = fn_map.get("main")
    if main_fn is not None:
        calls_configure = any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "_configure_stdio"
            for n in ast.walk(main_fn)
        )
        if not calls_configure:
            findings.append(
                LintFinding(
                    rule_id="CLI001",
                    path=rel,
                    line=main_fn.lineno,
                    message="`main()` must invoke `_configure_stdio()` before printing any output.",
                    hint="Call `_configure_stdio()` at the top of `main()`.",
                )
            )

    # Forbid hardcoded `json.dumps(..., ensure_ascii=False)` anywhere in cli.py
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "dumps"
        ):
            for kw in node.keywords:
                if (
                    kw.arg == "ensure_ascii"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is False
                ):
                    findings.append(
                        LintFinding(
                            rule_id="CLI001",
                            path=rel,
                            line=node.lineno,
                            message="Hardcoded `json.dumps(..., ensure_ascii=False)` breaks on Windows cp1252 pipes.",
                            hint="Use `_print_json(payload)` which sets `ensure_ascii=not _is_utf8_stream(sys.stdout)`.",
                        )
                    )

    return findings


@register_rule(
    "CLI002",
    title="CLI must expose `self-check` and `lint` subcommands",
    category="cli",
    description=(
        "Build scripts and developers rely on `a8vc self-check` (runtime smoke test) and "
        "`a8vc lint` (extensible repo lint suite)."
    ),
    history="Added after v0.2.0 to enforce pre-release verification in both dev and CI.",
)
def check_cli_verification_subcommands(ctx: LintContext) -> list[LintFinding]:
    rel = "src/a8vc/cli.py"
    text = ctx.read_text(rel) or ""
    findings: list[LintFinding] = []
    for subcmd in ("self-check", "lint"):
        if f'"{subcmd}"' not in text and f"'{subcmd}'" not in text:
            findings.append(
                LintFinding(
                    rule_id="CLI002",
                    path=rel,
                    message=f"CLI is missing `{subcmd}` subcommand.",
                    hint=f"Register `{subcmd}` in `_build_parser()` and dispatch it in `main()`.",
                )
            )
    return findings
