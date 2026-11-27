"""Windows Desktop GUI launcher & pywebview reliability lint rules (DSKxxx)."""

from __future__ import annotations

import re

from a8vc.lint.core import LintContext, LintFinding, register_rule


@register_rule(
    "DSK001",
    title="desktop.py must guard Windows pywebview icon format, MOTW unblocking, WebView2 check, and shown-event fallback",
    category="desktop",
    description=(
        "On Windows, `pywebview` passes `icon=` directly to .NET `System.Drawing.Icon(path)`, "
        "which throws `System.ArgumentException` inside the WinForms STA thread if given a `.png` "
        "file, crashing `Audio8-VC.exe` before any window opens. `desktop.py` must never pass `.png` "
        "on Windows, must unblock `:Zone.Identifier` (Mark of the Web) on bundled DLLs before "
        "importing `webview`, must check WebView2 availability, and must verify `window.events.shown` "
        "before skipping browser fallback."
    ),
    history=(
        "v0.2.1 passed `get_icon_path('png')` to `webview.start(icon=...)`, causing `Audio8-VC.exe` "
        "to fail to open on every Windows 10/11 laptop; extracted portable ZIPs also carry NTFS "
        "`:Zone.Identifier` which blocks .NET Framework from loading `Python.Runtime.dll`."
    ),
)
def check_desktop_windows_gui_safety(ctx: LintContext) -> list[LintFinding]:
    rel = "src/a8vc/desktop.py"
    text = ctx.read_text(rel)
    if text is None:
        return [
            LintFinding(
                rule_id="DSK001",
                path=rel,
                message="Missing src/a8vc/desktop.py.",
            )
        ]

    findings: list[LintFinding] = []

    # Catch the exact v0.2.1 regression: unconditionally preferring PNG before ICO
    if re.search(r'get_icon_path\(\s*["\']png["\']\s*\)\s+or\s+get_icon_path\(\s*["\']ico["\']\s*\)', text):
        findings.append(
            LintFinding(
                rule_id="DSK001",
                path=rel,
                line=ctx.find_line(text, r'get_icon_path\(\s*["\']png["\']\s*\)'),
                message=(
                    "desktop.py passes a PNG icon to `webview.start(icon=...)` on Windows; "
                    ".NET `System.Drawing.Icon` rejects `.png` and crashes WinForms startup."
                ),
                hint=(
                    "Use `select_webview_icon()` which returns `None` for frozen Windows builds "
                    "(extracting the embedded EXE `.ico`) and never returns `.png` on `win32`."
                ),
            )
        )

    for token, desc in (
        ("def select_webview_icon", "Windows-safe `select_webview_icon()` helper"),
        ("def unblock_windows_bundle_dlls", "Mark-of-the-Web (`:Zone.Identifier`) DLL unblocker"),
        ("Zone.Identifier", "NTFS `:Zone.Identifier` ADS stripping for portable ZIPs"),
        ("def has_windows_webview2", "Windows Edge WebView2 Runtime registry check"),
        ("shown", "window `shown` event verification before skipping browser fallback"),
    ):
        if token not in text:
            findings.append(
                LintFinding(
                    rule_id="DSK001",
                    path=rel,
                    message=f"desktop.py is missing {desc} (`{token}`).",
                    hint=f"Keep `{token}` in {rel} to guarantee `Audio8-VC.exe` opens reliably on Windows.",
                )
            )

    return findings


@register_rule(
    "DSK002",
    title="Windows GUI entry point must show error dialogs on fatal crashes and desktop.py must support --smoke-test",
    category="desktop",
    description=(
        "Because `Audio8-VC.exe` is built with `console=False`, silent startup exits leave users "
        "with no visible feedback. `entry_desktop.py` and `desktop.py` must log tracebacks and "
        "display a Windows `MessageBoxW` error dialog on fatal startup errors (unless suppressed "
        "in CI), and `desktop.py` must expose `--smoke-test` + `--smoke-report` for automated GUI CI."
    ),
    history=(
        "v0.2.1 CI only checked the PE 'MZ' header of `Audio8-VC.exe` without launching it, "
        "allowing a 100% reproducible Windows GUI startup crash to ship."
    ),
)
def check_desktop_crash_reporting_and_smoke_mode(ctx: LintContext) -> list[LintFinding]:
    findings: list[LintFinding] = []

    entry_rel = "packaging/windows/entry_desktop.py"
    entry_text = ctx.read_text(entry_rel)
    if entry_text is None:
        findings.append(
            LintFinding(
                rule_id="DSK002",
                path=entry_rel,
                message="Missing packaging/windows/entry_desktop.py.",
            )
        )
    else:
        for token, desc in (
            ("MessageBoxW", "native Windows error dialog on early startup crash"),
            ("traceback", "traceback logging on early startup crash"),
        ):
            if token not in entry_text:
                findings.append(
                    LintFinding(
                        rule_id="DSK002",
                        path=entry_rel,
                        message=f"{entry_rel} is missing {desc} (`{token}`).",
                    )
                )

    desktop_rel = "src/a8vc/desktop.py"
    desktop_text = ctx.read_text(desktop_rel) or ""
    for token, desc in (
        ("MessageBoxW", "Windows error dialog on fatal desktop startup error"),
        ("--smoke-test", "CLI `--smoke-test` mode for launching real `Audio8-VC.exe` in CI"),
        ("--smoke-report", "CLI `--smoke-report` JSON output for `console=False` CI verification"),
    ):
        if token not in desktop_text:
            findings.append(
                LintFinding(
                    rule_id="DSK002",
                    path=desktop_rel,
                    message=f"{desktop_rel} is missing {desc} (`{token}`).",
                )
            )

    return findings
