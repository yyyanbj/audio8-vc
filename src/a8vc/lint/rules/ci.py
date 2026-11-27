"""GitHub Actions and CI workflow lint rules (CIxxx)."""

from __future__ import annotations

import re

from a8vc.lint.core import LintContext, LintFinding, register_rule


@register_rule(
    "CI001",
    title="PowerShell scripts must use hashtable splatting in GitHub Actions",
    category="ci",
    description=(
        "Invoking build_windows.ps1 with PowerShell array splatting (@array) fails to bind "
        "named parameters like -Provider; always use hashtable splatting ($params = @{ ... })."
    ),
    history="GitHub Actions Windows build failed in v0.1.x before commit 97e1621 due to array splatting.",
)
def check_ps_hashtable_splatting(ctx: LintContext) -> list[LintFinding]:
    findings: list[LintFinding] = []
    for rel in ctx.glob(".github/workflows/*.yml") + ctx.glob(".github/workflows/*.yaml"):
        text = ctx.read_text(rel) or ""
        for m_call in re.finditer(r"([\w.-]+\.ps1)\s+@(\w+)", text):
            script_name = m_call.group(1)
            var_name = m_call.group(2)
            array_assign = re.search(rf"\${re.escape(var_name)}\s*=\s*@\(", text)
            hash_assign = re.search(rf"\${re.escape(var_name)}\s*=\s*@\{{", text)
            if array_assign or not hash_assign:
                line = ctx.find_line(text, rf"{re.escape(script_name)}\s+@{re.escape(var_name)}")
                findings.append(
                    LintFinding(
                        rule_id="CI001",
                        path=rel,
                        line=line,
                        message=(
                            f"{script_name} uses @{var_name} without a PowerShell hashtable "
                            f"`${var_name} = @{{ ... }}` definition."
                        ),
                        hint=(
                            f"Define `${var_name} = @{{ ... }}` and splat with "
                            f"`& .\\...\\{script_name} @{var_name}`."
                        ),
                    )
                )
    return findings


@register_rule(
    "CI002",
    title="Release workflow must run lint, ruff, pytest, and frozen self-check",
    category="ci",
    description=(
        "The Windows build/release workflow must execute the Audio8-VC lint suite, ruff, "
        "pytest, and `a8vc.exe self-check` on the installed build before publishing artifacts."
    ),
    history="v0.2.0 released without a lint gate or frozen self-check, shipping broken sherpa-onnx and WebSocket packaging.",
)
def check_ci_gates(ctx: LintContext) -> list[LintFinding]:
    rel = ".github/workflows/build-windows.yml"
    text = ctx.read_text(rel)
    if text is None:
        return [
            LintFinding(
                rule_id="CI002",
                path=rel,
                message="Missing Windows release workflow file.",
                hint="Keep .github/workflows/build-windows.yml with lint, test, and self-check gates.",
            )
        ]

    findings: list[LintFinding] = []
    required_checks = [
        (r"(?:a8vc\s+lint|scripts/lint\.py)", "project lint suite (`a8vc lint` or `scripts/lint.py`)"),
        (r"ruff\s+check", "`ruff check`"),
        (r"pytest", "`pytest`"),
        (r"\$cli\s+self-check", "installed Windows executable smoke check (`& $cli self-check`)"),
    ]
    for pattern, desc in required_checks:
        if not re.search(pattern, text):
            findings.append(
                LintFinding(
                    rule_id="CI002",
                    path=rel,
                    message=f"Windows release workflow is missing required gate: {desc}.",
                    hint=f"Add a CI step in {rel} that executes {desc} before releasing.",
                )
            )
    return findings


@register_rule(
    "CI003",
    title="Release workflow must verify CPU, DirectML, CUDA flavors and built-in voices",
    category="ci",
    description=(
        "Windows installer smoke tests must verify configured_provider, asr_provider=='cpu', "
        "DmlExecutionProvider/CUDAExecutionProvider presence, and >= 15 bundled built-in voices."
    ),
    history="Ensures multi-provider releases (97e1621) and bundled built-in voices (71e6636, 96b9cf7) stay verified.",
)
def check_ci_provider_and_voice_verification(ctx: LintContext) -> list[LintFinding]:
    rel = ".github/workflows/build-windows.yml"
    text = ctx.read_text(rel)
    if text is None:
        return []

    findings: list[LintFinding] = []
    markers = [
        ("DmlExecutionProvider", "DirectML provider check"),
        ("CUDAExecutionProvider", "CUDA provider check"),
        ("asr_provider", "ASR CPU provider check"),
        ("voice builtins --json", "bundled built-in voice catalog check"),
    ]
    for token, label in markers:
        if token not in text:
            findings.append(
                LintFinding(
                    rule_id="CI003",
                    path=rel,
                    message=f"Windows release workflow is missing {label} (`{token}`).",
                    hint=f"Verify {label} during the installed Windows smoke test in {rel}.",
                )
            )
    return findings


@register_rule(
    "CI004",
    title="CI and release workflows must run Win10 (windows-2022) and Win11 (windows-2025) CPU matrix gates",
    category="ci",
    description=(
        "Both `.github/workflows/ci.yml` and `.github/workflows/build-windows.yml` must verify "
        "built Windows CPU packages across `windows-2022` (Win10 baseline) and `windows-2025` "
        "(Win11 24H2 baseline) via `verify_windows_e2e.ps1`, and the GitHub Release job must "
        "depend on the Windows matrix verification job."
    ),
    history=(
        "Added pre-release Win10/Win11 Windows CPU environment matrix testing after v0.2.0 "
        "shipped runtime and browser WebSocket regressions."
    ),
)
def check_ci_win10_win11_matrix_gates(ctx: LintContext) -> list[LintFinding]:
    findings: list[LintFinding] = []
    for rel in (".github/workflows/ci.yml", ".github/workflows/build-windows.yml"):
        text = ctx.read_text(rel)
        if text is None:
            findings.append(
                LintFinding(
                    rule_id="CI004",
                    path=rel,
                    message=f"Missing required GitHub Actions workflow `{rel}`.",
                )
            )
            continue
        for token, label in (
            ("windows-2022", "Win10 compatibility baseline runner (`windows-2022`)"),
            ("windows-2025", "Win11 24H2 compatibility baseline runner (`windows-2025`)"),
            ("verify_windows_e2e.ps1", "Windows E2E matrix script (`verify_windows_e2e.ps1`)"),
            ("actions/cache@v4", "model cache step (`actions/cache@v4`)"),
        ):
            if token not in text:
                findings.append(
                    LintFinding(
                        rule_id="CI004",
                        path=rel,
                        message=f"Workflow `{rel}` is missing {label}.",
                        hint=f"Include `{token}` in `{rel}` to maintain the Win10/Win11 CPU E2E matrix.",
                    )
                )

    rel_build = ".github/workflows/build-windows.yml"
    build_text = ctx.read_text(rel_build) or ""
    if build_text and not re.search(
        r"release:\s*\n(?:[ \t]+.*\n)*?[ \t]+needs:\s*\[[^\]]*verify-windows-cpu-matrix[^\]]*\]",
        build_text,
    ):
        findings.append(
            LintFinding(
                rule_id="CI004",
                path=rel_build,
                line=ctx.find_line(build_text, r"^\s*release:\s*$"),
                message="The `release` job must depend on `verify-windows-cpu-matrix` before publishing.",
                hint="Set `needs: [build-windows, verify-windows-cpu-matrix]` on the `release` job.",
            )
        )
    return findings


@register_rule(
    "CI005",
    title="Windows E2E harness must verify installer, portable ZIP, real models, and headless browser /ws/vc",
    category="ci",
    description=(
        "`packaging/windows/verify_windows_e2e.ps1` and `scripts/e2e_verify.py` must verify "
        "Inno Setup install/reinstall/uninstall, Portable ZIP isolation, unified `onnxruntime.dll`, "
        "CP1252 console safety, real-model TTS/ASR/VAD, and headless Chromium/Edge fake-mic `/ws/vc`."
    ),
    history=(
        "Ensures every release passes full real-model and real-browser voice changing end-to-end "
        "on both Win10 and Win11 Windows CPU environments."
    ),
)
def check_windows_e2e_harness(ctx: LintContext) -> list[LintFinding]:
    findings: list[LintFinding] = []

    ps1_rel = "packaging/windows/verify_windows_e2e.ps1"
    ps1_text = ctx.read_text(ps1_rel)
    if ps1_text is None:
        findings.append(
            LintFinding(
                rule_id="CI005",
                path=ps1_rel,
                message=f"Missing Windows E2E verification script `{ps1_rel}`.",
            )
        )
    else:
        for token, desc in (
            ("-setup.exe", "Inno Setup installer verification"),
            ("-portable.zip", "Portable ZIP verification"),
            ("portable.flag", "portable mode marker verification"),
            ("onnxruntime.dll", "bundled onnxruntime.dll unification check"),
            ("__stale_upgrade_sentinel__.dll", "[InstallDelete] upgrade cleanup verification"),
            ("unins000.exe", "silent uninstaller verification"),
            ("e2e_verify.py", "Python real-model & browser E2E runner invocation"),
        ):
            if token not in ps1_text:
                findings.append(
                    LintFinding(
                        rule_id="CI005",
                        path=ps1_rel,
                        message=f"{ps1_rel} is missing {desc} (`{token}`).",
                    )
                )

    py_rel = "scripts/e2e_verify.py"
    py_text = ctx.read_text(py_rel)
    if py_text is None:
        findings.append(
            LintFinding(
                rule_id="CI005",
                path=py_rel,
                message=f"Missing real-model & browser E2E verifier `{py_rel}`.",
            )
        )
    else:
        for token, desc in (
            ("--use-fake-device-for-media-stream", "headless browser fake microphone flag"),
            ("--use-file-for-fake-audio-capture=", "headless browser real-speech fake audio capture"),
            ("AudioServiceSandbox", "audio service sandbox bypass for fake audio capture"),
            ("/ws/vc", "WebSocket real-time voice changer verification"),
            ("/api/tts/stream", "streaming TTS API verification"),
            ("/icon.svg", "brand icon HTTP verification"),
            ("cp1252", "legacy Windows CP1252 console codepage verification"),
        ):
            if token not in py_text:
                findings.append(
                    LintFinding(
                        rule_id="CI005",
                        path=py_rel,
                        message=f"{py_rel} is missing {desc} (`{token}`).",
                    )
                )

    return findings

