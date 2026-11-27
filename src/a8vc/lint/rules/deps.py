"""Dependency pinning, lockfile completeness, and ABI alignment lint rules (DEPxxx)."""

from __future__ import annotations

import re

from a8vc.lint.core import LintContext, LintFinding, register_rule


def _parse_exact_pin(deps: list[str], pkg_name: str) -> str | None:
    """Return exact version `X.Y.Z` if `pkg_name==X.Y.Z` is in `deps`, else None."""
    norm_target = pkg_name.lower().replace("_", "-")
    for raw in deps:
        spec = raw.split(";", 1)[0].strip()
        m = re.match(r"^([A-Za-z0-9_.-]+)\s*(.*)$", spec)
        if not m:
            continue
        name = m.group(1).lower().replace("_", "-")
        constraint = m.group(2).strip()
        if name == norm_target:
            m_eq = re.match(r"^==\s*([0-9]+(?:\.[0-9]+)+)$", constraint)
            if m_eq:
                return m_eq.group(1)
            return ""
    return None


@register_rule(
    "DEP001",
    title="onnxruntime, sherpa-onnx, sherpa-onnx-core, and build_windows.ps1 must have ABI-aligned exact pins",
    category="deps",
    description=(
        "On Windows, `onnxruntime.dll` is process-global by basename. `onnxruntime`, `sherpa-onnx`, "
        "`sherpa-onnx-core`, and `build_windows.ps1` `$OrtVersion` must all be exact-pinned and aligned."
    ),
    history=(
        "v0.2.0 had onnxruntime<1.24 alongside sherpa-onnx 1.13.8 (ORT 1.24.x), and omitted explicit "
        "sherpa-onnx-core pinning causing uv to drop native libraries."
    ),
)
def check_ort_sherpa_abi_pins(ctx: LintContext) -> list[LintFinding]:
    findings: list[LintFinding] = []
    pyproject = ctx.parse_toml("pyproject.toml")
    if pyproject is None:
        return [
            LintFinding(
                rule_id="DEP001",
                path="pyproject.toml",
                message="Missing pyproject.toml.",
            )
        ]

    project = pyproject.get("project", {})
    base_deps = list(project.get("dependencies", []))
    asr_deps = list(project.get("optional-dependencies", {}).get("asr", []))

    ort_pin = _parse_exact_pin(base_deps, "onnxruntime")
    if not ort_pin:
        findings.append(
            LintFinding(
                rule_id="DEP001",
                path="pyproject.toml",
                message="`onnxruntime` in [project].dependencies must be exact-pinned (`onnxruntime==X.Y.Z`).",
                hint="Pin `onnxruntime==1.24.4` to match `sherpa-onnx-core` and Windows DML/CUDA wheels.",
            )
        )

    sherpa_pin = _parse_exact_pin(asr_deps, "sherpa-onnx")
    sherpa_core_pin = _parse_exact_pin(asr_deps, "sherpa-onnx-core")
    if not sherpa_pin:
        findings.append(
            LintFinding(
                rule_id="DEP001",
                path="pyproject.toml",
                message="`sherpa-onnx` in [project.optional-dependencies].asr must be exact-pinned (`sherpa-onnx==X.Y.Z`).",
                hint="Pin `sherpa-onnx==1.13.0` (matching the ORT ABI version).",
            )
        )
    if not sherpa_core_pin:
        findings.append(
            LintFinding(
                rule_id="DEP001",
                path="pyproject.toml",
                message=(
                    "`sherpa-onnx-core` must be explicitly exact-pinned in [project.optional-dependencies].asr."
                ),
                hint=(
                    "sherpa-onnx 1.13.x declares `Dynamic: requires-dist`, causing uv to drop transitive "
                    "`sherpa-onnx-core` unless declared explicitly."
                ),
            )
        )
    if sherpa_pin and sherpa_core_pin and sherpa_pin != sherpa_core_pin:
        findings.append(
            LintFinding(
                rule_id="DEP001",
                path="pyproject.toml",
                message=(
                    f"Version mismatch: sherpa-onnx=={sherpa_pin} != sherpa-onnx-core=={sherpa_core_pin}."
                ),
                hint="Pin `sherpa-onnx` and `sherpa-onnx-core` to the exact same version.",
            )
        )

    ps1_rel = "packaging/windows/build_windows.ps1"
    ps1_text = ctx.read_text(ps1_rel)
    if ps1_text and ort_pin:
        m = re.search(r'\$OrtVersion\s*=\s*"([^"]+)"', ps1_text)
        if not m:
            findings.append(
                LintFinding(
                    rule_id="DEP001",
                    path=ps1_rel,
                    message="Could not find `$OrtVersion = \"...\"` in build_windows.ps1.",
                    hint=f'Set `$OrtVersion = "{ort_pin}"` to match pyproject.toml.',
                )
            )
        elif m.group(1) != ort_pin:
            findings.append(
                LintFinding(
                    rule_id="DEP001",
                    path=ps1_rel,
                    line=ctx.find_line(ps1_text, r"\$OrtVersion\s*="),
                    message=(
                        f"build_windows.ps1 `$OrtVersion = \"{m.group(1)}\"` does not match "
                        f"pyproject.toml `onnxruntime=={ort_pin}`."
                    ),
                    hint=f'Update `$OrtVersion` in {ps1_rel} to `"{ort_pin}"`.',
                )
            )
    return findings


@register_rule(
    "DEP002",
    title="uv.lock must lock sherpa-onnx-core alongside sherpa-onnx and match pinned ORT version",
    category="deps",
    description=(
        "Verifies `uv.lock` actually contains `sherpa-onnx-core` at the same version as `sherpa-onnx` "
        "and binds it into the `asr`/`all`/`windows` extras, preventing missing native DLL/.so crashes."
    ),
    history=(
        "uv's universal lockfile omitted sherpa-onnx-core in v0.2.0 due to `Dynamic: requires-dist`, "
        "causing ImportError (`libonnxruntime.so` / `DLL load failed`) on startup."
    ),
)
def check_uv_lock_completeness(ctx: LintContext) -> list[LintFinding]:
    findings: list[LintFinding] = []
    lock = ctx.parse_toml("uv.lock")
    pyproject = ctx.parse_toml("pyproject.toml")
    if lock is None:
        return [
            LintFinding(
                rule_id="DEP002",
                path="uv.lock",
                message="Missing uv.lock file.",
                hint="Run `uv lock` to generate uv.lock.",
            )
        ]

    pkgs_by_name: dict[str, list[dict[str, object]]] = {}
    for pkg in lock.get("package", []):
        name = str(pkg.get("name", "")).lower().replace("_", "-")
        pkgs_by_name.setdefault(name, []).append(pkg)

    sherpa_pkgs = pkgs_by_name.get("sherpa-onnx", [])
    sherpa_core_pkgs = pkgs_by_name.get("sherpa-onnx-core", [])

    if sherpa_pkgs and not sherpa_core_pkgs:
        findings.append(
            LintFinding(
                rule_id="DEP002",
                path="uv.lock",
                message="uv.lock locks `sherpa-onnx` but is MISSING `sherpa-onnx-core`!",
                hint=(
                    "Declare `sherpa-onnx-core==<version>` in pyproject.toml `[project.optional-dependencies].asr` "
                    "and run `uv lock`."
                ),
            )
        )
    elif sherpa_pkgs and sherpa_core_pkgs:
        sv = {str(p.get("version")) for p in sherpa_pkgs}
        cv = {str(p.get("version")) for p in sherpa_core_pkgs}
        if sv != cv:
            findings.append(
                LintFinding(
                    rule_id="DEP002",
                    path="uv.lock",
                    message=f"uv.lock version mismatch: sherpa-onnx={sorted(sv)} vs sherpa-onnx-core={sorted(cv)}.",
                    hint="Keep sherpa-onnx and sherpa-onnx-core locked at the exact same version.",
                )
            )

    # Verify audio8-vc package optional-dependencies in uv.lock include sherpa-onnx-core
    root_pkgs = pkgs_by_name.get("audio8-vc", [])
    if root_pkgs:
        opt_deps = root_pkgs[0].get("optional-dependencies", {})
        if isinstance(opt_deps, dict):
            for extra in ("asr", "all", "windows"):
                entries = opt_deps.get(extra, [])
                names = {
                    str(e.get("name", "")).lower().replace("_", "-")
                    for e in entries
                    if isinstance(e, dict)
                }
                if "sherpa-onnx" in names and "sherpa-onnx-core" not in names:
                    findings.append(
                        LintFinding(
                            rule_id="DEP002",
                            path="uv.lock",
                            message=f"uv.lock audio8-vc extra `{extra}` includes `sherpa-onnx` without `sherpa-onnx-core`.",
                            hint="Run `uv lock` after adding `sherpa-onnx-core` to pyproject.toml.",
                        )
                    )

    if pyproject is not None:
        base_deps = list(pyproject.get("project", {}).get("dependencies", []))
        ort_pin = _parse_exact_pin(base_deps, "onnxruntime")
        ort_locked = {str(p.get("version")) for p in pkgs_by_name.get("onnxruntime", [])}
        if ort_pin and ort_locked and ort_pin not in ort_locked:
            findings.append(
                LintFinding(
                    rule_id="DEP002",
                    path="uv.lock",
                    message=f"uv.lock onnxruntime version {sorted(ort_locked)} does not match pyproject.toml pin `{ort_pin}`.",
                    hint="Run `uv lock` to synchronize uv.lock with pyproject.toml.",
                )
            )

    return findings


@register_rule(
    "DEP003",
    title="Package version must be consistent across pyproject.toml, __init__.py, installer.iss, and version_info.txt",
    category="deps",
    description="Prevents stale version metadata in Windows installers and PE file headers.",
    history="Keeps release tags, PyInstaller PE version resources, and Inno Setup defaults aligned.",
)
def check_version_consistency(ctx: LintContext) -> list[LintFinding]:
    pyproject = ctx.parse_toml("pyproject.toml")
    if pyproject is None:
        return []
    expected = str(pyproject.get("project", {}).get("version", "")).strip()
    if not expected:
        return []

    findings: list[LintFinding] = []

    init_rel = "src/a8vc/__init__.py"
    init_text = ctx.read_text(init_rel) or ""
    m_init = re.search(r'__version__\s*=\s*"([^"]+)"', init_text)
    if not m_init or m_init.group(1) != expected:
        findings.append(
            LintFinding(
                rule_id="DEP003",
                path=init_rel,
                line=ctx.find_line(init_text, r"__version__"),
                message=(
                    f"__version__ is {m_init.group(1) if m_init else 'missing'!r}, "
                    f"expected {expected!r} from pyproject.toml."
                ),
                hint=f'Set `__version__ = "{expected}"` in {init_rel}.',
            )
        )

    iss_rel = "packaging/windows/installer.iss"
    iss_text = ctx.read_text(iss_rel) or ""
    m_iss = re.search(r'#define\s+AppVersion\s+"([^"]+)"', iss_text)
    if m_iss and m_iss.group(1) != expected:
        findings.append(
            LintFinding(
                rule_id="DEP003",
                path=iss_rel,
                line=ctx.find_line(iss_text, r"#define\s+AppVersion"),
                message=f"installer.iss default AppVersion is {m_iss.group(1)!r}, expected {expected!r}.",
                hint=f'Update `#define AppVersion "{expected}"` in {iss_rel}.',
            )
        )

    ver_rel = "packaging/windows/version_info.txt"
    ver_text = ctx.read_text(ver_rel) or ""
    m_ver = re.search(r"StringStruct\('ProductVersion',\s*'([^']+)'\)", ver_text)
    if m_ver and m_ver.group(1) != expected:
        findings.append(
            LintFinding(
                rule_id="DEP003",
                path=ver_rel,
                line=ctx.find_line(ver_text, r"ProductVersion"),
                message=f"version_info.txt ProductVersion is {m_ver.group(1)!r}, expected {expected!r}.",
                hint=f"Update ProductVersion / FileVersion / filevers / prodvers in {ver_rel} to {expected}.",
            )
        )

    return findings
