"""Windows packaging, PyInstaller spec, and Inno Setup lint rules (PKGxxx)."""

from __future__ import annotations

import ast
import re
import struct
import zlib

from a8vc.lint.core import LintContext, LintFinding, register_rule


def _extract_assigned_strings(tree: ast.Module, name: str) -> set[str] | None:
    """Extract string literal elements from a top-level list/tuple/set assignment."""
    for node in tree.body:
        target_name: str | None = None
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    target_name = t.id
                    value = node.value
                    break
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == name:
                target_name = node.target.id
                value = node.value
        if target_name is None or value is None:
            continue
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            out: set[str] = set()
            for elt in value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    out.add(elt.value)
            return out
    return None


@register_rule(
    "PKG001",
    title="build_windows.ps1 must patch only [runtime].provider and write UTF-8 without BOM",
    category="packaging",
    description=(
        "Patching default.toml for GPU flavors must scope edits strictly to the [runtime] section "
        "so [asr].provider remains 'cpu', and must write UTF-8 without BOM for Python tomllib."
    ),
    history="Commit 97e1621 fixed global provider regex replacement mutating [asr].provider to dml/cuda.",
)
def check_ps1_scoped_provider_patch(ctx: LintContext) -> list[LintFinding]:
    rel = "packaging/windows/build_windows.ps1"
    text = ctx.read_text(rel)
    if text is None:
        return [
            LintFinding(
                rule_id="PKG001",
                path=rel,
                message="Missing Windows build script packaging/windows/build_windows.ps1.",
            )
        ]

    findings: list[LintFinding] = []
    if re.search(r"-replace\s+['\"][^'\"]*provider\s*=", text, re.IGNORECASE):
        findings.append(
            LintFinding(
                rule_id="PKG001",
                path=rel,
                line=ctx.find_line(text, r"-replace\s+['\"][^'\"]*provider\s*="),
                message="Global `-replace` on `provider =` mutates [asr].provider alongside [runtime].provider.",
                hint="Track TOML `[section]` headers line-by-line and mutate `provider` only when `$section -eq 'runtime'`.",
            )
        )
    if '$section -eq "runtime"' not in text and "$section -eq 'runtime'" not in text:
        findings.append(
            LintFinding(
                rule_id="PKG001",
                path=rel,
                message="build_windows.ps1 does not appear to scope default.toml provider patching to [runtime].",
                hint="Check `$section -eq \"runtime\"` before replacing `provider = ...` in default.toml.",
            )
        )
    if "UTF8Encoding($false)" not in text:
        findings.append(
            LintFinding(
                rule_id="PKG001",
                path=rel,
                message="build_windows.ps1 should use `System.Text.UTF8Encoding($false)` (no BOM) when writing TOML/text files.",
                hint="Windows PowerShell 5.1 writes UTF-8 BOM by default, which breaks Python tomllib.",
            )
        )
    return findings


@register_rule(
    "PKG002",
    title="Inno Setup installer must clean previous _internal directory on upgrade",
    category="packaging",
    description=(
        "Upgrading between CPU, DirectML, and CUDA builds without deleting `{app}\\_internal` "
        "leaves conflicting ONNX Runtime DLLs behind."
    ),
    history="Commit 97e1621 added [InstallDelete] cleanup for {app}\\_internal in installer.iss.",
)
def check_inno_install_delete(ctx: LintContext) -> list[LintFinding]:
    rel = "packaging/windows/installer.iss"
    text = ctx.read_text(rel)
    if text is None:
        return [
            LintFinding(
                rule_id="PKG002",
                path=rel,
                message="Missing Inno Setup script packaging/windows/installer.iss.",
            )
        ]

    has_section = bool(re.search(r"^\s*\[InstallDelete\]\s*$", text, re.MULTILINE))
    has_internal_delete = bool(
        re.search(
            r'Type:\s*filesandordirs;\s*Name:\s*"\{app\}\\_internal"',
            text,
            re.IGNORECASE,
        )
    )
    if not (has_section and has_internal_delete):
        return [
            LintFinding(
                rule_id="PKG002",
                path=rel,
                line=ctx.find_line(text, r"\[InstallDelete\]"),
                message="installer.iss must define [InstallDelete] removing `{app}\\_internal`.",
                hint='Add `[InstallDelete]` with `Type: filesandordirs; Name: "{app}\\_internal"`.',
            )
        ]
    return []


@register_rule(
    "PKG003",
    title="PyInstaller spec must bundle uvicorn WebSocket sans-IO and legacy backends",
    category="packaging",
    description=(
        "uvicorn + websockets>=14 defaults to `uvicorn.protocols.websockets.websockets_sansio_impl`; "
        "omitting it in audio8_vc.spec breaks `/ws/vc` browser voice changing in frozen builds."
    ),
    history="v0.2.0 omitted websockets_sansio_impl from audio8_vc.spec, breaking browser WebSocket inference.",
)
def check_spec_websocket_hiddenimports(ctx: LintContext) -> list[LintFinding]:
    rel = "packaging/windows/audio8_vc.spec"
    tree = ctx.parse_python(rel)
    if tree is None:
        return [
            LintFinding(
                rule_id="PKG003",
                path=rel,
                message="Missing PyInstaller spec packaging/windows/audio8_vc.spec.",
            )
        ]

    findings: list[LintFinding] = []
    hidden = _extract_assigned_strings(tree, "hiddenimports") or set()
    required_hidden = [
        "uvicorn.protocols.websockets.auto",
        "uvicorn.protocols.websockets.websockets_impl",
        "uvicorn.protocols.websockets.websockets_sansio_impl",
    ]
    for mod in required_hidden:
        if mod not in hidden:
            findings.append(
                LintFinding(
                    rule_id="PKG003",
                    path=rel,
                    message=f"audio8_vc.spec hiddenimports is missing `{mod}`.",
                    hint=f"Add `{mod!r}` to `hiddenimports` so frozen uvicorn supports WebSocket `/ws/vc`.",
                )
            )

    req_submodules = _extract_assigned_strings(tree, "_REQUIRED_COLLECT_SUBMODULES") or set()
    for mod in ("uvicorn", "fastapi", "starlette", "websockets", "a8vc"):
        if mod not in req_submodules:
            findings.append(
                LintFinding(
                    rule_id="PKG003",
                    path=rel,
                    message=f"audio8_vc.spec `_REQUIRED_COLLECT_SUBMODULES` is missing `{mod}`.",
                    hint=f"Include `{mod!r}` in `_REQUIRED_COLLECT_SUBMODULES` so missing modules fail the build.",
                )
            )
    return findings


@register_rule(
    "PKG004",
    title="PyInstaller spec must require native ASR/ORT packages and unify onnxruntime.dll",
    category="packaging",
    description=(
        "`onnxruntime`, `sherpa_onnx`, `soundfile`, and `tokenizers` must be collected as required "
        "packages (never silently skipped in try/except), and Windows builds must unify `onnxruntime.dll` "
        "to the active ORT flavor so sherpa-onnx-core's CPU DLL cannot shadow DirectML/CUDA."
    ),
    history="v0.2.0 silently swallowed collect_all('sherpa_onnx') failures and risked duplicate onnxruntime.dll copies.",
)
def check_spec_required_native_packages(ctx: LintContext) -> list[LintFinding]:
    rel = "packaging/windows/audio8_vc.spec"
    tree = ctx.parse_python(rel)
    text = ctx.read_text(rel) or ""
    if tree is None:
        return []

    findings: list[LintFinding] = []
    req_collect = _extract_assigned_strings(tree, "_REQUIRED_COLLECT_ALL") or set()
    for pkg in ("onnxruntime", "sherpa_onnx", "soundfile", "tokenizers"):
        if pkg not in req_collect:
            findings.append(
                LintFinding(
                    rule_id="PKG004",
                    path=rel,
                    message=f"audio8_vc.spec `_REQUIRED_COLLECT_ALL` must include `{pkg}`.",
                    hint=f"Move `{pkg!r}` into `_REQUIRED_COLLECT_ALL` so packaging fails fast if native libs are missing.",
                )
            )

    if "onnxruntime.dll" not in text or "_CANONICAL_ORT_DLL" not in text:
        findings.append(
            LintFinding(
                rule_id="PKG004",
                path=rel,
                message="audio8_vc.spec must unify Windows `onnxruntime.dll` binaries to the active ORT flavor.",
                hint="Map any `onnxruntime.dll` entry collected from sherpa_onnx/Scripts to `onnxruntime/capi/onnxruntime.dll`.",
            )
        )
    return findings


@register_rule(
    "PKG005",
    title="build_windows.ps1 must assert ORT ABI version and execute a8vc.exe self-check",
    category="packaging",
    description=(
        "The Windows build pipeline must verify `$info.ort_version -eq $OrtVersion` and run "
        "`a8vc.exe self-check` before archiving the Portable ZIP or Inno Setup installer."
    ),
    history="v0.2.0 smoke tests only checked --version and info --json without testing sherpa_onnx or WebSocket startup.",
)
def check_ps1_smoke_and_selfcheck(ctx: LintContext) -> list[LintFinding]:
    rel = "packaging/windows/build_windows.ps1"
    text = ctx.read_text(rel)
    if text is None:
        return []

    findings: list[LintFinding] = []
    if not re.search(r'\$OrtVersion\s*=\s*"\d+\.\d+\.\d+"', text):
        findings.append(
            LintFinding(
                rule_id="PKG005",
                path=rel,
                message="build_windows.ps1 must declare an exact `$OrtVersion` pin (e.g. `1.24.4`).",
                hint='Set `$OrtVersion = "1.24.4"` matching pyproject.toml `onnxruntime==...`.',
            )
        )
    if "$info.ort_version -ne $OrtVersion" not in text:
        findings.append(
            LintFinding(
                rule_id="PKG005",
                path=rel,
                message="build_windows.ps1 must assert `$info.ort_version` matches `$OrtVersion`.",
                hint="Fail the Windows build if the bundled ORT wheel version drifts from `$OrtVersion`.",
            )
        )
    if '"self-check"' not in text and "'self-check'" not in text:
        findings.append(
            LintFinding(
                rule_id="PKG005",
                path=rel,
                message="build_windows.ps1 must execute `a8vc.exe self-check` after PyInstaller build.",
                hint='Add `Invoke-Checked $BuiltCli @("self-check")` to the smoke-test step.',
            )
        )
    return findings


def _decode_png_rgba(data: bytes) -> tuple[int, int, bytes]:
    """Minimal stdlib PNG decoder supporting 8-bit RGBA (filter type 0) for icon lint."""
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Not a valid PNG signature")
    pos = 8
    width = height = bit_depth = color_type = 0
    idat_parts: list[bytes] = []
    while pos + 12 <= len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        ctype = data[pos + 4 : pos + 8]
        chunk = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if ctype == b"IHDR":
            width, height, bit_depth, color_type = struct.unpack(">IIBB", chunk[:10])
        elif ctype == b"IDAT":
            idat_parts.append(chunk)
        elif ctype == b"IEND":
            break
    if bit_depth != 8 or color_type != 6:
        raise ValueError(f"Expected 8-bit RGBA PNG, got bit_depth={bit_depth} color_type={color_type}")
    raw = zlib.decompress(b"".join(idat_parts))
    stride = width * 4
    out = bytearray(width * height * 4)
    for y in range(height):
        row_start = y * (stride + 1)
        if raw[row_start] != 0:
            raise ValueError(f"Unsupported PNG scanline filter {raw[row_start]}")
        out[y * stride : (y + 1) * stride] = raw[row_start + 1 : row_start + 1 + stride]
    return width, height, bytes(out)


@register_rule(
    "PKG006",
    title="Application icon assets (.ico/.png/.svg) must exist, be visible (non-transparent), multi-res, and synced to WebUI",
    category="packaging",
    description=(
        "Verifies `packaging/windows/assets/audio8_vc.{ico,png,svg}` and `src/a8vc/server/static/"
        "{favicon.ico,icon.png,icon.svg}` exist, match, contain multi-resolution ICO frames "
        "(16/32/48/256), and have non-transparent visible pixels."
    ),
    history=(
        "v0.2.0 generate_icon.py forgot to multiply normalized alpha by 255, producing an almost "
        "invisible (alpha=1/255) Windows icon and favicon."
    ),
)
def check_icon_assets_visible_and_synced(ctx: LintContext) -> list[LintFinding]:
    findings: list[LintFinding] = []
    pairs = [
        ("packaging/windows/assets/audio8_vc.ico", "src/a8vc/server/static/favicon.ico"),
        ("packaging/windows/assets/audio8_vc.png", "src/a8vc/server/static/icon.png"),
        ("packaging/windows/assets/audio8_vc.svg", "src/a8vc/server/static/icon.svg"),
    ]
    for pkg_rel, web_rel in pairs:
        pkg_path = ctx.resolve(pkg_rel)
        web_path = ctx.resolve(web_rel)
        if not pkg_path.is_file():
            findings.append(
                LintFinding(
                    rule_id="PKG006",
                    path=pkg_rel,
                    message=f"Missing application icon asset `{pkg_rel}`.",
                    hint="Run `uv run python packaging/windows/generate_icon.py`.",
                )
            )
            continue
        if not web_path.is_file():
            findings.append(
                LintFinding(
                    rule_id="PKG006",
                    path=web_rel,
                    message=f"Missing synced WebUI icon asset `{web_rel}`.",
                    hint="Run `uv run python packaging/windows/generate_icon.py` to sync static icons.",
                )
            )
            continue
        if pkg_path.read_bytes() != web_path.read_bytes():
            findings.append(
                LintFinding(
                    rule_id="PKG006",
                    path=web_rel,
                    message=f"WebUI icon `{web_rel}` is out of sync with `{pkg_rel}`.",
                    hint="Re-run `uv run python packaging/windows/generate_icon.py`.",
                )
            )

    # Validate PNG visibility (catches alpha=[0..1] unscaled bug!)
    png_rel = "packaging/windows/assets/audio8_vc.png"
    png_path = ctx.resolve(png_rel)
    if png_path.is_file():
        try:
            w, h, rgba = _decode_png_rgba(png_path.read_bytes())
            if w != 256 or h != 256:
                findings.append(
                    LintFinding(
                        rule_id="PKG006",
                        path=png_rel,
                        message=f"Expected 256x256 icon PNG, got {w}x{h}.",
                    )
                )
            total_px = w * h
            opaque_px = sum(1 for i in range(3, len(rgba), 4) if rgba[i] >= 200)
            bright_px = sum(
                1
                for i in range(0, len(rgba), 4)
                if rgba[i + 3] >= 200 and max(rgba[i], rgba[i + 1], rgba[i + 2]) >= 160
            )
            if total_px > 0 and (opaque_px / total_px) < 0.5:
                findings.append(
                    LintFinding(
                        rule_id="PKG006",
                        path=png_rel,
                        message=(
                            f"Icon PNG is nearly transparent ({opaque_px}/{total_px} pixels with alpha>=200)."
                        ),
                        hint="Ensure RGBA alpha is scaled to 0..255 in packaging/windows/generate_icon.py.",
                    )
                )
            if total_px > 0 and (bright_px / total_px) < 0.05:
                findings.append(
                    LintFinding(
                        rule_id="PKG006",
                        path=png_rel,
                        message="Icon PNG lacks bright foreground contrast (nearly solid black).",
                        hint="Ensure foreground emblem renders visibly over the background tile.",
                    )
                )
        except Exception as exc:
            findings.append(
                LintFinding(
                    rule_id="PKG006",
                    path=png_rel,
                    message=f"Failed to validate icon PNG: {exc}",
                )
            )

    # Validate ICO multi-resolution entries
    ico_rel = "packaging/windows/assets/audio8_vc.ico"
    ico_path = ctx.resolve(ico_rel)
    if ico_path.is_file():
        raw_ico = ico_path.read_bytes()
        if len(raw_ico) < 6:
            findings.append(
                LintFinding(rule_id="PKG006", path=ico_rel, message="Icon ICO file is truncated.")
            )
        else:
            reserved, ico_type, count = struct.unpack("<HHH", raw_ico[:6])
            sizes: set[int] = set()
            for idx in range(count):
                off = 6 + idx * 16
                if off + 16 <= len(raw_ico):
                    dim = raw_ico[off]
                    sizes.add(256 if dim == 0 else dim)
            required_sizes = {16, 32, 48, 256}
            if reserved != 0 or ico_type != 1 or not required_sizes.issubset(sizes):
                findings.append(
                    LintFinding(
                        rule_id="PKG006",
                        path=ico_rel,
                        message=(
                            f"Invalid or incomplete ICO entries (found sizes={sorted(sizes)}, "
                            f"need {sorted(required_sizes)})."
                        ),
                        hint="Regenerate multi-resolution ICO with `packaging/windows/generate_icon.py`.",
                    )
                )

    return findings
