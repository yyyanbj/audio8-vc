"""Tests for the extensible Audio8-VC project lint suite (`a8vc lint` / `scripts/lint.py`)."""

from __future__ import annotations

import json
from pathlib import Path

from a8vc.cli import main as cli_main
from a8vc.lint import (
    LintContext,
    LintFinding,
    LintRegistry,
    Severity,
    default_registry,
    discover_repo_root,
    format_report_text,
)
from a8vc.lint.runner import main as lint_runner_main


def _write(root: Path, rel: str, content: str) -> None:
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def test_repo_passes_all_lint_rules() -> None:
    repo_root = discover_repo_root()
    report = default_registry.run(repo_root)
    assert len(report.rules_run) >= 20
    assert report.findings == [], format_report_text(report)
    assert report.ok(strict=True)


def test_lint_registry_extensibility_and_filtering(tmp_path: Path) -> None:
    registry = LintRegistry()
    # Prevent auto-discover from mixing built-in rules into isolated test registry
    registry._discovered = True

    @registry.rule(
        "CUSTOM001",
        title="Custom error rule",
        category="custom",
        description="Ensures README exists.",
        history="Added dynamically in unit test.",
    )
    def _check_readme(ctx: LintContext) -> list[LintFinding]:
        if not ctx.exists("README.md"):
            return [LintFinding(rule_id="CUSTOM001", path="README.md", message="Missing README.md")]
        return []

    @registry.rule(
        "CUSTOM002",
        title="Custom warning rule",
        category="custom",
        description="Emits a warning if CHANGELOG.md is absent.",
        history="Added dynamically in unit test.",
        severity=Severity.WARNING,
    )
    def _check_changelog(ctx: LintContext) -> list[LintFinding]:
        if not ctx.exists("CHANGELOG.md"):
            return [
                LintFinding(
                    rule_id="CUSTOM002",
                    severity=Severity.WARNING,
                    path="CHANGELOG.md",
                    message="Missing CHANGELOG.md",
                )
            ]
        return []

    rep_all = registry.run(tmp_path)
    assert rep_all.rules_run == ["CUSTOM001", "CUSTOM002"]
    assert rep_all.error_count == 1
    assert rep_all.warning_count == 1
    assert not rep_all.ok()

    rep_warn_only = registry.run(tmp_path, ignore=["CUSTOM001"])
    assert rep_warn_only.rules_run == ["CUSTOM002"]
    assert rep_warn_only.ok(strict=False)
    assert not rep_warn_only.ok(strict=True)

    _write(tmp_path, "README.md", "# Demo\n")
    rep_sel = registry.run(tmp_path, select=["CUSTOM001"])
    assert rep_sel.ok(strict=True)


def test_lint_catches_ci_and_packaging_regressions(tmp_path: Path) -> None:
    # CI001: PowerShell array splatting regression
    _write(
        tmp_path,
        ".github/workflows/build-windows.yml",
        """
jobs:
  build:
    steps:
      - run: |
          $buildArgs = @("-Provider", "cpu")
          & .\\packaging\\windows\\build_windows.ps1 @buildArgs
""",
    )
    # PKG001 + PKG005: global provider replace, missing UTF-8 no-BOM, missing self-check
    _write(
        tmp_path,
        "packaging/windows/build_windows.ps1",
        '$raw = $raw -replace \'provider = "cpu"\', \'provider = "dml"\'\n',
    )
    # PKG002: installer.iss without [InstallDelete] _internal cleanup
    _write(tmp_path, "packaging/windows/installer.iss", "[Setup]\nAppName=Audio8-VC\n")
    # PKG003 + PKG004: spec missing websockets_sansio_impl, sherpa_onnx required, and ort unification
    _write(
        tmp_path,
        "packaging/windows/audio8_vc.spec",
        """
hiddenimports = ["uvicorn.protocols.websockets.auto"]
_REQUIRED_COLLECT_ALL = ("onnxruntime",)
_REQUIRED_COLLECT_SUBMODULES = ("a8vc",)
""",
    )

    report = default_registry.run(tmp_path, categories=["ci", "packaging"])
    rule_ids = {f.rule_id for f in report.findings}
    assert {
        "CI001",
        "CI002",
        "CI003",
        "CI004",
        "CI005",
        "PKG001",
        "PKG002",
        "PKG003",
        "PKG004",
        "PKG005",
        "PKG006",
    } <= rule_ids


def test_lint_pkg006_catches_invisible_alpha_icon(tmp_path: Path) -> None:
    """Verify PKG006 catches the v0.2.0 alpha=1/255 invisible icon regression."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "generate_icon",
        discover_repo_root() / "packaging" / "windows" / "generate_icon.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Craft a 256x256 PNG with alpha=1 everywhere (reproducing the historical bug)
    bad_rgba = bytes([60, 140, 255, 1] * (256 * 256))
    bad_png = mod.encode_png(256, bad_rgba)

    for rel in (
        "packaging/windows/assets/audio8_vc.png",
        "src/a8vc/server/static/icon.png",
    ):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(bad_png)

    for rel in (
        "packaging/windows/assets/audio8_vc.svg",
        "src/a8vc/server/static/icon.svg",
    ):
        _write(tmp_path, rel, "<svg/>")

    for rel in (
        "packaging/windows/assets/audio8_vc.ico",
        "src/a8vc/server/static/favicon.ico",
    ):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\x00\x00\x01\x00\x00\x00")

    report = default_registry.run(tmp_path, select=["PKG006"])
    assert not report.ok()
    assert any("nearly transparent" in f.message for f in report.findings)


def test_lint_catches_dependency_abi_and_uvlock_regressions(tmp_path: Path) -> None:
    # DEP001 + DEP002 + DEP003: unpinned ORT, missing sherpa-onnx-core in pyproject + uv.lock, version drift
    _write(
        tmp_path,
        "pyproject.toml",
        """
[project]
name = "audio8-vc"
version = "0.2.0"
dependencies = ["onnxruntime>=1.22,<1.24"]

[project.optional-dependencies]
asr = ["sherpa-onnx==1.13.0"]
""",
    )
    _write(
        tmp_path,
        "uv.lock",
        """
[[package]]
name = "audio8-vc"
version = "0.2.0"

[package.optional-dependencies]
asr = [{ name = "sherpa-onnx" }]

[[package]]
name = "sherpa-onnx"
version = "1.13.0"
""",
    )
    _write(tmp_path, "src/a8vc/__init__.py", '__version__ = "0.1.0"\n')

    report = default_registry.run(tmp_path, categories=["deps"])
    rule_ids = {f.rule_id for f in report.findings}
    assert {"DEP001", "DEP002", "DEP003"} <= rule_ids


def test_lint_catches_runtime_server_web_cli_and_voice_regressions(tmp_path: Path) -> None:
    # RT001: SileroVADOnnx missing 'x' tensor support and create_vad missing Path/try fallback
    _write(
        tmp_path,
        "src/a8vc/audio/vad.py",
        """
class SileroVADOnnx:
    def run(self):
        return {"input": 1, "sr": 16000}

def create_vad(cfg, vad_model_path=None):
    return vad_model_path.is_file()
""",
    )
    # RT002: missing select_model_provider and fallback
    _write(tmp_path, "src/a8vc/runtime/providers.py", "def resolve_providers(): pass\n")
    _write(tmp_path, "src/a8vc/runtime/audio8_onnx.py", "class Audio8OnnxRuntime: pass\n")
    # RT003: importing sherpa_onnx before onnxruntime
    _write(
        tmp_path,
        "src/a8vc/asr/sherpa.py",
        """
def load():
    import sherpa_onnx
    return sherpa_onnx
""",
    )
    # API001 + API002 + API003: missing global exception handler, unprimed tts_stream, unguarded ws_voice_changer
    _write(
        tmp_path,
        "src/a8vc/server/app.py",
        """
def create_app():
    async def tts_stream():
        return StreamingResponse(gen())
    async def ws_voice_changer(websocket):
        engine = srv_state.create_vc_engine()
""",
    )
    # WEB001 + WEB002: raw res.json() and missing AudioContext/zeroGain
    _write(
        tmp_path,
        "src/a8vc/server/static/index.html",
        "<script>async function load(){ const d = await res.json(); }</script>\n",
    )
    # CLI001 + CLI002: ensure_ascii=False and missing _configure_stdio / self-check
    _write(
        tmp_path,
        "src/a8vc/cli.py",
        """
import json
def main():
    print(json.dumps({"a": "中文"}, ensure_ascii=False))
""",
    )
    # VC001 + VC002: missing catalog.json and incomplete .gitignore
    (tmp_path / "src/a8vc/voices/builtins").mkdir(parents=True)
    _write(tmp_path, ".gitignore", "__pycache__/\n")

    report = default_registry.run(
        tmp_path,
        categories=["runtime", "server", "web", "cli", "voices"],
    )
    rule_ids = {f.rule_id for f in report.findings}
    assert {
        "RT001",
        "RT002",
        "RT003",
        "API001",
        "API002",
        "API003",
        "WEB001",
        "WEB002",
        "CLI001",
        "CLI002",
        "VC001",
        "VC002",
    } <= rule_ids


def test_lint_cli_and_runner_json_output(capsys: object) -> None:
    repo_root = discover_repo_root()
    rc = cli_main(["lint", "--root", str(repo_root), "--json"])
    assert rc == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    payload = json.loads(captured.out)
    assert payload["ok"] is True
    assert payload["error_count"] == 0
    assert payload["rules_count"] >= 20

    rc2 = lint_runner_main(["--list-rules", "--json"])
    assert rc2 == 0
    rules_json = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert any(r["rule_id"] == "DEP002" for r in rules_json)
    assert any(r["rule_id"] == "DSK001" for r in rules_json)
    assert any(r["rule_id"] == "DSK002" for r in rules_json)


def test_lint_catches_desktop_windows_gui_regressions(tmp_path: Path) -> None:
    # DSK001: passing PNG to pywebview on Windows (exact v0.2.1 bug)
    _write(
        tmp_path,
        "src/a8vc/desktop.py",
        'icon_file = get_icon_path("png") or get_icon_path("ico")\n'
        'webview.start(gui="edgechromium", icon=str(icon_file))\n',
    )
    rep_dsk1 = default_registry.run(tmp_path, select={"DSK001"})
    assert any(f.rule_id == "DSK001" for f in rep_dsk1.findings)

    # DSK002: missing MessageBoxW / --smoke-test in entry_desktop.py & desktop.py
    _write(
        tmp_path,
        "packaging/windows/entry_desktop.py",
        "from a8vc.desktop import main\nraise SystemExit(main())\n",
    )
    rep_dsk2 = default_registry.run(tmp_path, select={"DSK002"})
    assert any(f.rule_id == "DSK002" and f.path == "packaging/windows/entry_desktop.py" for f in rep_dsk2.findings)
    assert any(f.rule_id == "DSK002" and f.path == "src/a8vc/desktop.py" for f in rep_dsk2.findings)

