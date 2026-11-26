"""Tests for the `a8vc` CLI voice pack management commands."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from a8vc.cli import main


def test_cli_voice_import_list_inspect_verify_export(
    tmp_path: Path, capsys: object
) -> None:
    voices_dir = tmp_path / "voices"
    cfg_path = tmp_path / "test.toml"
    cfg_path.write_text(f'[paths]\nvoices_dir = "{voices_dir.as_posix()}"\n', encoding="utf-8")

    # Create a fake official Audio8 voice directory
    official = tmp_path / "official_voice"
    official.mkdir()
    codes = np.arange(10 * 16, dtype=np.uint16).reshape(10, 16) % 4096
    np.save(str(official / "codes.npy"), codes, allow_pickle=False)
    (official / "meta.json").write_text(
        json.dumps({"name": "demo_voice", "reference_text": "你好世界", "sample_rate": 44100}),
        encoding="utf-8",
    )

    assert main(["--config", str(cfg_path), "voice", "import-official", str(official), "--id", "demo"]) == 0
    assert (voices_dir / "demo.a8voice").is_file()

    assert main(["--config", str(cfg_path), "voice", "list"]) == 0
    assert main(["--config", str(cfg_path), "voice", "inspect", "demo"]) == 0
    assert main(["--config", str(cfg_path), "voice", "verify", str(voices_dir / "demo.a8voice")]) == 0

    exported = tmp_path / "exported"
    assert main(["--config", str(cfg_path), "voice", "export-official", "demo", str(exported)]) == 0
    assert (exported / "demo" / "codes.npy").is_file()
    assert (exported / "demo" / "meta.json").is_file()


def test_cli_info_json(tmp_path: Path, capsys: object) -> None:
    cfg_path = tmp_path / "info.toml"
    cfg_path.write_text('[runtime]\nprovider = "cpu"\n[asr]\nprovider = "cpu"\n', encoding="utf-8")

    assert main(["--config", str(cfg_path), "info", "--json"]) == 0
    out = capsys.readouterr().out  # type: ignore[attr-defined]
    payload = json.loads(out)
    assert payload["configured_provider"] == "cpu"
    assert payload["asr_provider"] == "cpu"
    assert "CPUExecutionProvider" in payload["available_providers"]
    assert payload["ort_flavor"] in {"cpu", "dml", "cuda", "coreml"}
