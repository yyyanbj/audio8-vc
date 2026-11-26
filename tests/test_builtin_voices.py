"""Tests for bundled open-source `.a8voice` built-in voices, seeding, API, and CLI."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from a8vc.asr.sherpa import MockASR
from a8vc.cli import main
from a8vc.config import AppConfig
from a8vc.paths import get_builtin_voices_dir
from a8vc.server.app import ServerState, create_app
from a8vc.voices.catalog import (
    BUILTIN_SEED_MARKER,
    ensure_builtins_seeded,
    get_builtin_voice_path,
    install_builtin_voices,
    list_builtin_voices,
    load_builtin_catalog,
)
from a8vc.voices.format import (
    A8VOICE_EXT,
    build_voice_pack,
    load_voice_pack,
    verify_voice_pack,
)
from a8vc.voices.store import VoiceLibrary

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_bundled_builtin_voice_packs_integrity() -> None:
    builtins_dir = get_builtin_voices_dir()
    assert builtins_dir.is_dir()
    assert (builtins_dir / "catalog.json").is_file()
    assert (builtins_dir / "ATTRIBUTION.md").is_file()

    catalog = load_builtin_catalog(builtins_dir)
    assert len(catalog) == 15

    catalog_ids = [item["voice_id"] for item in catalog]
    assert len(set(catalog_ids)) == len(catalog_ids)

    # Every .a8voice file in builtins_dir matches catalog.json 1-to-1
    disk_files = sorted(p.name for p in builtins_dir.glob(f"*{A8VOICE_EXT}"))
    catalog_files = sorted(item["filename"] for item in catalog)
    assert disk_files == catalog_files

    categories = {item["category"] for item in catalog}
    assert {"anime", "game", "zh", "en"}.issubset(categories)

    # Every bundled voice pack verifies cleanly with strict SHA-256 and codec bounds
    for item in catalog:
        pack_path = get_builtin_voice_path(item["voice_id"], builtins_dir=builtins_dir)
        report = verify_voice_pack(pack_path)
        assert report["valid"] is True
        assert report["voice_id"] == item["voice_id"]
        assert report["num_codebooks"] == 10
        assert report["num_frames"] == item["num_frames"]
        assert report["codes_sha256"] == item["codes_sha256"]
        assert report["size_bytes"] == item["size_bytes"]

        pack = load_voice_pack(pack_path, verify_checksum=True)
        assert pack.manifest.sample_rate == 44100
        assert pack.manifest.reference_text.strip()
        assert pack.manifest.model_fingerprint == "audio8-tts-preview-0.1b-int8-v1"
        assert pack.codes.shape == (10, item["num_frames"])
        assert item["license"]
        assert item["source_project"]
        assert item["source_url"].startswith("https://")


def test_catalog_fallback_without_catalog_json(tmp_path: Path) -> None:
    src_pack = get_builtin_voice_path("voxcpm_tsundere")
    fallback_dir = tmp_path / "builtins_no_json"
    fallback_dir.mkdir()
    shutil.copy2(src_pack, fallback_dir / src_pack.name)

    entries = load_builtin_catalog(fallback_dir)
    assert len(entries) == 1
    assert entries[0]["voice_id"] == "voxcpm_tsundere"
    assert entries[0]["size_bytes"] == src_pack.stat().st_size


def test_install_and_auto_seed_behavior(tmp_path: Path) -> None:
    # 1. Brand-new empty library auto-seeds once and writes marker
    lib = VoiceLibrary(tmp_path / "voices_fresh")
    seeded = ensure_builtins_seeded(lib)
    assert len(seeded) == 15
    assert (lib.root / BUILTIN_SEED_MARKER).is_file()
    assert len(lib.list_voices()) == 15

    # 2. Deleting a voice does NOT cause it to be re-seeded on subsequent launch
    assert lib.delete("edge0_default") is True
    assert not lib.exists("edge0_default")
    assert ensure_builtins_seeded(lib) == []
    assert not lib.exists("edge0_default")

    # 3. list_builtin_voices accurately reflects installed vs missing state
    installed_ids = {v["voice_id"] for v in lib.list_voices() if "error" not in v}
    listed = {item["voice_id"]: item["installed"] for item in list_builtin_voices(installed_ids=installed_ids)}
    assert listed["edge0_default"] is False
    assert listed["voxcpm_tsundere"] is True

    # 4. Selective install restores only the missing voice without touching others
    res = install_builtin_voices(lib, voice_ids=["edge0_default", "voxcpm_tsundere"], overwrite=False)
    assert res["installed"] == ["edge0_default"]
    assert res["skipped"] == ["voxcpm_tsundere"]
    assert lib.exists("edge0_default")

    # 5. Overwrite=True reinstalls existing built-ins
    res_overwrite = install_builtin_voices(lib, voice_ids=["edge0_default"], overwrite=True)
    assert res_overwrite["installed"] == ["edge0_default"]
    assert res_overwrite["skipped"] == []

    # 6. Unknown built-in voice ID fails before modifying the library
    lib2 = VoiceLibrary(tmp_path / "voices_atomic")
    with pytest.raises(FileNotFoundError):
        install_builtin_voices(lib2, voice_ids=["voxcpm_tsundere", "does_not_exist"])
    assert not lib2.exists("voxcpm_tsundere")

    # 7. Existing non-empty custom library is NOT auto-seeded
    custom_lib = VoiceLibrary(tmp_path / "voices_custom")
    rng = np.random.default_rng(1)
    custom_pack = build_voice_pack(
        voice_id="my_custom",
        codes=rng.integers(0, 4096, size=(10, 12), dtype=np.int64),
        reference_text="Custom voice",
    )
    custom_lib.save(custom_pack)
    assert ensure_builtins_seeded(custom_lib) == []
    assert (custom_lib.root / BUILTIN_SEED_MARKER).is_file()
    assert [v["voice_id"] for v in custom_lib.list_voices()] == ["my_custom"]


def test_server_builtin_voices_seeding_and_endpoints(tmp_path: Path) -> None:
    cfg = AppConfig()
    cfg.paths.voices_dir = tmp_path / "server_voices"

    # ServerState with library=None automatically seeds built-in voices
    state = ServerState(cfg, library=None, asr=MockASR("ok"))
    app = create_app(cfg, state=state)
    client = TestClient(app)

    installed_resp = client.get("/api/voices")
    assert installed_resp.status_code == 200
    installed = installed_resp.json()["voices"]
    assert len(installed) == 15

    builtins_resp = client.get("/api/voices/builtins")
    assert builtins_resp.status_code == 200
    builtins = builtins_resp.json()["builtins"]
    assert len(builtins) == 15
    assert all(b["installed"] is True for b in builtins)

    # Delete one voice and verify catalog reflects installed=False, then reinstall via API
    del_resp = client.delete("/api/voices/kokoro_heart_en")
    assert del_resp.status_code == 200

    builtins_after_del = {
        b["voice_id"]: b["installed"]
        for b in client.get("/api/voices/builtins").json()["builtins"]
    }
    assert builtins_after_del["kokoro_heart_en"] is False

    install_resp = client.post(
        "/api/voices/builtins/install",
        json={"voice_ids": ["kokoro_heart_en"], "overwrite": False},
    )
    assert install_resp.status_code == 200
    payload = install_resp.json()
    assert payload["installed"] == ["kokoro_heart_en"]
    assert payload["skipped"] == []

    # Invalid voice_id returns 404
    bad_resp = client.post(
        "/api/voices/builtins/install",
        json={"voice_ids": ["nonexistent_voice"]},
    )
    assert bad_resp.status_code == 404


def test_cli_builtin_voice_commands(tmp_path: Path, capsys: object) -> None:
    voices_dir = tmp_path / "cli_voices"
    cfg_path = tmp_path / "cli.toml"
    cfg_path.write_text(f'[paths]\nvoices_dir = "{voices_dir.as_posix()}"\n', encoding="utf-8")

    # 1. List built-ins as JSON
    assert main(["--config", str(cfg_path), "voice", "builtins", "--json"]) == 0
    raw = capsys.readouterr().out  # type: ignore[attr-defined]
    items = json.loads(raw)
    assert len(items) == 15
    assert all(item["installed"] is False for item in items)

    # 2. Install two selected built-in voices
    assert (
        main(
            [
                "--config",
                str(cfg_path),
                "voice",
                "install-builtins",
                "--id",
                "voxcpm_tsundere",
                "--id",
                "game_announcer_en",
            ]
        )
        == 0
    )
    capsys.readouterr()  # type: ignore[attr-defined]
    assert (voices_dir / "voxcpm_tsundere.a8voice").is_file()
    assert (voices_dir / "game_announcer_en.a8voice").is_file()
    assert not (voices_dir / "cosyvoice_sweet.a8voice").exists()

    # 3. Install remaining built-in voices
    assert main(["--config", str(cfg_path), "voice", "install-builtins"]) == 0
    capsys.readouterr()  # type: ignore[attr-defined]
    assert len(list(voices_dir.glob(f"*{A8VOICE_EXT}"))) == 15

    # 4. Human-readable builtins table prints cleanly
    assert main(["--config", str(cfg_path), "voice", "builtins"]) == 0
    table = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "voxcpm_tsundere" in table
    assert "kokoro_heart_en" in table


def test_packaging_and_ui_include_builtin_voices() -> None:
    spec_text = (REPO_ROOT / "packaging" / "windows" / "audio8_vc.spec").read_text(encoding="utf-8")
    assert '"builtins"' in spec_text
    assert "a8vc/voices/builtins" in spec_text

    html_text = (REPO_ROOT / "src" / "a8vc" / "server" / "static" / "index.html").read_text(
        encoding="utf-8"
    )
    assert "/api/voices/builtins" in html_text
    assert "/api/voices/builtins/install" in html_text
    assert "builtin-voice-list" in html_text
