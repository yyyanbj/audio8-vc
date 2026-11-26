"""Tests for `.a8voice` single-file voice pack format, checksum verification, and library store."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from a8vc.voices.format import (
    build_voice_pack,
    export_official_voice_dir,
    import_official_voice_dir,
    load_voice_pack,
    save_voice_pack,
    validate_voice_id,
    verify_voice_pack,
)
from a8vc.voices.store import VoiceLibrary


def _make_sample_codes(frames: int = 40) -> np.ndarray:
    rng = np.random.default_rng(123)
    return rng.integers(0, 4096, size=(10, frames), dtype=np.int64)


def test_voice_pack_roundtrip_and_verify(tmp_path: Path) -> None:
    codes = _make_sample_codes(48)
    pack = build_voice_pack(
        voice_id="aria_01",
        codes=codes,
        reference_text="你好，这是一段参考音色测试。",
        name="Aria",
        language="zh",
        tags=["anime", "bright"],
    )
    out_file = save_voice_pack(pack, tmp_path / "aria_01.a8voice")
    assert out_file.is_file()
    assert out_file.stat().st_size < 16_384  # Compact single file (~few KB)

    loaded = load_voice_pack(out_file, verify_checksum=True)
    assert loaded.voice_id == "aria_01"
    assert loaded.manifest.name == "Aria"
    assert loaded.manifest.tags == ["anime", "bright"]
    assert loaded.manifest.num_frames == 48
    assert loaded.codes.dtype == np.int64
    assert np.array_equal(loaded.codes, codes)

    report = verify_voice_pack(out_file)
    assert report["valid"] is True
    assert report["voice_id"] == "aria_01"


def test_voice_pack_detects_tampered_codes(tmp_path: Path) -> None:
    codes = _make_sample_codes(20)
    pack = build_voice_pack(voice_id="tamper_test", codes=codes, reference_text="hello world")
    out_file = save_voice_pack(pack, tmp_path / "tamper_test.a8voice")

    # Tamper with codes.npy inside the zip without updating manifest.json
    tampered = tmp_path / "tampered.a8voice"
    with zipfile.ZipFile(out_file, "r") as zin, zipfile.ZipFile(tampered, "w") as zout:
        zout.writestr("manifest.json", zin.read("manifest.json"))
        bad_bytes = bytearray(zin.read("codes.npy"))
        bad_bytes[-1] ^= 0xFF
        zout.writestr("codes.npy", bytes(bad_bytes))

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_voice_pack(tampered, verify_checksum=True)


def test_voice_id_validation_rejects_traversal() -> None:
    assert validate_voice_id("aria-v1.0_zh") == "aria-v1.0_zh"
    for bad in ("", " ", "..", ".", "../evil", "foo/bar", "a\\b", "has space"):
        with pytest.raises(ValueError):
            validate_voice_id(bad)


def test_codes_bounds_validation() -> None:
    bad_codes = np.zeros((10, 12), dtype=np.int64)
    bad_codes[0, 3] = 4096  # out of [0, 4096)
    with pytest.raises(ValueError, match="out of range"):
        build_voice_pack(voice_id="bad_code", codes=bad_codes, reference_text="test")


def test_official_directory_bidirectional_interop(tmp_path: Path) -> None:
    official_dir = tmp_path / "official" / "default_speaker"
    official_dir.mkdir(parents=True)
    codes = _make_sample_codes(32).astype(np.uint16)
    np.save(str(official_dir / "codes.npy"), codes, allow_pickle=False)
    (official_dir / "meta.json").write_text(
        json.dumps(
            {
                "name": "default_speaker",
                "reference_text": "Hello from Audio8 official runtime.",
                "sample_rate": 44100,
                "model_fingerprint": "fp-abc123",
            }
        ),
        encoding="utf-8",
    )

    imported = import_official_voice_dir(official_dir, language="en")
    assert imported.voice_id == "default_speaker"
    assert imported.manifest.model_fingerprint == "fp-abc123"
    assert np.array_equal(imported.codes, codes.astype(np.int64))

    exported_dir = export_official_voice_dir(imported, tmp_path / "exported")
    assert (exported_dir / "meta.json").is_file()
    assert (exported_dir / "codes.npy").is_file()
    reloaded_codes = np.load(str(exported_dir / "codes.npy"), allow_pickle=False)
    assert np.array_equal(reloaded_codes, codes)


def test_voice_library_crud(tmp_path: Path) -> None:
    lib = VoiceLibrary(tmp_path / "voices")
    pack = build_voice_pack(
        voice_id="nova",
        codes=_make_sample_codes(16),
        reference_text="测试语音包",
    )
    saved = lib.save(pack)
    assert saved.name == "nova.a8voice"
    assert lib.exists("nova")

    with pytest.raises(FileExistsError):
        lib.save(pack, overwrite=False)

    voices = lib.list_voices()
    assert len(voices) == 1
    assert voices[0]["voice_id"] == "nova"

    # Import raw bytes into another library
    lib2 = VoiceLibrary(tmp_path / "voices2")
    imported = lib2.import_bytes(saved.read_bytes())
    assert imported.voice_id == "nova"
    assert lib2.delete("nova") is True
    assert lib2.delete("nova") is False
