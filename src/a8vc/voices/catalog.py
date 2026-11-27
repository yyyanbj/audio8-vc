"""Curated built-in `.a8voice` preset catalog and one-click/first-run installation."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from a8vc.paths import get_builtin_voices_dir
from a8vc.voices.format import A8VOICE_EXT, load_voice_pack, validate_voice_id
from a8vc.voices.store import VoiceLibrary

logger = logging.getLogger("a8vc.voices.builtins")

BUILTIN_SEED_MARKER = ".builtins_seeded"


def load_builtin_catalog(builtins_dir: Path | None = None) -> list[dict[str, Any]]:
    """Load the curated built-in voice catalog metadata from `builtins/catalog.json`."""
    root = builtins_dir if builtins_dir is not None else get_builtin_voices_dir()
    if not root.is_dir():
        return []

    catalog_path = root / "catalog.json"
    entries: list[dict[str, Any]] = []
    if catalog_path.is_file():
        try:
            raw = json.loads(catalog_path.read_text(encoding="utf-8"))
            for item in raw.get("voices", []):
                vid = str(item.get("voice_id", "")).strip()
                if not vid:
                    continue
                pack_path = root / str(item.get("filename") or f"{vid}{A8VOICE_EXT}")
                if not pack_path.is_file():
                    continue
                enriched = dict(item)
                enriched["filename"] = pack_path.name
                enriched["size_bytes"] = pack_path.stat().st_size
                entries.append(enriched)
            return entries
        except Exception as exc:
            logger.warning("Failed to parse built-in voice catalog %s: %s", catalog_path, exc)

    # Fallback: scan any `.a8voice` files directly if catalog.json is absent or unreadable
    for file_path in sorted(root.glob(f"*{A8VOICE_EXT}")):
        if file_path.name.startswith("."):
            continue
        try:
            pack = load_voice_pack(file_path, verify_checksum=False)
            entries.append(
                {
                    "voice_id": pack.voice_id,
                    "filename": file_path.name,
                    "name": pack.manifest.name,
                    "category": "builtin",
                    "category_label": "内置",
                    "language": pack.manifest.language,
                    "description": pack.manifest.description,
                    "tags": list(pack.manifest.tags),
                    "duration_sec": pack.manifest.duration_sec,
                    "num_frames": pack.manifest.num_frames,
                    "size_bytes": file_path.stat().st_size,
                    "codes_sha256": pack.manifest.codes_sha256,
                    "model_fingerprint": pack.manifest.model_fingerprint,
                    "source_project": "",
                    "source_rank_note": "",
                    "source_url": "",
                    "license": "",
                    "attribution": "",
                }
            )
        except Exception as exc:
            logger.warning("Skipping invalid built-in voice pack %s: %s", file_path, exc)
    return entries


def list_builtin_voices(
    *,
    installed_ids: set[str] | None = None,
    builtins_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Return built-in voice catalog items with installation state."""
    installed_set = installed_ids or set()
    items: list[dict[str, Any]] = []
    for entry in load_builtin_catalog(builtins_dir=builtins_dir):
        item = dict(entry)
        item["installed"] = item["voice_id"] in installed_set
        items.append(item)
    return items


def get_builtin_voice_path(voice_id: str, *, builtins_dir: Path | None = None) -> Path:
    """Resolve the path of a bundled built-in `.a8voice` file by `voice_id`."""
    vid = validate_voice_id(voice_id)
    root = builtins_dir if builtins_dir is not None else get_builtin_voices_dir()
    path = root / f"{vid}{A8VOICE_EXT}"
    if not path.is_file():
        raise FileNotFoundError(f"Built-in voice {vid!r} not found in {root}")
    return path


def install_builtin_voices(
    library: VoiceLibrary,
    *,
    voice_ids: list[str] | None = None,
    overwrite: bool = False,
    builtins_dir: Path | None = None,
) -> dict[str, Any]:
    """Install all or selected bundled built-in `.a8voice` packs into `library`."""
    catalog = load_builtin_catalog(builtins_dir=builtins_dir)
    if not catalog:
        return {"installed": [], "skipped": [], "total_builtins": 0}

    if voice_ids:
        target_ids = [validate_voice_id(v) for v in voice_ids]
        # Validate all requested IDs exist before modifying library
        paths = [get_builtin_voice_path(vid, builtins_dir=builtins_dir) for vid in target_ids]
    else:
        target_ids = [item["voice_id"] for item in catalog]
        paths = [get_builtin_voice_path(vid, builtins_dir=builtins_dir) for vid in target_ids]

    installed: list[str] = []
    skipped: list[str] = []
    for vid, pack_path in zip(target_ids, paths, strict=True):
        if library.exists(vid) and not overwrite:
            skipped.append(vid)
            continue
        pack = load_voice_pack(pack_path, verify_checksum=True)
        library.save(pack, overwrite=overwrite)
        installed.append(pack.voice_id)

    return {
        "installed": installed,
        "skipped": skipped,
        "total_builtins": len(catalog),
    }


def ensure_builtins_seeded(
    library: VoiceLibrary,
    *,
    builtins_dir: Path | None = None,
) -> list[str]:
    """Auto-install built-in voices once when initializing a brand-new empty voice directory.

    Writes a lightweight `.builtins_seeded` marker so voices intentionally deleted later
    by the user are not unexpectedly re-added on subsequent launches (users can still
    restore any built-in voice at any time via UI or `a8vc voice install-builtins`).
    """
    marker = library.root / BUILTIN_SEED_MARKER
    if marker.exists():
        return []

    existing_packs = [
        p for p in library.root.glob(f"*{A8VOICE_EXT}") if not p.name.startswith(".")
    ]
    if existing_packs:
        _write_seed_marker(marker, seeded_ids=[])
        return []

    result = install_builtin_voices(library, overwrite=False, builtins_dir=builtins_dir)
    installed = list(result["installed"])
    if installed:
        _write_seed_marker(marker, seeded_ids=installed)
        logger.info("Seeded %d built-in .a8voice packs into %s", len(installed), library.root)
    return installed


def _write_seed_marker(marker: Path, *, seeded_ids: list[str]) -> None:
    try:
        marker.write_text(
            json.dumps(
                {
                    "seeded_at": datetime.now(UTC).isoformat(),
                    "seeded_ids": seeded_ids,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("Could not write built-in seed marker %s: %s", marker, exc)
