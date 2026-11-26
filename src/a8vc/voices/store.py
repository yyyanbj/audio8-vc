"""Voice library manager for storing, listing, loading, and hot-reloading `.a8voice` files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from a8vc.voices.format import (
    A8VOICE_EXT,
    VoicePack,
    export_official_voice_dir,
    import_official_voice_dir,
    load_voice_pack,
    load_voice_pack_bytes,
    save_voice_pack,
    validate_voice_id,
)


class VoiceLibrary:
    """Manages a local directory of single-file `.a8voice` voice packs."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, voice_id: str) -> Path:
        vid = validate_voice_id(voice_id)
        return self.root / f"{vid}{A8VOICE_EXT}"

    def exists(self, voice_id: str) -> bool:
        return self.path_for(voice_id).is_file()

    def list_voices(self) -> list[dict[str, Any]]:
        """List all valid `.a8voice` packs sorted by voice_id."""
        items: list[dict[str, Any]] = []
        if not self.root.is_dir():
            return items
        for file_path in sorted(self.root.glob(f"*{A8VOICE_EXT}")):
            if file_path.name.startswith("."):
                continue
            try:
                pack = load_voice_pack(file_path, verify_checksum=False)
                entry = pack.manifest.model_dump(mode="json")
                entry["path"] = str(file_path)
                entry["size_bytes"] = file_path.stat().st_size
                items.append(entry)
            except Exception as exc:
                items.append(
                    {
                        "voice_id": file_path.stem,
                        "path": str(file_path),
                        "size_bytes": file_path.stat().st_size,
                        "error": str(exc),
                    }
                )
        return items

    def load(self, voice_id_or_path: str | Path, *, verify_checksum: bool = True) -> VoicePack:
        """Load a voice pack either by voice_id in this library or by explicit `.a8voice` path."""
        candidate = Path(voice_id_or_path)
        if candidate.is_file() and candidate.suffix == A8VOICE_EXT:
            return load_voice_pack(candidate, verify_checksum=verify_checksum)
        path = self.path_for(str(voice_id_or_path))
        if not path.is_file():
            raise FileNotFoundError(
                f"Voice {voice_id_or_path!r} not found in {self.root} (expected {path.name})"
            )
        return load_voice_pack(path, verify_checksum=verify_checksum)

    def save(self, pack: VoicePack, *, overwrite: bool = False) -> Path:
        target = self.path_for(pack.voice_id)
        if target.exists() and not overwrite:
            raise FileExistsError(
                f"Voice {pack.voice_id!r} already exists at {target}. Pass overwrite=True to replace."
            )
        return save_voice_pack(pack, target)

    def import_bytes(self, raw_bytes: bytes, *, overwrite: bool = False) -> VoicePack:
        """Verify and install an uploaded `.a8voice` file into the library."""
        pack = load_voice_pack_bytes(raw_bytes, verify_checksum=True)
        self.save(pack, overwrite=overwrite)
        return pack

    def import_official(
        self,
        official_dir: str | Path,
        *,
        voice_id: str | None = None,
        name: str | None = None,
        language: str = "zh",
        overwrite: bool = False,
    ) -> VoicePack:
        pack = import_official_voice_dir(
            official_dir,
            voice_id=voice_id,
            name=name,
            language=language,
        )
        self.save(pack, overwrite=overwrite)
        return pack

    def export_official(self, voice_id: str, output_dir: str | Path) -> Path:
        pack = self.load(voice_id)
        return export_official_voice_dir(pack, output_dir)

    def delete(self, voice_id: str) -> bool:
        path = self.path_for(voice_id)
        if not path.is_file():
            return False
        path.unlink()
        return True
