"""Single-file portable `.a8voice` voice pack specification and codec-token validation.

A `.a8voice` file is a standard ZIP archive containing:
- `manifest.json`: UTF-8 JSON metadata, transcript, audio parameters, and SHA-256 digest
- `codes.npy`: 2D NumPy array (`allow_pickle=False`), shape `[num_codebooks, T]`, dtype `uint16`
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

A8VOICE_EXT = ".a8voice"
A8VOICE_FORMAT = "a8voice"
A8VOICE_SCHEMA_VERSION = 1

_VOICE_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")
_MAX_MANIFEST_BYTES = 1 * 1024 * 1024  # 1 MiB
_MAX_CODES_BYTES = 32 * 1024 * 1024  # 32 MiB


def validate_voice_id(voice_id: str) -> str:
    """Validate that voice_id is a safe single-component identifier."""
    cleaned = voice_id.strip()
    if not cleaned or cleaned in {".", ".."} or not _VOICE_ID_RE.match(cleaned):
        raise ValueError(
            f"Invalid voice_id {voice_id!r}: must match ^[a-zA-Z0-9][a-zA-Z0-9._-]{{0,63}}$"
        )
    return cleaned


class VoiceManifest(BaseModel):
    format: str = A8VOICE_FORMAT
    schema_version: int = A8VOICE_SCHEMA_VERSION
    voice_id: str
    name: str
    description: str = ""
    language: str = "zh"
    tags: list[str] = Field(default_factory=list)
    reference_text: str
    num_codebooks: int = 10
    codebook_size: int = 4096
    num_frames: int
    sample_rate: int = 44100
    codec_hop_length: int = 2048
    duration_sec: float = 0.0
    codes_dtype: str = "uint16"
    codes_sha256: str = ""
    source_audio_name: str = ""
    source_sample_rate: int = 0
    source_sha256: str = ""
    model_family: str = "audio8-tts-0.1b"
    model_fingerprint: str = ""
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )


@dataclass(slots=True)
class VoicePack:
    """In-memory representation of a loaded `.a8voice` voice pack."""

    manifest: VoiceManifest
    codes: np.ndarray  # shape [num_codebooks, T], dtype int64 for runtime convenience
    path: Path | None = None

    @property
    def voice_id(self) -> str:
        return self.manifest.voice_id

    @property
    def cache_key(self) -> tuple[str, str, str]:
        """Stable cache identity for in-memory prefix KV/SSM state caching."""
        text_hash = hashlib.sha256(self.manifest.reference_text.encode("utf-8")).hexdigest()[:16]
        codes_hash = self.manifest.codes_sha256[:16] if self.manifest.codes_sha256 else _hash_codes(self.codes)[:16]
        return (self.manifest.voice_id, codes_hash, text_hash)


def _serialize_codes_npy(codes: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(codes, dtype=np.uint16), allow_pickle=False)
    return buf.getvalue()


def _hash_codes(codes: np.ndarray) -> str:
    return hashlib.sha256(_serialize_codes_npy(codes)).hexdigest()


def validate_codes_array(
    codes: np.ndarray,
    num_codebooks: int = 10,
    codebook_size: int = 4096,
) -> np.ndarray:
    """Validate codec code tensor `[num_codebooks, T]` and return contiguous int64."""
    arr = np.asarray(codes)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D codes array [num_codebooks, T], got shape {arr.shape}")
    if arr.shape[0] != num_codebooks:
        raise ValueError(
            f"Expected {num_codebooks} codebooks on axis 0, got {arr.shape[0]}"
        )
    if arr.shape[1] <= 0:
        raise ValueError("Voice codes must have at least 1 time frame (shape[1] > 0)")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"Voice codes must be integer dtype, got {arr.dtype}")
    arr_i64 = np.ascontiguousarray(arr, dtype=np.int64)
    min_val = int(arr_i64.min())
    max_val = int(arr_i64.max())
    if min_val < 0 or max_val >= codebook_size:
        raise ValueError(
            f"Code values out of range [0, {codebook_size}): min={min_val}, max={max_val}"
        )
    return arr_i64


def build_voice_pack(
    voice_id: str,
    codes: np.ndarray,
    reference_text: str,
    *,
    name: str | None = None,
    description: str = "",
    language: str = "zh",
    tags: list[str] | None = None,
    num_codebooks: int = 10,
    codebook_size: int = 4096,
    sample_rate: int = 44100,
    codec_hop_length: int = 2048,
    source_audio_name: str = "",
    source_sample_rate: int = 0,
    source_sha256: str = "",
    model_fingerprint: str = "",
) -> VoicePack:
    """Construct and validate a `VoicePack` object from discrete codes and metadata."""
    vid = validate_voice_id(voice_id)
    ref_text = " ".join(reference_text.strip().split())
    if not ref_text:
        raise ValueError("reference_text must be a non-empty string")
    validated_codes = validate_codes_array(
        codes, num_codebooks=num_codebooks, codebook_size=codebook_size
    )
    codes_bytes = _serialize_codes_npy(validated_codes)
    codes_sha256 = hashlib.sha256(codes_bytes).hexdigest()
    num_frames = int(validated_codes.shape[1])
    duration_sec = round((num_frames * codec_hop_length) / float(sample_rate), 3)

    manifest = VoiceManifest(
        voice_id=vid,
        name=(name or vid).strip() or vid,
        description=description.strip(),
        language=language.strip() or "zh",
        tags=[t.strip() for t in (tags or []) if t.strip()],
        reference_text=ref_text,
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        num_frames=num_frames,
        sample_rate=sample_rate,
        codec_hop_length=codec_hop_length,
        duration_sec=duration_sec,
        codes_dtype="uint16",
        codes_sha256=codes_sha256,
        source_audio_name=source_audio_name,
        source_sample_rate=source_sample_rate,
        source_sha256=source_sha256,
        model_fingerprint=model_fingerprint,
    )
    return VoicePack(manifest=manifest, codes=validated_codes)


def save_voice_pack(pack: VoicePack, output_path: str | Path) -> Path:
    """Atomically write a `VoicePack` to a `.a8voice` file."""
    target = Path(output_path)
    if target.suffix != A8VOICE_EXT:
        target = target.with_suffix(A8VOICE_EXT)
    target.parent.mkdir(parents=True, exist_ok=True)

    codes_i64 = validate_codes_array(
        pack.codes,
        num_codebooks=pack.manifest.num_codebooks,
        codebook_size=pack.manifest.codebook_size,
    )
    codes_bytes = _serialize_codes_npy(codes_i64)
    digest = hashlib.sha256(codes_bytes).hexdigest()
    manifest = pack.manifest.model_copy(
        update={
            "num_frames": int(codes_i64.shape[1]),
            "codes_dtype": "uint16",
            "codes_sha256": digest,
        }
    )
    manifest_bytes = json.dumps(
        manifest.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{manifest.voice_id}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as raw_fh:
            with zipfile.ZipFile(
                raw_fh, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
            ) as zf:
                zf.writestr("manifest.json", manifest_bytes)
                zf.writestr("codes.npy", codes_bytes)
        os.replace(tmp_path, target)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

    pack.manifest = manifest
    pack.path = target
    return target


def _read_zip_member_bounded(zf: zipfile.ZipFile, name: str, max_bytes: int) -> bytes:
    try:
        info = zf.getinfo(name)
    except KeyError as exc:
        raise ValueError(f"Invalid .a8voice archive: missing required entry {name!r}") from exc
    if info.file_size > max_bytes:
        raise ValueError(
            f"Entry {name!r} exceeds maximum allowed size ({info.file_size} > {max_bytes} bytes)"
        )
    with zf.open(info, "r") as fh:
        data = fh.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"Entry {name!r} decompressed payload exceeds limit ({max_bytes} bytes)")
    return data


def load_voice_pack_bytes(
    raw_bytes: bytes,
    source_path: Path | None = None,
    *,
    verify_checksum: bool = True,
) -> VoicePack:
    """Load and validate a `.a8voice` archive from bytes."""
    buf = io.BytesIO(raw_bytes)
    if not zipfile.is_zipfile(buf):
        raise ValueError("Not a valid .a8voice archive (ZIP header missing)")
    buf.seek(0)
    with zipfile.ZipFile(buf, mode="r") as zf:
        manifest_raw = _read_zip_member_bounded(zf, "manifest.json", _MAX_MANIFEST_BYTES)
        codes_raw = _read_zip_member_bounded(zf, "codes.npy", _MAX_CODES_BYTES)

    try:
        manifest_dict = json.loads(manifest_raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Corrupted manifest.json in .a8voice: {exc}") from exc

    manifest = VoiceManifest.model_validate(manifest_dict)
    if manifest.format != A8VOICE_FORMAT:
        raise ValueError(f"Unsupported voice pack format: {manifest.format!r}")
    if manifest.schema_version != A8VOICE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported .a8voice schema_version {manifest.schema_version} "
            f"(expected {A8VOICE_SCHEMA_VERSION})"
        )
    validate_voice_id(manifest.voice_id)
    if not manifest.reference_text.strip():
        raise ValueError("manifest.reference_text must be non-empty")

    actual_sha256 = hashlib.sha256(codes_raw).hexdigest()
    if verify_checksum and manifest.codes_sha256 and actual_sha256 != manifest.codes_sha256:
        raise ValueError(
            f"codes.npy SHA-256 mismatch: expected {manifest.codes_sha256}, got {actual_sha256}"
        )

    codes_arr = np.load(io.BytesIO(codes_raw), allow_pickle=False)
    codes_i64 = validate_codes_array(
        codes_arr,
        num_codebooks=manifest.num_codebooks,
        codebook_size=manifest.codebook_size,
    )
    if codes_i64.shape[1] != manifest.num_frames:
        raise ValueError(
            f"Frame count mismatch: manifest.num_frames={manifest.num_frames}, "
            f"codes.shape[1]={codes_i64.shape[1]}"
        )
    return VoicePack(manifest=manifest, codes=codes_i64, path=source_path)


def load_voice_pack(path: str | Path, *, verify_checksum: bool = True) -> VoicePack:
    """Load and validate a `.a8voice` file from disk."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Voice pack file not found: {p}")
    return load_voice_pack_bytes(p.read_bytes(), source_path=p, verify_checksum=verify_checksum)


def verify_voice_pack(path: str | Path) -> dict[str, Any]:
    """Verify `.a8voice` structural & cryptographic integrity and return a report."""
    p = Path(path)
    pack = load_voice_pack(p, verify_checksum=True)
    return {
        "valid": True,
        "path": str(p),
        "size_bytes": p.stat().st_size,
        "voice_id": pack.manifest.voice_id,
        "name": pack.manifest.name,
        "language": pack.manifest.language,
        "num_codebooks": pack.manifest.num_codebooks,
        "num_frames": pack.manifest.num_frames,
        "duration_sec": pack.manifest.duration_sec,
        "codes_sha256": pack.manifest.codes_sha256,
        "model_fingerprint": pack.manifest.model_fingerprint,
    }


def import_official_voice_dir(
    official_dir: str | Path,
    *,
    voice_id: str | None = None,
    name: str | None = None,
    language: str = "zh",
) -> VoicePack:
    """Import an official Audio8 ONNX voice folder (`meta.json` + `codes.npy`) into a `VoicePack`."""
    root = Path(official_dir)
    meta_path = root / "meta.json"
    codes_path = root / "codes.npy"
    if not meta_path.is_file() or not codes_path.is_file():
        raise FileNotFoundError(
            f"Official voice directory must contain meta.json and codes.npy: {root}"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    ref_text = str(meta.get("reference_text", "")).strip()
    if not ref_text:
        raise ValueError(f"Official voice meta.json missing non-empty reference_text: {meta_path}")

    codes = np.load(str(codes_path), allow_pickle=False)
    vid = validate_voice_id(voice_id or str(meta.get("name") or root.name))
    return build_voice_pack(
        voice_id=vid,
        codes=codes,
        reference_text=ref_text,
        name=name or str(meta.get("name") or vid),
        language=language,
        num_codebooks=int(codes.shape[0]),
        sample_rate=int(meta.get("sample_rate", 44100)),
        source_audio_name=str(meta.get("source_audio", "")),
        source_sample_rate=int(meta.get("source_sample_rate", 0)),
        source_sha256=str(meta.get("source_sha256", "")),
        model_fingerprint=str(meta.get("model_fingerprint", "")),
    )


def export_official_voice_dir(pack: VoicePack, output_dir: str | Path) -> Path:
    """Export a `.a8voice` `VoicePack` back to official Audio8 ONNX folder (`meta.json` + `codes.npy`)."""
    dest = Path(output_dir) / pack.voice_id
    dest.mkdir(parents=True, exist_ok=True)
    np.save(
        str(dest / "codes.npy"),
        np.ascontiguousarray(pack.codes, dtype=np.uint16),
        allow_pickle=False,
    )
    meta = {
        "name": pack.voice_id,
        "reference_text": pack.manifest.reference_text,
        "shape": [int(pack.codes.shape[0]), int(pack.codes.shape[1])],
        "dtype": "uint16",
        "sample_rate": pack.manifest.sample_rate,
        "source_audio": pack.manifest.source_audio_name,
        "source_sample_rate": pack.manifest.source_sample_rate,
        "source_sha256": pack.manifest.source_sha256,
        "model_fingerprint": pack.manifest.model_fingerprint,
        "created_at": pack.manifest.created_at,
        "source_kind": "a8vc_export",
    }
    (dest / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return dest
