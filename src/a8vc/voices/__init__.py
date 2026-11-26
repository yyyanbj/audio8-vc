"""Portable .a8voice voice pack format, library management, and one-click enrollment."""

from a8vc.voices.catalog import (
    ensure_builtins_seeded,
    get_builtin_voice_path,
    install_builtin_voices,
    list_builtin_voices,
    load_builtin_catalog,
)
from a8vc.voices.enroll import VoiceEnroller
from a8vc.voices.format import (
    A8VOICE_EXT,
    A8VOICE_FORMAT,
    A8VOICE_SCHEMA_VERSION,
    VoiceManifest,
    VoicePack,
    export_official_voice_dir,
    import_official_voice_dir,
    load_voice_pack,
    load_voice_pack_bytes,
    save_voice_pack,
    validate_voice_id,
    verify_voice_pack,
)
from a8vc.voices.store import VoiceLibrary

__all__ = [
    "A8VOICE_EXT",
    "A8VOICE_FORMAT",
    "A8VOICE_SCHEMA_VERSION",
    "VoiceEnroller",
    "VoiceLibrary",
    "VoiceManifest",
    "VoicePack",
    "ensure_builtins_seeded",
    "export_official_voice_dir",
    "get_builtin_voice_path",
    "import_official_voice_dir",
    "install_builtin_voices",
    "list_builtin_voices",
    "load_builtin_catalog",
    "load_voice_pack",
    "load_voice_pack_bytes",
    "save_voice_pack",
    "validate_voice_id",
    "verify_voice_pack",
]
