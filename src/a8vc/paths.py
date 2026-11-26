"""Cross-platform path resolution for source, portable, and frozen Windows installs."""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "Audio8-VC"


def is_frozen() -> bool:
    """Return True when running inside a PyInstaller / frozen executable bundle."""
    return bool(getattr(sys, "frozen", False))


def get_executable_dir() -> Path:
    """Directory containing the packaged executable (or current working directory in dev)."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path.cwd().resolve()


def get_bundle_dir() -> Path:
    """Read-only resource root (`sys._MEIPASS` when frozen, else repo/package root)."""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass).resolve()
    # src/a8vc/paths.py -> parents[2] is repo root when running from source checkout
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "configs").is_dir() or (candidate / "pyproject.toml").is_file():
        return candidate
    return Path(__file__).resolve().parent


def is_portable_mode() -> bool:
    """Return True if portable mode is enabled via env var or marker file next to the exe."""
    flag = os.environ.get("A8VC_PORTABLE", "").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return True
    exe_dir = get_executable_dir()
    return (exe_dir / "portable.flag").is_file() or (exe_dir / ".portable").is_file()


def get_user_data_dir() -> Path:
    """Return the per-user writable data directory for installed desktop usage.

    - Override: `A8VC_DATA_DIR` environment variable
    - Portable mode: `<exe_dir>/data`
    - Windows: `%LOCALAPPDATA%\\Audio8-VC`
    - macOS: `~/Library/Application Support/Audio8-VC`
    - Linux/other: `$XDG_DATA_HOME/Audio8-VC` or `~/.local/share/Audio8-VC`
    """
    override = os.environ.get("A8VC_DATA_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()

    if is_portable_mode():
        return (get_executable_dir() / "data").resolve()

    if sys.platform == "win32" or os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
        if local_app_data:
            return Path(local_app_data) / APP_NAME
        return Path.home() / "AppData" / "Local" / APP_NAME

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME

    xdg_data = os.environ.get("XDG_DATA_HOME", "").strip()
    if xdg_data:
        return Path(xdg_data) / APP_NAME
    return Path.home() / ".local" / "share" / APP_NAME


def get_data_dir(*, desktop: bool = False) -> Path:
    """Resolve active writable root directory for models, voices, logs, and user config.

    When running from source in CLI dev mode (`desktop=False` and not frozen and no
    `A8VC_DATA_DIR`/`A8VC_PORTABLE` override), keeps current working directory so existing
    relative paths (`models/`, `voices/`) work unchanged.
    """
    if os.environ.get("A8VC_DATA_DIR", "").strip() or is_portable_mode() or is_frozen() or desktop:
        data_dir = get_user_data_dir()
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir
    return Path.cwd()


def get_logs_dir(*, desktop: bool = False) -> Path:
    logs_dir = get_data_dir(desktop=desktop) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def get_default_config_path() -> Path | None:
    """Locate bundled `configs/default.toml` across dev and frozen layouts."""
    if is_frozen():
        candidates = [
            get_bundle_dir() / "configs" / "default.toml",
            get_executable_dir() / "configs" / "default.toml",
            Path("configs/default.toml"),
        ]
    else:
        candidates = [
            Path("configs/default.toml"),
            get_bundle_dir() / "configs" / "default.toml",
            get_executable_dir() / "configs" / "default.toml",
        ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def get_static_dir() -> Path:
    """Locate `server/static` directory in both source and PyInstaller frozen builds."""
    pkg_static = Path(__file__).resolve().parent / "server" / "static"
    if pkg_static.is_dir():
        return pkg_static
    bundle_static = get_bundle_dir() / "a8vc" / "server" / "static"
    if bundle_static.is_dir():
        return bundle_static
    return pkg_static


def get_icon_path(ext: str = "ico") -> Path | None:
    """Locate bundled application icon (`.ico` or `.png`)."""
    filename = f"audio8_vc.{ext.lstrip('.')}"
    candidates = [
        get_bundle_dir() / "assets" / filename,
        get_bundle_dir() / "packaging" / "windows" / "assets" / filename,
        get_executable_dir() / "assets" / filename,
        Path(__file__).resolve().parents[2] / "packaging" / "windows" / "assets" / filename,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def resolve_readonly_or_writable_path(
    rel_or_abs: Path,
    *,
    data_dir: Path,
    must_be_writable: bool = False,
) -> Path:
    """Resolve a configured path against `data_dir`, allowing bundled read-only model fallback.

    For read-only model artifacts (when `must_be_writable=False`):
    1. If `rel_or_abs` is absolute, return it as-is.
    2. If `data_dir / rel_or_abs` already exists, prefer the user's writable copy.
    3. Otherwise, if a bundled copy exists in `<exe_dir>` or `<bundle_dir>` (e.g., an
       offline all-in-one Windows installer), return that bundled read-only copy.
    4. Otherwise, return `data_dir / rel_or_abs` so in-app downloads save into `data_dir`.
    """
    if rel_or_abs.is_absolute():
        return rel_or_abs

    writable_target = data_dir / rel_or_abs
    if must_be_writable or writable_target.exists():
        return writable_target

    for readonly_root in (get_executable_dir(), get_bundle_dir()):
        candidate = readonly_root / rel_or_abs
        if candidate.exists():
            return candidate

    return writable_target
