# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 6.x Windows build spec for Audio8-VC.

Produces a shared onedir distribution (`dist/Audio8-VC/`) containing:
- `Audio8-VC.exe` : Windowed desktop app (WebView2 GUI + embedded FastAPI/WS server)
- `a8vc.exe`      : Console CLI companion sharing the same `_internal/` runtime
"""

from __future__ import annotations

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

SPEC_DIR = Path(SPECPATH).resolve()  # noqa: F821 (provided by PyInstaller)
REPO_ROOT = SPEC_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
ICON_PATH = SPEC_DIR / "assets" / "audio8_vc.ico"
VERSION_FILE = SPEC_DIR / "version_info.txt"

datas: list[tuple[str, str]] = [
    (str(REPO_ROOT / "configs" / "default.toml"), "configs"),
    (str(SRC_DIR / "a8vc" / "server" / "static"), "a8vc/server/static"),
    (str(SPEC_DIR / "assets" / "audio8_vc.ico"), "assets"),
    (str(SPEC_DIR / "assets" / "audio8_vc.png"), "assets"),
    (str(REPO_ROOT / "README.md"), "."),
]
binaries: list[tuple[str, str]] = []
hiddenimports: list[str] = [
    # pywebview Windows EdgeChromium (WebView2) + pythonnet/.NET interop
    "webview",
    "webview.platforms.winforms",
    "webview.platforms.edgechromium",
    "webview.platforms.win32",
    "clr",
    "clr_loader",
    "pythonnet",
    # uvicorn / FastAPI / WebSocket dynamic modules
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.loops.asyncio",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.http.httptools_impl",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.protocols.websockets.websockets_impl",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",
    "multipart",
    "python_multipart",
    # Windows multiprocessing & audio resampling
    "multiprocessing.popen_spawn_win32",
    "scipy.signal",
]

# Collect native DLLs, data files, and submodules for runtime packages
_COLLECT_ALL_PACKAGES = (
    "onnxruntime",
    "sherpa_onnx",
    "sounddevice",
    "_sounddevice_data",
    "soundfile",
    "_soundfile_data",
    "tokenizers",
    "webview",
    "clr_loader",
    "pythonnet",
)

for pkg in _COLLECT_ALL_PACKAGES:
    try:
        pkg_datas, pkg_bins, pkg_hidden = collect_all(pkg)
        datas += pkg_datas
        binaries += pkg_bins
        hiddenimports += pkg_hidden
    except Exception as exc:
        print(f"[audio8_vc.spec] Note: skipped collect_all({pkg!r}): {exc}")

for pkg in ("a8vc", "uvicorn", "fastapi", "starlette", "websockets", "anyio", "huggingface_hub"):
    try:
        hiddenimports += collect_submodules(pkg)
    except Exception as exc:
        print(f"[audio8_vc.spec] Note: skipped collect_submodules({pkg!r}): {exc}")

# Optional: bundle NVIDIA CUDA 12 / cuDNN 9 DLLs from `nvidia-*-cu12` wheels when requested
if os.environ.get("A8VC_BUNDLE_CUDA_RUNTIME", "").strip().lower() in {"1", "true", "yes"}:
    import importlib.util

    nvidia_spec = importlib.util.find_spec("nvidia")
    if nvidia_spec is not None and nvidia_spec.submodule_search_locations:
        for loc in nvidia_spec.submodule_search_locations:
            nvidia_root = Path(loc)
            for dll_path in sorted(nvidia_root.glob("*/bin/*.dll")):
                rel_dir = dll_path.parent.relative_to(nvidia_root)
                binaries.append((str(dll_path), str(Path("nvidia") / rel_dir)))
        print(f"[audio8_vc.spec] Collected optional NVIDIA CUDA runtime DLLs (total binaries={len(binaries)})")

# Trim unrelated heavy packages and non-Windows GUI toolkits
excludes = [
    "torch",
    "torchaudio",
    "torchvision",
    "transformers",
    "datasets",
    "accelerate",
    "deepspeed",
    "matplotlib",
    "tkinter",
    "pytest",
    "IPython",
    "jupyter",
    "webview.platforms.cocoa",
    "webview.platforms.gtk",
    "webview.platforms.qt",
    "webview.platforms.cef",
    "webview.platforms.android",
]

a_gui = Analysis(  # noqa: F821
    [str(SPEC_DIR / "entry_desktop.py")],
    pathex=[str(SRC_DIR)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

a_cli = Analysis(  # noqa: F821
    [str(SPEC_DIR / "entry_cli.py")],
    pathex=[str(SRC_DIR)],
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz_gui = PYZ(a_gui.pure)  # noqa: F821
pyz_cli = PYZ(a_cli.pure)  # noqa: F821

icon_arg = str(ICON_PATH) if ICON_PATH.is_file() else None
version_arg = str(VERSION_FILE) if VERSION_FILE.is_file() and os.name == "nt" else None

exe_gui = EXE(  # noqa: F821
    pyz_gui,
    a_gui.scripts,
    [],
    exclude_binaries=True,
    name="Audio8-VC",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=icon_arg,
    version=version_arg,
)

exe_cli = EXE(  # noqa: F821
    pyz_cli,
    a_cli.scripts,
    [],
    exclude_binaries=True,
    name="a8vc",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=icon_arg,
    version=version_arg,
)

coll = COLLECT(  # noqa: F821
    exe_gui,
    a_gui.binaries,
    a_gui.datas,
    exe_cli,
    a_cli.binaries,
    a_cli.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Audio8-VC",
)
