# PyInstaller spec for the W.H.I.S.P.R. GUI (one-dir bundle).
#
# Build:
#     pip install "silvance-whisper[gui,bundle]"
#     python packaging/fetch_assets.py ffmpeg
#     python packaging/fetch_assets.py models small,medium,large-v3
#     pyinstaller --noconfirm packaging/whispr.spec
#
# The resulting dist/whispr/ folder is fully self-contained (Python runtime, all
# dependencies, the ffmpeg binary, and the Whisper models) and can be copied to an
# air-gapped machine and run with no network access.

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

# SPECPATH is injected by PyInstaller and is the directory containing this spec
# (i.e. the packaging/ folder). Paths in the spec are resolved relative to it, so
# anchor everything explicitly: the entry script lives beside the spec, and the
# offline assets live in the repo root (one level up).
SPEC_DIR = Path(SPECPATH)  # noqa: F821 - provided by PyInstaller at exec time
REPO_ROOT = SPEC_DIR.parent

datas = []
binaries = []
hiddenimports = []

# Collect the native libraries / data files these packages need at runtime.
for package in (
    "faster_whisper",
    "ctranslate2",
    "av",
    "onnxruntime",
    "tokenizers",
    "ttkbootstrap",
):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

# Bundle the offline assets (ffmpeg + models) fetched by fetch_assets.py.
assets = REPO_ROOT / "whispr_assets"
if assets.is_dir():
    datas.append((str(assets), "whispr_assets"))

# This build is CPU-only (the GUI always runs device="cpu"), so strip the CUDA /
# cuDNN libraries that ctranslate2 and onnxruntime ship in their wheels - they are
# never loaded and account for the bulk of the bundle size.
_CUDA_MARKERS = (
    "cudnn",
    "cublas",
    "cudart",
    "cufft",
    "curand",
    "cusolver",
    "cusparse",
    "cupti",
    "nvrtc",
    "nvtx",
    "libcuda",
    "onnxruntime_providers_cuda",
    "onnxruntime_providers_tensorrt",
)


def _is_cuda_lib(name) -> bool:
    base = os.path.basename(str(name)).lower()
    return any(marker in base for marker in _CUDA_MARKERS)


binaries = [b for b in binaries if not _is_cuda_lib(b[0])]
datas = [d for d in datas if not _is_cuda_lib(d[0])]

a = Analysis(
    [os.path.join(SPEC_DIR, "whispr_entry.py")],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["torch", "tensorflow"],
    noarchive=False,
)

# Drop any CUDA libraries the dependency analysis pulled in as well.
a.binaries = [b for b in a.binaries if not _is_cuda_lib(b[0])]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="whispr",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="whispr",
)
