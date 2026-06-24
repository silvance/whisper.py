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

from pathlib import Path

from PyInstaller.utils.hooks import collect_all

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
assets = Path("whispr_assets")
if assets.is_dir():
    datas.append((str(assets), "whispr_assets"))

a = Analysis(
    ["packaging/whispr_entry.py"],
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
