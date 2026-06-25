# PyInstaller spec for the Whispers GUI (one-dir bundle).
#
# Build (sherpa-onnx diarizer, smaller):
#     pip install "silvance-whisper[gui,bundle]"
#     python packaging/fetch_assets.py ffmpeg
#     python packaging/fetch_assets.py models small,medium,large-v3
#     python packaging/fetch_assets.py diarization
#     pyinstaller --noconfirm packaging/whispr.spec
#
# Build (pyannote diarizer, best quality on hard audio; adds CPU PyTorch):
#     pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
#     pip install "silvance-whisper[gui,bundle,pyannote]"
#     python packaging/fetch_assets.py ffmpeg
#     python packaging/fetch_assets.py models small,medium,large-v3
#     HF_TOKEN=... python packaging/fetch_assets.py pyannote
#     pyinstaller --noconfirm packaging/whispr.spec
#
# Optional text translation (offline Argos Translate):
#     pip install "silvance-whisper[gui,bundle,translate]"
#     python packaging/fetch_assets.py argos ar,ru,zh,fa,uk,he,ko
#     (plus the steps above, then pyinstaller)
#
# The spec auto-detects whether pyannote.audio / argostranslate are installed: it
# bundles the PyTorch stack + offline pyannote cache and/or the Argos stack +
# language packs accordingly; otherwise it builds the lighter sherpa-onnx-only
# bundle and excludes torch.
#
# The resulting dist/whispr/ folder is fully self-contained (Python runtime, all
# dependencies, the ffmpeg binary, and the Whisper + diarization models) and can be
# copied to an air-gapped machine and run with no network access.

import importlib.util
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, copy_metadata

# SPECPATH is injected by PyInstaller and is the directory containing this spec
# (i.e. the packaging/ folder). Paths in the spec are resolved relative to it, so
# anchor everything explicitly: the entry script lives beside the spec, and the
# offline assets live in the repo root (one level up).
SPEC_DIR = Path(SPECPATH)  # noqa: F821 - provided by PyInstaller at exec time
REPO_ROOT = SPEC_DIR.parent

# pyannote.audio (PyTorch) is bundled only when it is installed in the build
# environment. When absent, the bundle uses the sherpa-onnx diarizer and torch is
# excluded entirely (it is never imported and is the bulk of the size).
PYANNOTE = importlib.util.find_spec("pyannote.audio") is not None

# argostranslate (offline text translation) is bundled only when installed. It
# hard-imports stanza, so stanza is pulled in too; spacy is optional (guarded) and
# left out via excludes below.
ARGOS = importlib.util.find_spec("argostranslate") is not None

datas = []
binaries = []
hiddenimports = []

# Native libraries / data files the always-present packages need at runtime.
packages = [
    "faster_whisper",
    "ctranslate2",
    "av",
    "onnxruntime",
    "tokenizers",
    "ttkbootstrap",
]
# The pyannote/PyTorch dependency tree. collect_all is wrapped in try/except so a
# package that isn't present (or has no collectable data) doesn't abort the build.
if PYANNOTE:
    packages += [
        "torch",
        "torchaudio",
        "soundfile",  # libsndfile - pyannote 3.1.1's audio backend
        "pyannote",
        "asteroid_filterbanks",
        "lightning_fabric",
        "pytorch_lightning",
        "sklearn",
        "scipy",
        "omegaconf",
        "networkx",
        "huggingface_hub",
        "transformers",
        "sympy",
    ]

# Argos Translate dependency tree (CTranslate2 is already collected above when
# present via faster-whisper). stanza is a hard import in argostranslate's sbd
# module; sentencepiece/sacremoses/minisbd are the tokenizers/sentence splitter.
if ARGOS:
    packages += [
        "argostranslate",
        "stanza",
        "sentencepiece",
        "sacremoses",
        "minisbd",
        "ctranslate2",
    ]

for package in packages:
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    except Exception as exc:  # noqa: BLE001 - best-effort collection
        print(f"whispr.spec: skipping collect_all({package!r}): {exc}")
        continue
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

# Some of these libraries read their own distribution metadata at runtime
# (importlib.metadata.version(...)); bundle it so they don't crash when frozen.
if PYANNOTE:
    for dist in (
        "torch",
        "pyannote.audio",
        "pytorch_lightning",
        "lightning_fabric",
        "asteroid_filterbanks",
        "huggingface_hub",
        "tqdm",
        "filelock",
        "regex",
        "requests",
        "packaging",
    ):
        try:
            datas += copy_metadata(dist)
        except Exception as exc:  # noqa: BLE001 - metadata may be absent
            print(f"whispr.spec: skipping copy_metadata({dist!r}): {exc}")

if ARGOS:
    for dist in (
        "argostranslate",
        "stanza",
        "sentencepiece",
        "sacremoses",
        "ctranslate2",
        "minisbd",
    ):
        try:
            datas += copy_metadata(dist)
        except Exception as exc:  # noqa: BLE001 - metadata may be absent
            print(f"whispr.spec: skipping copy_metadata({dist!r}): {exc}")

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
    # Exclude torch only for the sherpa-onnx build; when pyannote is bundled we
    # need torch. tensorflow is never used. spacy is an optional argostranslate
    # dependency we don't use (we force the MiniSBD sentence splitter), so drop it.
    excludes=(
        ["tensorflow"]
        + ([] if PYANNOTE else ["torch"])
        + (["spacy"] if ARGOS else [])
    ),
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
