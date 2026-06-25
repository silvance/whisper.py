"""Locate bundled, offline resources (ffmpeg binary and Whisper models).

This lets the GUI run on an air-gapped machine with no network access. It works
both when running from source and when frozen with PyInstaller (one-dir or
one-file), and an explicit override directory can be supplied via the
``WHISPR_ASSETS`` environment variable.

Expected layout of an assets directory::

    whispr_assets/
        ffmpeg/ffmpeg            (or ffmpeg.exe on Windows)
        models/
            small/model.bin ...
            medium/model.bin ...
            large-v3/model.bin ...
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from shutil import which
from typing import Dict, List, Optional, Tuple

ASSETS_DIRNAME = "whispr_assets"
ENV_ASSETS = "WHISPR_ASSETS"


def _candidate_asset_dirs() -> List[Path]:
    """Ordered candidate ``whispr_assets`` directories (existing or not)."""
    candidates: List[Path] = []

    override = os.environ.get(ENV_ASSETS)
    if override:
        candidates.append(Path(override))

    # PyInstaller one-file: resources are unpacked under sys._MEIPASS.
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / ASSETS_DIRNAME)

    # PyInstaller one-dir: resources sit next to the executable.
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent / ASSETS_DIRNAME)

    # Running from source: repo root (parent of the whispr package).
    candidates.append(Path(__file__).resolve().parent.parent / ASSETS_DIRNAME)

    # Last resort: current working directory.
    candidates.append(Path.cwd() / ASSETS_DIRNAME)

    return candidates


def asset_dirs() -> List[Path]:
    """Existing assets directories, de-duplicated and in priority order."""
    out: List[Path] = []
    seen = set()
    for candidate in _candidate_asset_dirs():
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_dir():
            out.append(resolved)
    return out


def find_bundled_ffmpeg() -> Optional[Path]:
    """Return the bundled ffmpeg executable, if one is shipped in the assets."""
    names = ("ffmpeg.exe", "ffmpeg") if os.name == "nt" else ("ffmpeg",)
    for base in asset_dirs():
        for directory in (base / "ffmpeg", base):
            for name in names:
                candidate = directory / name
                if candidate.is_file():
                    return candidate
    return None


def find_ffmpeg() -> Optional[Path]:
    """Return the ffmpeg to use: a bundled one if present, else one on PATH."""
    bundled = find_bundled_ffmpeg()
    if bundled is not None:
        return bundled
    on_path = which("ffmpeg")
    return Path(on_path) if on_path else None


def bundled_models() -> Dict[str, Path]:
    """Map of bundled model name -> directory (a CTranslate2 model folder).

    A directory is considered a model if it contains a ``model.bin`` file.
    """
    models: Dict[str, Path] = {}
    for base in asset_dirs():
        models_dir = base / "models"
        if not models_dir.is_dir():
            continue
        for child in sorted(models_dir.iterdir()):
            if child.is_dir() and (child / "model.bin").is_file():
                models.setdefault(child.name, child)
    return models


def bundled_diarization_models() -> Optional[Tuple[Path, Path]]:
    """Return ``(segmentation.onnx, embedding.onnx)`` if both are bundled."""
    for base in asset_dirs():
        directory = base / "diarization"
        segmentation = directory / "segmentation.onnx"
        embedding = directory / "embedding.onnx"
        if segmentation.is_file() and embedding.is_file():
            return segmentation, embedding
    return None


def pyannote_cache_dir() -> Optional[Path]:
    """Return a bundled offline Hugging Face cache for pyannote, if present.

    The build copies the downloaded pyannote models into ``whispr_assets/pyannote``
    (an HF cache layout); at runtime this is used with ``HF_HUB_OFFLINE`` so the
    air-gapped app needs no network or token.
    """
    for base in asset_dirs():
        directory = base / "pyannote"
        if directory.is_dir():
            return directory
    return None


def configure_offline_hf_cache() -> Optional[Path]:
    """Point Hugging Face at the bundled pyannote cache, in offline mode.

    Returns the configured cache directory, or ``None`` if none is bundled.

    This MUST run at process startup, before ``huggingface_hub`` (or anything that
    imports it - ``faster_whisper`` and ``pyannote.audio`` both do) is first
    imported: huggingface_hub reads ``HF_HOME`` / ``HF_HUB_OFFLINE`` into module
    constants at import time, so setting them later (e.g. inside the diarization
    call) has no effect and the app falls back to the network.

    No-op when no bundled cache is present (e.g. running from source), so normal
    online Hugging Face behaviour is unchanged.

    When a bundle IS present these are set unconditionally (not setdefault): the
    air-gapped app must use its own cache regardless of any HF_HOME / HF_HUB_CACHE
    / HF_HUB_OFFLINE the operator's machine happens to have set, which would
    otherwise point Hugging Face at the wrong (empty) cache in offline mode.
    """
    cache = pyannote_cache_dir()
    if cache is None:
        return None
    hub = str(cache / "hub")
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_HUB_CACHE"] = hub
    os.environ["HF_HUB_OFFLINE"] = "1"
    # pyannote.audio loads its sub-models (segmentation + speaker embedding) via
    # Model.from_pretrained, whose cache_dir defaults to PYANNOTE_CACHE (NOT
    # HF_HUB_CACHE) - and that default is captured into a module constant at
    # import time. Point it at the bundled cache here, before pyannote is imported,
    # so the sub-models resolve from the bundle too.
    os.environ["PYANNOTE_CACHE"] = hub
    return cache


def bundled_argos_data_dir() -> Optional[Path]:
    """Return the bundled Argos Translate data dir, if present.

    The build populates ``whispr_assets/argos/argos-translate`` (Argos's data dir,
    holding ``packages/`` and the ``minisbd/`` sentence-splitter cache). Argos
    derives all of these from ``XDG_DATA_HOME``, so the parent is used at runtime.
    """
    for base in asset_dirs():
        directory = base / "argos" / "argos-translate"
        if directory.is_dir():
            return directory
    return None


def configure_offline_translation() -> Optional[Path]:
    """Point Argos Translate at the bundled data dir, offline.

    Returns the data dir, or ``None`` if none is bundled. Like the HF config this
    MUST run at startup before ``argostranslate`` is imported (it reads
    ``XDG_DATA_HOME`` / ``ARGOS_CHUNK_TYPE`` into module state at import). No-op
    when no bundle is present, so from-source use is unchanged.

    Argos resolves its data dir as ``XDG_DATA_HOME/argos-translate`` and puts the
    language packs and the MiniSBD model cache under it, so pointing XDG_DATA_HOME
    at the bundle wires up both. ``ARGOS_CHUNK_TYPE=MINISBD`` forces the lightweight
    sentence splitter whose models we pre-cache at build time (no stanza models, no
    runtime downloads).
    """
    data = bundled_argos_data_dir()
    if data is None:
        return None
    os.environ["XDG_DATA_HOME"] = str(data.parent)
    os.environ["ARGOS_CHUNK_TYPE"] = "MINISBD"
    return data
