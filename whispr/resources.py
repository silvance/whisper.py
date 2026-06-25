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
