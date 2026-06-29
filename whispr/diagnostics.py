"""A build self-test: report which engines, models and assets are actually present.

Bundles are assembled by a CI workflow with several optional pieces (diarizers,
translation packs, OCR), and it's easy to ship one that's missing something. This
gathers a plain report so whoever hands out a build can confirm it's complete
*before* it reaches an air-gapped machine. It only inspects (``find_spec`` +
cheap filesystem checks) - it never loads the heavy libraries.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import List

from . import resources
from .playback import playback_available


@dataclass
class Check:
    """One diagnostic line: a label, an OK/missing flag, and a detail string."""

    label: str
    ok: bool
    detail: str


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _tessdata_languages() -> List[str]:
    directory = resources.bundled_tessdata_dir()
    if directory is None:
        return []
    return sorted(p.stem for p in directory.glob("*.traineddata"))


def gather() -> List[Check]:
    """Collect the diagnostic checks for the current install/bundle."""
    checks: List[Check] = []

    # --- Transcription (always required) ---
    checks.append(
        Check(
            "Transcription engine (faster-whisper)",
            _installed("faster_whisper"),
            "installed" if _installed("faster_whisper") else "MISSING",
        )
    )
    models = sorted(resources.bundled_models())
    checks.append(
        Check(
            "Bundled models",
            bool(models),
            ", ".join(models) if models else "none bundled (will need a model path)",
        )
    )
    ffmpeg = resources.find_ffmpeg()
    checks.append(
        Check("ffmpeg", ffmpeg is not None, str(ffmpeg) if ffmpeg else "not found")
    )

    # --- Diarization (either backend works) ---
    pyannote = _installed("pyannote.audio")
    checks.append(
        Check(
            "Diarization: pyannote.audio",
            pyannote,
            "installed" if pyannote else "not installed",
        )
    )
    checks.append(
        Check(
            "Diarization: pyannote model cache",
            resources.pyannote_cache_dir() is not None,
            "bundled" if resources.pyannote_cache_dir() else "not bundled",
        )
    )
    sherpa = _installed("sherpa_onnx")
    checks.append(
        Check(
            "Diarization: sherpa-onnx",
            sherpa,
            "installed" if sherpa else "not installed",
        )
    )
    checks.append(
        Check(
            "Diarization: sherpa models",
            resources.bundled_diarization_models() is not None,
            "bundled" if resources.bundled_diarization_models() else "not bundled",
        )
    )

    # --- Translation ---
    argos = _installed("argostranslate")
    checks.append(
        Check(
            "Translation engine (Argos)",
            argos,
            "installed" if argos else "not installed",
        )
    )
    checks.append(
        Check(
            "Translation: bundled language packs",
            resources.bundled_argos_data_dir() is not None,
            "bundled" if resources.bundled_argos_data_dir() else "not bundled",
        )
    )

    # --- OCR ---
    tesseract = resources.find_tesseract()
    checks.append(
        Check(
            "OCR: Tesseract binary",
            tesseract is not None,
            str(tesseract) if tesseract else "not found",
        )
    )
    langs = _tessdata_languages()
    checks.append(
        Check(
            "OCR: tessdata languages",
            bool(langs),
            ", ".join(langs) if langs else "none bundled",
        )
    )
    ocr_libs = (
        _installed("pytesseract") and _installed("PIL") and _installed("pypdfium2")
    )
    checks.append(
        Check(
            "OCR: Python libraries",
            ocr_libs,
            "installed" if ocr_libs else "pytesseract / Pillow / pypdfium2 missing",
        )
    )

    # --- UX add-ons ---
    checks.append(
        Check(
            "Word export (python-docx)",
            _installed("docx"),
            "installed" if _installed("docx") else "not installed",
        )
    )
    checks.append(
        Check(
            "Auto-detect (langdetect)",
            _installed("langdetect"),
            "installed" if _installed("langdetect") else "not installed",
        )
    )
    checks.append(
        Check(
            "Drag-and-drop (tkinterdnd2)",
            _installed("tkinterdnd2"),
            "installed" if _installed("tkinterdnd2") else "not installed",
        )
    )
    checks.append(
        Check(
            "Audio playback",
            playback_available(),
            "available"
            if playback_available()
            else "unavailable (needs ffmpeg + a player)",
        )
    )
    return checks


def format_report(checks: "List[Check] | None" = None) -> str:
    """Render the checks as an aligned, human-readable report."""
    if checks is None:
        checks = gather()
    width = max((len(c.label) for c in checks), default=0)
    lines = ["Whispers build self-test", "=" * 24, ""]
    for check in checks:
        mark = "OK " if check.ok else "-- "
        lines.append(f"[{mark}] {check.label.ljust(width)}  {check.detail}")
    return "\n".join(lines)
