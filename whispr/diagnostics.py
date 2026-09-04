"""A build self-test: deployment verification before a bundle leaves for the field.

Bundles are assembled by a CI workflow with several optional pieces (diarizers,
translation packs, OCR), and it's easy to ship one that's missing something. This
answers the questions that decide whether a build is usable for an operation:

    Can this build transcribe?
    Can this build diarize?
    Can this build create speaker profiles?
    Can this build compare voices?
    Which speaker-embedding model does it use, and what is its SHA-256?
    Can it export reports?
    Which decision thresholds are in force, and were any overridden?
    Is every required local asset present?

and summarises them as READY / NOT READY, with the build identity (build id,
version, commit) and the hashes of the bundled models. Whoever hands out a build
runs this *before* it reaches an air-gapped machine.

It only inspects - ``find_spec`` plus cheap filesystem reads (hashing a model is
the one heavier operation, and is done lazily) - and never loads the heavy
libraries or touches the network.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import List

from . import resources, thresholds
from .buildinfo import build_info
from .hashing import short
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
    # The speaker-embedding model powers speaker profiles + comparison; two
    # builds must share it to compare voiceprints, so name it here - as the file
    # it actually is, not as a broad capability claim.
    embedding = resources.bundled_embedding_model()
    described = resources.describe_bundled_embedding_model()
    checks.append(
        Check(
            "Speaker embedding (speaker profiles)",
            embedding is not None,
            (described or "bundled (model not recorded by this build)")
            if embedding is not None
            else "not bundled",
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


# -- Capability verification -----------------------------------------------


@dataclass
class Capability:
    """One thing an operator needs the build to do, and whether it can."""

    name: str
    available: bool
    detail: str
    # Critical capabilities gate the READY verdict; the rest merely degrade.
    critical: bool = True


def _can_transcribe() -> Capability:
    engine = _installed("faster_whisper")
    models = sorted(resources.bundled_models())
    ok = engine and bool(models)
    if not engine:
        detail = "faster-whisper is not installed"
    elif not models:
        detail = "no transcription models are bundled"
    else:
        detail = f"yes - models: {', '.join(models)}"
    return Capability("Can transcribe", ok, detail)


def _can_diarize() -> Capability:
    pyannote = _installed("pyannote.audio") and (
        resources.pyannote_cache_dir() is not None
    )
    sherpa = _installed("sherpa_onnx") and (
        resources.bundled_diarization_models() is not None
    )
    engines = [
        name for name, ready in (("pyannote", pyannote), ("sherpa", sherpa)) if ready
    ]
    return Capability(
        "Can separate speakers",
        bool(engines),
        f"yes - {', '.join(engines)}" if engines else "no diarization engine is ready",
    )


def _speaker_embedding() -> Capability:
    """Speaker enrolment and comparison both rest on this one model."""
    model = resources.bundled_embedding_model()
    engine = _installed("sherpa_onnx")
    if model is None:
        return Capability(
            "Speaker embedding model", False, "not bundled", critical=True
        )
    if not engine:
        return Capability(
            "Speaker embedding model",
            False,
            "bundled, but sherpa-onnx is not installed to run it",
        )
    name = resources.bundled_embedding_model_name() or "unknown"
    digest = build_info().model_sha256(name)
    if digest is None:
        from .hashing import sha256_file_or_none

        digest = sha256_file_or_none(model)
    described = resources.describe_bundled_embedding_model() or name
    return Capability(
        "Speaker embedding model", True, f"{described} (sha256 {short(digest)})"
    )


def _can_export_reports() -> Capability:
    docx = _installed("docx")
    return Capability(
        "Can export reports",
        True,  # text reports always work; DOCX is the richer option
        "yes - Word (.docx) and text" if docx else "text only (python-docx missing)",
        critical=False,
    )


def capabilities() -> List[Capability]:
    """The operator-facing capability answers for this build."""
    embedding = _speaker_embedding()
    ffmpeg = resources.find_ffmpeg()
    items = [
        Capability(
            "Can read audio/video (ffmpeg)",
            ffmpeg is not None,
            str(ffmpeg) if ffmpeg else "ffmpeg not found",
        ),
        _can_transcribe(),
        _can_diarize(),
        embedding,
        Capability(
            "Can create speaker profiles",
            embedding.available,
            "yes" if embedding.available else "needs the speaker-embedding model",
        ),
        Capability(
            "Can compare voices",
            embedding.available,
            "yes" if embedding.available else "needs the speaker-embedding model",
        ),
        _can_export_reports(),
    ]
    return items


def readiness(items: "List[Capability] | None" = None) -> "tuple[bool, List[str]]":
    """``(ready, blockers)`` - ready only when every critical capability is present."""
    if items is None:
        items = capabilities()
    blockers = [c.name for c in items if c.critical and not c.available]
    return (not blockers), blockers


def _wrap(text: str, width: int = 76) -> List[str]:
    """Wrap a paragraph for the fixed-width self-test window."""
    import textwrap

    return textwrap.wrap(text, width=width) or [""]


def format_report(checks: "List[Check] | None" = None) -> str:
    """Render the full self-test: identity, readiness, capabilities and details."""
    if checks is None:
        checks = gather()
    items = capabilities()
    ready, blockers = readiness(items)
    info = build_info()

    lines = ["Whispers build self-test", "=" * 24, ""]
    lines.append("READY" if ready else "NOT READY")
    if blockers:
        lines.append("Missing: " + ", ".join(blockers))
    lines.append("")

    lines += ["Build", "-----"]
    lines += info.describe()
    lines.append(
        "Runtime network access: none - Whispers performs no network calls, "
        "downloads no models and sends no telemetry."
    )
    lines.append("")

    lines += ["Capabilities", "------------"]
    width = max((len(c.name) for c in items), default=0)
    for capability in items:
        mark = "OK " if capability.available else "-- "
        lines.append(f"[{mark}] {capability.name.ljust(width)}  {capability.detail}")
    lines.append("")

    lines += ["Assets and components", "---------------------"]
    width = max((len(c.label) for c in checks), default=0)
    for check in checks:
        mark = "OK " if check.ok else "-- "
        lines.append(f"[{mark}] {check.label.ljust(width)}  {check.detail}")

    lines += ["", *thresholds.describe_active()]

    caveat = str(resources.bundled_embedding_model_info().get("caveat") or "")
    if caveat:
        lines += ["", "Speaker embedding model", "-----------------------"]
        for chunk in _wrap(caveat):
            lines.append(chunk)

    models = info.models
    if models:
        lines += ["", "Bundled model hashes", "--------------------"]
        for entry in models:
            lines.append(
                f"  {str(entry.get('friendly_name', '?')):<20} "
                f"{str(entry.get('kind', '')):<18} "
                f"{short(entry.get('sha256'))}"
            )
    return "\n".join(lines)
