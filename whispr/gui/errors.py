"""Map raw backend exceptions to plain-English messages for non-technical users.

The GUI shows the friendly line and still logs the full traceback to the Status
pane for troubleshooting.
"""

from __future__ import annotations


def friendly_error(exc: Exception) -> str:
    """Return a plain-English one-liner describing ``exc``."""
    name = type(exc).__name__
    msg = str(exc)
    low = msg.lower()
    if isinstance(exc, FileNotFoundError) or "no such file" in low:
        return f"A required file was missing: {msg}"
    if "ffmpeg" in low:
        return (
            "Couldn't run ffmpeg, which is needed to read this file. The "
            "packaged app includes ffmpeg; if running from source, install "
            "ffmpeg and make sure it's on your PATH."
        )
    if "faster-whisper is not installed" in low or "faster_whisper" in low:
        return (
            "The transcription engine isn't installed "
            "(pip install 'silvance-whisper[gui]')."
        )
    if "sherpa-onnx is not installed" in low:
        return (
            "The sherpa speaker engine isn't installed. Switch Engine to "
            "pyannote, or install sherpa-onnx."
        )
    if "pyannote.audio is not installed" in low:
        return (
            "The pyannote speaker engine isn't installed. Switch Engine to "
            "sherpa, or use a build that includes pyannote."
        )
    if (
        "no diarization models" in low
        or "offline mode" in low
        or "localentrynotfound" in name.lower()
    ):
        return (
            "Couldn't load the speaker models. Use a build that bundles them, "
            "switch the Engine, or untick 'Identify speakers' to transcribe only."
        )
    if "memoryerror" in name.lower() or "out of memory" in low:
        return (
            "Ran out of memory. Try a smaller model (e.g. base.en) or a "
            "shorter recording."
        )
    if (
        "tesseract" in low
        or "pytesseract" in low
        or "pypdfium2" in low
        or "pillow" in low
    ):
        detail = msg.splitlines()[0] if msg else name
        return (
            "OCR couldn't run in this build (it must be built with OCR bundled — "
            f"'ocr_langs' set). Details: {detail}"
        )
    short = msg.splitlines()[0] if msg else name
    return f"Something went wrong ({name}): {short}"
