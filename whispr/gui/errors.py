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
    # A model chosen in the GUI that this offline build doesn't contain. The
    # message is already written for the operator, so surface it as-is.
    if "isn't in this build" in low:
        return msg.splitlines()[0]
    # These name the control the operator has to change, which now lives under
    # Advanced options > Speaker separation > Method.
    if "sherpa-onnx is not installed" in low:
        return (
            "One of the speaker-separation methods (sherpa) isn't installed in "
            "this build. Under Advanced options, set Speaker separation > "
            "Method to the pyannote option."
        )
    if "pyannote.audio is not installed" in low:
        return (
            "One of the speaker-separation methods (pyannote) isn't installed "
            "in this build. Under Advanced options, set Speaker separation > "
            "Method to the sherpa option."
        )
    if (
        "no diarization models" in low
        or "offline mode" in low
        or "localentrynotfound" in name.lower()
    ):
        return (
            "Couldn't load the speaker-separation models. Use a build that "
            "includes them, try the other Method under Advanced options, or "
            "untick 'Identify who is speaking' to transcribe only."
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
