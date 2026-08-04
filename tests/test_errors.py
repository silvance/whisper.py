import pytest

# whispr.gui imports tkinter (via the package __init__); skip cleanly where the
# Tk runtime isn't available. errors.py itself has no GUI dependency.
errors = pytest.importorskip("whispr.gui.errors")
friendly_error = errors.friendly_error


def test_model_not_in_build_surfaces_message_verbatim():
    exc = RuntimeError(
        "The model 'medium.en' isn't in this build. Open Self-test… to see the "
        "bundled models and pick one of those, or rebuild the bundle with "
        "'medium.en' included."
    )
    out = friendly_error(exc)
    assert out.startswith("The model 'medium.en' isn't in this build.")
    # Not wrapped in the generic "Something went wrong" fallback.
    assert "Something went wrong" not in out


def test_faster_whisper_missing_is_friendly():
    out = friendly_error(RuntimeError("faster-whisper is not installed"))
    assert "transcription engine isn't installed" in out


def test_unknown_error_falls_back():
    out = friendly_error(ValueError("some obscure failure"))
    assert out.startswith("Something went wrong (ValueError):")
