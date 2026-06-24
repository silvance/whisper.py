import pytest

from whispr.transcription import (
    Segment,
    TranscriptionResult,
    _format_timestamp,
    is_supported_media,
    transcribe_audio,
)


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0, "00:00:00,000"),
        (1.5, "00:00:01,500"),
        (61.25, "00:01:01,250"),
        (3661.007, "01:01:01,007"),
        (-5, "00:00:00,000"),
    ],
)
def test_format_timestamp(seconds, expected):
    assert _format_timestamp(seconds) == expected


@pytest.mark.parametrize(
    "name,expected",
    [
        ("clip.mp3", True),
        ("clip.WAV", True),
        ("recording.mp4", True),
        ("notes.txt", False),
        ("archive.zip", False),
    ],
)
def test_is_supported_media(name, expected):
    assert is_supported_media(name) is expected


def test_to_srt_formats_segments():
    result = TranscriptionResult(
        text="hello\nworld",
        language="en",
        language_probability=0.99,
        duration=2.0,
        segments=[
            Segment(start=0.0, end=1.0, text="hello"),
            Segment(start=1.0, end=2.0, text="world"),
        ],
    )
    srt = result.to_srt()
    assert "1\n00:00:00,000 --> 00:00:01,000\nhello" in srt
    assert "2\n00:00:01,000 --> 00:00:02,000\nworld" in srt


def test_to_txt_returns_text():
    result = TranscriptionResult(
        text="some text", language="en", language_probability=1.0, duration=1.0
    )
    assert result.to_txt() == "some text"


def test_transcribe_audio_missing_file(tmp_path):
    # FileNotFoundError is raised before the optional backend is needed.
    with pytest.raises(FileNotFoundError):
        transcribe_audio(tmp_path / "does-not-exist.mp3")


def test_transcribe_audio_without_backend(tmp_path):
    """When faster-whisper is not installed, a clear RuntimeError is raised."""
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("faster-whisper is installed; backend-missing path not exercised")

    media = tmp_path / "clip.mp3"
    media.write_bytes(b"\x00")  # file must exist so we reach the import guard
    with pytest.raises(RuntimeError, match="faster-whisper is not installed"):
        transcribe_audio(media)
