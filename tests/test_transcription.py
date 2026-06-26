import pytest

from whispr import transcription
from whispr.transcription import (
    Segment,
    TranscriptionResult,
    Word,
    _format_timestamp,
    convert_to_wav,
    is_low_confidence_segment,
    is_low_confidence_word,
    is_supported_media,
    is_video,
    transcribe_audio,
)


@pytest.mark.parametrize(
    "prob,expected",
    [(0.95, False), (0.55, False), (0.4, True), (None, False)],
)
def test_is_low_confidence_word(prob, expected):
    word = Word(start=0.0, end=1.0, word="x", probability=prob)
    assert is_low_confidence_word(word) is expected


@pytest.mark.parametrize(
    "logprob,expected",
    [(-0.1, False), (-0.7, False), (-1.2, True), (None, False)],
)
def test_is_low_confidence_segment(logprob, expected):
    seg = Segment(start=0.0, end=1.0, text="x", avg_logprob=logprob)
    assert is_low_confidence_segment(seg) is expected


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


@pytest.mark.parametrize(
    "name,expected",
    [
        ("movie.mp4", True),
        ("clip.MKV", True),
        ("recording.mov", True),
        ("audio.mp3", False),
        ("audio.wav", False),
    ],
)
def test_is_video(name, expected):
    assert is_video(name) is expected


def test_convert_to_wav_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        convert_to_wav(tmp_path / "missing.mp4")


def test_convert_to_wav_without_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setattr(transcription, "find_ffmpeg", lambda: None)
    media = tmp_path / "movie.mp4"
    media.write_bytes(b"\x00")
    with pytest.raises(RuntimeError, match="ffmpeg was not found"):
        convert_to_wav(media)


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


def test_to_txt_with_speakers():
    result = TranscriptionResult(
        text="hello\nworld",
        language="en",
        language_probability=1.0,
        duration=2.0,
        segments=[
            Segment(start=0.0, end=1.0, text="hello", speaker="SPEAKER_00"),
            Segment(start=1.0, end=2.0, text="world", speaker="SPEAKER_01"),
        ],
    )
    assert result.has_speakers
    assert result.to_txt() == "[SPEAKER_00] hello\n[SPEAKER_01] world"
    # blank_lines puts an empty line between turns.
    assert result.to_txt(blank_lines=True) == "[SPEAKER_00] hello\n\n[SPEAKER_01] world"


def test_to_txt_blank_lines_without_speakers():
    result = TranscriptionResult(
        text="one\ntwo",
        language="en",
        language_probability=1.0,
        duration=2.0,
        segments=[
            Segment(start=0.0, end=1.0, text="one"),
            Segment(start=1.0, end=2.0, text="two"),
        ],
    )
    assert result.to_txt() == "one\ntwo"
    assert result.to_txt(blank_lines=True) == "one\n\ntwo"


def test_to_srt_with_speakers():
    result = TranscriptionResult(
        text="hi",
        language="en",
        language_probability=1.0,
        duration=1.0,
        segments=[Segment(start=0.0, end=1.0, text="hi", speaker="SPEAKER_02")],
    )
    assert "[SPEAKER_02] hi" in result.to_srt()


def test_speaker_names_remap_txt_and_srt():
    result = TranscriptionResult(
        text="hello\nworld",
        language="en",
        language_probability=1.0,
        duration=2.0,
        segments=[
            Segment(start=0.0, end=1.0, text="hello", speaker="SPEAKER_00"),
            Segment(start=1.0, end=2.0, text="world", speaker="SPEAKER_01"),
        ],
    )
    names = {"SPEAKER_00": "Xin"}
    txt = result.to_txt(names)
    assert "[Xin] hello" in txt
    assert "[SPEAKER_01] world" in txt  # unmapped id falls through unchanged
    assert "[Xin] hello" in result.to_srt(names)


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
