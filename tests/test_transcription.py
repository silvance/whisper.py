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


# -- silence skipping -------------------------------------------------------
# Field report: with "Skip silence" on, a pause was followed by speech being
# dropped until somebody spoke up. Silero enters a speech region at `threshold`
# but only leaves below `neg_threshold`, so on muffled audio - where the speech
# probability sits around 0.3-0.5 - a genuine pause ends the region and the
# quieter speech after it never reaches faster-whisper's default 0.5 to start a
# new one. These pin the settings that answer that, and the reporting that
# makes the loss visible when it still happens.


def test_speech_is_entered_below_faster_whispers_default():
    """0.5 is tuned for clean speech and discards muffled speech after a pause."""
    assert transcription.VAD_THRESHOLD < 0.5
    assert transcription.vad_options()["threshold"] == transcription.VAD_THRESHOLD


def test_leaving_speech_is_harder_than_entering_it():
    """Hysteresis has to stay in the right direction, or regions never close."""
    assert transcription.VAD_NEG_THRESHOLD < transcription.VAD_THRESHOLD


def test_a_pause_longer_than_a_conversation_beat_is_needed_to_close_a_region():
    assert transcription.VAD_MIN_SILENCE_MS >= 2000


def test_word_onsets_are_padded_more_than_the_default():
    """The start of a word is its quietest part, and the first thing clipped."""
    assert transcription.VAD_SPEECH_PAD_MS > 400


def test_every_tuned_value_reaches_faster_whisper():
    options = transcription.vad_options()
    assert options == {
        "threshold": transcription.VAD_THRESHOLD,
        "neg_threshold": transcription.VAD_NEG_THRESHOLD,
        "min_silence_duration_ms": transcription.VAD_MIN_SILENCE_MS,
        "speech_pad_ms": transcription.VAD_SPEECH_PAD_MS,
    }


# -- reporting what was skipped --------------------------------------------


def _result(duration, after_vad):
    return transcription.TranscriptionResult(
        text="",
        language="en",
        language_probability=1.0,
        duration=duration,
        duration_after_vad=after_vad,
    )


def test_a_recording_that_was_heard_whole_reports_nothing_skipped():
    result = _result(600.0, 600.0)
    assert result.skipped_seconds == 0.0
    assert result.kept_fraction == pytest.approx(1.0)


def test_what_the_filter_removed_is_measurable():
    result = _result(600.0, 150.0)
    assert result.skipped_seconds == pytest.approx(450.0)
    assert result.kept_fraction == pytest.approx(0.25)


def test_an_unfiltered_run_reports_no_fraction_rather_than_a_full_one():
    """Silence skipping off is not the same as it having kept everything."""
    result = _result(600.0, None)
    assert result.kept_fraction is None
    assert result.skipped_seconds == 0.0


def test_a_zero_length_recording_does_not_divide_by_zero():
    assert _result(0.0, 0.0).kept_fraction is None


# -- the settings have to survive the trip to the model ---------------------


class _FakeInfo:
    language = "en"
    language_probability = 1.0
    duration = 600.0
    duration_after_vad = 120.0


class _FakeModel:
    """Records what transcribe() was handed, and returns one empty segment."""

    def __init__(self):
        self.kwargs = None

    def transcribe(self, path, **kwargs):
        self.kwargs = kwargs
        return iter(()), _FakeInfo()


def _run(monkeypatch, tmp_path, **overrides):
    model = _FakeModel()
    monkeypatch.setattr(transcription, "_load_model", lambda *a, **k: model)
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"\x00")
    result = transcription.transcribe_audio(audio, **overrides)
    return model, result


def test_the_tuned_settings_are_handed_to_the_model(monkeypatch, tmp_path):
    model, _ = _run(monkeypatch, tmp_path, vad_filter=True)
    assert model.kwargs["vad_filter"] is True
    assert model.kwargs["vad_parameters"] == transcription.vad_options()


def test_no_settings_are_sent_when_silence_skipping_is_off(monkeypatch, tmp_path):
    model, _ = _run(monkeypatch, tmp_path, vad_filter=False)
    assert model.kwargs["vad_filter"] is False
    assert model.kwargs["vad_parameters"] is None


def test_how_much_was_skipped_comes_back_on_the_result(monkeypatch, tmp_path):
    _, result = _run(monkeypatch, tmp_path, vad_filter=True)
    assert result.duration_after_vad == pytest.approx(120.0)
    assert result.kept_fraction == pytest.approx(0.2)


def test_an_unfiltered_run_does_not_claim_a_filtered_duration(monkeypatch, tmp_path):
    """The model still reports duration_after_vad; it means nothing here."""
    _, result = _run(monkeypatch, tmp_path, vad_filter=False)
    assert result.duration_after_vad is None
    assert result.kept_fraction is None
