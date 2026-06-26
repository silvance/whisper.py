from whispr.project import PROJECT_SUFFIX, load_project, save_project
from whispr.transcription import Segment, TranscriptionResult, Word


def _result():
    return TranscriptionResult(
        text="hello world",
        language="en",
        language_probability=0.99,
        duration=2.0,
        segments=[
            Segment(
                start=0.0,
                end=1.0,
                text="hello",
                speaker="SPEAKER_00",
                avg_logprob=-0.2,
                words=[Word(start=0.0, end=1.0, word=" hello", probability=0.9)],
            ),
            Segment(
                start=1.0,
                end=2.0,
                text="world",
                speaker="SPEAKER_01",
                words=[Word(start=1.0, end=2.0, word=" world", probability=0.4)],
            ),
        ],
    )


def test_save_then_load_round_trips(tmp_path):
    result = _result()
    names = {"SPEAKER_00": "Alice"}
    path = tmp_path / ("clip" + PROJECT_SUFFIX)
    save_project(path, result, names, source="/audio/clip.mp3")
    assert path.exists()

    loaded, loaded_names, source = load_project(path)
    assert source == "/audio/clip.mp3"
    assert loaded_names == names
    assert loaded.language == "en"
    assert loaded.duration == 2.0
    assert [s.speaker for s in loaded.segments] == ["SPEAKER_00", "SPEAKER_01"]
    assert [s.text for s in loaded.segments] == ["hello", "world"]
    assert loaded.segments[0].words[0].probability == 0.9
    assert loaded.segments[0].avg_logprob == -0.2
    assert loaded.has_speakers


def test_load_tolerates_missing_optional_fields(tmp_path):
    path = tmp_path / ("min" + PROJECT_SUFFIX)
    path.write_text(
        '{"version": 1, "result": {"segments": [{"text": "hi"}]}}', encoding="utf-8"
    )
    loaded, names, source = load_project(path)
    assert names == {}
    assert source is None
    assert loaded.segments[0].text == "hi"
    assert loaded.segments[0].speaker is None


def test_round_trip_preserves_non_diarized(tmp_path):
    result = TranscriptionResult(
        text="one two",
        language="en",
        language_probability=1.0,
        duration=1.0,
        segments=[Segment(start=0.0, end=1.0, text="one two")],
    )
    path = tmp_path / ("nd" + PROJECT_SUFFIX)
    save_project(path, result)
    loaded, names, source = load_project(path)
    assert not loaded.has_speakers
    assert loaded.segments[0].text == "one two"
    assert names == {}
