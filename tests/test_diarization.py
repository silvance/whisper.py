import pytest

from whispr.diarization import SpeakerSegment, assign_speakers, diarize
from whispr.transcription import Segment, Word


def test_assign_speakers_by_overlap():
    segments = [
        Segment(start=0.0, end=2.0, text="hello"),
        Segment(start=2.0, end=4.0, text="world"),
        Segment(start=4.0, end=6.0, text="again"),
    ]
    speakers = [
        SpeakerSegment(start=0.0, end=2.1, speaker="SPEAKER_00"),
        SpeakerSegment(start=2.1, end=6.0, speaker="SPEAKER_01"),
    ]
    assign_speakers(segments, speakers)
    assert [s.speaker for s in segments] == [
        "SPEAKER_00",
        "SPEAKER_01",
        "SPEAKER_01",
    ]


def test_assign_speakers_no_overlap_leaves_none():
    segments = [Segment(start=10.0, end=12.0, text="orphan")]
    speakers = [SpeakerSegment(start=0.0, end=5.0, speaker="SPEAKER_00")]
    assign_speakers(segments, speakers)
    assert segments[0].speaker is None


def test_assign_speakers_picks_majority_overlap():
    # Segment 1-5: overlaps SPEAKER_00 for 1s (1-2) and SPEAKER_01 for 3s (2-5).
    segments = [Segment(start=1.0, end=5.0, text="mostly one")]
    speakers = [
        SpeakerSegment(start=0.0, end=2.0, speaker="SPEAKER_00"),
        SpeakerSegment(start=2.0, end=8.0, speaker="SPEAKER_01"),
    ]
    assign_speakers(segments, speakers)
    assert segments[0].speaker == "SPEAKER_01"


def test_assign_speakers_splits_segment_by_word():
    # One Whisper segment spanning a speaker change mid-way.
    segment = Segment(
        start=0.0,
        end=4.0,
        text="hello there how are you",
        words=[
            Word(start=0.0, end=1.0, word=" hello"),
            Word(start=1.0, end=2.0, word=" there"),
            Word(start=2.0, end=3.0, word=" how"),
            Word(start=3.0, end=4.0, word=" you"),
        ],
    )
    speakers = [
        SpeakerSegment(start=0.0, end=2.0, speaker="SPEAKER_00"),
        SpeakerSegment(start=2.0, end=4.0, speaker="SPEAKER_01"),
    ]
    out = assign_speakers([segment], speakers)
    assert [s.speaker for s in out] == ["SPEAKER_00", "SPEAKER_01"]
    assert out[0].text == "hello there"
    assert out[1].text == "how you"


def test_assign_speakers_gap_word_uses_nearest():
    segment = Segment(
        start=10.0,
        end=11.0,
        text="orphan",
        words=[Word(start=10.0, end=11.0, word="orphan")],
    )
    speakers = [SpeakerSegment(start=0.0, end=5.0, speaker="SPEAKER_00")]
    out = assign_speakers([segment], speakers)
    assert out[0].speaker == "SPEAKER_00"  # nearest turn, not None


def test_assign_speakers_smooths_spurious_short_run():
    # A spurious 1-word island (from overlap/jitter) wedged between two longer
    # runs of the same speaker should be absorbed back into them.
    segment = Segment(
        start=0.0,
        end=4.0,
        text="alpha beta gamma delta epsilon",
        words=[
            Word(start=0.0, end=0.8, word=" alpha"),
            Word(start=0.8, end=1.6, word=" beta"),
            Word(start=1.6, end=1.9, word=" gamma"),  # spurious single-word blip
            Word(start=1.9, end=2.8, word=" delta"),
            Word(start=2.8, end=4.0, word=" epsilon"),
        ],
    )
    speakers = [
        SpeakerSegment(start=0.0, end=1.6, speaker="SPEAKER_00"),
        SpeakerSegment(start=1.6, end=1.9, speaker="SPEAKER_01"),  # blip turn
        SpeakerSegment(start=1.9, end=4.0, speaker="SPEAKER_00"),
    ]
    out = assign_speakers([segment], speakers)
    # Without smoothing this would split into three runs; the blip is absorbed.
    assert [s.speaker for s in out] == ["SPEAKER_00"]
    assert out[0].text == "alpha beta gamma delta epsilon"


def test_assign_speakers_keeps_real_short_turn_at_min_length():
    # A clearly-spoken short turn longer than the smoothing window survives.
    segment = Segment(
        start=0.0,
        end=4.0,
        text="long opening yes long closing",
        words=[
            Word(start=0.0, end=1.2, word=" long"),
            Word(start=1.2, end=1.6, word=" opening"),
            Word(start=1.6, end=2.6, word=" yes"),  # 1.0s > 0.8s window
            Word(start=2.6, end=3.2, word=" long"),
            Word(start=3.2, end=4.0, word=" closing"),
        ],
    )
    speakers = [
        SpeakerSegment(start=0.0, end=1.6, speaker="SPEAKER_00"),
        SpeakerSegment(start=1.6, end=2.6, speaker="SPEAKER_01"),
        SpeakerSegment(start=2.6, end=4.0, speaker="SPEAKER_00"),
    ]
    out = assign_speakers([segment], speakers)
    assert [s.speaker for s in out] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]


def test_diarize_without_backend(tmp_path):
    # sherpa-onnx is not installed in this environment -> clear RuntimeError.
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"\x00")
    with pytest.raises(RuntimeError, match="sherpa-onnx is not installed"):
        diarize(wav)
