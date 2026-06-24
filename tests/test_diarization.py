import pytest

from whispr.diarization import SpeakerSegment, assign_speakers, diarize
from whispr.transcription import Segment


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


def test_diarize_without_backend(tmp_path):
    # sherpa-onnx is not installed in this environment -> clear RuntimeError.
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"\x00")
    with pytest.raises(RuntimeError, match="sherpa-onnx is not installed"):
        diarize(wav)
