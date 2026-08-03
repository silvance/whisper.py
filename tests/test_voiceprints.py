import math

from whispr.diarization import SpeakerSegment
from whispr.voiceprints import (
    Voiceprint,
    best_match,
    centroid,
    cosine_similarity,
    recognize,
)


def test_normalize_and_cosine_of_identical_directions():
    a = [3.0, 0.0, 0.0]
    b = [10.0, 0.0, 0.0]
    assert cosine_similarity(a, b) == 1.0


def test_cosine_of_orthogonal_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_handles_empty_and_mismatched():
    assert cosine_similarity([], [1.0]) == 0.0
    assert cosine_similarity([1.0, 2.0], [1.0]) == 0.0


def test_centroid_is_unit_length_mean_direction():
    c = centroid([[2.0, 0.0], [0.0, 2.0]])
    # Mean direction is (1,1) normalized -> both components 1/sqrt(2).
    assert math.isclose(c[0], 1 / math.sqrt(2), rel_tol=1e-9)
    assert math.isclose(c[1], 1 / math.sqrt(2), rel_tol=1e-9)


def test_voiceprint_add_caps_and_normalizes():
    vp = Voiceprint(name="Alpha")
    vp.add([5.0, 0.0])
    assert math.isclose(vp.vectors[0][0], 1.0)  # normalized on the way in
    # The centroid of one east-pointing vector is east.
    assert math.isclose(vp.centroid[0], 1.0)


def test_best_match_picks_closest_above_threshold():
    east = Voiceprint(name="East", vectors=[[1.0, 0.0]])
    north = Voiceprint(name="North", vectors=[[0.0, 1.0]])
    name, score = best_match([0.9, 0.1], [east, north], threshold=0.5)
    assert name == "East"
    assert score > 0.5


def test_best_match_returns_none_below_threshold():
    east = Voiceprint(name="East", vectors=[[1.0, 0.0]])
    name, score = best_match([0.0, 1.0], [east], threshold=0.5)
    assert name is None


class _FakeEmbedder:
    """Returns a fixed vector per turn by looking up its start time."""

    def __init__(self, by_start):
        self._by_start = by_start

    def embed_span(self, _wav, start, _end):
        return self._by_start.get(round(start, 3))


def test_recognize_reattributes_quiet_turn_to_its_own_voice():
    # Two enrolled voices, orthogonal so matching is unambiguous.
    loud = Voiceprint(name="Loud", vectors=[[1.0, 0.0]])
    quiet = Voiceprint(name="Quiet", vectors=[[0.0, 1.0]])

    # The diarizer wrongly lumped everything into one cluster (SPEAKER_00), but
    # the middle turn is actually the quiet speaker.
    segments = [
        SpeakerSegment(start=0.0, end=3.0, speaker="SPEAKER_00"),
        SpeakerSegment(start=3.0, end=6.0, speaker="SPEAKER_00"),  # actually Quiet
        SpeakerSegment(start=6.0, end=9.0, speaker="SPEAKER_00"),
    ]
    embedder = _FakeEmbedder(
        {0.0: [1.0, 0.0], 3.0: [0.0, 1.0], 6.0: [1.0, 0.0]}
    )
    out, name_map = recognize("x.wav", segments, [loud, quiet], embedder)
    speakers = [s.speaker for s in out]
    assert speakers[0] == "voice::Loud"
    assert speakers[1] == "voice::Quiet"  # rescued from the loud cluster
    assert speakers[2] == "voice::Loud"
    assert name_map == {"voice::Loud": "Loud", "voice::Quiet": "Quiet"}


def test_recognize_names_whole_cluster_and_skips_short_turns():
    loud = Voiceprint(name="Loud", vectors=[[1.0, 0.0]])
    segments = [
        SpeakerSegment(start=0.0, end=2.0, speaker="SPEAKER_00"),
        SpeakerSegment(start=2.0, end=2.3, speaker="SPEAKER_00"),  # too short to embed
    ]
    embedder = _FakeEmbedder({0.0: [1.0, 0.0]})
    out, name_map = recognize("x.wav", segments, [loud], embedder)
    # The short turn inherits the cluster's dominant name via the fallback.
    assert [s.speaker for s in out] == ["voice::Loud", "voice::Loud"]
    assert name_map == {"voice::Loud": "Loud"}


def test_recognize_no_voiceprints_is_identity():
    segments = [SpeakerSegment(start=0.0, end=2.0, speaker="SPEAKER_00")]
    out, name_map = recognize("x.wav", segments, [], _FakeEmbedder({}))
    assert [s.speaker for s in out] == ["SPEAKER_00"]
    assert name_map == {}


def test_recognize_leaves_unmatched_turns_alone():
    east = Voiceprint(name="East", vectors=[[1.0, 0.0]])
    segments = [SpeakerSegment(start=0.0, end=2.0, speaker="SPEAKER_00")]
    # Turn embedding is orthogonal to the only voiceprint -> no match.
    embedder = _FakeEmbedder({0.0: [0.0, 1.0]})
    out, name_map = recognize("x.wav", segments, [east], embedder)
    assert [s.speaker for s in out] == ["SPEAKER_00"]
    assert name_map == {}
