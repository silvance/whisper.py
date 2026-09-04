import math

from whispr.diarization import SpeakerSegment
from whispr.voiceprints import (
    Voiceprint,
    best_match,
    centroid,
    compare_voiceprints,
    cosine_similarity,
    decide_identity,
    rank_candidates,
    recognize,
    similarity_band,
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


def test_compare_voiceprints_same_direction_is_high():
    a = Voiceprint(name="A", vectors=[[1.0, 0.0], [0.9, 0.1]])
    b = Voiceprint(name="B", vectors=[[1.0, 0.0]])
    assert compare_voiceprints(a, b) > 0.9


def test_compare_voiceprints_orthogonal_is_zero():
    a = Voiceprint(name="A", vectors=[[1.0, 0.0]])
    b = Voiceprint(name="B", vectors=[[0.0, 1.0]])
    assert compare_voiceprints(a, b) == 0.0


def test_similarity_band_uses_similarity_not_identity_language():
    high, high_text = similarity_band(0.9)
    mid, _ = similarity_band(0.5)
    low, low_text = similarity_band(0.1)
    assert high == "High similarity"
    assert mid == "Intermediate similarity"
    assert low == "Low similarity"
    # Never asserts identity, probability or "same speaker".
    for text in (high_text, low_text, high, low):
        lowered = text.lower()
        assert "same speaker" not in lowered
        assert "probability" not in lowered
        assert "%" not in lowered


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
    embedder = _FakeEmbedder({0.0: [1.0, 0.0], 3.0: [0.0, 1.0], 6.0: [1.0, 0.0]})
    out, name_map = recognize("x.wav", segments, [loud, quiet], embedder)
    speakers = [s.speaker for s in out]
    assert speakers[0] == "voice::Loud"
    assert speakers[1] == "voice::Quiet"  # rescued from the loud cluster
    assert speakers[2] == "voice::Loud"
    assert name_map == {"voice::Loud": "Loud", "voice::Quiet": "Quiet"}


def test_identity_does_not_propagate_to_unmatched_short_turns():
    """A short turn must not inherit a known identity from its cluster."""
    loud = Voiceprint(name="Loud", vectors=[[1.0, 0.0]])
    segments = [
        SpeakerSegment(start=0.0, end=2.0, speaker="SPEAKER_00"),
        SpeakerSegment(start=2.0, end=2.3, speaker="SPEAKER_00"),  # too short to embed
    ]
    embedder = _FakeEmbedder({0.0: [1.0, 0.0]})
    out, name_map = recognize("x.wav", segments, [loud], embedder)
    # The matched turn is attributed; the unmatched short one stays unknown.
    assert [s.speaker for s in out] == ["voice::Loud", "SPEAKER_00"]
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


# -- conservative open-set decisions ---------------------------------------


def _candidates(*pairs):
    return [(name, list(vec)) for name, vec in pairs]


def _at_cosine(target):
    """A unit vector whose cosine similarity to [1, 0] is exactly ``target``."""
    return [target, math.sqrt(max(0.0, 1.0 - target * target))]


def test_decide_accepts_clear_winner():
    """The brief's example: 0.81 vs 0.54 may be accepted."""
    decision = decide_identity(
        [1.0, 0.0],
        _candidates(("Actor_A", _at_cosine(0.81)), ("Actor_B", _at_cosine(0.54))),
        acceptance=0.62,
        margin=0.08,
    )
    assert decision.accepted is True
    assert decision.best_name == "Actor_A"
    assert math.isclose(decision.best_score, 0.81, abs_tol=1e-6)
    assert math.isclose(decision.second_score, 0.54, abs_tol=1e-6)
    assert decision.margin > 0.08


def test_decide_rejects_below_acceptance():
    decision = decide_identity(
        [1.0, 0.0],
        _candidates(
            ("Actor_A", _at_cosine(0.55)),
        ),
        acceptance=0.62,
        margin=0.08,
    )
    assert decision.accepted is False
    assert decision.best_name == "Actor_A"  # reported for diagnostics...
    assert "below the acceptance threshold" in decision.reason


def test_decide_rejects_ambiguous_top_two():
    """The brief's example: 0.68 vs 0.67 is ambiguous - leave it unknown."""
    decision = decide_identity(
        [1.0, 0.0],
        _candidates(("Actor_A", _at_cosine(0.68)), ("Actor_B", _at_cosine(0.67))),
        acceptance=0.62,
        margin=0.08,
    )
    assert decision.accepted is False
    assert "Ambiguous" in decision.reason
    assert decision.second_name == "Actor_B"
    assert decision.margin < 0.08


def test_decide_rejects_insufficient_speech():
    decision = decide_identity(
        [1.0, 0.0],
        _candidates(
            ("Actor_A", (1.0, 0.0)),
        ),
        acceptance=0.5,
        margin=0.08,
        speech_seconds=1.0,
        min_speech_seconds=3.0,
    )
    assert decision.accepted is False
    assert "at least" in decision.reason


def test_decide_with_no_candidates():
    decision = decide_identity([1.0, 0.0], [])
    assert decision.accepted is False
    assert decision.best_name is None
    assert "No enrolled voices" in decision.reason


def test_decision_diagnostics_round_trip():
    decision = decide_identity(
        [1.0, 0.0],
        _candidates(("Actor_A", (1.0, 0.0)), ("Actor_B", (0.0, 1.0))),
        speech_seconds=12.5,
    )
    data = decision.to_dict()
    assert data["best_name"] == "Actor_A"
    assert data["second_name"] == "Actor_B"
    assert data["margin"] == decision.margin
    assert data["speech_seconds"] == 12.5
    assert data["acceptance_threshold"] == decision.acceptance_threshold


def test_rank_candidates_orders_by_score():
    ranked = rank_candidates(
        [1.0, 0.0],
        _candidates(("Far", (0.0, 1.0)), ("Near", (1.0, 0.0)), ("Mid", (0.7, 0.7))),
    )
    assert [name for name, _ in ranked] == ["Near", "Mid", "Far"]


def test_recognize_leaves_ambiguous_turn_unknown():
    """Two known actors scoring 0.68/0.67 must not yield an attribution."""
    a = Voiceprint(name="Actor_A", vectors=[_at_cosine(0.68)])
    b = Voiceprint(name="Actor_B", vectors=[_at_cosine(0.67)])
    segments = [SpeakerSegment(start=0.0, end=5.0, speaker="SPEAKER_00")]
    embedder = _FakeEmbedder({0.0: [1.0, 0.0]})
    out, name_map = recognize("x.wav", segments, [a, b], embedder)
    assert out[0].speaker == "SPEAKER_00"
    assert name_map == {}
