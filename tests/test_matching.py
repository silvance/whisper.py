import math

from whispr.matching import (
    ComparisonResult,
    compare_embedding_to_profile,
    search_gallery,
)
from whispr.quality import GOOD, INSUFFICIENT, POOR
from whispr.speaker_profiles import (
    EmbeddingModelIdentity,
    EnrollmentSample,
    SpeakerProfile,
)
from whispr.thresholds import (
    BAND_HIGH,
    BAND_INSUFFICIENT,
    BAND_LOW,
    DISCLAIMER,
)

MODEL = EmbeddingModelIdentity(
    name="titanet-large", sha256="a" * 64, vector_dimension=2
)
OTHER_MODEL = EmbeddingModelIdentity(
    name="campplus", sha256="b" * 64, vector_dimension=2
)


def _at_cosine(target):
    return [target, math.sqrt(max(0.0, 1.0 - target * target))]


def _profile(name="Subject A", vec=(1.0, 0.0), seconds=20.0, quality=GOOD, model=MODEL):
    profile = SpeakerProfile(display_name=name, embedding_model=model)
    profile.add_reference_sample(
        EnrollmentSample(
            embedding=list(vec),
            speech_duration=seconds,
            quality={
                "assessment": quality,
                "voiced_seconds": seconds,
                "duration_seconds": seconds,
                "warnings": [],
            },
        )
    )
    return profile


# -- 1:1 comparison --------------------------------------------------------


def test_high_similarity_comparison():
    result = compare_embedding_to_profile(
        [1.0, 0.0],
        _profile(),
        questioned_seconds=15.0,
        questioned_quality=GOOD,
        questioned_model=MODEL,
    )
    assert result.refused is False
    assert result.band == BAND_HIGH
    assert math.isclose(result.score, 1.0, abs_tol=1e-9)
    assert result.conclusive is True


def test_low_similarity_comparison():
    result = compare_embedding_to_profile(
        _at_cosine(0.2),
        _profile(),
        questioned_seconds=15.0,
        questioned_quality=GOOD,
        questioned_model=MODEL,
    )
    assert result.band == BAND_LOW


def test_result_never_reports_a_percentage_or_probability():
    result = compare_embedding_to_profile(
        [1.0, 0.0],
        _profile(),
        questioned_seconds=15.0,
        questioned_quality=GOOD,
        questioned_model=MODEL,
    )
    text = "\n".join(result.format_lines()).lower()
    assert "similarity score: 1.00 / 1.00" in text
    for banned in (
        "%",
        "probability",
        "confidence of identity",
        "same person",
        "positive identification",
        "likelihood of identity",
    ):
        assert banned not in text


def test_format_lines_include_thresholds_durations_and_quality():
    result = compare_embedding_to_profile(
        [1.0, 0.0],
        _profile(seconds=38.4),
        questioned_seconds=14.2,
        questioned_quality="Fair",
        questioned_model=MODEL,
    )
    text = "\n".join(result.format_lines())
    assert "Operational threshold:" in text
    assert "Reference speech: 38.4 sec" in text
    assert "Questioned speech: 14.2 sec" in text
    assert "Reference quality: Good" in text
    assert "Questioned quality: Fair" in text


def test_disclaimer_is_carried_in_the_record():
    result = compare_embedding_to_profile(
        [1.0, 0.0],
        _profile(),
        questioned_seconds=15.0,
        questioned_quality=GOOD,
        questioned_model=MODEL,
    )
    assert result.to_dict()["disclaimer"] == DISCLAIMER
    assert "not forensic speaker identification" in DISCLAIMER


# -- adequacy overrides the number ----------------------------------------


def test_short_questioned_speech_is_insufficient_despite_perfect_score():
    result = compare_embedding_to_profile(
        [1.0, 0.0],
        _profile(),
        questioned_seconds=1.5,
        questioned_quality=GOOD,
        questioned_model=MODEL,
    )
    assert result.band == BAND_INSUFFICIENT
    assert result.conclusive is False
    assert any("Questioned speech is only" in w for w in result.warnings)


def test_thin_reference_profile_is_insufficient():
    result = compare_embedding_to_profile(
        [1.0, 0.0],
        _profile(seconds=4.0),
        questioned_seconds=20.0,
        questioned_quality=GOOD,
        questioned_model=MODEL,
    )
    assert result.band == BAND_INSUFFICIENT
    assert any("reference profile holds only" in w for w in result.warnings)


def test_insufficient_questioned_quality_blocks_assessment():
    result = compare_embedding_to_profile(
        [1.0, 0.0],
        _profile(),
        questioned_seconds=20.0,
        questioned_quality=INSUFFICIENT,
        questioned_model=MODEL,
    )
    assert result.band == BAND_INSUFFICIENT


def test_poor_quality_still_assessed_but_warned():
    result = compare_embedding_to_profile(
        [1.0, 0.0],
        _profile(),
        questioned_seconds=20.0,
        questioned_quality=POOR,
        questioned_warnings=["Audio is very quiet."],
        questioned_model=MODEL,
    )
    assert result.band == BAND_HIGH
    assert "Audio is very quiet." in result.warnings


def test_profile_without_trusted_samples_is_insufficient():
    empty = SpeakerProfile(display_name="Nobody", embedding_model=MODEL)
    result = compare_embedding_to_profile(
        [1.0, 0.0],
        empty,
        questioned_seconds=20.0,
        questioned_quality=GOOD,
        questioned_model=MODEL,
    )
    assert result.band == BAND_INSUFFICIENT
    assert result.refused is False


# -- model compatibility ---------------------------------------------------


def test_mismatched_model_refuses_comparison():
    result = compare_embedding_to_profile(
        [1.0, 0.0],
        _profile(model=OTHER_MODEL),
        questioned_seconds=20.0,
        questioned_quality=GOOD,
        questioned_model=MODEL,
    )
    assert result.refused is True
    assert "not comparable" in result.refusal_reason
    assert "Comparison refused." in result.format_lines()


def test_legacy_profile_refused_by_default():
    result = compare_embedding_to_profile(
        [1.0, 0.0],
        _profile(model=None),
        questioned_seconds=20.0,
        questioned_quality=GOOD,
        questioned_model=MODEL,
    )
    assert result.refused is True
    assert result.compatibility is not None
    assert result.compatibility.needs_confirmation is True


def test_legacy_profile_comparable_only_with_explicit_confirmation():
    result = compare_embedding_to_profile(
        [1.0, 0.0],
        _profile(model=None),
        questioned_seconds=20.0,
        questioned_quality=GOOD,
        questioned_model=MODEL,
        allow_unverified_model=True,
    )
    assert result.refused is False
    assert any("could not be verified" in w for w in result.warnings)


def test_proven_mismatch_cannot_be_overridden():
    result = compare_embedding_to_profile(
        [1.0, 0.0],
        _profile(model=OTHER_MODEL),
        questioned_seconds=20.0,
        questioned_quality=GOOD,
        questioned_model=MODEL,
        allow_unverified_model=True,  # must not rescue a real mismatch
    )
    assert result.refused is True


# -- gallery search --------------------------------------------------------


def test_gallery_ranks_and_accepts_clear_winner():
    gallery = [
        _profile("Actor A", _at_cosine(0.81)),
        _profile("Actor D", _at_cosine(0.59)),
        _profile("Actor B", _at_cosine(0.42)),
    ]
    result = search_gallery(
        [1.0, 0.0], gallery, questioned_seconds=20.0, questioned_model=MODEL
    )
    assert [m.display_name for m in result.matches] == ["Actor A", "Actor D", "Actor B"]
    assert result.searched == 3
    assert result.accepted_name == "Actor A"
    text = "\n".join(result.summary_lines())
    assert "Further review is warranted" in text
    assert "%" not in text


def test_gallery_refuses_ambiguous_top_two():
    gallery = [
        _profile("Actor A", _at_cosine(0.68)),
        _profile("Actor B", _at_cosine(0.67)),
    ]
    result = search_gallery(
        [1.0, 0.0], gallery, questioned_seconds=20.0, questioned_model=MODEL
    )
    assert result.accepted_name is None
    assert "No known profile produced a sufficiently strong match." in "\n".join(
        result.summary_lines()
    )


def test_gallery_reports_nothing_when_all_below_threshold():
    gallery = [_profile("Actor A", _at_cosine(0.4))]
    result = search_gallery(
        [1.0, 0.0], gallery, questioned_seconds=20.0, questioned_model=MODEL
    )
    assert result.accepted_name is None
    assert result.matches[0].band == BAND_LOW


def test_gallery_skips_incompatible_profiles():
    gallery = [_profile("Actor A", model=OTHER_MODEL), _profile("Actor B")]
    result = search_gallery(
        [1.0, 0.0], gallery, questioned_seconds=20.0, questioned_model=MODEL
    )
    assert result.searched == 1
    assert any("Actor A" in reason for reason in result.skipped)


def test_gallery_short_questioned_speech_accepts_nothing():
    gallery = [_profile("Actor A", _at_cosine(0.95))]
    result = search_gallery(
        [1.0, 0.0], gallery, questioned_seconds=1.0, questioned_model=MODEL
    )
    assert result.accepted_name is None


def test_gallery_record_round_trip():
    gallery = [
        _profile("Actor A", _at_cosine(0.81)),
        _profile("Actor B", _at_cosine(0.3)),
    ]
    result = search_gallery(
        [1.0, 0.0], gallery, questioned_seconds=20.0, questioned_model=MODEL
    )
    data = result.to_dict()
    assert data["searched"] == 2
    assert data["decision"]["best_name"] == "Actor A"
    assert data["decision"]["margin"] > 0
    assert data["disclaimer"] == DISCLAIMER


def test_comparison_result_defaults_are_inconclusive():
    assert ComparisonResult().conclusive is False
