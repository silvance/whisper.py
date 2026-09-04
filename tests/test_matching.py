import math

import pytest

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
    BAND_INTERMEDIATE,
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
        [1.0, 0.0],
        gallery,
        questioned_seconds=20.0,
        questioned_quality=GOOD,
        questioned_model=MODEL,
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
        [1.0, 0.0],
        gallery,
        questioned_seconds=20.0,
        questioned_quality=GOOD,
        questioned_model=MODEL,
    )
    assert result.accepted_name is None
    assert "No known profile produced a sufficiently strong match." in "\n".join(
        result.summary_lines()
    )


def test_gallery_reports_nothing_when_all_below_threshold():
    gallery = [_profile("Actor A", _at_cosine(0.4))]
    result = search_gallery(
        [1.0, 0.0],
        gallery,
        questioned_seconds=20.0,
        questioned_quality=GOOD,
        questioned_model=MODEL,
    )
    assert result.accepted_name is None
    assert result.matches[0].band == BAND_LOW


def test_gallery_skips_incompatible_profiles():
    gallery = [_profile("Actor A", model=OTHER_MODEL), _profile("Actor B")]
    result = search_gallery(
        [1.0, 0.0],
        gallery,
        questioned_seconds=20.0,
        questioned_quality=GOOD,
        questioned_model=MODEL,
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
        [1.0, 0.0],
        gallery,
        questioned_seconds=20.0,
        questioned_quality=GOOD,
        questioned_model=MODEL,
    )
    data = result.to_dict()
    assert data["searched"] == 2
    assert data["decision"]["best_name"] == "Actor A"
    assert data["decision"]["margin"] > 0
    assert data["disclaimer"] == DISCLAIMER


def test_comparison_result_defaults_are_inconclusive():
    assert ComparisonResult().conclusive is False


def test_a_configured_threshold_override_is_honoured(monkeypatch):
    """The comparison band edge comes from the active set, not a frozen default."""
    from whispr import thresholds as th

    monkeypatch.setattr(th, "_active", th.Thresholds(comparison_high=0.99))
    profile = _profile(seconds=30.0)
    result = compare_embedding_to_profile(
        _at_cosine(0.90),
        profile,
        questioned_seconds=20.0,
        questioned_quality=GOOD,
        questioned_model=MODEL,
    )
    # 0.90 clears the shipped 0.62 but not the configured 0.99.
    assert result.score == pytest.approx(0.90, abs=1e-6)
    assert result.band == BAND_INTERMEDIATE
    assert result.operational_threshold == 0.99


# -- The gallery and the 1:1 path must agree about the same clip -------------


def _both_paths(seconds, quality):
    """Run the same questioned audio through both comparison paths."""
    profile = _profile("Actor A", _at_cosine(0.95))
    one_to_one = compare_embedding_to_profile(
        [1.0, 0.0],
        profile,
        questioned_seconds=seconds,
        questioned_quality=quality,
        questioned_model=MODEL,
    )
    gallery = search_gallery(
        [1.0, 0.0],
        [profile],
        questioned_seconds=seconds,
        questioned_quality=quality,
        questioned_model=MODEL,
    )
    return one_to_one, gallery


def test_insufficient_quality_blocks_a_gallery_lead_as_well_as_a_comparison():
    """The same clip must not be 'Insufficient data' one way and a lead the other."""
    one_to_one, gallery = _both_paths(seconds=30.0, quality=INSUFFICIENT)
    assert one_to_one.band == BAND_INSUFFICIENT
    assert gallery.accepted_name is None
    assert gallery.inadequate_reason
    summary = " ".join(gallery.summary_lines())
    assert BAND_INSUFFICIENT in summary
    assert "supports no conclusion" in summary
    # The high score is still visible; it is the conclusion that is withheld.
    assert gallery.matches and gallery.matches[0].score > 0.9


def test_too_little_speech_blocks_a_gallery_lead_as_well_as_a_comparison():
    one_to_one, gallery = _both_paths(seconds=1.0, quality=GOOD)
    assert one_to_one.band == BAND_INSUFFICIENT
    assert gallery.accepted_name is None
    assert "1.0s" in gallery.inadequate_reason


def test_adequate_audio_produces_a_lead_on_both_paths():
    one_to_one, gallery = _both_paths(seconds=30.0, quality=GOOD)
    assert one_to_one.band == BAND_HIGH
    assert gallery.accepted_name == "Actor A"
    assert gallery.inadequate_reason == ""


def test_gallery_defaults_to_refusing_rather_than_assuming_good_audio():
    # A caller that forgets to say gets the conservative answer.
    gallery = search_gallery(
        [1.0, 0.0],
        [_profile("Actor A", _at_cosine(0.95))],
        questioned_seconds=30.0,
        questioned_model=MODEL,
    )
    assert gallery.accepted_name is None


# -- Two subjects with the same display name --------------------------------


def test_a_repeated_display_name_is_disambiguated_in_the_ranking():
    first = _profile("J. Smith", _at_cosine(0.9))
    second = _profile("J. Smith", _at_cosine(0.3))
    result = search_gallery(
        [1.0, 0.0],
        [first, second],
        questioned_seconds=30.0,
        questioned_quality=GOOD,
        questioned_model=MODEL,
    )
    # Both subjects are searched and ranked separately - one cannot stand in
    # for the other.
    assert result.searched == 2
    assert len(result.matches) == 2
    names = [m.display_name for m in result.matches]
    assert len(set(names)) == 2
    assert first.subject_id in names[0]
    subject_ids = [m.subject_id for m in result.matches]
    assert subject_ids == [first.subject_id, second.subject_id]


def test_a_lead_against_a_repeated_name_says_which_subject():
    first = _profile("J. Smith", _at_cosine(0.95))
    second = _profile("J. Smith", _at_cosine(0.2))
    result = search_gallery(
        [1.0, 0.0],
        [first, second],
        questioned_seconds=30.0,
        questioned_quality=GOOD,
        questioned_model=MODEL,
    )
    assert result.accepted_name is not None
    assert first.subject_id in result.accepted_name


# -- Comparison provenance ---------------------------------------------------


def test_a_comparison_can_name_the_recording_it_measured():
    result = compare_embedding_to_profile(
        [1.0, 0.0],
        _profile(),
        questioned_seconds=20.0,
        questioned_quality=GOOD,
        questioned_model=MODEL,
    )
    result.questioned_source_filename = "intercept-042.wav"
    result.questioned_source_sha256 = "c" * 64
    result.questioned_selection = "diarized speaker (SPEAKER_01)"
    result.questioned_window_count = 3
    text = "\n".join(result.format_lines())
    assert "intercept-042.wav" in text
    assert ("c" * 64) in text
    assert "diarized speaker (SPEAKER_01)" in text
    assert "measured across 3 window(s)" in text
    assert result.to_dict()["questioned_source_sha256"] == "c" * 64


def test_a_refused_comparison_still_names_the_recording():
    result = compare_embedding_to_profile(
        [1.0, 0.0],
        _profile(model=OTHER_MODEL),
        questioned_seconds=20.0,
        questioned_quality=GOOD,
        questioned_model=MODEL,
    )
    result.questioned_source_filename = "intercept-042.wav"
    result.questioned_source_sha256 = "d" * 64
    assert result.refused
    text = "\n".join(result.format_lines())
    assert "intercept-042.wav" in text and ("d" * 64) in text


def test_a_comparison_without_recorded_provenance_claims_none():
    result = compare_embedding_to_profile(
        [1.0, 0.0],
        _profile(),
        questioned_seconds=20.0,
        questioned_quality=GOOD,
        questioned_model=MODEL,
    )
    text = "\n".join(result.format_lines())
    assert "Questioned recording:" not in text
    assert "Source SHA-256" not in text


def test_a_refused_gallery_search_does_not_print_an_acceptance_rationale():
    """The refusal must not be followed by the reasoning for a lead it withheld."""
    gallery = search_gallery(
        [1.0, 0.0],
        [_profile("Actor A", _at_cosine(0.95))],
        questioned_seconds=30.0,
        questioned_quality=INSUFFICIENT,
        questioned_model=MODEL,
    )
    text = " ".join(gallery.summary_lines())
    assert BAND_INSUFFICIENT in text
    assert gallery.decision is None
    # No ">= threshold with margin >=" style justification anywhere.
    assert ">=" not in text
    assert "margin" not in text.lower()


def test_an_adequate_gallery_search_still_explains_its_decision():
    gallery = search_gallery(
        [1.0, 0.0],
        [_profile("Actor A", _at_cosine(0.68)), _profile("Actor B", _at_cosine(0.67))],
        questioned_seconds=30.0,
        questioned_quality=GOOD,
        questioned_model=MODEL,
    )
    assert gallery.decision is not None
    assert gallery.accepted_name is None
    # Ambiguous, and it says why.
    assert any("margin" in line.lower() for line in gallery.summary_lines())
