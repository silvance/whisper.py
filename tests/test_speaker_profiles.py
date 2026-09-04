import json

import pytest

import whispr.speaker_profiles as sp
from whispr.speaker_profiles import (
    SAMPLE_LEARNED,
    SAMPLE_REFERENCE,
    EmbeddingModelIdentity,
    EnrollmentSample,
    ProfileError,
    SpeakerProfile,
    check_compatibility,
    list_speaker_profiles,
    load_profile_file,
    save_speaker_profile,
)


def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "speakers_dir", lambda: tmp_path / "speakers")


def _model(sha="a" * 64, dim=2, name="titanet-large"):
    return EmbeddingModelIdentity(name=name, sha256=sha, vector_dimension=dim)


def _sample(vec=(1.0, 0.0), **kw):
    return EnrollmentSample(embedding=list(vec), speech_duration=12.0, **kw)


# -- v2 round trip ---------------------------------------------------------


def test_v2_save_load_round_trip(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    profile = SpeakerProfile(display_name="Subject A", embedding_model=_model())
    profile.add_reference_sample(_sample((1.0, 0.0)))
    path = save_speaker_profile(profile)

    (loaded,) = load_profile_file(path)
    assert loaded.display_name == "Subject A"
    assert loaded.subject_id == profile.subject_id
    assert loaded.schema_version == sp.SCHEMA_VERSION
    assert loaded.embedding_model is not None
    assert loaded.embedding_model.sha256 == "a" * 64
    assert loaded.embedding_model.vector_dimension == 2
    assert len(loaded.trusted_samples()) == 1
    assert loaded.is_legacy is False


def test_saved_file_records_schema_and_summary(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    profile = SpeakerProfile(display_name="B", embedding_model=_model())
    profile.add_reference_sample(_sample())
    path = save_speaker_profile(profile)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == sp.SCHEMA_VERSION
    assert data["summary"]["reference_sample_count"] == 1
    assert data["summary"]["total_reference_seconds"] == 12.0


def test_list_speaker_profiles(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    for name in ("One", "Two"):
        prof = SpeakerProfile(display_name=name, embedding_model=_model())
        prof.add_reference_sample(_sample())
        save_speaker_profile(prof)
    assert {p.display_name for p in list_speaker_profiles()} == {"One", "Two"}


def test_save_is_atomic_no_partial_file_on_failure(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    profile = SpeakerProfile(display_name="Atomic", embedding_model=_model())
    profile.add_reference_sample(_sample())
    path = save_speaker_profile(profile)
    original = path.read_text(encoding="utf-8")

    # A failure part-way through a re-save must leave the old file intact.
    def _boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(sp.os, "replace", _boom)
    with pytest.raises(ProfileError):
        save_speaker_profile(profile, path)
    assert path.read_text(encoding="utf-8") == original


# -- trust model -----------------------------------------------------------


def test_learned_sample_is_not_trusted_until_approved():
    profile = SpeakerProfile(display_name="C", embedding_model=_model())
    profile.add_reference_sample(_sample((1.0, 0.0)))
    sample, consistent = profile.propose_learned_sample(_sample((0.99, 0.1)))
    assert consistent is True
    assert sample.approved is False
    # Excluded from the trusted model until approved.
    assert sample not in profile.trusted_samples()
    assert profile.pending_samples() == [sample]

    assert profile.approve_sample(sample.sample_id) is True
    assert sample in profile.trusted_samples()


def test_mistaken_learned_sample_does_not_move_the_centroid():
    profile = SpeakerProfile(display_name="D", embedding_model=_model())
    profile.add_reference_sample(_sample((1.0, 0.0)))
    before = profile.centroid()
    # A wildly different (mis-corrected) voice.
    profile.propose_learned_sample(_sample((0.0, 1.0)))
    assert profile.centroid() == before


def test_outlier_learned_sample_is_flagged_for_review():
    profile = SpeakerProfile(display_name="E", embedding_model=_model())
    profile.add_reference_sample(_sample((1.0, 0.0)))
    sample, consistent = profile.propose_learned_sample(_sample((0.0, 1.0)))
    assert consistent is False
    assert "Outlier" in sample.notes
    assert sample.approved is False


def test_reference_samples_are_trusted_immediately():
    profile = SpeakerProfile(display_name="F", embedding_model=_model())
    sample = profile.add_reference_sample(_sample())
    assert sample.sample_type == SAMPLE_REFERENCE
    assert sample.is_trusted is True
    assert profile.total_reference_seconds == 12.0


def test_learned_samples_are_capped_but_reference_kept():
    profile = SpeakerProfile(display_name="G", embedding_model=_model())
    for _ in range(3):
        profile.add_reference_sample(_sample())
    for _ in range(sp.MAX_LEARNED_SAMPLES + 5):
        profile.propose_learned_sample(_sample((1.0, 0.0)))
    assert len(profile.reference_samples()) == 3
    assert len(profile.learned_samples()) == sp.MAX_LEARNED_SAMPLES


def test_remove_sample():
    profile = SpeakerProfile(display_name="H", embedding_model=_model())
    sample = profile.add_reference_sample(_sample())
    assert profile.remove_sample(sample.sample_id) is True
    assert profile.samples == []
    assert profile.remove_sample("nope") is False


def test_summary_and_source_files():
    profile = SpeakerProfile(display_name="I", embedding_model=_model())
    profile.add_reference_sample(_sample(source_filename="a.wav"))
    profile.add_reference_sample(_sample(source_filename="a.wav"))
    profile.add_reference_sample(_sample(source_filename="b.wav"))
    assert profile.source_files() == ["a.wav", "b.wav"]
    summary = profile.summary()
    assert summary["reference_sample_count"] == 3
    assert summary["source_file_count"] == 2


# -- model compatibility ---------------------------------------------------


def test_compatible_same_model():
    result = check_compatibility(_model(), _model())
    assert result.ok is True


def test_incompatible_different_model_hash():
    result = check_compatibility(_model(sha="a" * 64), _model(sha="b" * 64))
    assert result.ok is False
    assert result.needs_confirmation is False
    assert "not comparable" in result.reason


def test_incompatible_vector_dimension_mismatch():
    result = check_compatibility(_model(dim=192), _model(dim=256))
    assert result.ok is False
    assert result.needs_confirmation is False
    assert "dimension" in result.reason.lower()


def test_missing_model_identity_needs_confirmation():
    result = check_compatibility(None, _model())
    assert result.ok is False
    assert result.needs_confirmation is True


def test_missing_hash_needs_confirmation():
    result = check_compatibility(_model(sha=None), _model())
    assert result.ok is False
    assert result.needs_confirmation is True


# -- migration -------------------------------------------------------------


def test_import_legacy_voiceprint_file(tmp_path):
    legacy = tmp_path / "old.whispr-voiceprint.json"
    legacy.write_text(
        json.dumps(
            {
                "kind": "whispr-voiceprint",
                "name": "Falcon",
                "vectors": [[1.0, 0.0], [0.9, 0.1]],
                "source_profile": "Op One",
            }
        ),
        encoding="utf-8",
    )
    (profile,) = load_profile_file(legacy)
    assert profile.display_name == "Falcon"
    assert len(profile.samples) == 2
    # Preserved as usable history, but provenance is unknown.
    assert all(s.sample_type == SAMPLE_LEARNED for s in profile.samples)
    assert profile.is_legacy is True
    assert len(profile.trusted_samples()) == 2
    assert "Op One" in profile.notes


def test_import_legacy_operation_profile_yields_one_subject_per_voice(tmp_path):
    legacy = tmp_path / "op.whispr-profile.json"
    legacy.write_text(
        json.dumps(
            {
                "name": "Op Two",
                "settings": {"model": "small.en"},
                "voiceprints": [
                    {"name": "Alpha", "vectors": [[1.0, 0.0]]},
                    {"name": "Bravo", "vectors": [[0.0, 1.0]]},
                ],
            }
        ),
        encoding="utf-8",
    )
    profiles = load_profile_file(legacy)
    assert {p.display_name for p in profiles} == {"Alpha", "Bravo"}
    assert all(p.is_legacy for p in profiles)
    assert profiles[0].settings["model"] == "small.en"


def test_legacy_profile_is_not_silently_discarded(tmp_path):
    legacy = tmp_path / "empty.whispr-profile.json"
    legacy.write_text(json.dumps({"name": "Op", "voiceprints": []}), encoding="utf-8")
    with pytest.raises(ProfileError, match="no enrolled voices"):
        load_profile_file(legacy)


def test_malformed_profile_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    with pytest.raises(ProfileError):
        load_profile_file(bad)


def test_unknown_shape_raises(tmp_path):
    bad = tmp_path / "other.json"
    bad.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
    with pytest.raises(ProfileError, match="not a Whispers speaker profile"):
        load_profile_file(bad)


def test_future_schema_version_refused(tmp_path):
    future = tmp_path / "future.whispr-speaker.json"
    future.write_text(
        json.dumps(
            {
                "kind": "whispr-speaker-profile",
                "schema_version": sp.SCHEMA_VERSION + 1,
                "display_name": "X",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProfileError, match="newer than this build"):
        load_profile_file(future)


def test_a_sample_that_does_not_state_its_trust_is_not_trusted():
    """Fail closed: an object with no trust metadata must not load as reference."""
    issues = []
    sample = EnrollmentSample.from_dict({"embedding": [1.0, 0.0]}, issues=issues)
    assert sample is not None
    assert sample.sample_type == SAMPLE_LEARNED
    assert not sample.approved
    assert not sample.is_trusted
    # Kept for review rather than discarded, and the demotion is stated.
    assert issues and "unapproved learned sample" in issues[0]
    assert "review before approving" in sample.notes
    assert sample.sample_id


def test_a_sample_claiming_reference_without_approval_is_not_trusted():
    issues = []
    sample = EnrollmentSample.from_dict(
        {"embedding": [1.0, 0.0], "sample_type": SAMPLE_REFERENCE}, issues=issues
    )
    assert sample is not None and not sample.is_trusted
    assert sample.sample_type == SAMPLE_LEARNED


def test_an_explicit_reference_sample_still_loads_as_trusted():
    sample = EnrollmentSample.from_dict(
        {
            "embedding": [1.0, 0.0],
            "sample_type": SAMPLE_REFERENCE,
            "approved": True,
            "speech_duration": 12.0,
        }
    )
    assert sample is not None
    assert sample.sample_type == SAMPLE_REFERENCE
    assert sample.is_trusted
    assert sample.speech_duration == 12.0


def test_an_unusable_embedding_is_dropped_rather_than_loaded():
    for bad in (
        {"embedding": []},
        {"embedding": "not-a-vector"},
        {"embedding": [1.0, float("nan")]},
        {"embedding": [1.0, float("inf")]},
        {"embedding": [1.0, "0.5"]},
        {"embedding": [True, False]},
        {},
    ):
        issues = []
        assert EnrollmentSample.from_dict(bad, issues=issues) is None
        # Never a silent drop.
        assert issues and "Dropped sample" in issues[0]


def test_saved_profiles_round_trip_their_trust_unchanged(tmp_path, monkeypatch):
    """Strictness bites malformed files, not files this build wrote."""
    _isolate(tmp_path, monkeypatch)
    profile = SpeakerProfile(display_name="Round Trip", embedding_model=_model())
    profile.add_reference_sample(_sample((1.0, 0.0)))
    profile.propose_learned_sample(_sample((0.0, 1.0)))
    (loaded,) = load_profile_file(save_speaker_profile(profile))
    assert len(loaded.trusted_samples()) == 1
    assert len(loaded.pending_samples()) == 1
    assert loaded.import_warnings == []


def test_vectors_of_the_wrong_dimension_are_dropped_with_a_reason(
    tmp_path, monkeypatch
):
    _isolate(tmp_path, monkeypatch)
    profile = SpeakerProfile(display_name="Mixed", embedding_model=_model(dim=2))
    profile.add_reference_sample(_sample((1.0, 0.0)))
    path = save_speaker_profile(profile)
    data = json.loads(path.read_text(encoding="utf-8"))
    # A sample the recorded model could not have produced.
    data["samples"].append(
        {
            "sample_id": "smp_wrong",
            "embedding": [1.0, 0.0, 0.0, 0.0],
            "sample_type": SAMPLE_REFERENCE,
            "approved": True,
            "speech_duration": 30.0,
        }
    )
    path.write_text(json.dumps(data), encoding="utf-8")

    (loaded,) = load_profile_file(path)
    assert len(loaded.samples) == 1
    assert loaded.import_warnings
    assert "192" not in loaded.import_warnings[0]
    assert "smp_wrong" in loaded.import_warnings[0]
    # The surviving sample is the one the model can account for.
    assert len(loaded.centroid()) == 2


def test_mixed_dimensions_without_a_model_keep_the_majority(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    profile = SpeakerProfile(display_name="No model")
    for vec in ((1.0, 0.0), (0.0, 1.0), (1.0, 0.0)):
        profile.add_reference_sample(_sample(vec))
    path = save_speaker_profile(profile)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["samples"].append(
        {
            "sample_id": "smp_odd",
            "embedding": [1.0, 0.0, 0.0],
            "sample_type": SAMPLE_REFERENCE,
            "approved": True,
        }
    )
    path.write_text(json.dumps(data), encoding="utf-8")
    (loaded,) = load_profile_file(path)
    assert len(loaded.samples) == 3
    assert any("smp_odd" in w for w in loaded.import_warnings)


# -- Telling two subjects with the same name apart ---------------------------


def test_display_labels_are_unique_when_two_subjects_share_a_name():
    a = SpeakerProfile(display_name="J. Smith")
    b = SpeakerProfile(display_name="J. Smith")
    c = SpeakerProfile(display_name="Someone Else")
    labels = sp.display_labels([a, b, c])
    assert labels[c.subject_id] == "Someone Else"
    assert labels[a.subject_id] != labels[b.subject_id]
    assert a.subject_id in labels[a.subject_id]
    assert b.subject_id in labels[b.subject_id]
    # Three distinct labels for three distinct subjects.
    assert len(set(labels.values())) == 3


# -- Finding a subject by the name a transcript uses -------------------------


def _stored(tmp_path, monkeypatch, names):
    monkeypatch.setattr(sp, "speakers_dir", lambda: tmp_path)
    for name in names:
        profile = SpeakerProfile(display_name=name)
        save_speaker_profile(profile)


def test_find_by_name_is_case_insensitive(tmp_path, monkeypatch):
    _stored(tmp_path, monkeypatch, ["John Doe"])
    found = sp.find_speaker_profile_by_name("  john doe ")
    assert found is not None and found.display_name == "John Doe"


def test_find_by_name_returns_nothing_for_an_unknown_speaker(tmp_path, monkeypatch):
    _stored(tmp_path, monkeypatch, ["John Doe"])
    assert sp.find_speaker_profile_by_name("SPEAKER_01") is None
    assert sp.find_speaker_profile_by_name("") is None


def test_an_ambiguous_name_matches_nothing_rather_than_guessing(tmp_path, monkeypatch):
    # Two subjects share a display name: picking one would attach a correction
    # to the wrong person's reference material.
    _stored(tmp_path, monkeypatch, ["J. Smith", "J. Smith"])
    assert sp.find_speaker_profile_by_name("J. Smith") is None
