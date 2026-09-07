import json

import pytest

from whispr import comparison_log
from whispr.comparison_log import (
    KIND_GALLERY,
    KIND_PROFILE,
    ComparisonRecord,
    RankedSubject,
    delete_comparison,
    export_csv,
    list_comparisons,
    load_comparison,
    record_from_comparison,
    record_from_gallery,
    save_comparison,
    subjects_in,
)
from whispr.matching import ComparisonResult, GalleryMatch, GalleryResult
from whispr.speaker_profiles import ProfileError


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    """Keep every test's records in its own directory."""
    monkeypatch.setattr(
        comparison_log, "comparisons_dir", lambda: tmp_path / "comparisons"
    )
    return tmp_path


def _result(**kw):
    data = dict(
        reference_name="A. Subject",
        reference_subject_id="subj_1",
        questioned_label="SPEAKER_00",
        score=0.71,
        band="High similarity",
        reference_seconds=430.0,
        questioned_seconds=62.5,
        reference_quality="Good",
        questioned_quality="Fair",
        embedding_model="titanet-large, dim 192",
        questioned_source_filename="clip.wav",
        questioned_source_sha256="abc123",
        questioned_selection="diarized speaker (SPEAKER_00, 9 turn(s))",
        questioned_window_count=9,
    )
    data.update(kw)
    return ComparisonResult(**data)


# -- what a record carries -------------------------------------------------


def test_record_carries_the_account_of_a_comparison():
    record = record_from_comparison(_result())
    assert record.kind == KIND_PROFILE
    assert record.reference_name == "A. Subject"
    assert record.questioned_filename == "clip.wav"
    assert record.questioned_sha256 == "abc123"
    assert record.score == pytest.approx(0.71)
    assert record.band == "High similarity"
    assert record.operational_threshold > 0
    assert record.recorded_utc.endswith("+00:00")


def test_record_never_carries_audio_or_embeddings():
    """The log is an account of activity, not a second copy of the evidence.

    A voice embedding is the sensitive artefact here; a record of activity must
    not quietly become a second store of them.
    """
    data = record_from_comparison(_result()).to_dict()
    # The only "embedding" in a record is the name of the model that made one.
    assert [k for k in data if "embedding" in k] == ["embedding_model"]
    assert isinstance(data["embedding_model"], str)
    for key, value in data.items():
        assert not (
            isinstance(value, list) and any(isinstance(v, float) for v in value)
        ), f"{key} holds raw numbers"
    payload = json.dumps(data).lower()
    for forbidden in ("vector", "waveform", "transcript", "audio_data"):
        assert forbidden not in payload


def test_a_refused_comparison_is_still_recorded():
    record = record_from_comparison(
        _result(refused=True, refusal_reason="Different embedding model.")
    )
    assert record.refused is True
    assert "embedding model" in record.refusal_reason


def test_gallery_record_keeps_the_whole_ranking():
    result = GalleryResult(
        questioned_label="SPEAKER_01",
        matches=[
            GalleryMatch("subj_1", "A. Subject", 0.66, "High similarity"),
            GalleryMatch("subj_2", "B. Subject", 0.41, "Low similarity"),
        ],
        searched=2,
        questioned_source_filename="street.wav",
    )
    record = record_from_gallery(result, questioned_seconds=48.0)
    assert record.kind == KIND_GALLERY
    assert [r.display_name for r in record.ranked] == ["A. Subject", "B. Subject"]
    # The runner-up matters: a thin margin should still read as thin later.
    assert record.ranked[1].score == pytest.approx(0.41)
    assert record.subjects_searched == 2
    assert record.subject_label == "A. Subject"


def test_gallery_record_carries_the_reason_it_supports_nothing():
    result = GalleryResult(
        matches=[GalleryMatch("subj_1", "A. Subject", 0.8, "High similarity")],
        searched=1,
        inadequate_reason="Only 4.0s of usable speech.",
    )
    record = record_from_gallery(result)
    assert record.warnings[0] == "Only 4.0s of usable speech."


def test_record_pins_what_the_reference_profile_was_at_the_time():
    """A profile grows; a record has to say what it was measured against.

    Without this, the same recording scoring differently next month reads as a
    contradiction rather than as a reference profile that has since gained
    samples.
    """
    from whispr.speaker_profiles import EnrollmentSample, SpeakerProfile

    profile = SpeakerProfile(display_name="A. Subject")
    for _ in range(3):
        profile.add_reference_sample(
            EnrollmentSample(embedding=[0.1, 0.2], speech_duration=6.0)
        )
    record = record_from_comparison(_result(), profile=profile)
    assert record.reference_sample_count == 3
    assert record.reference_updated_utc == profile.updated_utc


def test_record_keeps_the_audio_it_was_measured_from_and_where_it_was():
    record = record_from_comparison(
        _result(),
        questioned_path="/evidence/clip.wav",
        questioned_spans=[(1.0, 9.0), (20.5, 28.5)],
    )
    assert record.questioned_path == "/evidence/clip.wav"
    assert record.questioned_spans == [(1.0, 9.0), (20.5, 28.5)]
    # The path is a pointer to the recording, never a copy of it.
    assert "audio" not in json.dumps(record.to_dict()).lower()


def test_record_keeps_every_threshold_in_force_not_just_the_one_that_decided():
    record = record_from_comparison(_result())
    assert record.thresholds["comparison_high"] > 0
    assert "min_questioned_seconds" in record.thresholds
    restored = ComparisonRecord.from_dict(record.to_dict())
    assert restored.thresholds == record.thresholds


def test_record_names_the_build_that_decided_it():
    record = record_from_comparison(_result())
    assert isinstance(record.app_version, str)


def test_gallery_record_keeps_the_margin_over_the_runner_up():
    from whispr.voiceprints import MatchDecision

    result = GalleryResult(
        matches=[
            GalleryMatch("subj_1", "A. Subject", 0.7, "High similarity"),
            GalleryMatch("subj_2", "B. Subject", 0.45, "Intermediate similarity"),
        ],
        searched=2,
        decision=MatchDecision(
            best_name="A. Subject",
            best_score=0.7,
            second_name="B. Subject",
            second_score=0.45,
            accepted=True,
        ),
    )
    record = record_from_gallery(result)
    assert record.runner_up_name == "B. Subject"
    assert record.margin == pytest.approx(0.25)


def test_spans_survive_a_round_trip_and_bad_ones_are_dropped():
    record = ComparisonRecord(questioned_spans=[(0.0, 8.0)])
    assert ComparisonRecord.from_dict(record.to_dict()).questioned_spans == [(0.0, 8.0)]
    data = record.to_dict()
    data["questioned_spans"] = [[1.0, 2.0], "nonsense", [3.0], [4.0, "x"]]
    assert ComparisonRecord.from_dict(data).questioned_spans == [(1.0, 2.0)]


# -- storage ---------------------------------------------------------------


def test_save_and_list_round_trip():
    saved = record_from_comparison(_result())
    path = save_comparison(saved)
    assert path.exists()
    (loaded,) = list_comparisons()
    assert loaded.record_id == saved.record_id
    assert loaded.questioned_sha256 == "abc123"
    assert loaded.score == pytest.approx(0.71)


def test_records_come_back_newest_first():
    for index, name in enumerate(("first", "second", "third")):
        record = record_from_comparison(_result(questioned_source_filename=name))
        record.recorded_utc = f"2026-01-0{index + 1}T00:00:00+00:00"
        save_comparison(record)
    assert [r.questioned_filename for r in list_comparisons()] == [
        "third",
        "second",
        "first",
    ]


def test_each_record_is_its_own_file_so_none_overwrites_another():
    for _ in range(3):
        save_comparison(record_from_comparison(_result()))
    assert len(list(comparison_log.comparisons_dir().iterdir())) == 3
    assert len(list_comparisons()) == 3


def test_an_unreadable_record_is_skipped_not_fatal():
    save_comparison(record_from_comparison(_result()))
    bad = comparison_log.comparisons_dir() / f"broken{comparison_log.RECORD_SUFFIX}"
    bad.write_text("{not json", encoding="utf-8")
    assert len(list_comparisons()) == 1
    # ...and it is left on disk: it is still evidence something was run.
    assert bad.exists()


def test_a_newer_schema_is_refused_rather_than_misread(tmp_path):
    path = tmp_path / f"future{comparison_log.RECORD_SUFFIX}"
    path.write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
    with pytest.raises(ProfileError, match="newer version"):
        load_comparison(path)


def test_delete_removes_one_record_only():
    keep = record_from_comparison(_result(questioned_source_filename="keep.wav"))
    drop = record_from_comparison(_result(questioned_source_filename="drop.wav"))
    save_comparison(keep)
    save_comparison(drop)
    assert delete_comparison(drop) is True
    assert [r.questioned_filename for r in list_comparisons()] == ["keep.wav"]


def test_listing_an_empty_store_is_empty_not_an_error():
    assert list_comparisons() == []


def test_limit_caps_how_many_are_read():
    for _ in range(5):
        save_comparison(record_from_comparison(_result()))
    assert len(list_comparisons(limit=2)) == 2


# -- views over the log ----------------------------------------------------


def test_subjects_are_listed_for_filtering():
    records = [
        record_from_comparison(_result(reference_name="B. Subject")),
        record_from_comparison(_result(reference_name="a. subject")),
        record_from_comparison(_result(reference_name="B. Subject")),
    ]
    assert subjects_in(records) == ["a. subject", "B. Subject"]


def test_a_record_with_no_subject_name_still_files_somewhere():
    record = record_from_comparison(_result(reference_name=""))
    assert record.subject_label == "(unnamed subject)"
    assert record.recording_label == "clip.wav"


def test_export_csv_writes_a_row_per_comparison(tmp_path):
    records = [
        record_from_comparison(_result(questioned_source_filename="one.wav")),
        record_from_comparison(_result(questioned_source_filename="two.wav")),
    ]
    out = export_csv(records, tmp_path / "log.csv")
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3  # header + two rows
    assert "questioned_sha256" in lines[0]
    assert "one.wav" in lines[1]
    # No percentage anywhere: a similarity score is not a probability.
    assert "%" not in out.read_text(encoding="utf-8")


def test_round_trip_through_dict_keeps_the_ranking():
    record = ComparisonRecord(
        kind=KIND_GALLERY,
        ranked=[RankedSubject("A. Subject", "subj_1", 0.5, "Intermediate similarity")],
    )
    restored = ComparisonRecord.from_dict(record.to_dict())
    assert restored.ranked[0].display_name == "A. Subject"
    assert restored.ranked[0].score == pytest.approx(0.5)
