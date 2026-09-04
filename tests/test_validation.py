"""Tests for the validation harness.

These verify the harness's *arithmetic and plumbing* on controlled inputs. They
deliberately do not claim to validate speaker-recognition accuracy - that is what
the harness itself is for, run over real recordings.
"""

import csv
import json

import pytest

from whispr.validation import (
    CorpusItem,
    EmbeddedItem,
    OperatingPoint,
    Trial,
    ValidationReport,
    bucket_for,
    build_trials,
    equal_error_rate,
    false_accept_rate,
    false_reject_rate,
    load_corpus,
    operating_points,
    summarise,
    write_json,
    write_roc_csv,
    write_trials_csv,
)


def _embedded(speaker, name, vec, seconds=20.0, condition=""):
    return EmbeddedItem(
        item=CorpusItem(
            speaker_id=speaker,
            path=__import__("pathlib").Path(name),
            condition=condition,
        ),
        embedding=list(vec),
        speech_seconds=seconds,
    )


# -- corpus loading --------------------------------------------------------


def test_load_corpus_from_directory(tmp_path):
    for speaker, files in (("SPK_A", ["a1.wav", "a2.wav"]), ("SPK_B", ["b1.wav"])):
        folder = tmp_path / speaker
        folder.mkdir()
        for name in files:
            (folder / name).write_bytes(b"\x00")
    (tmp_path / "SPK_A" / "notes.txt").write_text("ignored", encoding="utf-8")

    items = load_corpus(tmp_path)
    assert len(items) == 3  # the .txt is not audio
    assert {i.speaker_id for i in items} == {"SPK_A", "SPK_B"}


def test_load_corpus_from_json_manifest(tmp_path):
    manifest = tmp_path / "corpus.json"
    manifest.write_text(
        json.dumps(
            [
                {"speaker_id": "A", "path": "a.wav", "condition": "phone"},
                {"speaker_id": "B", "path": "b.wav", "condition": "room"},
            ]
        ),
        encoding="utf-8",
    )
    items = load_corpus(manifest)
    assert [i.speaker_id for i in items] == ["A", "B"]
    assert items[0].condition == "phone"
    # Relative paths resolve against the manifest.
    assert items[0].path == tmp_path / "a.wav"


def test_load_corpus_from_csv_manifest(tmp_path):
    manifest = tmp_path / "corpus.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["speaker_id", "path", "condition"])
        writer.writerow(["A", "a.wav", "phone"])
    items = load_corpus(manifest)
    assert items[0].speaker_id == "A" and items[0].condition == "phone"


def test_load_corpus_rejects_unknown_input(tmp_path):
    bad = tmp_path / "corpus.txt"
    bad.write_text("nope", encoding="utf-8")
    with pytest.raises(ValueError):
        load_corpus(bad)


def test_manifest_row_missing_fields_raises(tmp_path):
    manifest = tmp_path / "corpus.json"
    manifest.write_text(json.dumps([{"speaker_id": "A"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="speaker_id and path"):
        load_corpus(manifest)


# -- trial generation ------------------------------------------------------


def test_build_trials_labels_genuine_and_impostor():
    embedded = [
        _embedded("A", "a1.wav", (1.0, 0.0)),
        _embedded("A", "a2.wav", (0.99, 0.14)),
        _embedded("B", "b1.wav", (0.0, 1.0)),
    ]
    trials = build_trials(embedded)
    assert len(trials) == 3  # 3 choose 2
    genuine = [t for t in trials if t.genuine]
    impostor = [t for t in trials if not t.genuine]
    assert len(genuine) == 1 and len(impostor) == 2
    # Same speaker, different recordings -> high; different speakers -> low.
    assert genuine[0].score > 0.9
    assert all(t.score < 0.3 for t in impostor)


def test_build_trials_records_min_duration_and_condition():
    embedded = [
        _embedded("A", "a1.wav", (1.0, 0.0), seconds=30.0, condition="phone"),
        _embedded("A", "a2.wav", (1.0, 0.0), seconds=4.0, condition="room"),
    ]
    trial = build_trials(embedded)[0]
    assert trial.min_speech_seconds == 4.0
    assert trial.same_condition is False


def test_build_trials_never_pairs_a_recording_with_itself():
    item = _embedded("A", "a1.wav", (1.0, 0.0))
    assert build_trials([item, item]) == []


# -- rates -----------------------------------------------------------------


def test_false_accept_and_reject_rates():
    impostor = [0.1, 0.3, 0.5, 0.7]
    genuine = [0.4, 0.6, 0.8, 0.9]
    assert false_accept_rate(impostor, 0.5) == 0.5  # 0.5 and 0.7 accepted
    assert false_reject_rate(genuine, 0.5) == 0.25  # only 0.4 rejected
    assert false_accept_rate([], 0.5) == 0.0
    assert false_reject_rate([], 0.5) == 0.0


def test_rates_move_in_opposite_directions():
    genuine = [0.5, 0.6, 0.7, 0.8]
    impostor = [0.1, 0.2, 0.3, 0.4]
    strict = operating_points(genuine, impostor, [0.75])[0]
    lenient = operating_points(genuine, impostor, [0.15])[0]
    assert strict.false_accept_rate <= lenient.false_accept_rate
    assert strict.false_reject_rate >= lenient.false_reject_rate


def test_perfectly_separated_scores_have_a_clean_threshold():
    genuine = [0.8, 0.85, 0.9]
    impostor = [0.1, 0.2, 0.3]
    points = operating_points(genuine, impostor)
    perfect = [
        p for p in points if p.false_accept_rate == 0.0 and p.false_reject_rate == 0.0
    ]
    assert perfect, "a separable set should admit a zero-error threshold"
    rate, threshold = equal_error_rate(points)
    assert rate == 0.0
    assert 0.3 < threshold <= 0.8


def test_equal_error_rate_of_overlapping_distributions():
    genuine = [0.4, 0.5, 0.6, 0.7]
    impostor = [0.3, 0.4, 0.5, 0.6]
    rate, threshold = equal_error_rate(operating_points(genuine, impostor))
    assert rate is not None and threshold is not None
    assert 0.0 < rate < 1.0


def test_equal_error_rate_of_nothing():
    assert equal_error_rate([]) == (None, None)


def test_genuine_accept_rate_is_the_roc_axis():
    point = OperatingPoint(threshold=0.5, false_accept_rate=0.1, false_reject_rate=0.2)
    assert point.genuine_accept_rate == pytest.approx(0.8)


def test_summarise_distribution():
    stats = summarise([0.1, 0.2, 0.3, 0.4, 0.5])
    assert stats["count"] == 5
    assert stats["min"] == 0.1
    assert stats["max"] == 0.5
    assert stats["median"] == pytest.approx(0.3)
    assert stats["mean"] == pytest.approx(0.3)


def test_summarise_empty():
    assert summarise([]) == {"count": 0}


def test_duration_buckets():
    assert bucket_for(1.0) == "under 5s"
    assert bucket_for(10.0) == "5-15s"
    assert bucket_for(20.0) == "15-30s"
    assert bucket_for(120.0) == "30s+"


# -- report ----------------------------------------------------------------


def _report():
    trials = [
        Trial(0.85, True, "A", "A", "a1", "a2", 30.0),
        Trial(0.80, True, "A", "A", "a1", "a3", 4.0),
        Trial(0.20, False, "A", "B", "a1", "b1", 30.0),
        Trial(0.25, False, "A", "B", "a2", "b1", 4.0),
    ]
    return ValidationReport(
        trials=trials, corpus_size=4, speaker_count=2, embedding_model="titanet-large"
    )


def test_report_separates_genuine_and_impostor():
    report = _report()
    assert report.genuine_scores == [0.85, 0.80]
    assert report.impostor_scores == [0.20, 0.25]


def test_report_groups_by_duration():
    groups = _report().by_duration()
    assert "30s+" in groups and "under 5s" in groups
    assert groups["30s+"]["genuine"]["count"] == 1


def test_report_dict_has_rates_and_caveat():
    data = _report().to_dict()
    assert data["trial_count"] == 4
    assert data["equal_error_rate"] is not None
    assert data["operating_points"]
    assert "representative of the intended deployment" in data["note"]


def test_report_summary_lines_are_readable():
    text = "\n".join(_report().summary_lines())
    assert "Corpus: 4 recording(s), 2 speaker(s)" in text
    assert "Equal error rate:" in text
    assert "Candidate thresholds" in text


def test_report_exports(tmp_path):
    report = _report()
    write_json(report, tmp_path / "validation.json")
    write_trials_csv(report, tmp_path / "trials.csv")
    write_roc_csv(report, tmp_path / "roc.csv")

    data = json.loads((tmp_path / "validation.json").read_text(encoding="utf-8"))
    assert data["speaker_count"] == 2

    with open(tmp_path / "trials.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert rows[0]["genuine"] == "1"

    with open(tmp_path / "roc.csv", encoding="utf-8") as handle:
        roc = list(csv.DictReader(handle))
    assert roc and "false_accept_rate" in roc[0]


def test_empty_report_does_not_crash():
    report = ValidationReport()
    assert report.to_dict()["trial_count"] == 0
    assert report.summary_lines()
