import pytest

from whispr.matching import ComparisonResult
from whispr.provenance import (
    AnalysisProvenance,
    DiarizationProvenance,
    SourceRecord,
    TranscriptionProvenance,
)
from whispr.quality import GOOD
from whispr.reports import (
    DISCLAIMER_HEADING,
    build_report_sections,
    render_text,
    write_analysis_report,
)
from whispr.thresholds import BAND_HIGH, BAND_INSUFFICIENT, DISCLAIMER
from whispr.transcription import Segment, TranscriptionResult

SOURCE_SHA = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"


def _provenance():
    return AnalysisProvenance(
        source=SourceRecord(
            filename="operation.wav",
            original_path="/cases/operation.wav",
            file_size=1024,
            sha256=SOURCE_SHA,
        ),
        transcription=TranscriptionProvenance(
            model_name="small.en", model_sha256="a" * 64, detected_language="en"
        ),
        diarization=DiarizationProvenance(
            engine="pyannote",
            embedding_model="titanet-large",
            embedding_model_sha256="c" * 64,
        ),
    )


def _result():
    return TranscriptionResult(
        text="hello there",
        language="en",
        language_probability=0.98,
        duration=12.0,
        segments=[
            Segment(start=0.0, end=6.0, text="hello", speaker="SPEAKER_00"),
            Segment(start=6.0, end=12.0, text="there", speaker="voice::Alpha"),
        ],
    )


def _comparison(**kw):
    defaults = dict(
        reference_name="Subject Alpha",
        questioned_label="operation.wav",
        score=0.78,
        band=BAND_HIGH,
        reference_seconds=38.4,
        questioned_seconds=14.2,
        reference_quality=GOOD,
        questioned_quality="Fair",
        embedding_model="titanet-large, dim 192, sha256 cccccccccccc",
    )
    defaults.update(kw)
    return ComparisonResult(**defaults)


def _text(**kw):
    return render_text(build_report_sections(**kw))


# -- content ---------------------------------------------------------------


def test_report_includes_source_hash_and_build():
    text = _text(result=_result(), provenance=_provenance())
    assert SOURCE_SHA in text
    assert "operation.wav" in text
    assert "Application version:" in text
    assert "Build ID:" in text


def test_report_includes_model_information():
    text = _text(result=_result(), provenance=_provenance())
    assert "small.en" in text
    assert "a" * 64 in text  # transcription model hash
    assert "titanet-large" in text
    assert "c" * 64 in text  # embedding model hash


def test_report_includes_transcript_with_timestamps_and_speakers():
    text = _text(
        result=_result(),
        speaker_names={"voice::Alpha": "Alpha"},
        provenance=_provenance(),
    )
    assert "[00:00-00:06]" in text
    assert "[SPEAKER_00] hello" in text
    assert "[Alpha] there" in text
    assert "Detected language: en" in text


def test_report_includes_comparison_details():
    text = _text(
        result=_result(), provenance=_provenance(), comparisons=[_comparison()]
    )
    assert "Reference subject: Subject Alpha" in text
    assert "Similarity score: 0.78 / 1.00" in text
    assert f"Assessment: {BAND_HIGH}" in text
    assert "Operational threshold:" in text
    assert "Reference speech: 38.4 sec" in text
    assert "Questioned speech: 14.2 sec" in text


def test_high_similarity_is_phrased_as_a_lead_not_an_identification():
    text = _text(
        result=_result(), provenance=_provenance(), comparisons=[_comparison()]
    )
    assert "Further review is warranted" in text
    lowered = text.lower()
    assert "positive identification" not in lowered
    assert "confirmed speaker" not in lowered
    assert "is john doe" not in lowered


def test_analysis_body_never_asserts_identity_or_a_percentage():
    """The findings must not claim identity.

    The disclaimer section is excluded on purpose: it legitimately contains
    phrases like "biometric probability of identity" *in order to deny them*.
    What matters is that the analysis body never asserts them.
    """
    sections = build_report_sections(
        result=_result(), provenance=_provenance(), comparisons=[_comparison()]
    )
    body = render_text([s for s in sections if s.heading != DISCLAIMER_HEADING]).lower()
    for banned in (
        "probability of identity",
        "same person",
        "match probability",
        "biometric",
        "positive identification",
        "confirmed speaker",
    ):
        assert banned not in body
    # The score is a raw value out of 1.00, never a percentage.
    assert "similarity score: 0.78 / 1.00" in body
    assert "78%" not in body


def test_disclaimer_is_present_verbatim():
    text = _text(
        result=_result(), provenance=_provenance(), comparisons=[_comparison()]
    )
    assert DISCLAIMER in text
    assert "Interpretation and limitations" in text
    assert "require validation against representative" in text


def test_report_records_active_thresholds():
    text = _text(
        result=_result(), provenance=_provenance(), comparisons=[_comparison()]
    )
    assert "Active thresholds:" in text
    assert "Recognition acceptance:" in text
    assert "Recognition margin:" in text


def test_refused_comparison_is_reported_as_refused():
    refused = _comparison(
        refused=True, refusal_reason="Different speaker-embedding models."
    )
    text = _text(result=_result(), provenance=_provenance(), comparisons=[refused])
    assert "comparison refused" in text.lower()
    assert "Different speaker-embedding models." in text
    assert "Similarity score:" not in text


def test_insufficient_comparison_does_not_claim_a_lead():
    weak = _comparison(band=BAND_INSUFFICIENT, questioned_seconds=1.0)
    text = _text(result=_result(), provenance=_provenance(), comparisons=[weak])
    assert BAND_INSUFFICIENT in text
    assert "Further review is warranted" not in text


def test_missing_provenance_is_stated_not_invented():
    text = _text(result=_result())
    assert "not recorded" in text
    assert "cannot be confirmed from this report" in text


def test_report_without_comparisons():
    text = _text(result=_result(), provenance=_provenance())
    assert "No speaker comparisons were performed." in text


def test_report_without_transcript():
    text = _text(provenance=_provenance(), comparisons=[_comparison()])
    assert "No transcript is attached" in text


def test_multiple_comparisons_are_numbered():
    text = _text(
        result=_result(),
        provenance=_provenance(),
        comparisons=[_comparison(), _comparison(reference_name="Subject Bravo")],
    )
    assert "Comparison 1" in text
    assert "Comparison 2" in text
    assert "Subject Bravo" in text


# -- writing ---------------------------------------------------------------


def test_write_text_report(tmp_path):
    path = tmp_path / "report.txt"
    write_analysis_report(
        path,
        result=_result(),
        provenance=_provenance(),
        comparisons=[_comparison()],
        case_notes="Operation Nightfall, tasking 42.",
    )
    text = path.read_text(encoding="utf-8")
    assert SOURCE_SHA in text
    assert "Operation Nightfall, tasking 42." in text
    assert DISCLAIMER in text


def test_write_docx_report(tmp_path):
    pytest.importorskip("docx")
    path = tmp_path / "report.docx"
    write_analysis_report(
        path, result=_result(), provenance=_provenance(), comparisons=[_comparison()]
    )
    assert path.exists() and path.stat().st_size > 0
