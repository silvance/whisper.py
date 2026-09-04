import json

from whispr.buildinfo import UNKNOWN, BuildInfo, build_info
from whispr.hashing import sha256_bytes, sha256_file, sha256_file_or_none, short
from whispr.project import (
    SCHEMA_VERSION,
    load_project,
    load_project_record,
    save_project,
)
from whispr.provenance import (
    AnalysisProvenance,
    DiarizationProvenance,
    SourceRecord,
    TranscriptionProvenance,
)
from whispr.transcription import Segment, TranscriptionResult

# Known-answer vector, so a refactor of the hashing helper cannot silently
# change what a recorded provenance hash means.
HELLO = b"hello world"
HELLO_SHA = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"


# -- hashing ---------------------------------------------------------------


def test_sha256_bytes_known_answer():
    assert sha256_bytes(HELLO) == HELLO_SHA


def test_sha256_file_matches_bytes(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(HELLO)
    assert sha256_file(path) == HELLO_SHA


def test_sha256_file_streams_large_input(tmp_path):
    import hashlib

    path = tmp_path / "big.bin"
    payload = b"x" * (3 * 1024 * 1024 + 17)  # spans several read chunks
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_sha256_file_or_none_missing():
    assert sha256_file_or_none("/no/such/file") is None
    assert sha256_file_or_none(None) is None


def test_short_digest():
    assert short(HELLO_SHA) == HELLO_SHA[:12]
    assert short(None) == "unknown"


def test_hashing_does_not_modify_the_source(tmp_path):
    path = tmp_path / "src.bin"
    path.write_bytes(HELLO)
    before = path.stat().st_mtime_ns
    sha256_file(path)
    assert path.read_bytes() == HELLO
    assert path.stat().st_mtime_ns == before


# -- source record ---------------------------------------------------------


def test_source_record_from_path(tmp_path):
    path = tmp_path / "recording.wav"
    path.write_bytes(HELLO)
    record = SourceRecord.from_path(path)
    assert record.filename == "recording.wav"
    assert record.sha256 == HELLO_SHA
    assert record.file_size == len(HELLO)
    assert record.original_path == str(path)


def test_source_record_from_none_is_empty():
    record = SourceRecord.from_path(None)
    assert record.filename == ""
    assert record.sha256 is None


def test_source_record_round_trip(tmp_path):
    path = tmp_path / "r.wav"
    path.write_bytes(HELLO)
    record = SourceRecord.from_path(path)
    assert SourceRecord.from_dict(record.to_dict()) == record


# -- analysis provenance ---------------------------------------------------


def _provenance(tmp_path):
    source = tmp_path / "op.wav"
    source.write_bytes(HELLO)
    return AnalysisProvenance(
        source=SourceRecord.from_path(source),
        transcription=TranscriptionProvenance(
            model_name="small.en",
            model_sha256="a" * 64,
            detected_language="en",
        ),
        diarization=DiarizationProvenance(
            engine="pyannote",
            embedding_model="titanet-large",
            embedding_model_sha256="c" * 64,
            expected_speaker_count=2,
        ),
    )


def test_provenance_round_trip(tmp_path):
    original = _provenance(tmp_path)
    restored = AnalysisProvenance.from_dict(original.to_dict())
    assert restored.source.sha256 == HELLO_SHA
    assert restored.transcription is not None
    assert restored.transcription.model_name == "small.en"
    assert restored.diarization is not None
    assert restored.diarization.embedding_model_sha256 == "c" * 64
    assert restored.diarization.expected_speaker_count == 2
    assert restored.thresholds.recognition_acceptance == (
        original.thresholds.recognition_acceptance
    )


def test_provenance_describe_covers_source_models_and_thresholds(tmp_path):
    text = "\n".join(_provenance(tmp_path).describe())
    assert HELLO_SHA in text
    assert "small.en" in text
    assert "titanet-large" in text
    assert "acceptance threshold" in text


def test_provenance_records_matching_thresholds(tmp_path):
    data = _provenance(tmp_path).to_dict()
    assert "acceptance_threshold" in data["speaker_matching"]
    assert "margin_threshold" in data["speaker_matching"]


def test_build_info_reports_unknown_without_a_manifest(monkeypatch):
    import whispr.buildinfo as bi

    monkeypatch.setattr(bi, "load_manifest", lambda: {})
    info = bi.build_info()
    assert info.build_id == UNKNOWN
    assert info.from_manifest is False
    assert any("cannot be confirmed" in line for line in info.describe())


def test_build_info_uses_manifest_when_present(monkeypatch):
    import whispr.buildinfo as bi

    monkeypatch.setattr(
        bi,
        "load_manifest",
        lambda: {
            "build_id": "2026.09.04-1",
            "git_commit": "abc1234",
            "build_timestamp": "2026-09-04T10:00:00Z",
            "application_version": "1.2.3",
            "dependencies": {"faster-whisper": "1.0.3"},
            "models": [{"friendly_name": "titanet-large", "sha256": "d" * 64}],
        },
    )
    info = bi.build_info()
    assert info.from_manifest is True
    assert info.build_id == "2026.09.04-1"
    assert info.dependencies["faster-whisper"] == "1.0.3"
    assert info.model_sha256("titanet-large") == "d" * 64
    assert info.model_sha256("nope") is None


def test_build_info_round_trip():
    info = BuildInfo(build_id="x", git_commit="y")
    assert info.to_dict()["build_id"] == "x"
    assert isinstance(build_info(), BuildInfo)


# -- project schema --------------------------------------------------------


def _result():
    return TranscriptionResult(
        text="hello",
        language="en",
        language_probability=0.99,
        duration=1.0,
        segments=[Segment(start=0.0, end=1.0, text="hello", speaker="SPEAKER_00")],
    )


def test_project_saves_and_loads_provenance(tmp_path):
    provenance = _provenance(tmp_path)
    path = tmp_path / "case.whispr.json"
    save_project(path, _result(), {"SPEAKER_00": "Alpha"}, "op.wav", provenance)

    record = load_project_record(path)
    assert record.schema_version == SCHEMA_VERSION
    assert record.has_provenance is True
    assert record.provenance is not None
    assert record.provenance.source.sha256 == HELLO_SHA
    assert record.speaker_names == {"SPEAKER_00": "Alpha"}
    assert record.result.segments[0].text == "hello"


def test_project_written_file_records_schema_and_hash(tmp_path):
    path = tmp_path / "case.whispr.json"
    save_project(path, _result(), None, "op.wav", _provenance(tmp_path))
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["provenance"]["source"]["sha256"] == HELLO_SHA


def test_schema_1_project_still_loads(tmp_path):
    """A pre-provenance project must keep working, reporting no provenance."""
    legacy = tmp_path / "old.whispr.json"
    legacy.write_text(
        json.dumps(
            {
                "version": 1,
                "source": "old.wav",
                "speaker_names": {"SPEAKER_00": "Bravo"},
                "result": {
                    "text": "old text",
                    "language": "en",
                    "language_probability": 0.5,
                    "duration": 2.0,
                    "segments": [
                        {"start": 0.0, "end": 2.0, "text": "old text", "words": []}
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    record = load_project_record(legacy)
    assert record.schema_version == 1
    assert record.has_provenance is False
    assert record.speaker_names == {"SPEAKER_00": "Bravo"}
    assert record.result.text == "old text"
    # The legacy 3-tuple API keeps working too.
    result, names, source = load_project(legacy)
    assert result.text == "old text"
    assert names == {"SPEAKER_00": "Bravo"}
    assert source == "old.wav"


def test_project_without_provenance_saves_cleanly(tmp_path):
    path = tmp_path / "plain.whispr.json"
    save_project(path, _result(), None, "op.wav")
    record = load_project_record(path)
    assert record.has_provenance is False
    assert record.schema_version == SCHEMA_VERSION


def test_project_save_is_atomic(tmp_path, monkeypatch):
    import whispr.speaker_profiles as sp

    path = tmp_path / "case.whispr.json"
    save_project(path, _result(), None, "op.wav")
    original = path.read_text(encoding="utf-8")

    def _boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(sp.os, "replace", _boom)
    try:
        save_project(path, _result(), None, "changed.wav")
    except OSError:
        pass
    assert path.read_text(encoding="utf-8") == original
