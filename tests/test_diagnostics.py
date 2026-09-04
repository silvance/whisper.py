from whispr import diagnostics
from whispr.buildinfo import BuildInfo
from whispr.diagnostics import Check, format_report, gather
from whispr.hashing import sha256_file


def test_gather_returns_checks_covering_each_area():
    checks = gather()
    labels = " ".join(c.label for c in checks)
    assert all(isinstance(c, Check) for c in checks)
    for area in ("Transcription", "Diarization", "Translation", "OCR", "playback"):
        assert area.lower() in labels.lower()


def test_format_report_marks_ok_and_missing():
    checks = [
        Check("Thing A", True, "installed"),
        Check("Thing B", False, "MISSING"),
    ]
    report = format_report(checks)
    assert "Whispers build self-test" in report
    assert "[OK ] Thing A" in report
    assert "[-- ] Thing B" in report
    assert "MISSING" in report


def test_format_report_defaults_to_gather():
    # Should run end-to-end with no argument.
    report = format_report()
    assert "build self-test" in report


def test_tessdata_languages_lists_stems(tmp_path, monkeypatch):
    tessdata = tmp_path / "tessdata"
    tessdata.mkdir()
    (tessdata / "eng.traineddata").write_bytes(b"\x00")
    (tessdata / "ara.traineddata").write_bytes(b"\x00")
    monkeypatch.setattr(diagnostics.resources, "bundled_tessdata_dir", lambda: tessdata)
    assert diagnostics._tessdata_languages() == ["ara", "eng"]


# -- Capability verification -------------------------------------------------


def _all_present(monkeypatch, tmp_path, *, embedding_name="titanet-large"):
    """Point every capability probe at a complete, bundled build."""
    ffmpeg = tmp_path / "ffmpeg"
    ffmpeg.write_bytes(b"\x00")
    embedding = tmp_path / "embedding.onnx"
    embedding.write_bytes(b"embedding-model-bytes")
    monkeypatch.setattr(diagnostics, "_installed", lambda module: True)
    monkeypatch.setattr(diagnostics.resources, "find_ffmpeg", lambda: ffmpeg)
    monkeypatch.setattr(diagnostics.resources, "bundled_models", lambda: ["base.en"])
    monkeypatch.setattr(diagnostics.resources, "pyannote_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(
        diagnostics.resources, "bundled_diarization_models", lambda: tmp_path
    )
    monkeypatch.setattr(
        diagnostics.resources, "bundled_embedding_model", lambda: embedding
    )
    monkeypatch.setattr(
        diagnostics.resources, "bundled_embedding_model_name", lambda: embedding_name
    )
    return embedding


def test_complete_bundle_is_ready(monkeypatch, tmp_path):
    _all_present(monkeypatch, tmp_path)
    items = diagnostics.capabilities()
    ready, blockers = diagnostics.readiness(items)
    assert ready and blockers == []
    names = {c.name for c in items}
    for expected in (
        "Can transcribe",
        "Can separate speakers",
        "Can create speaker profiles",
        "Can compare voices",
    ):
        assert expected in names


def test_missing_embedding_model_blocks_speaker_work(monkeypatch, tmp_path):
    _all_present(monkeypatch, tmp_path)
    monkeypatch.setattr(diagnostics.resources, "bundled_embedding_model", lambda: None)
    ready, blockers = diagnostics.readiness()
    assert not ready
    # Enrolment and comparison both rest on that one model, so both are blocked.
    assert "Speaker embedding model" in blockers
    assert "Can create speaker profiles" in blockers
    assert "Can compare voices" in blockers


def test_missing_transcription_model_blocks_readiness(monkeypatch, tmp_path):
    _all_present(monkeypatch, tmp_path)
    monkeypatch.setattr(diagnostics.resources, "bundled_models", lambda: [])
    ready, blockers = diagnostics.readiness()
    assert not ready
    assert "Can transcribe" in blockers


def test_diarization_needs_an_engine_and_its_models(monkeypatch, tmp_path):
    _all_present(monkeypatch, tmp_path)
    monkeypatch.setattr(diagnostics.resources, "pyannote_cache_dir", lambda: None)
    monkeypatch.setattr(
        diagnostics.resources, "bundled_diarization_models", lambda: None
    )
    ready, blockers = diagnostics.readiness()
    assert not ready
    assert "Can separate speakers" in blockers
    # An installed library with no bundled models is not a usable diarizer.
    assert diagnostics._can_diarize().detail == "no diarization engine is ready"


def test_report_export_degrades_without_docx_but_stays_ready(monkeypatch, tmp_path):
    _all_present(monkeypatch, tmp_path)
    monkeypatch.setattr(diagnostics, "_installed", lambda module: module != "docx")
    export = diagnostics._can_export_reports()
    assert export.available and not export.critical
    assert "text only" in export.detail
    ready, _ = diagnostics.readiness()
    assert ready


def test_embedding_capability_names_the_model_and_its_hash(monkeypatch, tmp_path):
    embedding = _all_present(monkeypatch, tmp_path, embedding_name="titanet-small")
    capability = diagnostics._speaker_embedding()
    assert capability.available
    assert "titanet-small" in capability.detail
    # Falls back to hashing the file when the manifest does not list it.
    digest = sha256_file(embedding)
    assert digest[:12] in capability.detail


def test_embedding_capability_uses_the_manifest_hash_when_present(
    monkeypatch, tmp_path
):
    _all_present(monkeypatch, tmp_path)
    recorded = "a" * 64
    monkeypatch.setattr(
        diagnostics,
        "build_info",
        lambda: BuildInfo(
            models=[{"friendly_name": "titanet-large", "sha256": recorded}]
        ),
    )
    assert recorded[:12] in diagnostics._speaker_embedding().detail


def test_bundled_embedding_without_sherpa_is_not_usable(monkeypatch, tmp_path):
    _all_present(monkeypatch, tmp_path)
    monkeypatch.setattr(
        diagnostics, "_installed", lambda module: module != "sherpa_onnx"
    )
    capability = diagnostics._speaker_embedding()
    assert not capability.available
    assert "sherpa-onnx" in capability.detail


def test_report_states_readiness_build_identity_and_offline_status(
    monkeypatch, tmp_path
):
    _all_present(monkeypatch, tmp_path)
    monkeypatch.setattr(
        diagnostics,
        "build_info",
        lambda: BuildInfo(
            build_id="run-42",
            git_commit="deadbeef",
            application_version="1.2.3",
            models=[
                {
                    "friendly_name": "titanet-large",
                    "kind": "speaker-embedding",
                    "sha256": "b" * 64,
                }
            ],
        ),
    )
    report = diagnostics.format_report()
    assert "READY" in report and "NOT READY" not in report
    assert "run-42" in report and "deadbeef" in report and "1.2.3" in report
    # The self-test must state the offline posture explicitly.
    assert "Runtime network access: none" in report
    # ...and the hash of the model that speaker comparison depends on.
    assert "Bundled model hashes" in report
    assert ("b" * 12) in report


def test_report_names_the_missing_pieces_when_not_ready(monkeypatch, tmp_path):
    _all_present(monkeypatch, tmp_path)
    monkeypatch.setattr(diagnostics.resources, "find_ffmpeg", lambda: None)
    report = diagnostics.format_report()
    assert "NOT READY" in report
    assert "Missing: " in report
    assert "Can read audio/video (ffmpeg)" in report
