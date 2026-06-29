from whispr import diagnostics
from whispr.diagnostics import Check, format_report, gather


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
