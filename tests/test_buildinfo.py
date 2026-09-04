"""Build identity: what the manifest says, and what is reported without one."""

import json

import pytest

from whispr import buildinfo
from whispr.buildinfo import UNKNOWN, BuildInfo, build_info, load_manifest


@pytest.fixture
def assets(tmp_path, monkeypatch):
    """A stand-in ``whispr_assets`` directory for the manifest to live in."""
    directory = tmp_path / "whispr_assets"
    directory.mkdir()
    monkeypatch.setattr(buildinfo, "asset_dirs", lambda: [directory])
    return directory


def _write(assets, payload):
    (assets / "build_manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def test_no_manifest_reports_unknown_rather_than_inventing_one(assets):
    info = build_info()
    assert not info.from_manifest
    assert info.build_id == UNKNOWN
    assert info.git_commit == UNKNOWN
    # The platform/python facts are live, so they are real either way.
    assert info.platform and info.python_version
    assert any("cannot be confirmed" in line for line in info.describe())


def test_manifest_supplies_build_identity_and_model_hashes(assets):
    _write(
        assets,
        {
            "build_id": "9876-1",
            "git_commit": "c0ffee",
            "build_timestamp": "2026-01-02T03:04:05+00:00",
            "application_version": "0.9.0",
            "platform": "Windows-10",
            "python_version": "3.11.9",
            "dependencies": {"faster-whisper": "1.2.0"},
            "models": [
                {
                    "friendly_name": "titanet-large",
                    "kind": "speaker-embedding",
                    "sha256": "f" * 64,
                }
            ],
        },
    )
    info = build_info()
    assert info.from_manifest
    assert info.build_id == "9876-1"
    assert info.git_commit == "c0ffee"
    assert info.application_version == "0.9.0"
    assert info.dependencies["faster-whisper"] == "1.2.0"
    assert info.model_sha256("titanet-large") == "f" * 64
    described = "\n".join(info.describe())
    assert "9876-1" in described and "cannot be confirmed" not in described


def test_model_sha256_is_none_for_a_model_the_manifest_does_not_list(assets):
    _write(assets, {"models": [{"friendly_name": "base.en", "sha256": "a" * 64}]})
    info = build_info()
    assert info.model_sha256("base.en") == "a" * 64
    assert info.model_sha256("large-v3") is None


def test_unreadable_manifest_degrades_to_unknown(assets):
    (assets / "build_manifest.json").write_text("{ not json", encoding="utf-8")
    assert load_manifest() == {}
    assert build_info().build_id == UNKNOWN


def test_manifest_of_the_wrong_shape_is_ignored(assets):
    _write(assets, ["not", "a", "mapping"])
    assert load_manifest() == {}


def test_malformed_entries_are_dropped_not_trusted(assets):
    _write(
        assets,
        {
            "build_id": "x1",
            "dependencies": "not-a-mapping",
            "models": ["not-a-mapping", {"friendly_name": "base.en"}],
        },
    )
    info = build_info()
    assert info.dependencies == {}
    assert len(info.models) == 1
    # An entry with no recorded hash reports no hash.
    assert info.model_sha256("base.en") is None


def test_to_dict_round_trips_the_recorded_identity():
    info = BuildInfo(build_id="b", git_commit="g", dependencies={"numpy": "1.26.4"})
    data = info.to_dict()
    assert data["build_id"] == "b"
    assert data["git_commit"] == "g"
    assert data["dependencies"] == {"numpy": "1.26.4"}


def test_runtime_summary_records_the_interpreter():
    summary = buildinfo.runtime_summary()
    assert summary["python_version"] and summary["platform"] and summary["executable"]
