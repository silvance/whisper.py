"""The build-time manifest writer: what a bundle records about itself."""

import importlib.util
from pathlib import Path

import pytest

from whispr.hashing import sha256_file

# build_manifest.py is a build script under packaging/ (not an installed
# module); load it by path, as tests/test_fetch_assets.py does.
_PATH = Path(__file__).resolve().parent.parent / "packaging" / "build_manifest.py"
_spec = importlib.util.spec_from_file_location("build_manifest", _PATH)
build_manifest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_manifest)


@pytest.fixture
def assets(tmp_path, monkeypatch):
    """A fake asset tree laid out the way fetch_assets.py leaves one."""
    root = tmp_path / "whispr_assets"
    models = root / "models" / "base.en"
    models.mkdir(parents=True)
    (models / "model.bin").write_bytes(b"transcription-weights")
    diarization = root / "diarization"
    diarization.mkdir()
    (diarization / "segmentation.onnx").write_bytes(b"segmentation-weights")
    (diarization / "embedding.onnx").write_bytes(b"embedding-weights")
    (diarization / "embedding_model.txt").write_text("titanet-large", encoding="utf-8")
    ffmpeg = root / "ffmpeg"
    ffmpeg.mkdir()
    (ffmpeg / "ffmpeg").write_bytes(b"ffmpeg-binary")
    monkeypatch.setattr(build_manifest, "ASSETS", root)
    monkeypatch.setattr(build_manifest, "MANIFEST", root / "build_manifest.json")
    monkeypatch.chdir(tmp_path)
    return root


def _by_name(entries):
    return {entry["friendly_name"]: entry for entry in entries}


def test_every_bundled_model_is_hashed(assets):
    entries = _by_name(build_manifest._model_entries())
    assert set(entries) == {"base.en", "segmentation", "titanet-large", "ffmpeg"}
    assert entries["base.en"]["kind"] == "transcription"
    assert entries["titanet-large"]["kind"] == "speaker-embedding"
    expected = sha256_file(assets / "diarization" / "embedding.onnx")
    assert entries["titanet-large"]["sha256"] == expected
    assert entries["titanet-large"]["size"] == len(b"embedding-weights")
    # Paths are recorded relative to the asset root, not as build-machine paths.
    assert not Path(entries["base.en"]["filename"]).is_absolute()


def test_the_embedding_model_is_named_from_what_was_actually_fetched(assets):
    (assets / "diarization" / "embedding_model.txt").write_text(
        "campplus\n", encoding="utf-8"
    )
    entries = _by_name(build_manifest._model_entries())
    assert "campplus" in entries
    assert "titanet-large" not in entries


def test_absent_assets_are_omitted_rather_than_recorded_as_present(assets):
    (assets / "diarization" / "segmentation.onnx").unlink()
    assert "segmentation" not in _by_name(build_manifest._model_entries())


def test_manifest_records_build_identity_and_dependency_versions(assets):
    manifest = build_manifest.build_manifest(build_id="99-1", commit="abc123")
    assert manifest["build_id"] == "99-1"
    assert manifest["git_commit"] == "abc123"
    assert manifest["build_timestamp"].endswith("+00:00")
    assert manifest["platform"] and manifest["python_version"]
    assert isinstance(manifest["dependencies"], dict)
    assert len(manifest["models"]) == 4


def test_commit_falls_back_to_the_ci_environment(monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "from-ci")
    assert build_manifest._git_commit(None) == "from-ci"
    # An explicit value always wins.
    assert build_manifest._git_commit("explicit") == "explicit"


def test_main_writes_the_manifest_where_the_app_looks_for_it(assets, capsys):
    assert build_manifest.main(["--build-id", "7-2", "--commit", "beef"]) == 0
    written = assets / "build_manifest.json"
    assert written.is_file()
    import json

    data = json.loads(written.read_text(encoding="utf-8"))
    assert data["build_id"] == "7-2"
    assert data["git_commit"] == "beef"
    assert "titanet-large" in capsys.readouterr().out


def test_main_refuses_to_write_a_manifest_with_no_assets(tmp_path, monkeypatch):
    monkeypatch.setattr(build_manifest, "ASSETS", tmp_path / "missing")
    with pytest.raises(SystemExit):
        build_manifest.main([])
