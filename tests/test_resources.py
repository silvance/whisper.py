import os

from whispr import resources


def _make_assets(tmp_path):
    assets = tmp_path / "whispr_assets"
    (assets / "ffmpeg").mkdir(parents=True)
    (assets / "models" / "small").mkdir(parents=True)
    (assets / "models" / "medium").mkdir(parents=True)
    # A bare directory without model.bin should be ignored.
    (assets / "models" / "not-a-model").mkdir(parents=True)
    ffmpeg_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    (assets / "ffmpeg" / ffmpeg_name).write_bytes(b"\x00")
    (assets / "models" / "small" / "model.bin").write_bytes(b"\x00")
    (assets / "models" / "medium" / "model.bin").write_bytes(b"\x00")
    return assets


def test_find_bundled_ffmpeg(tmp_path, monkeypatch):
    assets = _make_assets(tmp_path)
    monkeypatch.setenv(resources.ENV_ASSETS, str(assets))
    found = resources.find_bundled_ffmpeg()
    assert found is not None
    assert found.parent == assets / "ffmpeg"


def test_find_ffmpeg_prefers_bundled(tmp_path, monkeypatch):
    assets = _make_assets(tmp_path)
    monkeypatch.setenv(resources.ENV_ASSETS, str(assets))
    # Even if PATH has ffmpeg, the bundled one wins.
    monkeypatch.setattr(resources, "which", lambda _: "/usr/bin/ffmpeg")
    found = resources.find_ffmpeg()
    assert found is not None and found.parent == assets / "ffmpeg"


def test_find_ffmpeg_falls_back_to_path(tmp_path, monkeypatch):
    monkeypatch.setenv(resources.ENV_ASSETS, str(tmp_path / "empty"))
    monkeypatch.setattr(resources, "which", lambda _: "/usr/bin/ffmpeg")
    assert str(resources.find_ffmpeg()) == "/usr/bin/ffmpeg"


def test_find_ffmpeg_none_when_absent(tmp_path, monkeypatch):
    monkeypatch.setenv(resources.ENV_ASSETS, str(tmp_path / "empty"))
    monkeypatch.setattr(resources, "which", lambda _: None)
    assert resources.find_ffmpeg() is None


def test_bundled_models(tmp_path, monkeypatch):
    assets = _make_assets(tmp_path)
    monkeypatch.setenv(resources.ENV_ASSETS, str(assets))
    models = resources.bundled_models()
    assert set(models) == {"small", "medium"}
    assert models["small"] == assets / "models" / "small"
    assert "not-a-model" not in models
