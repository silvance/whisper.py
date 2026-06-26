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


def test_bundled_diarization_models_present(tmp_path, monkeypatch):
    assets = tmp_path / "whispr_assets"
    diar = assets / "diarization"
    diar.mkdir(parents=True)
    (diar / "segmentation.onnx").write_bytes(b"\x00")
    (diar / "embedding.onnx").write_bytes(b"\x00")
    monkeypatch.setenv(resources.ENV_ASSETS, str(assets))
    found = resources.bundled_diarization_models()
    assert found == (diar / "segmentation.onnx", diar / "embedding.onnx")


def test_bundled_diarization_models_absent(tmp_path, monkeypatch):
    monkeypatch.setenv(resources.ENV_ASSETS, str(tmp_path / "empty"))
    assert resources.bundled_diarization_models() is None


def test_configure_offline_hf_cache_present(tmp_path, monkeypatch):
    assets = tmp_path / "whispr_assets"
    (assets / "pyannote" / "hub").mkdir(parents=True)
    monkeypatch.setenv(resources.ENV_ASSETS, str(assets))
    for var in ("HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE"):
        monkeypatch.delenv(var, raising=False)

    cache = resources.configure_offline_hf_cache()
    hub = str(assets / "pyannote" / "hub")
    assert cache == assets / "pyannote"
    assert os.environ["HF_HOME"] == str(assets / "pyannote")
    assert os.environ["HF_HUB_CACHE"] == hub
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    # pyannote sub-models read PYANNOTE_CACHE, not HF_HUB_CACHE.
    assert os.environ["PYANNOTE_CACHE"] == hub


def test_configure_offline_hf_cache_overrides_stale_env(tmp_path, monkeypatch):
    # An operator machine with pre-set HF vars must not win: the bundle has to
    # use its own cache, or offline lookups land in the wrong (empty) cache.
    assets = tmp_path / "whispr_assets"
    (assets / "pyannote" / "hub").mkdir(parents=True)
    monkeypatch.setenv(resources.ENV_ASSETS, str(assets))
    monkeypatch.setenv("HF_HOME", "/some/stale/hf")
    monkeypatch.setenv("HF_HUB_CACHE", "/some/stale/hf/hub")
    monkeypatch.setenv("PYANNOTE_CACHE", "/some/stale/pyannote")

    resources.configure_offline_hf_cache()
    hub = str(assets / "pyannote" / "hub")
    assert os.environ["HF_HOME"] == str(assets / "pyannote")
    assert os.environ["HF_HUB_CACHE"] == hub
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["PYANNOTE_CACHE"] == hub


def test_configure_offline_hf_cache_absent_is_noop(tmp_path, monkeypatch):
    monkeypatch.setenv(resources.ENV_ASSETS, str(tmp_path / "empty"))
    for var in ("HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE"):
        monkeypatch.delenv(var, raising=False)

    assert resources.configure_offline_hf_cache() is None
    # No bundled cache -> normal online HF behaviour is left untouched.
    assert "HF_HUB_OFFLINE" not in os.environ


def test_configure_offline_translation_present(tmp_path, monkeypatch):
    assets = tmp_path / "whispr_assets"
    data = assets / "argos" / "argos-translate"
    data.mkdir(parents=True)
    monkeypatch.setenv(resources.ENV_ASSETS, str(assets))
    for var in ("XDG_DATA_HOME", "ARGOS_CHUNK_TYPE"):
        monkeypatch.delenv(var, raising=False)

    result = resources.configure_offline_translation()
    assert result == data
    # XDG_DATA_HOME is the parent so Argos resolves data dir + minisbd cache here.
    assert os.environ["XDG_DATA_HOME"] == str(assets / "argos")
    assert os.environ["ARGOS_CHUNK_TYPE"] == "MINISBD"


def test_configure_offline_translation_absent_is_noop(tmp_path, monkeypatch):
    monkeypatch.setenv(resources.ENV_ASSETS, str(tmp_path / "empty"))
    for var in ("XDG_DATA_HOME", "ARGOS_CHUNK_TYPE"):
        monkeypatch.delenv(var, raising=False)

    assert resources.configure_offline_translation() is None
    assert "ARGOS_CHUNK_TYPE" not in os.environ


def test_find_bundled_tesseract(tmp_path, monkeypatch):
    assets = tmp_path / "whispr_assets"
    (assets / "tesseract").mkdir(parents=True)
    name = "tesseract.exe" if os.name == "nt" else "tesseract"
    (assets / "tesseract" / name).write_bytes(b"\x00")
    monkeypatch.setenv(resources.ENV_ASSETS, str(assets))
    found = resources.find_bundled_tesseract()
    assert found is not None and found.parent == assets / "tesseract"


def test_find_tesseract_prefers_bundled(tmp_path, monkeypatch):
    assets = tmp_path / "whispr_assets"
    (assets / "tesseract").mkdir(parents=True)
    name = "tesseract.exe" if os.name == "nt" else "tesseract"
    (assets / "tesseract" / name).write_bytes(b"\x00")
    monkeypatch.setenv(resources.ENV_ASSETS, str(assets))
    monkeypatch.setattr(resources, "which", lambda _: "/usr/bin/tesseract")
    found = resources.find_tesseract()
    assert found is not None and found.parent == assets / "tesseract"


def test_find_tesseract_falls_back_to_path(tmp_path, monkeypatch):
    monkeypatch.setenv(resources.ENV_ASSETS, str(tmp_path / "empty"))
    monkeypatch.setattr(resources, "which", lambda _: "/usr/bin/tesseract")
    assert str(resources.find_tesseract()) == "/usr/bin/tesseract"


def test_find_tesseract_none_when_absent(tmp_path, monkeypatch):
    monkeypatch.setenv(resources.ENV_ASSETS, str(tmp_path / "empty"))
    monkeypatch.setattr(resources, "which", lambda _: None)
    assert resources.find_tesseract() is None


def test_configure_offline_ocr_present(tmp_path, monkeypatch):
    assets = tmp_path / "whispr_assets"
    tessdata = assets / "tesseract" / "tessdata"
    tessdata.mkdir(parents=True)
    monkeypatch.setenv(resources.ENV_ASSETS, str(assets))
    monkeypatch.delenv("TESSDATA_PREFIX", raising=False)

    result = resources.configure_offline_ocr()
    assert result == assets / "tesseract"
    assert os.environ["TESSDATA_PREFIX"] == str(tessdata)
    var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
    assert str(assets / "tesseract") in os.environ[var].split(os.pathsep)


def test_configure_offline_ocr_absent_is_noop(tmp_path, monkeypatch):
    monkeypatch.setenv(resources.ENV_ASSETS, str(tmp_path / "empty"))
    monkeypatch.delenv("TESSDATA_PREFIX", raising=False)
    assert resources.configure_offline_ocr() is None
    assert "TESSDATA_PREFIX" not in os.environ


def test_find_bundled_tesseract_at_relocated_frozen_root(tmp_path, monkeypatch):
    # PyInstaller can move the .exe out of whispr_assets/tesseract to the bundle
    # root; find_bundled_tesseract should still locate it via the frozen root.
    monkeypatch.setenv(resources.ENV_ASSETS, str(tmp_path / "empty"))
    name = "tesseract.exe" if os.name == "nt" else "tesseract"
    (tmp_path / name).write_bytes(b"\x00")
    monkeypatch.setattr(resources.sys, "_MEIPASS", str(tmp_path), raising=False)
    found = resources.find_bundled_tesseract()
    assert found == tmp_path / name


def test_configure_offline_ocr_adds_binary_and_root_to_path(tmp_path, monkeypatch):
    assets = tmp_path / "whispr_assets"
    tessdata = assets / "tesseract" / "tessdata"
    tessdata.mkdir(parents=True)
    name = "tesseract.exe" if os.name == "nt" else "tesseract"
    (assets / "tesseract" / name).write_bytes(b"\x00")
    monkeypatch.setenv(resources.ENV_ASSETS, str(assets))
    var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
    monkeypatch.setenv(var, "/pre-existing")

    resources.configure_offline_ocr()
    entries = os.environ[var].split(os.pathsep)
    assert str(assets / "tesseract") in entries  # binary's folder is searchable
    assert "/pre-existing" in entries  # existing entries preserved
