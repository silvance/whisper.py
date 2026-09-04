import importlib.util
from pathlib import Path

import pytest

# fetch_assets.py is a build script under packaging/ (not an installed module);
# load it by path. Its heavy imports are all inside functions, so this is cheap.
_PATH = Path(__file__).resolve().parent.parent / "packaging" / "fetch_assets.py"
_spec = importlib.util.spec_from_file_location("fetch_assets", _PATH)
fetch_assets = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch_assets)


def test_resolve_embedding_alias():
    name, url = fetch_assets._resolve_embedding("titanet-large")
    assert name == "titanet-large"
    assert url.endswith("nemo_en_titanet_large.onnx")


def test_resolve_embedding_zh_cn_alias():
    name, url = fetch_assets._resolve_embedding("campplus")
    assert name == "campplus"
    assert url.startswith(fetch_assets.EMBEDDING_RELEASE_BASE)
    assert url.endswith(".onnx")


def test_resolve_embedding_direct_url():
    url_in = "https://example.com/models/my_custom_embed.onnx"
    name, url = fetch_assets._resolve_embedding(url_in)
    assert url == url_in
    assert name == "my_custom_embed"


def test_resolve_embedding_unknown_raises():
    with pytest.raises(SystemExit):
        fetch_assets._resolve_embedding("not-a-real-model")


def test_default_embedding_is_a_known_alias():
    assert fetch_assets.DEFAULT_EMBEDDING in fetch_assets.EMBEDDING_MODELS


def test_embedding_metadata_records_what_the_file_actually_is():
    name, url = fetch_assets._resolve_embedding("campplus")
    meta = fetch_assets.embedding_metadata(name, url)
    # The published CAM++/ERes2Net assets are the zh-cn checkpoints; record that
    # rather than calling them broadly multilingual.
    assert "zh-cn" in meta["label"]
    assert "Mandarin" in meta["training_data"]
    assert meta["filename"].endswith(".onnx")
    assert meta["source_url"] == url
    assert "varies with language" in meta["caveat"]


def test_no_embedding_alias_claims_to_be_multilingual():
    for entry in fetch_assets.EMBEDDING_MODELS.values():
        blob = " ".join(str(v) for v in entry.values()).lower()
        assert "multilingual" not in blob


def test_embedding_metadata_invents_nothing_for_a_direct_url():
    name, url = fetch_assets._resolve_embedding("https://example.invalid/custom.onnx")
    meta = fetch_assets.embedding_metadata(name, url)
    assert meta["name"] == "custom"
    # Nothing is known about an arbitrary file beyond its name and where it came
    # from, so the descriptive fields stay empty.
    assert meta["architecture"] == ""
    assert meta["training_data"] == ""
    assert "sha256" not in meta


def test_embedding_metadata_hashes_the_downloaded_file(tmp_path):
    model = tmp_path / "embedding.onnx"
    model.write_bytes(b"weights")
    name, url = fetch_assets._resolve_embedding("titanet-large")
    meta = fetch_assets.embedding_metadata(name, url, model)
    from whispr.hashing import sha256_file

    assert meta["sha256"] == sha256_file(model)
    assert meta["size"] == len(b"weights")
