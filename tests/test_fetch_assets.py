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


def test_resolve_embedding_multilingual_alias():
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
