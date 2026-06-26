import importlib.util

import pytest

from whispr import translation
from whispr.translation import detect_language


def test_chunk_text_rejoins_and_bounds():
    text = "line one\nline two\n\nparagraph two with more words\n"
    chunks = translation._chunk_text(text, max_chars=10)
    # Concatenation is lossless (structure preserved).
    assert "".join(chunks) == text
    # Every chunk respects the cap.
    assert all(len(c) <= 10 for c in chunks)


def test_chunk_text_hard_splits_long_line():
    text = "x" * 25
    chunks = translation._chunk_text(text, max_chars=10)
    assert "".join(chunks) == text
    assert all(len(c) <= 10 for c in chunks)
    assert len(chunks) == 3


def test_chunk_text_empty():
    assert translation._chunk_text("") == [""]


def test_translate_text_without_argos(monkeypatch):
    # argostranslate is not installed in this environment -> clear RuntimeError.
    with pytest.raises(RuntimeError, match="Argos Translate is not installed"):
        translation.translate_text("hola", from_code="es", to_code="en")


def test_available_source_languages_without_argos():
    with pytest.raises(RuntimeError, match="Argos Translate is not installed"):
        translation.available_source_languages()


def test_translate_files_without_argos(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hola", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Argos Translate is not installed"):
        translation.translate_files([f], from_code="es", to_code="en")


def test_detect_language_empty_is_none():
    assert detect_language("") is None
    assert detect_language("   ") is None


def test_detect_language_without_langdetect():
    if importlib.util.find_spec("langdetect") is not None:
        pytest.skip("langdetect installed; missing-dependency path not exercised")
    assert detect_language("this is plainly english text") is None


def test_detect_language_english():
    pytest.importorskip("langdetect")
    assert detect_language("This is a clearly English sentence about the weather.") == (
        "en"
    )
