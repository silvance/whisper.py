import importlib.util

import pytest

from whispr import ocr
from whispr.ocr import (
    extract_text,
    is_image,
    is_ocr_file,
    is_pdf,
    ocr_available,
    tesseract_lang,
)


@pytest.mark.parametrize(
    "code,expected",
    [
        ("ar", "ara"),
        ("fa", "fas"),
        ("he", "heb"),
        ("zh", "chi_sim"),
        ("ko", "kor"),
        ("en", "eng"),
        ("chi_tra", "chi_tra"),  # already a Tesseract code -> passthrough
        ("xx", "xx"),  # unknown -> passthrough
    ],
)
def test_tesseract_lang_mapping(code, expected):
    assert tesseract_lang(code) == expected


@pytest.mark.parametrize(
    "name,image,pdf,ocrable",
    [
        ("scan.png", True, False, True),
        ("photo.JPG", True, False, True),
        ("doc.pdf", False, True, True),
        ("notes.txt", False, False, False),
        ("clip.mp3", False, False, False),
    ],
)
def test_file_type_predicates(name, image, pdf, ocrable):
    assert is_image(name) is image
    assert is_pdf(name) is pdf
    assert is_ocr_file(name) is ocrable


def test_ocr_available_false_without_binary(monkeypatch):
    monkeypatch.setattr(ocr, "find_tesseract", lambda: None)
    assert ocr_available() is False


def test_extract_text_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_text(tmp_path / "missing.png", lang="eng")


def test_extract_text_rejects_non_ocr_file(tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="image or PDF"):
        extract_text(notes, lang="eng")


def test_extract_image_requires_pillow(tmp_path):
    if importlib.util.find_spec("PIL") is not None:
        pytest.skip("Pillow is installed; missing-dependency path not exercised")
    image = tmp_path / "scan.png"
    image.write_bytes(b"\x00")
    with pytest.raises(RuntimeError, match="Pillow is not installed"):
        extract_text(image, lang="eng")


def test_extract_pdf_requires_pypdfium2(tmp_path):
    if importlib.util.find_spec("pypdfium2") is not None:
        pytest.skip("pypdfium2 is installed; missing-dependency path not exercised")
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    with pytest.raises(RuntimeError, match="pypdfium2 is not installed"):
        extract_text(pdf, lang="eng")
