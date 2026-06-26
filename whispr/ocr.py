"""Offline OCR for images and PDFs, feeding the translation workflow.

This lets an operator drop a photo of a document or a scanned PDF into the
Translate tab and get the foreign text out (which can then be reviewed and
translated to English). It mirrors the rest of the package: the heavy
dependencies (``pytesseract``/Tesseract, ``pypdfium2``, ``Pillow``) are imported
lazily so importing this module never requires them, and long jobs are
cancellable and report progress.

Text extraction has two layers:

* **Images** (`.png`, `.jpg`, ...) are read with Pillow and OCR'd with Tesseract.
* **PDFs** are opened with pypdfium2; each page's embedded text is used directly
  when present (digital PDFs), and pages with little/no text (scans) are
  rendered to an image and OCR'd. So both digital and scanned PDFs work.

Tesseract is used because of its strong coverage of the team's priority
languages, including right-to-left scripts (Arabic, Farsi, Hebrew) and
CJK/Cyrillic/Korean. The language data files (``tessdata``) and the Tesseract
binary are bundled into the offline build (see ``whispr.resources``); from
source it uses a system Tesseract on PATH.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from .resources import bundled_tessdata_dir, find_tesseract
from .transcription import CancelCallback, CancelledError, ProgressCallback

PathLike = Union[str, Path]

# Image containers we offer for OCR (anything Pillow + Tesseract can read).
IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".gif",
    ".webp",
)
PDF_EXTENSIONS = (".pdf",)
OCR_EXTENSIONS = IMAGE_EXTENSIONS + PDF_EXTENSIONS

# Argos / ISO 639-1 source codes -> Tesseract language codes (639-2/3). Covers the
# bundled intel-leaning set plus a few common extras; unknown codes pass through
# (so a raw Tesseract code like "chi_tra" also works if given directly).
_TESSERACT_LANG = {
    "ar": "ara",
    "ru": "rus",
    "zh": "chi_sim",
    "fa": "fas",
    "uk": "ukr",
    "he": "heb",
    "ko": "kor",
    "en": "eng",
    "es": "spa",
    "fr": "fra",
    "de": "deu",
    "it": "ita",
    "pt": "por",
    "nl": "nld",
    "pl": "pol",
    "tr": "tur",
    "ja": "jpn",
    "hi": "hin",
    "ur": "urd",
    "ps": "pus",
}

# Rendering DPI for scanned PDF pages handed to Tesseract. 300 is the usual sweet
# spot for OCR accuracy without ballooning memory.
_PDF_RENDER_DPI = 300

# A page yielding fewer than this many characters of embedded text is treated as a
# scan and rendered + OCR'd instead.
_MIN_EMBEDDED_CHARS = 16


def tesseract_lang(code: str) -> str:
    """Map an ISO 639-1 / Argos code to a Tesseract language code."""
    return _TESSERACT_LANG.get(code, code)


def is_image(path: PathLike) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def is_pdf(path: PathLike) -> bool:
    return Path(path).suffix.lower() in PDF_EXTENSIONS


def is_ocr_file(path: PathLike) -> bool:
    """True if ``path`` is an image or PDF we can extract text from."""
    return Path(path).suffix.lower() in OCR_EXTENSIONS


def ocr_available() -> bool:
    """True if OCR can run: a Tesseract binary plus the Python deps are present.

    Lets the GUI hide/disable OCR controls (and a lean build present cleanly) when
    the engine isn't bundled, without importing the heavy libraries just to check.
    """
    if find_tesseract() is None:
        return False
    try:
        import importlib.util

        return (
            importlib.util.find_spec("pytesseract") is not None
            and importlib.util.find_spec("PIL") is not None
        )
    except ModuleNotFoundError:
        return False


def _configure_pytesseract():
    """Return the pytesseract module wired up to the bundled/Tesseract binary."""
    try:
        import pytesseract
    except ImportError as exc:
        raise RuntimeError(
            "pytesseract is not installed. Install the OCR extra with:  "
            "pip install 'silvance-whisper[ocr]'"
        ) from exc
    binary = find_tesseract()
    if binary is None:
        raise RuntimeError(
            "Tesseract OCR was not found (neither bundled in whispr_assets nor on "
            "PATH). Use a build that bundles it, or install Tesseract."
        )
    pytesseract.pytesseract.tesseract_cmd = str(binary)
    return pytesseract


def _tessdata_config() -> str:
    """Tesseract CLI config pointing at bundled tessdata, if present."""
    data = bundled_tessdata_dir()
    return f'--tessdata-dir "{data}"' if data is not None else ""


def _ocr_image(image, lang: str) -> str:
    """OCR a Pillow image with Tesseract and return stripped text."""
    pytesseract = _configure_pytesseract()
    return pytesseract.image_to_string(
        image, lang=lang, config=_tessdata_config()
    ).strip()


def _extract_image_file(path: Path, lang: str) -> str:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is not installed. Install the OCR extra with:  "
            "pip install 'silvance-whisper[ocr]'"
        ) from exc
    with Image.open(path) as image:
        return _ocr_image(image, lang)


def _extract_pdf_file(
    path: Path,
    lang: str,
    progress: Optional[ProgressCallback],
    cancelled: Optional[CancelCallback],
) -> str:
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RuntimeError(
            "pypdfium2 is not installed. Install the OCR extra with:  "
            "pip install 'silvance-whisper[ocr]'"
        ) from exc

    pdf = pdfium.PdfDocument(str(path))
    try:
        pages: list[str] = []
        total = len(pdf)
        for index in range(total):
            if cancelled is not None and cancelled():
                raise CancelledError("OCR cancelled.")
            page = pdf[index]
            # Use embedded text when the page has it (digital PDF); otherwise the
            # page is a scan, so render it and OCR the image.
            textpage = page.get_textpage()
            text = textpage.get_text_range().strip()
            if len(text) < _MIN_EMBEDDED_CHARS:
                if progress is not None:
                    progress(f"OCR page {index + 1}/{total} (scanned)…")
                bitmap = page.render(scale=_PDF_RENDER_DPI / 72)
                image = bitmap.to_pil()
                try:
                    text = _ocr_image(image, lang)
                finally:
                    image.close()
            elif progress is not None:
                progress(f"Read page {index + 1}/{total} (text)…")
            pages.append(text)
        return "\n\n".join(pages).strip()
    finally:
        pdf.close()


def extract_text(
    path: PathLike,
    *,
    lang: str = "eng",
    progress: Optional[ProgressCallback] = None,
    cancelled: Optional[CancelCallback] = None,
) -> str:
    """Extract text from an image or PDF.

    Parameters
    ----------
    path
        An image or PDF file (see ``OCR_EXTENSIONS``).
    lang
        A Tesseract language code (e.g. ``"ara"``). Use :func:`tesseract_lang`
        to map an ISO 639-1 / Argos code (``"ar"``) to it.
    progress, cancelled
        Optional status callback and cooperative-cancel predicate.
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Input file does not exist: {src}")
    if is_pdf(src):
        return _extract_pdf_file(src, lang, progress, cancelled)
    if is_image(src):
        if progress is not None:
            progress(f"OCR {src.name}…")
        return _extract_image_file(src, lang)
    raise ValueError(f"Not an OCR-able image or PDF: {src.name}")
