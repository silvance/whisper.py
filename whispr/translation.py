"""Offline text translation backend using Argos Translate (CTranslate2 / CPU).

This mirrors :mod:`whispr.transcription`: ``argostranslate`` is imported lazily so
importing this module does not require the optional dependency, and long jobs are
processed in bounded chunks and are cancellable, so a large batch cannot exhaust
memory (the failure mode of the old tool).

Translation runs fully offline from bundled Argos language packs (see
``whispr.resources.configure_offline_translation``); from source it uses whatever
packs Argos has installed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

from .transcription import CancelCallback, CancelledError, ProgressCallback

PathLike = Union[str, Path]

# Default target language for the foreign -> English workflow.
DEFAULT_TARGET = "en"

# Cap the text handed to a single translate() call so peak memory stays bounded
# no matter how large the input file is.
MAX_CHUNK_CHARS = 2000


def _argos_translate():
    """Import argostranslate.translate, with a clear error if it's missing."""
    try:
        import argostranslate.translate as translate_mod
    except ImportError as exc:
        raise RuntimeError(
            "Argos Translate is not installed. Install the translation extra "
            "with:  pip install 'silvance-whisper[translate]'"
        ) from exc
    return translate_mod


def available_source_languages(
    target: str = DEFAULT_TARGET,
) -> List[Tuple[str, str]]:
    """Return ``(code, name)`` for installed languages translatable to ``target``.

    Used to populate the GUI's "From" dropdown. Empty when no packs are installed.
    """
    translate_mod = _argos_translate()
    languages = translate_mod.get_installed_languages()
    target_lang = next((lang for lang in languages if lang.code == target), None)
    if target_lang is None:
        return []
    out: List[Tuple[str, str]] = []
    for lang in languages:
        if lang.code == target:
            continue
        if lang.get_translation(target_lang) is not None:
            out.append((lang.code, lang.name))
    return sorted(out, key=lambda pair: pair[1])


def _make_translator(from_code: str, to_code: str):
    """Build a reusable translation object (loads the model once)."""
    translate_mod = _argos_translate()
    translation = translate_mod.get_translation_from_codes(from_code, to_code)
    if translation is None:
        raise RuntimeError(
            f"No installed translation for {from_code!r} -> {to_code!r}. "
            "Bundle or install the matching Argos language pack."
        )
    return translation


def _chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> List[str]:
    """Split ``text`` into <= ``max_chars`` pieces on line boundaries.

    Newline separators are preserved inside the chunks, so concatenating the
    translated chunks keeps the original line/paragraph structure. Pathologically
    long single lines are hard-split as a last resort.
    """
    chunks: List[str] = []
    buf = ""
    # Keep the "\n" separators (odd indices) so structure is preserved.
    for part in re.split(r"(\n)", text):
        if buf and len(buf) + len(part) > max_chars:
            chunks.append(buf)
            buf = ""
        buf += part
        while len(buf) > max_chars:
            chunks.append(buf[:max_chars])
            buf = buf[max_chars:]
    if buf:
        chunks.append(buf)
    return chunks or [""]


def translate_text(
    text: str,
    *,
    from_code: str,
    to_code: str = DEFAULT_TARGET,
    on_progress: Optional[Callable[[float], None]] = None,
    cancelled: Optional[CancelCallback] = None,
) -> str:
    """Translate ``text`` from ``from_code`` to ``to_code``, preserving structure.

    The text is translated chunk-by-chunk (bounded memory) and the job can be
    cancelled between chunks.
    """
    translation = _make_translator(from_code, to_code)
    chunks = _chunk_text(text)
    out: List[str] = []
    for index, chunk in enumerate(chunks):
        if cancelled is not None and cancelled():
            raise CancelledError("Translation cancelled.")
        out.append(translation.translate(chunk) if chunk.strip() else chunk)
        if on_progress is not None:
            on_progress((index + 1) / len(chunks))
    return "".join(out)


def translate_files(
    paths: Sequence[PathLike],
    *,
    from_code: str,
    to_code: str = DEFAULT_TARGET,
    progress: Optional[ProgressCallback] = None,
    on_progress: Optional[Callable[[float], None]] = None,
    cancelled: Optional[CancelCallback] = None,
) -> List[Path]:
    """Translate each text file, writing ``<name>.<to_code><suffix>`` beside it.

    One translation model is loaded and reused across all files, and each file is
    streamed through ``_chunk_text`` so peak memory does not grow with the batch
    size. Returns the list of output paths.
    """
    translation = _make_translator(from_code, to_code)
    outputs: List[Path] = []
    total = max(1, len(paths))
    for file_index, raw_path in enumerate(paths):
        if cancelled is not None and cancelled():
            raise CancelledError("Translation cancelled.")
        src = Path(raw_path)
        if progress is not None:
            progress(f"Translating {src.name}...")
        text = src.read_text(encoding="utf-8", errors="replace")
        parts: List[str] = []
        for chunk in _chunk_text(text):
            if cancelled is not None and cancelled():
                raise CancelledError("Translation cancelled.")
            parts.append(translation.translate(chunk) if chunk.strip() else chunk)
        dest = src.with_name(f"{src.stem}.{to_code}{src.suffix}")
        dest.write_text("".join(parts), encoding="utf-8")
        outputs.append(dest)
        if progress is not None:
            progress(f"Wrote {dest.name}")
        if on_progress is not None:
            on_progress((file_index + 1) / total)
    return outputs
