"""SHA-256 helpers for provenance (source media, model files, exported records).

Analysis results have to be traceable back to the exact recording and models that
produced them, so several subsystems need the same digest. This is the single
implementation they share.

Hashing is read-only and streams the file, so it never modifies or fully loads
source media.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]

# Read in 1 MiB blocks: large enough to keep the syscall count low on multi-GB
# media, small enough not to matter for memory.
_CHUNK_BYTES = 1024 * 1024


def sha256_file(path: PathLike) -> str:
    """Hex SHA-256 of a file's contents, streamed (never loads it whole)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file_or_none(path: Optional[PathLike]) -> Optional[str]:
    """Like :func:`sha256_file`, but ``None`` for a missing/unreadable path.

    Provenance is best-effort for optional assets: a missing model file should
    leave the hash blank rather than abort an analysis.
    """
    if path is None:
        return None
    try:
        return sha256_file(path)
    except OSError:
        return None


def sha256_bytes(data: bytes) -> str:
    """Hex SHA-256 of an in-memory value."""
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """Hex SHA-256 of UTF-8 encoded ``text``."""
    return sha256_bytes(text.encode("utf-8"))


def short(digest: Optional[str], length: int = 12) -> str:
    """Abbreviate a digest for display (full value stays in the record)."""
    if not digest:
        return "unknown"
    return digest[:length]
