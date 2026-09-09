"""Check that an installed bundle is the one the build produced.

A desktop bundle is thousands of files copied onto removable media and carried
to a machine with no network. Copies go wrong: an archive extracts short, a
scanner quarantines a library, a drive returns a bad read. The application then
fails at whichever missing piece it happens to reach first, with a traceback
about that piece rather than about the copy - and the same folder can fail two
different ways on two runs, which is the clearest sign that the files, not the
software, are the problem.

The build records an inventory of every file it shipped, with size and SHA-256.
``whispr --verify`` re-reads the installed copy and says plainly which files are
missing, truncated or altered. That turns "it crashes differently each time"
into a list.

The inventory is a statement about *this build's own files*. It is not a
security control: anyone who can alter the files can alter the inventory beside
them. It answers "did this copy arrive intact", not "is this copy trustworthy".
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .hashing import sha256_file

PathLike = Union[str, Path]
ProgressFn = Callable[[str], None]

INVENTORY_NAME = "bundle-inventory.json"
SCHEMA_VERSION = 1


def install_root() -> Optional[Path]:
    """The directory the running bundle was installed into, if it is one.

    Frozen one-dir builds put the executable at the top of the copied folder,
    so that is what an operator extracted and what an inventory describes.
    Returns ``None`` when running from source, where there is nothing to check.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return None


def inventory_path(root: Optional[PathLike] = None) -> Optional[Path]:
    """Where the inventory for ``root`` lives (``None`` if there is no root)."""
    base = Path(root) if root is not None else install_root()
    if base is None:
        return None
    return Path(base) / INVENTORY_NAME


@dataclass
class FileRecord:
    """One file the build shipped."""

    path: str
    size: int
    sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return {"path": self.path, "size": self.size, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, data: Any) -> "Optional[FileRecord]":
        if not isinstance(data, dict):
            return None
        path = str(data.get("path") or "")
        if not path:
            return None
        try:
            size = int(data.get("size") or 0)
        except (TypeError, ValueError):
            size = 0
        return cls(path=path, size=size, sha256=str(data.get("sha256") or ""))


@dataclass
class VerifyResult:
    """What the check found, in the terms an operator needs to act on."""

    root: str = ""
    checked: int = 0
    # Recorded by the build but not present now.
    missing: List[str] = field(default_factory=list)
    # Present, but not the file the build shipped.
    altered: List[str] = field(default_factory=list)
    # Present but unreadable - a scanner holding it, or a bad read.
    unreadable: List[str] = field(default_factory=list)
    # Present and not in the inventory. Reported, never a failure: an operator
    # may keep their own files beside the application.
    extra: List[str] = field(default_factory=list)
    # Set when there is no inventory to check against at all.
    reason: str = ""

    @property
    def intact(self) -> bool:
        return not (self.missing or self.altered or self.unreadable)

    @property
    def usable(self) -> bool:
        """False when the check could not be run, rather than found problems."""
        return not self.reason

    def summary_lines(self) -> List[str]:
        if not self.usable:
            return [self.reason]
        lines = [f"Checked {self.checked} file(s) in {self.root}."]
        if self.intact:
            lines.append("This copy matches the build that produced it.")
        else:
            lines.append(
                "This copy does NOT match the build that produced it. The "
                "application may fail in different places on different runs "
                "until it is replaced."
            )
        for label, entries in (
            ("Missing", self.missing),
            ("Altered or truncated", self.altered),
            ("Could not be read", self.unreadable),
        ):
            if entries:
                lines.append("")
                lines.append(f"{label} ({len(entries)}):")
                lines.extend(f"  {name}" for name in entries[:40])
                if len(entries) > 40:
                    lines.append(f"  ... and {len(entries) - 40} more")
        if self.extra:
            lines.append("")
            lines.append(
                f"{len(self.extra)} file(s) present that the build did not ship "
                "(not a problem in itself)."
            )
        return lines


def _walk(root: Path) -> List[Path]:
    out: List[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            out.append(Path(dirpath) / name)
    return out


def build_inventory(
    root: PathLike, *, progress: Optional[ProgressFn] = None
) -> Dict[str, Any]:
    """Hash every file under ``root`` and return the inventory to write there.

    The inventory itself is excluded: it does not exist yet when this runs, and
    a file cannot record its own digest.
    """
    base = Path(root).resolve()
    records: List[FileRecord] = []
    for path in sorted(_walk(base)):
        relative = path.relative_to(base).as_posix()
        if relative == INVENTORY_NAME:
            continue
        if progress is not None:
            progress(relative)
        records.append(
            FileRecord(
                path=relative,
                size=path.stat().st_size,
                sha256=sha256_file(path),
            )
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "file_count": len(records),
        "total_bytes": sum(r.size for r in records),
        "files": [r.to_dict() for r in records],
    }


def load_inventory(path: PathLike) -> List[FileRecord]:
    """Read an inventory file into records (empty list if it holds none)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("inventory is not an object")
    version = data.get("schema_version")
    if isinstance(version, int) and version > SCHEMA_VERSION:
        raise ValueError(
            f"inventory was written by a newer version of Whispers "
            f"(schema {version}, this build understands {SCHEMA_VERSION})"
        )
    raw = data.get("files")
    if not isinstance(raw, list):
        return []
    records = [FileRecord.from_dict(item) for item in raw]
    return [r for r in records if r is not None]


def verify(
    root: Optional[PathLike] = None, *, progress: Optional[ProgressFn] = None
) -> VerifyResult:
    """Compare the installed copy at ``root`` against the build's inventory."""
    base = Path(root).resolve() if root is not None else install_root()
    if base is None:
        return VerifyResult(
            reason=(
                "This is not an installed bundle - there is nothing to check. "
                "Run --verify on a copy built by the release workflow."
            )
        )
    result = VerifyResult(root=str(base))
    manifest = Path(base) / INVENTORY_NAME
    try:
        records = load_inventory(manifest)
    except FileNotFoundError:
        result.reason = (
            f"No {INVENTORY_NAME} beside the application, so there is nothing to "
            "check this copy against. It was either built before this check "
            "existed, or the file did not survive the copy."
        )
        return result
    except (OSError, ValueError) as exc:
        result.reason = f"Could not read {manifest}: {exc}"
        return result

    listed = {r.path for r in records}
    for record in records:
        if progress is not None:
            progress(record.path)
        target = base / record.path
        result.checked += 1
        if not target.is_file():
            result.missing.append(record.path)
            continue
        try:
            if target.stat().st_size != record.size:
                # Reported separately from a hash mismatch: a short file is the
                # signature of a copy that stopped early.
                result.altered.append(record.path)
                continue
            if record.sha256 and sha256_file(target) != record.sha256:
                result.altered.append(record.path)
        except OSError:
            result.unreadable.append(record.path)

    for path in _walk(base):
        relative = path.relative_to(base).as_posix()
        if relative != INVENTORY_NAME and relative not in listed:
            result.extra.append(relative)
    return result
