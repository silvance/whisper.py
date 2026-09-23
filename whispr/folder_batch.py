"""Turning "here is a folder" into "here are the recordings in it".

An operator with a case folder wants to hand over the folder, not pick its
contents out of a file dialog one at a time. What they hand over is rarely
only recordings: there are notes, photographs, a spreadsheet, the transcripts
from the last run, and usually subfolders - by day, by source, by target.

So this walks it and decides what is a recording, and it says what it passed
over rather than quietly shortening the list. A folder of 50 that arrives as a
queue of 12 has to explain itself, because the other 38 are either irrelevant
or the whole point, and only the operator knows which.

Two rules are worth stating plainly:

* A file the operator named themselves is taken at its word. Extensions are a
  guess about content, and someone who points at a specific file knows more
  about it than the guess does. Only the contents of a *folder* are filtered.
* Nothing here follows a symlink out of the folder it was given. A recording
  the operator did not put there is not in scope, and a loop would walk
  forever.

Kept out of the GUI so the walking and the wording can be tested without a
display.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Set

from .transcription import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS

# Files that are not recordings and never worth mentioning as "ignored" - the
# operating system's own litter, which the operator did not put there.
LITTER = frozenset({".ds_store", "thumbs.db", "desktop.ini"})


def is_media(path: Path) -> bool:
    """Whether a file's name says it is something we can transcribe."""
    return path.suffix.lower() in AUDIO_EXTENSIONS


@dataclass
class Found:
    """What a walk turned up, including what it decided against."""

    recordings: List[Path] = field(default_factory=list)
    folders: int = 0
    ignored: int = 0
    unreadable: List[str] = field(default_factory=list)
    converted: List[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.recordings)


def expand(paths: "Sequence[Path]", *, recursive: bool = True) -> Found:
    """Resolve a mixture of files and folders into the recordings to transcribe.

    Folders are walked - every subfolder too, unless ``recursive`` is off -
    and their contents filtered to media files. Files given directly are kept
    as given. The order is the order an operator would read them in: each
    folder's own files, sorted by name, before the folders inside it.
    """
    found = Found()
    seen: Set[Path] = set()
    for path in paths:
        if path.is_dir():
            _walk(path, found, seen, recursive=recursive)
        elif path not in seen:
            seen.add(path)
            found.recordings.append(path)
    _drop_our_own_conversions(found)
    return found


def _walk(folder: Path, found: Found, seen: Set[Path], *, recursive: bool) -> None:
    def note(error: OSError) -> None:
        # A folder that cannot be read is reported, not raised: one unreadable
        # subfolder should not cost the operator the other forty.
        found.unreadable.append(str(error.filename or folder))

    for root, dirs, names in os.walk(folder, onerror=note):
        found.folders += 1
        dirs.sort(key=str.lower)
        here = Path(root)
        for name in sorted(names, key=str.lower):
            path = here / name
            if name.lower() in LITTER or name.startswith("."):
                continue
            if not is_media(path):
                found.ignored += 1
                continue
            if path not in seen:
                seen.add(path)
                found.recordings.append(path)
        if not recursive:
            dirs[:] = []


def _drop_our_own_conversions(found: Found) -> None:
    """Leave behind the WAVs this application wrote beside a video last time.

    A batch converts video to WAV next to the source, so a folder run twice
    would find ``interview.mp4`` and the ``interview.wav`` we made from it, and
    transcribe the same hour twice. The pair is only assumed when both are in
    the same folder and both are in this batch; the file is named in the status
    log, and an operator who really does have separate audio of the same event
    can add it back by choosing it directly.
    """
    videos = {
        (path.parent, path.stem)
        for path in found.recordings
        if path.suffix.lower() in VIDEO_EXTENSIONS
    }
    if not videos:
        return
    kept: List[Path] = []
    for path in found.recordings:
        if path.suffix.lower() == ".wav" and (path.parent, path.stem) in videos:
            found.converted.append(path.name)
            continue
        kept.append(path)
    found.recordings = kept


def plural(count: int, one: str, many: str = "") -> str:
    """ "1 recording", "3 recordings" - written out, because people read this."""
    return f"{count} {one if count == 1 else (many or one + 's')}"


def describe(found: Found, *, queued: "Sequence[Path]" = (), sample: int = 4) -> str:
    """The sentence under the queue: what is in it, and what was left out.

    Silence about the files it passed over would be the wrong kind of tidy. An
    operator who dropped a folder of 50 and sees 12 queued needs to know the
    other 38 were not recordings - not wonder whether the tool lost them.
    """
    recordings = list(queued) or found.recordings
    if not recordings:
        if found.folders:
            passed = (
                f" {plural(found.ignored, 'file')} in there "
                f"{'is not' if found.ignored == 1 else 'are not'} audio or video."
                if found.ignored
                else ""
            )
            return "No recordings in that folder." + passed
        return ""

    opening = plural(len(recordings), "recording") + " queued"
    if found.folders > 1:
        opening += f" from {plural(found.folders, 'folder')}"
    names = ", ".join(path.name for path in recordings[:sample])
    if len(recordings) > sample:
        names += f" (+{len(recordings) - sample} more)"
    sentence = f"{opening} — {names}."

    asides: List[str] = []
    if found.ignored:
        asides.append(f"{plural(found.ignored, 'other file')} ignored")
    if found.converted:
        asides.append(
            f"{plural(len(found.converted), 'already-converted copy', 'already-converted copies')}"
            f" left out ({', '.join(found.converted[:3])})"
        )
    if found.unreadable:
        asides.append(f"{plural(len(found.unreadable), 'folder')} could not be read")
    if asides:
        sentence += " " + "; ".join(asides).capitalize() + "."
    return sentence


__all__ = ["Found", "LITTER", "describe", "expand", "is_media", "plural"]
