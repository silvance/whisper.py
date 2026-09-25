"""Giving every recording in a batch an output name of its own.

Outputs were named after the source file and nothing else. That was fine while
a batch was a handful of files picked by hand, and stopped being fine the day a
whole folder could be dropped on the window: a case folder organised by day
holds ``2026-09-01/interview.wav`` and ``2026-09-02/interview.wav``, and both
wanted to be ``interview.wav.txt`` in the output folder. The second overwrote
the first, silently, and the run still finished green.

The rule here is that a name is extended only as far as it has to be. A
recording whose name is unique in the batch keeps it. Where two or more would
collide, each takes as much of its folder path as is needed to tell them
apart:

    2026-09-01/interview.wav  ->  2026-09-01 - interview.wav
    2026-09-02/interview.wav  ->  2026-09-02 - interview.wav
    carpark.m4a               ->  carpark.m4a

So the output folder stays flat - no directories the operator did not ask for
- nothing is ever overwritten by another recording in the same run, and the
names say where the ambiguous ones came from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

# Between the folder name and the file name. Spaces either side so the join is
# visible at a glance; " - " is legal on Windows where "/" and ":" are not.
JOIN = " - "

# Characters a filename cannot carry on Windows, which is where this ships.
ILLEGAL = '<>:"/\\|?*'


def clean(part: str) -> str:
    """A path component made safe to use inside a file name."""
    out = "".join("_" if ch in ILLEGAL else ch for ch in part).strip(" .")
    return out or "_"


def components(path: Path) -> "List[str]":
    """A path's folder names, nearest first: the order to disambiguate in."""
    return [clean(part) for part in reversed(path.parent.parts)]


def plan(sources: "Sequence[Path]") -> "Dict[Path, str]":
    """Map each source to the output name it should be written under.

    The returned names are unique across ``sources``, so no two recordings in
    one run can write over each other. Sources that share a name are extended
    with their folders until they differ; everything else is left alone.
    """
    by_name: "Dict[str, List[Path]]" = {}
    for source in sources:
        by_name.setdefault(clean(source.name), []).append(source)

    names: "Dict[Path, str]" = {}
    for base, group in by_name.items():
        if len(group) == 1:
            names[group[0]] = base
            continue
        for source in group:
            names[source] = _extended(source, base, group)
    # Extending one group can land on a name another group already holds - rare,
    # but silent overwriting is what this module exists to stop.
    return _make_unique(sources, names)


def _make_unique(
    sources: "Sequence[Path]", names: "Dict[Path, str]"
) -> "Dict[Path, str]":
    """The last guard: no two sources leave here with the same name.

    Walked in path order rather than queue order, so the same batch produces
    the same names however it was assembled - a run that has to be repeated
    should overwrite its own output, not write a second copy of it.
    """
    final: "Dict[Path, str]" = {}
    seen: "set[str]" = set()
    for source in sorted(sources, key=str):
        name = names[source]
        if name in seen:
            stem, dot, suffix = name.partition(".")
            nth = 2
            while f"{stem} ({nth}){dot}{suffix}" in seen:
                nth += 1
            name = f"{stem} ({nth}){dot}{suffix}"
        seen.add(name)
        final[source] = name
    return final


def _extended(source: Path, base: str, group: "Sequence[Path]") -> str:
    """The shortest folder-prefixed name that no other member of the group has."""
    mine = components(source)
    others = [components(other) for other in group if other != source]
    for depth in range(1, max((len(mine), *(len(o) for o in others)), default=0) + 1):
        prefix = mine[:depth]
        if all(other[:depth] != prefix for other in others):
            return JOIN.join([*reversed(prefix), base])
    return JOIN.join([*reversed(mine), base])


def transcript_name(base: str) -> str:
    """``interview.wav`` -> ``interview.wav.txt``, the convention already shipped."""
    return base + ".txt"


def subtitle_name(base: str) -> str:
    return base + ".srt"


def converted_audio_name(base: str) -> str:
    """The WAV a video is converted to, beside the rest of that file's output."""
    return Path(base).stem + ".wav"


__all__ = [
    "JOIN",
    "clean",
    "components",
    "converted_audio_name",
    "plan",
    "subtitle_name",
    "transcript_name",
]
