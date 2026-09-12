"""Which name goes on which speaker, and when a name may not travel.

Two kinds of name end up on a diarized transcript, and they are not equally
trustworthy:

* a name *recognised* from a voiceprint, measured against the turns in front of
  it. It is earned, every time, by the audio it is put on.
* a name *typed* by an operator into the Speaker 1..N fields before a run. It is
  positional, and the diarizer's numbering is arbitrary, so even on a first run
  it may need swapping - which the interface says.

The distinction matters when a transcript is split a second time. The new split
numbers its speakers from scratch: "Speaker 1" afterwards need not be the person
"Speaker 1" was before. A recognised name survives that because it is worked out
again. A typed name must not, because carrying it across would put an operator's
word - often a subject's name - onto somebody else, while looking like the tool
had preserved their work rather than guessed.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence


def preset_names(
    speaker_ids: "Iterable[str]",
    recognized: "Mapping[str, str]",
    typed: "Sequence[str]",
    *,
    use_typed: bool = True,
) -> "Dict[str, str]":
    """Map speaker id -> display name for one result.

    Recognised voices are applied first; whatever is left over is matched to
    the typed Speaker N fields in label order. ``use_typed`` is False when the
    split is being redone, and then only the recognised names apply.
    """
    ids = sorted({sid for sid in speaker_ids if sid})
    names: Dict[str, str] = {sid: recognized[sid] for sid in ids if sid in recognized}
    if not use_typed:
        return names
    unrecognised = [sid for sid in ids if sid not in names]
    for sid, name in zip(unrecognised, typed):
        cleaned = name.strip()
        if cleaned:
            names[sid] = cleaned
    return names
