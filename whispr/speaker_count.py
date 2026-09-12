"""How many people are on the recording: the choice, and what it means.

Diarization can work the speaker count out for itself, and on clean audio it
usually does. On a long, poor-quality call it can be badly wrong in a way that
costs hours: a ninety-minute two-handed phone call came back split across five
speakers, and the only way to correct it was to run the whole recording again
with the count set.

So the count is no longer an advanced setting left at its default. The operator
answers a question before the run starts - a number, or "not sure" - and either
answer is fine. What is not fine is the question never being asked.

The vocabulary lives here rather than in the GUI so it can be tested without a
display, and so a stored setting and a dropdown label can never drift apart.
"""

from __future__ import annotations

from typing import List, Optional

# Shown until the operator answers. Never a usable answer itself.
UNSET = "Choose…"

# The explicit "work it out" answer. A real choice, not the absence of one.
UNKNOWN = "Not sure — work it out"

# Above this, naming each speaker up front stops being how anyone works, and
# the count is better left to the engine.
MAX_SPEAKERS = 10


def choices() -> "List[str]":
    """Every dropdown entry, in the order they are offered."""
    return [UNSET, UNKNOWN] + [_count_label(n) for n in range(1, MAX_SPEAKERS + 1)]


def _count_label(count: int) -> str:
    return "1 person" if count == 1 else f"{count} people"


def is_answered(label: str) -> bool:
    """True once the operator has actually answered the question.

    "Not sure" answers it. An unrecognised label - a stale value from an older
    build, say - does not, so a bad setting asks again rather than running with
    a count nobody chose.
    """
    return label in choices() and label != UNSET


def to_setting(label: str) -> str:
    """The stored value for a dropdown label.

    Empty means "no count given to the engine", which is what both "not sure"
    and an unanswered question come to. The difference between those two is
    whether the run is allowed to start, and that is :func:`is_answered`.
    """
    for count in range(1, MAX_SPEAKERS + 1):
        if label == _count_label(count):
            return str(count)
    return ""


def from_setting(value: object) -> str:
    """The dropdown label for a stored count.

    A positive count is an answer somebody gave; anything else (blank, zero,
    nonsense from a hand-edited file) is not, and comes back as :data:`UNSET`
    so it is asked again.
    """
    count = parse(value)
    if count is None:
        return UNSET
    return _count_label(count)


def parse(value: object) -> "Optional[int]":
    """A stored count as a number, or None when there is no usable count."""
    if isinstance(value, bool):  # bool is an int; a checkbox value is not a count
        return None
    if isinstance(value, int):
        count = value
    else:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            count = int(text)
        except ValueError:
            return None
    if count < 1 or count > MAX_SPEAKERS:
        return None
    return count
