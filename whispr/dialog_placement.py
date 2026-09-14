"""Where a dialog belongs: over the window that asked the question.

A second monitor turns a small default into a real failure. Tk, left to
itself, lets the window manager place a new dialog, and on Windows that is
usually the top-left of the *primary* display - not the display the
application is on. An analyst working with Whispers on the second screen gets
a question they never see, behind whatever is on their first screen, while the
application sits there apparently frozen, waiting on an answer to a dialog
that is somewhere else entirely.

So placement is arithmetic here rather than a default: centre the dialog on
the parent window's own rectangle. That rectangle is where the operator is
already looking, whichever monitor it is on, and it needs no knowledge of how
many screens there are or where they sit relative to each other - which is as
well, because Tk will not reliably say.

Kept out of the GUI package so it can be reasoned about, and tested, without a
display.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Rect:
    """A window's position and size in screen coordinates."""

    x: int
    y: int
    width: int
    height: int

    @property
    def usable(self) -> bool:
        """Whether this rectangle says anything worth placing against.

        A window that has not been drawn yet reports 1x1 at 0,0. Centring on
        that would put every dialog in the corner of the primary screen, which
        is the very thing this module exists to avoid.
        """
        return self.width > 1 and self.height > 1


def place(size: Tuple[int, int], parent: Rect) -> "Optional[Tuple[int, int]]":
    """Top-left corner for a dialog of ``size`` opening over ``parent``.

    Centred on the parent, and kept within it: a dialog that fits cannot land
    beyond an edge of the window it belongs to, so it cannot stray onto
    another monitor. A dialog too large to fit is aligned to the parent's
    corresponding edge instead, which keeps its title bar - and its close
    button - where the operator can reach them.

    ``None`` when the parent has no meaningful geometry yet; the caller should
    then leave placement alone rather than guess.
    """
    if not parent.usable:
        return None
    width, height = (max(1, int(n)) for n in size)
    return (
        _axis(width, parent.x, parent.width),
        _axis(height, parent.y, parent.height),
    )


def _axis(length: int, origin: int, extent: int) -> int:
    if length >= extent:
        return origin
    return origin + (extent - length) // 2


def geometry(size: Tuple[int, int], position: Tuple[int, int]) -> str:
    """The ``WxH+X+Y`` string Tk's ``wm geometry`` wants.

    Negative coordinates are written as ``+-N``: in Tk's grammar a bare ``-N``
    measures from the opposite edge of the screen, which on a monitor to the
    left of the primary one would throw the dialog across the desktop.
    """
    width, height = (max(1, int(n)) for n in size)
    x, y = (int(n) for n in position)
    return f"{width}x{height}+{x}+{y}"
