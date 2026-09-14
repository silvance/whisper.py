"""Where a panel's own window opens, and what the button that sends it says.

A transcript is a document, and a document wants a screen - often the second
screen, next to the case file or the map - rather than a box at the foot of a
page of settings. The GUI does the moving (see :mod:`whispr.gui.popout`); this
is the part of the decision that is arithmetic and wording, kept out of the
GUI so it can be tested without a display.
"""

from __future__ import annotations

from typing import Optional

from .dialog_placement import Rect, geometry, place

# The same button, both ways round. The wording says what will happen rather
# than naming the mechanism: nobody outside the code thinks in "panes".
POPOUT_LABEL = "Open in its own window"
DOCK_LABEL = "Put it back on this page"

# Below these a window would be worse than no window at all.
MIN_WIDTH = 480
MIN_HEIGHT = 360


def fit_geometry(
    size: str,
    screen_width: int,
    screen_height: int,
    over: "Optional[Rect]" = None,
) -> str:
    """Where to open the panel's window: the wanted size, cut down to the screen.

    An operator who moves Whispers onto a laptop panel after working on a
    desktop monitor should not get a transcript window taller than the screen
    with its buttons off the bottom edge, so the wanted size is a ceiling
    rather than a promise. The window is only *placed* here; the point of the
    feature is that they then move and maximise it wherever they like.

    ``over`` is the main window, and the new window opens on top of it. Without
    it the arithmetic can only centre on the screen Tk reports, which on
    Windows is the primary one - so a transcript sent to its own window from an
    application running on the second monitor would arrive on the first. That
    is the opposite of what this feature is for.
    """
    want_w, want_h = (int(n) for n in size.split("x"))
    width = min(want_w, max(MIN_WIDTH, screen_width - 80))
    height = min(want_h, max(MIN_HEIGHT, screen_height - 120))
    if over is not None:
        position = place((width, height), over)
        if position is not None:
            return geometry((width, height), position)
    x = max(0, (screen_width - width) // 2)
    y = max(0, (screen_height - height) // 3)
    return f"{width}x{height}+{x}+{y}"
