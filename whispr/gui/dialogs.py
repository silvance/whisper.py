"""Opening a window where the operator is looking.

Tk leaves a new dialog's position to the window manager, which on Windows
means the top-left of the primary display. With the application on a second
monitor the result is a question asked off-screen: the operator sees a window
that has stopped responding and no reason for it, because the reason is on
another screen behind a browser.

Everything here is about that: put the window over the one it came from, raise
it, and make sure it can be answered - or dismissed - without hunting for it.

The arithmetic lives in :mod:`whispr.dialog_placement`, which has no display to
depend on. What is left here is the Tk: asking a window where it is, and
telling another one where to go.
"""

from __future__ import annotations

import tkinter as tk
from typing import Callable, Optional

from ..dialog_placement import Rect, geometry, place


def active_window(widget: tk.Misc) -> tk.Misc:
    """The window a dialog raised from ``widget`` should open over.

    Normally the window ``widget`` is in. When the transcript has been sent to
    a screen of its own, though, that window is where the operator is working
    and its buttons are the ones being pressed, so a question raised from it
    belongs there and not back on the settings window.

    Which makes ``winfo_toplevel`` the wrong question to ask: the detached
    transcript is a frame handed to the window manager, not a ``Toplevel``, and
    Tk answers that its toplevel is still the main window. What actually
    distinguishes a window from a panel is who lays it out - the window manager
    or a parent - so that is what this walks up looking for.
    """
    try:
        focused = widget.focus_displayof()
    except (tk.TclError, KeyError):  # pragma: no cover - platform dependent
        focused = None
    node: "Optional[tk.Misc]" = focused if focused is not None else widget
    try:
        while node is not None:
            if node.winfo_manager() == "wm":
                return node
            parent = node.winfo_parent()
            if not parent:
                break
            node = node.nametowidget(parent)
    except (tk.TclError, KeyError):  # pragma: no cover - a window can die mid-walk
        pass
    return widget.winfo_toplevel()


def rect_of(window: tk.Misc) -> Rect:
    """Where a window is, in screen coordinates."""
    return Rect(
        window.winfo_rootx(),
        window.winfo_rooty(),
        window.winfo_width(),
        window.winfo_height(),
    )


def place_over(window: tk.Toplevel, parent: "Optional[tk.Misc]" = None) -> bool:
    """Centre ``window`` on ``parent`` and raise it. True if it was placed.

    Called after the dialog's widgets exist, so that the size being centred is
    the size it will actually be. A parent that has not been drawn yet has no
    rectangle worth centring on, and placement is then left alone rather than
    guessed at - a guess would land in the corner of the primary screen, which
    is the failure this avoids.
    """
    over = active_window(parent if parent is not None else window.master or window)
    try:
        window.update_idletasks()
        size = (window.winfo_reqwidth(), window.winfo_reqheight())
        position = place(size, rect_of(over))
        if position is None:
            return False
        window.geometry(geometry(size, position))
    except tk.TclError:  # pragma: no cover - platform dependent
        return False
    return True


def present(
    window: tk.Toplevel,
    parent: tk.Misc,
    *,
    modal: bool = False,
    on_close: "Optional[Callable[[], object]]" = None,
) -> None:
    """Place, raise and focus a dialog, and make Escape dismiss it.

    ``modal`` adds the grab that stops the rest of the application being used
    while the question stands. The focus is taken deliberately: the application
    is waiting on an answer, and a window that quietly appears behind another
    program is indistinguishable from one that has hung. Where the operating
    system will not hand focus to a background application it flashes the task
    bar instead, which is the signal we want in that case anyway.
    """
    over = active_window(parent)
    try:
        window.transient(over)  # type: ignore[call-overload]
    except tk.TclError:  # pragma: no cover - platform dependent
        window.transient(parent.winfo_toplevel())  # type: ignore[call-overload]
    place_over(window, over)
    if modal:
        try:
            window.grab_set()
        except tk.TclError:  # pragma: no cover - platform dependent
            pass
    close: Callable[[], object] = on_close if on_close is not None else window.destroy
    window.protocol("WM_DELETE_WINDOW", close)
    window.bind("<Escape>", lambda _e: close())
    try:
        window.lift()
        window.focus_force()
    except tk.TclError:  # pragma: no cover - a WM may refuse either
        pass
