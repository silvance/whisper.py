"""Give a panel its own window, and take it back again.

A transcript is a document. Reading one in a box at the foot of a page of
settings is the wrong shape for the work: an analyst wants it filling a screen
- often the second screen, next to the case file or the map - while the
settings that produced it stay where they are.

Tk can do this without rebuilding the panel or keeping two copies of it in
step. ``wm manage`` takes a frame out of its parent's layout and hands it to
the window manager as a real, resizable, maximisable window; ``wm forget``
gives it back. The widgets inside never move, so state, scroll position,
selection and bindings all survive the trip in both directions.

Those two calls have no tkinter wrapper for an arbitrary frame, hence this
module: the Tcl underneath, plus the bookkeeping that makes a popped panel
behave like a window somebody can close.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

from ..panel_window import DOCK_LABEL, POPOUT_LABEL, fit_geometry
from .dialogs import rect_of


class PopoutError(RuntimeError):
    """The window manager would not take the panel, or would not give it back."""


class Popout:
    """One panel that can live in the page or in a window of its own.

    ``restore`` is called to put the panel back into its parent's layout; it
    has to be supplied because only the caller knows how the panel was laid
    out in the first place.
    """

    def __init__(
        self,
        widget: tk.Misc,
        *,
        title: str,
        restore: Callable[[], None],
        on_change: "Optional[Callable[[bool], None]]" = None,
        min_size: str = "720x480",
        size: str = "1100x780",
    ) -> None:
        self._widget = widget
        self._root = widget.winfo_toplevel()
        self._title = title
        self._restore = restore
        self._on_change = on_change
        self._min_size = min_size
        self._size = size
        self._out = False

    @property
    def is_out(self) -> bool:
        """True while the panel has a window of its own."""
        return self._out

    def toggle(self) -> None:
        self.dock() if self._out else self.pop()

    def pop(self) -> None:
        """Hand the panel to the window manager as a window of its own.

        Two calls have to succeed together or not at all: ``wm manage``, which
        takes the panel out of the page, and the close protocol, without which
        the window's X would *destroy* the panel and take the transcript with
        it. If the second fails the first is undone, so the promise made to the
        operator - "it stays on the page" - is one this can keep.

        Everything after those two is cosmetic. A window manager that refuses a
        title or a size still gave us a window, and refusing to use it over
        that would be the wrong trade.
        """
        if self._out:
            self.focus()
            return
        tk_ = self._root.tk
        path = str(self._widget)
        try:
            tk_.call("wm", "manage", path)
        except tk.TclError as exc:  # pragma: no cover - platform dependent
            raise PopoutError(str(exc)) from exc
        try:
            tk_.call(
                "wm",
                "protocol",
                path,
                "WM_DELETE_WINDOW",
                self._root.register(self.dock),
            )
        except tk.TclError as exc:  # pragma: no cover - platform dependent
            self._roll_back(path)
            raise PopoutError(str(exc)) from exc
        for call in (
            ("wm", "title", path, self._title),
            ("wm", "minsize", path, *(int(n) for n in self._min_size.split("x"))),
            ("wm", "geometry", path, self._geometry()),
        ):
            try:
                tk_.call(*call)
            except tk.TclError:  # pragma: no cover - cosmetic; the window stands
                pass
        self._out = True
        self._changed()

    def _roll_back(self, path: str) -> None:
        """Undo a half-finished pop, so the panel is somewhere rather than nowhere."""
        try:
            self._root.tk.call("wm", "forget", path)
            self._restore()
        except tk.TclError:  # pragma: no cover - nothing further we can do
            pass

    def dock(self) -> None:
        """Put the panel back where it came from."""
        if not self._out:
            return
        try:
            self._root.tk.call("wm", "forget", str(self._widget))
        except tk.TclError as exc:  # pragma: no cover - platform dependent
            raise PopoutError(str(exc)) from exc
        # ``wm forget`` leaves the widget unmapped and with no geometry manager:
        # without this it would simply vanish from both places.
        self._restore()
        self._out = False
        self._changed()

    def focus(self) -> None:
        """Bring the panel's window to the front, if it has one."""
        if not self._out:
            return
        path = str(self._widget)
        try:
            self._root.tk.call("raise", path)
            self._root.tk.call("focus", "-force", path)
        except tk.TclError:  # pragma: no cover - a WM may refuse either
            pass

    def _geometry(self) -> str:
        try:
            return fit_geometry(
                self._size,
                self._root.winfo_screenwidth(),
                self._root.winfo_screenheight(),
                rect_of(self._root),
            )
        except tk.TclError:  # pragma: no cover - no display
            return self._size

    def _changed(self) -> None:
        if self._on_change is not None:
            self._on_change(self._out)


class PanelWindow:
    """A titled card that can sit in the page or stand as its own window.

    Bundles the three pieces that always go together: the classic frame Tk
    insists on for ``wm manage``, the card the caller fills, and the stand-in
    left in the page so the way back is where the panel used to be rather than
    somewhere in the taskbar.
    """

    def __init__(
        self,
        parent: tk.Misc,
        *,
        title: str,
        window_title: str,
        note: str,
        pack_options: "Optional[dict]" = None,
        on_error: "Optional[Callable[[str], None]]" = None,
    ) -> None:
        from .theme import SPACE_MD, SPACE_SM, Style, palette
        from .widgets import Card, secondary_button, subtle_button

        self._pack_options = dict(pack_options or {"fill": "both", "expand": True})
        self._on_error = on_error
        self._buttons: "list[ttk.Button]" = []

        self.host = tk.Frame(
            parent, background=palette().background, highlightthickness=0
        )
        self.host.pack(**self._pack_options)
        self.card = Card(self.host, title)
        self.card.pack(fill="both", expand=True)

        self._placeholder = Card(parent, title)
        ttk.Label(
            self._placeholder.body,
            text=note,
            style=Style.MUTED,
            wraplength=560,
            justify="left",
        ).pack(anchor="w")
        row = ttk.Frame(self._placeholder.body, style=Style.CARD_INNER)
        row.pack(fill="x", pady=(SPACE_MD, 0))
        secondary_button(row, "Bring it back here", self.dock).pack(side="left")
        subtle_button(row, "Show me that window", self.focus).pack(
            side="left", padx=(SPACE_SM, 0)
        )

        self._popout = Popout(
            self.host,
            title=window_title,
            restore=self._restore,
            on_change=self._changed,
        )

    def toggle_button(self, parent: tk.Misc) -> "ttk.Button":
        """A button that sends the panel out and brings it back.

        Put it inside :attr:`card`: popped out, it travels with the panel, so
        the way back is on the window the operator is actually looking at.
        """
        from .widgets import subtle_button

        button = subtle_button(parent, POPOUT_LABEL, self.toggle)
        self._buttons.append(button)
        return button

    @property
    def is_out(self) -> bool:
        return self._popout.is_out

    def toggle(self) -> None:
        try:
            self._popout.toggle()
        except PopoutError as exc:
            # Tk can refuse on an unusual window manager. The panel is still on
            # the page, which is the part that matters.
            if self._on_error is not None:
                self._on_error(str(exc))

    def dock(self) -> None:
        if self._popout.is_out:
            self.toggle()

    def focus(self) -> None:
        self._popout.focus()

    def _restore(self) -> None:
        self._placeholder.pack_forget()
        self.host.pack(**self._pack_options)

    def _changed(self, out: bool) -> None:
        for button in self._buttons:
            button.configure(text=DOCK_LABEL if out else POPOUT_LABEL)
        if out:
            self._placeholder.pack(fill="x", **self._placeholder_pady())

    def _placeholder_pady(self) -> dict:
        pady = self._pack_options.get("pady")
        return {"pady": pady} if pady is not None else {}
