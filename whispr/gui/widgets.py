"""Reusable Tkinter widgets and helpers shared by the GUI tabs.

Pulled out of the former monolithic ``whispr.app`` so the tab modules can share
the collapsible section, the scrollable page, mouse-wheel binding, file
drag-and-drop wiring, and the thread-safe text helpers.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Callable, List


class CollapsibleSection(ttk.Frame):
    """A titled section whose body can be collapsed to a single header row.

    Clicking the header toggles the body. Collapsing the settings frees vertical
    space for the transcript and lets the window be resized down without clipping
    controls. Put child widgets in ``.body``.
    """

    def __init__(
        self,
        parent: tk.Misc,
        title: str,
        *,
        expanded: bool = True,
        body_padding: tuple = (10, 6, 10, 10),
    ) -> None:
        super().__init__(parent)
        self.title = title
        self.expanded = expanded
        self.header = ttk.Button(self, command=self.toggle)
        self.header.pack(fill="x")
        self.body = ttk.Frame(self, padding=body_padding)
        if expanded:
            self.body.pack(fill="both", expand=True)
        self._refresh_header()

    def _refresh_header(self) -> None:
        arrow = "▼" if self.expanded else "▶"
        self.header.configure(text=f"{arrow}  {self.title}")

    def toggle(self) -> None:
        self.set_expanded(not self.expanded)

    def set_expanded(self, value: bool) -> None:
        if value == self.expanded:
            return
        self.expanded = value
        if value:
            self.body.pack(fill="both", expand=True)
        else:
            self.body.forget()
        self._refresh_header()


def scrollable_body(parent: tk.Misc) -> "tuple[tk.Canvas, ttk.Frame]":
    """Wrap a scrollable region in ``parent`` and return ``(canvas, inner)``.

    Settings sections can stack taller than the window (especially on small
    screens), which previously pushed the Run button and transcript off the
    bottom with no way to reach them. This puts everything in a vertically
    scrollable canvas: the scrollbar always works, and the mouse wheel is wired
    up by :func:`bind_wheel`.
    """
    canvas = tk.Canvas(parent, highlightthickness=0)
    vsb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vsb.set)
    vsb.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    inner = ttk.Frame(canvas, padding=12)
    window = canvas.create_window((0, 0), window=inner, anchor="nw")

    def _sync() -> None:
        # Match the inner frame's width to the canvas, and let it stretch to fill
        # the viewport when the content is shorter than it (so widgets that expand
        # look right) while still allowing it to overflow + scroll.
        canvas.itemconfigure(window, width=canvas.winfo_width())
        canvas.itemconfigure(
            window, height=max(inner.winfo_reqheight(), canvas.winfo_height())
        )
        canvas.configure(scrollregion=canvas.bbox("all"))

    # Re-sync both when the viewport resizes and when the content grows or shrinks
    # (e.g. as settings sections are expanded/collapsed).
    canvas.bind("<Configure>", lambda _e: _sync())
    inner.bind("<Configure>", lambda _e: _sync())
    return canvas, inner


def bind_wheel(canvas: tk.Canvas, root_widget: tk.Misc) -> None:
    """Make the mouse wheel scroll ``canvas`` while over its content.

    Bound recursively to every widget except ``tk.Text`` (and its ``ScrolledText``
    subclass), which keep their own native scrolling so the transcript/status
    panes don't fight the page scroll.
    """

    def _on_wheel(event: "tk.Event[tk.Misc]") -> None:
        if getattr(event, "num", None) == 4:
            canvas.yview_scroll(-1, "units")
        elif getattr(event, "num", None) == 5:
            canvas.yview_scroll(1, "units")
        else:
            canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

    def _bind(widget: tk.Misc) -> None:
        if not isinstance(widget, tk.Text):
            widget.bind("<MouseWheel>", _on_wheel, add="+")  # Windows / macOS
            widget.bind("<Button-4>", _on_wheel, add="+")  # Linux scroll up
            widget.bind("<Button-5>", _on_wheel, add="+")  # Linux scroll down
        for child in widget.winfo_children():
            _bind(child)

    _bind(canvas)
    _bind(root_widget)


def register_drop(
    root: tk.Misc,
    enabled: bool,
    widget: tk.Misc,
    handler: Callable[[List[Path]], None],
) -> None:
    """Register ``widget`` as a file-drop target calling ``handler(paths)``.

    No-op when ``enabled`` is False (tkdnd not loaded). Uses tkinterdnd2's wrapper
    methods directly on the widget (the root is a themed ttkbootstrap window, not a
    ``TkinterDnD.Tk``), and parses the platform-specific drop payload into clean
    ``Path`` objects.
    """
    if not enabled:
        return
    try:
        from tkinterdnd2 import DND_FILES, TkinterDnD
    except Exception:  # noqa: BLE001 - convenience feature only
        return

    def _on_drop(event: object) -> None:
        data = getattr(event, "data", "")
        try:
            raw = root.tk.splitlist(data)
        except Exception:  # noqa: BLE001 - fall back to a naive split
            raw = str(data).split()
        paths = [Path(item) for item in raw if item]
        if paths:
            handler(paths)

    try:
        TkinterDnD.DnDWrapper.drop_target_register(widget, DND_FILES)
        TkinterDnD.DnDWrapper.dnd_bind(widget, "<<Drop>>", _on_drop)
    except Exception:  # noqa: BLE001 - never let DnD wiring break the UI
        pass


# -- Thread-safe text-widget helpers (schedule onto the Tk main loop) ----------


def append_line(widget: tk.Text, text: str) -> None:
    """Append ``text`` + newline to a read-only text widget."""

    def _do() -> None:
        widget.configure(state="normal")
        widget.insert("end", str(text) + "\n")
        widget.see("end")
        widget.configure(state="disabled")

    widget.after(0, _do)


def clear_text(widget: tk.Text) -> None:
    """Clear a read-only text widget."""

    def _do() -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.configure(state="disabled")

    widget.after(0, _do)


def set_readonly_text(widget: tk.Text, text: str) -> None:
    """Replace the contents of a read-only text widget."""

    def _do() -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", text)
        widget.configure(state="disabled")

    widget.after(0, _do)


def set_text(widget: tk.Text, text: str) -> None:
    """Replace the contents of an editable text widget."""

    def _do() -> None:
        widget.delete("1.0", "end")
        widget.insert("end", text)

    widget.after(0, _do)
