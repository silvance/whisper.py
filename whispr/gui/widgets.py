"""Reusable presentation components shared by the screens.

The mechanics (scrollable page, mouse wheel, drag-and-drop, thread-safe text
helpers) plus the small set of components every screen is built from: a page
header that says what the page is for, cards instead of framed boxes, status
banners, empty states, and buttons with an actual hierarchy.

Presentation only - no screen logic lives here, and the visual values come from
:mod:`whispr.gui.theme` rather than being spelled out again per screen.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Callable, List, Optional

from .theme import (
    CARD_PADDING,
    PAGE_PADDING,
    SPACE_LG,
    SPACE_MD,
    SPACE_SM,
    SPACE_XS,
    Style,
    palette,
)


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
    canvas = tk.Canvas(parent, highlightthickness=0, background=palette().background)
    vsb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vsb.set)
    vsb.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    inner = ttk.Frame(canvas, padding=PAGE_PADDING, style=Style.PAGE)
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
            raw: tuple = tuple(root.tk.splitlist(data))
        except Exception:  # noqa: BLE001 - fall back to a naive split
            raw = tuple(str(data).split())
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


# -- Presentation components -------------------------------------------------


class PageHeader(ttk.Frame):
    """A page's title and the one sentence explaining what it is for.

    Every screen opens with this. An operator who has not used Whispers in a
    month should be able to read the top of a page and know what it does without
    inferring it from the controls.
    """

    def __init__(self, parent: tk.Misc, title: str, description: str = "") -> None:
        super().__init__(parent, style=Style.PAGE)
        ttk.Label(self, text=title, style=Style.PAGE_TITLE).pack(anchor="w")
        if description:
            ttk.Label(
                self,
                text=description,
                style=Style.PAGE_SUBTITLE,
                wraplength=760,
                justify="left",
            ).pack(anchor="w", pady=(SPACE_XS, 0))


class Card(ttk.Frame):
    """A titled panel. Put content in ``.body``.

    Separated from its neighbours by ground colour and space rather than by a
    drawn box, which is what stops a page of these reading as a stack of forms.
    """

    def __init__(
        self,
        parent: tk.Misc,
        title: str = "",
        description: str = "",
        *,
        padding: tuple = CARD_PADDING,
    ) -> None:
        super().__init__(parent, style=Style.CARD, padding=padding)
        if title:
            ttk.Label(self, text=title, style=Style.SECTION_TITLE).pack(anchor="w")
        if description:
            ttk.Label(
                self,
                text=description,
                style=Style.MUTED,
                wraplength=700,
                justify="left",
            ).pack(anchor="w", pady=(SPACE_XS, 0))
        self.body = ttk.Frame(self, style=Style.CARD_INNER)
        self.body.pack(
            fill="both",
            expand=True,
            pady=(SPACE_MD if (title or description) else 0, 0),
        )


class StatusBanner(ttk.Frame):
    """A single line of state - success, warning, error - that cannot be missed.

    Colour alone never carries the meaning: each kind has its own leading mark
    and its own words, so it still reads correctly in greyscale or to an
    operator who cannot distinguish the colours.
    """

    MARKS = {
        "success": "✓",
        "warning": "!",
        "error": "×",
        "info": "i",
        "busy": "•",
    }

    def __init__(
        self,
        parent: tk.Misc,
        *,
        wraplength: int = 640,
        before: Optional[tk.Misc] = None,
    ) -> None:
        super().__init__(parent, style=Style.CARD_INNER)
        self._wraplength = wraplength
        # Where to insert itself when shown. Without this it would append after
        # whatever is already packed - i.e. below the result it is reporting on.
        self._before = before
        self._mark = ttk.Label(self, text="", style=Style.MUTED)
        self._mark.pack(side="left", anchor="n", padx=(0, SPACE_SM))
        self._text = ttk.Label(
            self, text="", style=Style.MUTED, wraplength=wraplength, justify="left"
        )
        self._text.pack(side="left", fill="x", expand=True)
        self._visible = False

    def insert_before(self, widget: tk.Misc) -> None:
        """Appear immediately above ``widget`` when shown."""
        self._before = widget

    def show(self, kind: str, message: str) -> None:
        """Display ``message`` as a ``success``/``warning``/``error``/``info`` line."""
        style = {
            "success": Style.SUCCESS,
            "warning": Style.WARNING,
            "error": Style.DANGER,
            "info": Style.MUTED,
            "busy": Style.ACCENT,
        }.get(kind, Style.MUTED)
        self._mark.configure(text=self.MARKS.get(kind, ""), style=style)
        self._text.configure(text=message, style=style)
        if not self._visible:
            if self._before is not None and self._before.winfo_manager() == "pack":
                self.pack(fill="x", pady=(SPACE_SM, 0), before=self._before)
            else:
                self.pack(fill="x", pady=(SPACE_SM, 0))
            self._visible = True

    def hide(self) -> None:
        if self._visible:
            self.pack_forget()
            self._visible = False


class EmptyState(ttk.Frame):
    """What a screen shows before it has anything to show.

    Blank space makes a first-time operator wonder whether something is broken;
    a sentence saying what to do next answers the question the Help window would
    otherwise have to.
    """

    def __init__(
        self,
        parent: tk.Misc,
        title: str,
        description: str = "",
        *,
        mark: str = "",
    ) -> None:
        super().__init__(parent, style=Style.CARD_INNER, padding=(0, SPACE_LG))
        if mark:
            ttk.Label(self, text=mark, style=Style.MUTED).pack()
        ttk.Label(self, text=title, style=Style.BODY).pack(pady=(SPACE_XS, 0))
        if description:
            ttk.Label(
                self,
                text=description,
                style=Style.MUTED,
                wraplength=440,
                justify="center",
            ).pack(pady=(SPACE_XS, 0))


def primary_button(
    parent: tk.Misc, text: str, command: Callable[[], None]
) -> ttk.Button:
    """The one action a screen exists to perform."""
    return ttk.Button(parent, text=text, command=command, style=Style.PRIMARY)


def secondary_button(
    parent: tk.Misc, text: str, command: Callable[[], None]
) -> ttk.Button:
    """Everything else: browse, copy, export, reset."""
    return ttk.Button(parent, text=text, command=command, style=Style.SECONDARY)


def subtle_button(
    parent: tk.Misc, text: str, command: Callable[[], None]
) -> ttk.Button:
    """Low-emphasis, for header and toolbar actions."""
    return ttk.Button(parent, text=text, command=command, style=Style.SUBTLE)


def danger_button(
    parent: tk.Misc, text: str, command: Callable[[], None]
) -> ttk.Button:
    """Destructive. Deliberately does not look like an ordinary action."""
    return ttk.Button(parent, text=text, command=command, style=Style.DANGER_BUTTON)


class Disclosure(ttk.Frame):
    """ "Advanced options ▸" - present, discoverable, and out of the way.

    Progressive disclosure is the whole approach to the technical depth in this
    application: nobody should need to know what VAD or a clustering threshold
    is in order to transcribe a recording, and nobody who does know should have
    to go without them.
    """

    def __init__(
        self,
        parent: tk.Misc,
        title: str,
        *,
        expanded: bool = False,
        style_name: str = Style.LINK,
    ) -> None:
        super().__init__(parent, style=Style.PAGE)
        self.title = title
        self.expanded = expanded
        self._button = ttk.Button(self, command=self.toggle, style=style_name)
        self._button.pack(anchor="w")
        self.body = ttk.Frame(self, style=Style.PAGE)
        if expanded:
            self.body.pack(fill="both", expand=True, pady=(SPACE_SM, 0))
        self._refresh()

    def _refresh(self) -> None:
        arrow = "▾" if self.expanded else "▸"
        self._button.configure(text=f"{arrow}  {self.title}")

    def toggle(self) -> None:
        self.set_expanded(not self.expanded)

    def set_expanded(self, value: bool) -> None:
        if value == self.expanded:
            return
        self.expanded = value
        if value:
            self.body.pack(fill="both", expand=True, pady=(SPACE_SM, 0))
        else:
            self.body.forget()
        self._refresh()


class KeyValueRow(ttk.Frame):
    """A label and its value, aligned - for details rather than for input."""

    def __init__(
        self,
        parent: tk.Misc,
        label: str,
        value: str = "",
        *,
        width: int = 22,
        value_style: str = Style.BODY,
    ) -> None:
        super().__init__(parent, style=Style.CARD_INNER)
        ttk.Label(self, text=label, style=Style.FIELD_LABEL, width=width).pack(
            side="left", anchor="w"
        )
        self.value = ttk.Label(
            self, text=value, style=value_style, justify="left", wraplength=420
        )
        self.value.pack(side="left", anchor="w")

    def set(self, value: str, *, style_name: Optional[str] = None) -> None:
        self.value.configure(text=value)
        if style_name:
            self.value.configure(style=style_name)


class Badge(ttk.Label):
    """A short state word - Trusted, Needs review - carried by text, not colour."""

    KINDS = {
        "success": Style.SUCCESS,
        "warning": Style.WARNING,
        "danger": Style.DANGER,
        "muted": Style.META,
        "accent": Style.ACCENT,
    }

    def __init__(self, parent: tk.Misc, text: str = "", kind: str = "muted") -> None:
        super().__init__(parent, text=text, style=self.KINDS.get(kind, Style.META))

    def set(self, text: str, kind: str = "muted") -> None:
        self.configure(text=text, style=self.KINDS.get(kind, Style.META))


def style_text_widget(widget: tk.Text) -> tk.Text:
    """Apply the theme to a raw text widget, which ttk styles cannot reach."""
    from .theme import text_widget_options

    colors = palette()
    try:
        widget.configure(**text_widget_options())
    except tk.TclError:  # pragma: no cover - a widget that rejects an option
        pass
    # A ScrolledText is a Text inside a Frame with a classic Scrollbar; both
    # keep the platform default otherwise, which is a pale box around a dark
    # transcript.
    master = widget.master
    if isinstance(master, tk.Frame):
        try:
            master.configure(background=colors.surface_alt, highlightthickness=0)
        except tk.TclError:  # pragma: no cover
            pass
    bar = getattr(widget, "vbar", None)
    if bar is not None:
        try:
            bar.configure(
                background=colors.border,
                troughcolor=colors.surface_alt,
                activebackground=colors.text_faint,
                highlightthickness=0,
                borderwidth=0,
                width=12,
            )
        except tk.TclError:  # pragma: no cover
            pass
    return widget


def divider(parent: tk.Misc) -> ttk.Separator:
    return ttk.Separator(parent, orient="horizontal", style=Style.SEPARATOR)


class FileDropZone(ttk.Frame):
    """Where a recording comes into the application.

    A path entry beside a Browse button is not wrong, but it puts a filesystem
    string at the centre of the screen and gives no hint that a file can simply
    be dropped. This shows the invitation when empty and the chosen file - name
    and size, the path kept as quiet metadata - once there is one.
    """

    def __init__(
        self,
        parent: tk.Misc,
        variable: tk.StringVar,
        on_choose: Callable[[], None],
        *,
        prompt: str = "Drop an audio or video file here",
        button_text: str = "Choose file",
        change_text: str = "Change file",
    ) -> None:
        super().__init__(parent, style=Style.CARD_INNER)
        self._variable = variable
        self._on_choose = on_choose

        # An outlined region reads as somewhere to drop something; a bare label
        # in the middle of a card does not. tk.Frame because ttk styles have no
        # way to draw this border.
        colors = palette()
        self.empty = tk.Frame(
            self,
            background=colors.surface,
            highlightthickness=1,
            highlightbackground=colors.border,
            highlightcolor=colors.border,
            padx=SPACE_MD,
            pady=SPACE_MD,
        )
        inner = ttk.Frame(self.empty, style=Style.CARD_INNER)
        inner.pack()
        ttk.Label(inner, text=prompt, style=Style.BODY).pack()
        ttk.Label(inner, text="or", style=Style.META).pack(pady=(SPACE_XS, SPACE_XS))
        secondary_button(inner, button_text, on_choose).pack()

        self.chosen = ttk.Frame(self, style=Style.CARD_INNER)
        details = ttk.Frame(self.chosen, style=Style.CARD_INNER)
        details.pack(side="left", fill="x", expand=True)
        self._name = ttk.Label(details, text="", style=Style.BODY)
        self._name.pack(anchor="w")
        self._meta = ttk.Label(details, text="", style=Style.META)
        self._meta.pack(anchor="w", pady=(SPACE_XS, 0))
        secondary_button(self.chosen, change_text, on_choose).pack(side="right")

        variable.trace_add("write", lambda *_a: self.refresh())
        self.refresh()

    def refresh(self) -> None:
        raw = self._variable.get().strip()
        if not raw:
            self.chosen.pack_forget()
            self.empty.pack(fill="x")
            return
        self.empty.pack_forget()
        path = Path(raw)
        self._name.configure(text=path.name or raw)
        self._meta.configure(text=self._describe(path))
        self.chosen.pack(fill="x")

    @staticmethod
    def _describe(path: Path) -> str:
        """Size and location - useful, but not the headline."""
        try:
            size = float(path.stat().st_size)
        except OSError:
            # A path that is not there yet is still worth showing; its folder is
            # the only useful thing left to say about it.
            return str(path.parent) if path.parent != Path(".") else ""
        shown = f"{size:.0f} bytes"
        for unit in ("KB", "MB", "GB"):
            size /= 1024.0
            if size < 1024 or unit == "GB":
                shown = f"{size:.1f} {unit}"
                break
        return f"{shown}  ·  {path.parent}"
