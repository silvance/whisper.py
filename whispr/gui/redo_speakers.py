"""Split an existing transcript by speaker again, with a different count.

Transcription is the expensive half: ninety minutes of poor phone audio is the
better part of an hour of a laptop's life. Working out who spoke when is a
small fraction of that, on audio the model has already been handed.

So when the speaker count comes back wrong - two people reported as five - the
remedy should not be to transcribe the recording a second time. The words were
never in doubt. This dialog asks for the count to try instead, and says plainly
what will be lost: the splitting is redone from the words as transcribed, so
any speaker tags corrected by hand go with it.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Dict, Optional

from .. import speaker_count
from .theme import SPACE_LG, SPACE_MD, SPACE_SM, SPACE_XL, SPACE_XS, Style, palette
from .widgets import primary_button, secondary_button


def ask_redo_speakers(
    root: tk.Misc,
    *,
    current: str,
    found: int,
    corrections: int = 0,
    recording: str = "",
) -> "Optional[str]":
    """Ask which count to split by instead. Returns a stored count, or None.

    ``current`` is the answer the last run used, ``found`` how many speakers it
    came back with, and ``corrections`` how many speaker tags have been changed
    by hand since - the thing the operator is about to spend. The return is the
    stored form ("" for "work it out"), or None if they thought better of it.
    """
    win = tk.Toplevel(root)
    win.title("Redo speaker separation")
    win.transient(root)  # type: ignore[call-overload]
    win.configure(background=palette().surface)
    win.resizable(False, False)

    frame = ttk.Frame(win, padding=SPACE_XL, style=Style.CARD)
    frame.pack(fill="both", expand=True)

    ttk.Label(frame, text="Redo speaker separation", style=Style.SECTION_TITLE).pack(
        anchor="w"
    )
    ttk.Label(
        frame,
        text=(
            "Splits the transcript by speaker again"
            + (f" for {recording}" if recording else "")
            + ". The recording is not transcribed again, so this takes a "
            "fraction of the time the first run did."
        ),
        style=Style.MUTED,
        wraplength=460,
        justify="left",
    ).pack(anchor="w", pady=(SPACE_XS, SPACE_LG))

    people = "speaker" if found == 1 else "speakers"
    ttk.Label(
        frame,
        text=f"Last run: {current} — came back with {found} {people}.",
        style=Style.META,
        wraplength=460,
        justify="left",
    ).pack(anchor="w", pady=(0, SPACE_MD))

    ttk.Label(frame, text="Try instead", style=Style.FIELD_LABEL).pack(anchor="w")
    # The prompt entry is not offered here: the operator has already answered
    # the question once, and this dialog exists to change that answer.
    options = [c for c in speaker_count.choices() if c != speaker_count.UNSET]
    choice_var = tk.StringVar(value=current if current in options else options[0])
    combo = ttk.Combobox(
        frame,
        textvariable=choice_var,
        values=options,
        state="readonly",
        width=30,
    )
    combo.pack(anchor="w", pady=(SPACE_XS, SPACE_MD))

    # Said plainly, because it is the one thing this costs: the new split
    # numbers its speakers from scratch, so "Speaker 1" afterwards need not be
    # the person "Speaker 1" was before, and no name is carried across.
    warning = (
        "Speakers are numbered again from scratch, so any names have to be put "
        "back afterwards. The words stay exactly as transcribed."
    )
    if corrections:
        tags = "tag" if corrections == 1 else "tags"
        warning = (
            f"The {corrections} speaker {tags} you corrected by hand will be "
            "replaced, and speakers are numbered again from scratch, so any "
            "names have to be put back afterwards. The words stay exactly as "
            "transcribed."
        )
    ttk.Label(
        frame,
        text=warning,
        style=Style.WARNING if corrections else Style.META,
        wraplength=460,
        justify="left",
    ).pack(anchor="w", pady=(0, SPACE_MD))

    chosen: Dict[str, Optional[str]] = {"value": None}

    def _confirm() -> None:
        chosen["value"] = speaker_count.to_setting(choice_var.get())
        win.destroy()

    buttons = ttk.Frame(frame, style=Style.CARD_INNER)
    buttons.pack(fill="x", pady=(SPACE_SM, 0))
    secondary_button(buttons, "Cancel", win.destroy).pack(side="right")
    go = primary_button(buttons, "Redo separation", _confirm)
    go.pack(side="right", padx=(0, SPACE_SM))

    win.bind("<Escape>", lambda _e: win.destroy())
    win.bind("<Return>", lambda _e: _confirm())
    combo.focus_set()
    win.update_idletasks()
    try:
        x = root.winfo_rootx() + (root.winfo_width() - win.winfo_width()) // 2
        y = root.winfo_rooty() + (root.winfo_height() - win.winfo_height()) // 3
        win.geometry(f"+{max(0, x)}+{max(0, y)}")
    except tk.TclError:  # no geometry yet; leave it to the window manager
        pass
    win.grab_set()
    win.wait_window()
    return chosen["value"]
