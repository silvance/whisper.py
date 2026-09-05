"""Send a corrected speaker from a transcript to a known subject's profile.

Correcting the speaker tags in a transcript is the point at which an operator
knows, better than any model does, which stretches of audio belong to whom.
This is the dialog that lets that work be kept: pick the speaker, pick (or
create) the subject, and the audio behind those lines is enrolled as *pending*
samples on that subject's profile.

Pending, always: a correction can be wrong, and a wrong sample in a known
actor's reference material is the failure mode this whole subsystem exists to
avoid. An analyst approves the samples in Speaker Profiles before they count
towards anything.

Dialog only - the enrolment itself runs on the caller's worker thread.
"""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk
from typing import Dict, List, Optional, Tuple

from ..speaker_profiles import SpeakerProfile, list_speaker_profiles
from .theme import SPACE_LG, SPACE_MD, SPACE_SM, SPACE_XL, SPACE_XS, Style, palette
from .widgets import primary_button, secondary_button


@dataclass
class SaveSpeakerChoice:
    """What the operator asked for: this speaker, onto that subject."""

    speaker: str
    profile: SpeakerProfile
    is_new: bool


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} sec"
    return f"{seconds / 60:.1f} min"


def ask_save_speaker(
    root: tk.Misc,
    speakers: List[Tuple[str, float]],
    *,
    recording: str = "",
) -> Optional[SaveSpeakerChoice]:
    """Ask which speaker to save and which subject to save them to.

    ``speakers`` is ``(display name, seconds of speech)``, longest first.
    Returns ``None`` if the operator cancelled.
    """
    subjects = list_speaker_profiles()
    by_label: Dict[str, SpeakerProfile] = {}
    for subject in subjects:
        label = subject.display_name
        if label in by_label:  # two subjects can share a display name
            label = f"{label} ({subject.subject_id})"
        by_label[label] = subject

    win = tk.Toplevel(root)
    win.title("Save speaker to profile")
    win.transient(root)  # type: ignore[call-overload]
    win.configure(background=palette().surface)
    win.resizable(False, False)

    frame = ttk.Frame(win, padding=SPACE_XL, style=Style.CARD)
    frame.pack(fill="both", expand=True)

    ttk.Label(frame, text="Save speaker to profile", style=Style.SECTION_TITLE).pack(
        anchor="w"
    )
    ttk.Label(
        frame,
        text=(
            "Adds the audio behind this speaker's lines"
            + (f" in {recording}" if recording else "")
            + " to a subject's reference profile."
        ),
        style=Style.MUTED,
        wraplength=460,
        justify="left",
    ).pack(anchor="w", pady=(SPACE_XS, SPACE_LG))

    # -- Which speaker ----------------------------------------------------
    speaker_labels = [
        f"{name} — {_duration(secs)} of speech" for name, secs in speakers
    ]
    speaker_var = tk.StringVar(value=speaker_labels[0])
    ttk.Label(frame, text="Speaker", style=Style.FIELD_LABEL).pack(anchor="w")
    ttk.Combobox(
        frame,
        textvariable=speaker_var,
        values=speaker_labels,
        state="readonly",
        width=44,
    ).pack(anchor="w", pady=(SPACE_XS, SPACE_LG))

    # -- Which subject ----------------------------------------------------
    ttk.Label(frame, text="Save to", style=Style.FIELD_LABEL).pack(anchor="w")
    target_var = tk.StringVar(value="existing" if by_label else "new")
    existing_var = tk.StringVar(value=next(iter(by_label), ""))
    new_var = tk.StringVar(value="")

    existing_row = ttk.Frame(frame, style=Style.CARD_INNER)
    ttk.Radiobutton(
        existing_row,
        text="An existing subject",
        value="existing",
        variable=target_var,
        style=Style.RADIO,
        state="normal" if by_label else "disabled",
    ).pack(side="left")
    existing_combo = ttk.Combobox(
        existing_row,
        textvariable=existing_var,
        values=list(by_label),
        state="readonly" if by_label else "disabled",
        width=28,
    )
    existing_combo.pack(side="left", padx=(SPACE_MD, 0))
    existing_row.pack(anchor="w", pady=(SPACE_XS, 0))

    new_row = ttk.Frame(frame, style=Style.CARD_INNER)
    ttk.Radiobutton(
        new_row,
        text="A new subject",
        value="new",
        variable=target_var,
        style=Style.RADIO,
    ).pack(side="left")
    new_entry = ttk.Entry(new_row, textvariable=new_var, width=30)
    new_entry.pack(side="left", padx=(SPACE_MD, 0))
    new_row.pack(anchor="w", pady=(SPACE_XS, SPACE_MD))

    ttk.Label(
        frame,
        text=(
            "Samples added this way are marked pending review. Approve them on "
            "the Speaker Profiles page before they are used as reference "
            "material — a corrected label can still be wrong, and a wrong "
            "sample would quietly distort every later comparison."
        ),
        style=Style.META,
        wraplength=460,
        justify="left",
    ).pack(anchor="w", pady=(0, SPACE_XS))

    error_var = tk.StringVar(value="")
    ttk.Label(
        frame,
        textvariable=error_var,
        style=Style.WARNING,
        wraplength=460,
        justify="left",
    ).pack(anchor="w", pady=(0, SPACE_MD))

    chosen: Dict[str, Optional[SaveSpeakerChoice]] = {"value": None}

    def _confirm() -> None:
        speaker = speakers[speaker_labels.index(speaker_var.get())][0]
        if target_var.get() == "existing":
            subject = by_label.get(existing_var.get())
            if subject is None:
                error_var.set("Choose a subject, or create a new one.")
                return
            chosen["value"] = SaveSpeakerChoice(speaker, subject, is_new=False)
        else:
            name = new_var.get().strip()
            if not name:
                error_var.set("Give the new subject a name.")
                new_entry.focus_set()
                return
            # Typing the name of a subject that already exists means that
            # subject, not a second one wearing the same name.
            match = [
                s
                for s in subjects
                if s.display_name.strip().casefold() == name.casefold()
            ]
            if len(match) == 1:
                chosen["value"] = SaveSpeakerChoice(speaker, match[0], is_new=False)
            elif match:
                error_var.set(
                    f"More than one subject is already called '{name}'. Pick the "
                    "one you mean from the list above."
                )
                return
            else:
                chosen["value"] = SaveSpeakerChoice(
                    speaker, SpeakerProfile(display_name=name), is_new=True
                )
        win.destroy()

    buttons = ttk.Frame(frame, style=Style.CARD_INNER)
    buttons.pack(fill="x", pady=(SPACE_SM, 0))
    secondary_button(buttons, "Cancel", win.destroy).pack(side="right")
    add = primary_button(buttons, "Add samples", _confirm)
    add.pack(side="right", padx=(0, SPACE_SM))

    win.bind("<Escape>", lambda _e: win.destroy())
    win.bind("<Return>", lambda _e: _confirm())
    add.focus_set()
    win.update_idletasks()
    # Centred on the window it came from, rather than wherever the window
    # manager would have dropped it.
    try:
        x = root.winfo_rootx() + (root.winfo_width() - win.winfo_width()) // 2
        y = root.winfo_rooty() + (root.winfo_height() - win.winfo_height()) // 3
        win.geometry(f"+{max(0, x)}+{max(0, y)}")
    except tk.TclError:  # no geometry yet; leave it to the window manager
        pass
    win.grab_set()
    win.wait_window()
    return chosen["value"]
