"""Speaker-comparison UI: export one speaker, and compare two voiceprints.

Extracted from :mod:`whispr.gui.transcribe_tab` (which was growing large) since
this tooling is self-contained: it needs only the window, the active profile, and
a status callback. The comparison maths lives in :mod:`whispr.voiceprints`; this
module is just the Tk dialogs around it.
"""

from __future__ import annotations

import tkinter as tk
from functools import partial
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Callable, Dict, List, Optional

from ..profiles import (
    PROFILE_SUFFIX,
    VOICEPRINT_SUFFIX,
    Profile,
    export_voiceprint,
    read_voiceprints,
)
from ..voiceprints import Voiceprint, compare_voiceprints, similarity_band
from .errors import friendly_error

# Reports a one-line status/result back to the caller (e.g. a status label).
StatusFn = Callable[[str], None]


def choose_speaker(
    root: tk.Misc, names: List[str], *, title: str = "Choose speaker"
) -> Optional[str]:
    """Modal picker for one speaker name from ``names`` (None if cancelled)."""
    if len(names) == 1:
        return names[0]
    win = tk.Toplevel(root)
    win.title(title)
    win.transient(root)  # type: ignore[call-overload]
    win.grab_set()
    var = tk.StringVar(value=names[0])
    chosen: Dict[str, Optional[str]] = {"name": None}
    ttk.Label(win, text="Speaker:").pack(side="left", padx=(12, 6), pady=12)
    ttk.Combobox(win, textvariable=var, values=names, state="readonly", width=24).pack(
        side="left", pady=12
    )

    def _ok() -> None:
        chosen["name"] = var.get()
        win.destroy()

    ttk.Button(win, text="Cancel", command=win.destroy).pack(
        side="right", padx=(0, 12), pady=12
    )
    ttk.Button(win, text="OK", command=_ok).pack(side="right", padx=6, pady=12)
    win.wait_window()
    return chosen["name"]


def export_speaker(
    root: tk.Misc, profile: Optional[Profile], on_status: StatusFn
) -> None:
    """Export one speaker's voiceprint for comparison in another profile."""
    if profile is None or not profile.voiceprints:
        on_status("Pick a profile with at least one learned voice first.")
        return
    name = choose_speaker(
        root, sorted(profile.voiceprints), title="Export which speaker?"
    )
    if not name:
        return
    path = filedialog.asksaveasfilename(
        title="Export speaker voiceprint",
        defaultextension=VOICEPRINT_SUFFIX,
        initialfile=f"{name}{VOICEPRINT_SUFFIX}",
        filetypes=[
            ("Whispers voiceprint", f"*{VOICEPRINT_SUFFIX}"),
            ("All files", "*.*"),
        ],
    )
    if not path:
        return
    try:
        export_voiceprint(profile.voiceprints[name], path, source_profile=profile.name)
        on_status(f"Exported {name}'s voiceprint to {Path(path).name}")
    except Exception as exc:  # noqa: BLE001 - surfaced to the user
        on_status(friendly_error(exc))


def load_voiceprint_from_file(
    root: tk.Misc, on_status: StatusFn
) -> Optional[Voiceprint]:
    """Browse for a voiceprint (or profile) file and return one speaker's print."""
    path = filedialog.askopenfilename(
        title="Load a speaker voiceprint (or profile)",
        filetypes=[
            (
                "Whispers voiceprint / profile",
                f"*{VOICEPRINT_SUFFIX} *{PROFILE_SUFFIX}",
            ),
            ("All files", "*.*"),
        ],
    )
    if not path:
        return None
    try:
        voiceprints = read_voiceprints(path)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user
        on_status(friendly_error(exc))
        return None
    name = choose_speaker(
        root, sorted(voiceprints), title="Which speaker from this file?"
    )
    return voiceprints.get(name) if name else None


def open_compare_dialog(root: tk.Misc) -> None:
    """Open a dialog to compare two voiceprints and score their similarity."""
    win = tk.Toplevel(root)
    win.title("Compare voices")
    win.transient(root)  # type: ignore[call-overload]
    frame = ttk.Frame(win, padding=12)
    frame.pack(fill="both", expand=True)
    frame.columnconfigure(1, weight=1)

    ttk.Label(
        frame,
        text="Compare two speaker voiceprints for similarity.",
        font=("", 11, "bold"),
    ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

    slots: Dict[str, Optional[Voiceprint]] = {"a": None, "b": None}
    name_vars = {
        "a": tk.StringVar(value="(none loaded)"),
        "b": tk.StringVar(value="(none loaded)"),
    }
    result_var = tk.StringVar(value="")

    def _load(slot: str) -> None:
        voiceprint = load_voiceprint_from_file(root, result_var.set)
        if voiceprint is not None:
            slots[slot] = voiceprint
            name_vars[slot].set(voiceprint.name)
            result_var.set("")

    for i, (slot, label) in enumerate((("a", "Speaker A"), ("b", "Speaker B")), 1):
        ttk.Label(frame, text=label).grid(
            row=i, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Label(frame, textvariable=name_vars[slot]).grid(
            row=i, column=1, sticky="w", pady=4
        )
        ttk.Button(frame, text="Load…", command=partial(_load, slot)).grid(
            row=i, column=2, sticky="e", pady=4
        )

    def _do_compare() -> None:
        a, b = slots["a"], slots["b"]
        if a is None or b is None:
            result_var.set("Load a voiceprint into both A and B first.")
            return
        score = compare_voiceprints(a, b)
        band, blurb = similarity_band(score)
        pct = max(0.0, min(1.0, score)) * 100.0
        result_var.set(
            f"Voice similarity: {pct:.0f}%  —  {band}\n{blurb.capitalize()}."
        )

    ttk.Button(frame, text="Compare", command=_do_compare).grid(
        row=3, column=0, sticky="w", pady=(8, 4)
    )
    ttk.Label(frame, textvariable=result_var, font=("", 10), justify="left").grid(
        row=4, column=0, columnspan=3, sticky="w", pady=(0, 8)
    )

    ttk.Label(
        frame,
        text=(
            "Investigative aid only — not forensic voice identification. The "
            "score is a similarity indicator that shifts with recording quality "
            "and how much speech each voiceprint was built from; treat it as a "
            "lead to verify, never as proof of identity. Both voiceprints must "
            "come from builds using the same speaker-embedding model."
        ),
        wraplength=420,
        justify="left",
        font=("", 8),
    ).grid(row=5, column=0, columnspan=3, sticky="w")
    ttk.Button(frame, text="Close", command=win.destroy).grid(
        row=6, column=2, sticky="e", pady=(10, 0)
    )
