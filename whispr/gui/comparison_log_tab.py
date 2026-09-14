"""The history page: every speaker comparison this copy has run.

Master and detail, like Speaker Profiles: the comparisons on the left, filtered
by subject, and everything behind the selected one on the right - which
recording, which speech in it, the score against the threshold that was in force
at the time, the audio quality on both sides, and any caveat that was attached.

The wording is the wording used everywhere else. A score is a score out of 1.00
next to the threshold it had to clear; there are no percentages here, nothing is
a "match", and a row that says the audio could not support an assessment says
that instead of showing a number that looks like one.
"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Dict, List, Optional, Tuple

from ..comparison_log import (
    KIND_GALLERY,
    ComparisonRecord,
    delete_comparison,
    export_csv,
    list_comparisons,
    subjects_in,
)
from ..speaker_profiles import ProfileError, list_speaker_profiles
from ..thresholds import BAND_HIGH, BAND_INSUFFICIENT, DISCLAIMER
from .dialogs import active_window
from .errors import friendly_error
from .theme import SPACE_LG, SPACE_MD, SPACE_SM, SPACE_XS, Style, palette
from .widgets import (
    Card,
    EmptyState,
    KeyValueRow,
    PageHeader,
    StatusBanner,
    bind_wheel,
    danger_button,
    scrollable_body,
    secondary_button,
    subtle_button,
)

ALL_SUBJECTS = "All subjects"
ALL_RECORDINGS = "All recordings"


class ComparisonLogTab:
    """Builds and drives the Comparison history page inside ``parent``."""

    def __init__(
        self,
        parent: ttk.Frame,
        root: tk.Misc,
        cancel_event: threading.Event,
        on_cancel: Callable[[], None],
    ) -> None:
        self.parent = parent
        self.root = root
        self._cancel_event = cancel_event
        self._records: List[ComparisonRecord] = []
        self._shown: List[ComparisonRecord] = []
        self._selected: Optional[ComparisonRecord] = None
        # Subject ids that still have a profile, so a record about a subject
        # since deleted can say so rather than look like a live one.
        self._live_subjects: set = set()
        self.status_var = tk.StringVar(value="Idle")
        self.subject_var = tk.StringVar(value=ALL_SUBJECTS)
        self.recording_var = tk.StringVar(value=ALL_RECORDINGS)
        self._build()
        self.refresh()

    # -- UI ----------------------------------------------------------------

    def _build(self) -> None:
        canvas, container = scrollable_body(self.parent)

        PageHeader(
            container,
            "Comparison History",
            "Every speaker comparison this copy has run, and the recordings and "
            "scores behind each one.",
        ).pack(fill="x", pady=(0, SPACE_LG))

        # Stacked rather than side by side: the list needs its full width for
        # the columns that matter - the score and the assessment - and at 1366
        # a half-width list pushes both off the right edge.
        self._build_list(container)
        self._build_detail(container)

        self.banner = StatusBanner(container)
        ttk.Label(
            container, textvariable=self.status_var, style=Style.PAGE_SUBTITLE
        ).pack(anchor="w", pady=(SPACE_MD, 0))
        bind_wheel(canvas, container)

    def _build_list(self, parent: tk.Misc) -> None:
        card = Card(parent, "Comparisons")
        card.pack(fill="x")

        filters = ttk.Frame(card.body, style=Style.CARD_INNER)
        filters.pack(fill="x", pady=(0, SPACE_SM))
        ttk.Label(filters, text="Subject", style=Style.FIELD_LABEL).pack(side="left")
        self.subject_combo = ttk.Combobox(
            filters,
            textvariable=self.subject_var,
            values=[ALL_SUBJECTS],
            state="readonly",
            width=24,
        )
        self.subject_combo.pack(side="left", padx=(SPACE_SM, SPACE_LG))
        self.subject_combo.bind("<<ComboboxSelected>>", lambda _e: self._repaint())
        subtle_button(filters, "Refresh", self.refresh).pack(side="right")

        # Filtering by recording is how the same clip's history reads as a
        # timeline: the score against a subject can move as that subject's
        # reference profile grows, and both runs stand in the record.
        ttk.Label(filters, text="Recording", style=Style.FIELD_LABEL).pack(side="left")
        self.recording_combo = ttk.Combobox(
            filters,
            textvariable=self.recording_var,
            values=[ALL_RECORDINGS],
            state="readonly",
            width=24,
        )
        self.recording_combo.pack(side="left", padx=(SPACE_SM, 0))
        self.recording_combo.bind("<<ComboboxSelected>>", lambda _e: self._repaint())

        # The list grows for as long as the office keeps working, so it carries
        # its own scrollbar rather than relying on the page's.
        self._table = ttk.Frame(card.body, style=Style.CARD_INNER)
        self.tree = ttk.Treeview(
            self._table,
            columns=("when", "kind", "subject", "recording", "score", "band"),
            show="headings",
            height=12,
            selectmode="browse",
        )
        rows_scroll = ttk.Scrollbar(
            self._table, orient="vertical", command=self.tree.yview
        )
        self.tree.configure(yscrollcommand=rows_scroll.set)
        rows_scroll.pack(side="right", fill="y")
        headings: List[Tuple[str, str, int, bool]] = [
            ("when", "When", 140, False),
            # 1:1 and a gallery search answer different questions, so a row must
            # not be mistaken for the other kind.
            ("kind", "Type", 110, False),
            ("subject", "Subject", 200, True),
            ("recording", "Recording", 180, True),
            ("score", "Score", 65, False),
            ("band", "Assessment", 170, False),
        ]
        for column, heading, width, stretch in headings:
            self.tree.heading(column, text=heading)
            self.tree.column(column, width=width, stretch=stretch, anchor="w")
        self.tree.column("score", anchor="e")
        self.tree.pack(side="left", fill="both", expand=True)
        self._table.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self._on_select())
        # Colour is never the only signal: the assessment column says the same
        # thing in words, so these only reinforce it.
        self.tree.tag_configure("high", foreground=palette().accent_active)
        self.tree.tag_configure("insufficient", foreground=palette().warning)
        self.tree.tag_configure("refused", foreground=palette().danger_active)

        self.empty = EmptyState(
            card.body,
            "No comparisons recorded yet",
            "Run a comparison on the Compare Speakers page and it will be listed "
            "here, with the recording and the score behind it.",
        )

        # Held so the table and the empty state, which are packed and unpacked as
        # the filter changes, can always be inserted above the actions.
        self._actions = ttk.Frame(card.body, style=Style.CARD_INNER)
        actions = self._actions
        actions.pack(fill="x", pady=(SPACE_MD, 0))
        secondary_button(actions, "Export as CSV…", self._export).pack(side="left")
        self._delete_button = danger_button(actions, "Delete entry", self._delete)
        self._delete_button.pack(side="right")

    def _build_detail(self, parent: tk.Misc) -> None:
        card = Card(parent, "Comparison detail")
        card.pack(fill="x", pady=(SPACE_MD, 0))

        self.detail_empty = EmptyState(
            card.body,
            "Nothing selected",
            "Choose a comparison above to see the recording, the speech "
            "that was measured, and the numbers behind the assessment.",
        )
        self.detail_empty.pack(fill="both", expand=True)

        self.detail_body = ttk.Frame(card.body, style=Style.CARD_INNER)

        self.headline_var = tk.StringVar(value="")
        self.headline = ttk.Label(
            self.detail_body,
            textvariable=self.headline_var,
            style=Style.SECTION_TITLE,
            wraplength=780,
            justify="left",
        )
        self.headline.pack(anchor="w")
        self.summary_var = tk.StringVar(value="")
        ttk.Label(
            self.detail_body,
            textvariable=self.summary_var,
            style=Style.MUTED,
            wraplength=780,
            justify="left",
        ).pack(anchor="w", pady=(SPACE_XS, SPACE_MD))

        facts = ttk.Frame(self.detail_body, style=Style.CARD_INNER)
        facts.pack(fill="x")
        self._rows: Dict[str, KeyValueRow] = {
            "when": KeyValueRow(facts, "Recorded"),
            "subject": KeyValueRow(facts, "Subject"),
            "score": KeyValueRow(facts, "Similarity score"),
            "threshold": KeyValueRow(facts, "Operational threshold"),
            "margin": KeyValueRow(facts, "Margin"),
            "questioned": KeyValueRow(facts, "Questioned speech"),
            "selection": KeyValueRow(facts, "Selection"),
            "recording": KeyValueRow(facts, "Recording"),
            "sha": KeyValueRow(facts, "Source SHA-256"),
            "where": KeyValueRow(facts, "Recording location"),
            "reference": KeyValueRow(facts, "Reference speech"),
            "snapshot": KeyValueRow(facts, "Reference at the time"),
            "quality": KeyValueRow(facts, "Audio quality"),
            "model": KeyValueRow(facts, "Voice model"),
            "build": KeyValueRow(facts, "Application version"),
        }
        for row in self._rows.values():
            row.pack(fill="x", pady=(0, SPACE_XS))

        # A gallery search ranked everyone; the runner-up is part of the record.
        self.ranking_label = ttk.Label(
            self.detail_body, text="Ranking", style=Style.FIELD_LABEL
        )
        self.ranking = ttk.Treeview(
            self.detail_body,
            columns=("subject", "score", "band"),
            show="headings",
            height=5,
            selectmode="none",
        )
        for column, heading, width in (
            ("subject", "Subject", 170),
            ("score", "Score", 70),
            ("band", "Assessment", 160),
        ):
            self.ranking.heading(column, text=heading)
            self.ranking.column(column, width=width, anchor="w")
        self.ranking.column("score", anchor="e")

        self.caveats_label = ttk.Label(
            self.detail_body,
            text="Caveats recorded at the time",
            style=Style.FIELD_LABEL,
        )
        self.caveats = ttk.Label(
            self.detail_body,
            text="",
            style=Style.WARNING,
            wraplength=780,
            justify="left",
        )

        # Held so the ranking and the caveats can be packed *above* it: they are
        # part of the finding, and a disclaimer that floats above them reads as
        # though it applies to something else.
        self._disclaimer = ttk.Label(
            self.detail_body,
            text=DISCLAIMER,
            style=Style.META,
            wraplength=780,
            justify="left",
        )
        self._disclaimer.pack(anchor="w", pady=(SPACE_MD, 0))

    # -- Data --------------------------------------------------------------

    def refresh(self) -> None:
        """Reload the stored records and repaint."""
        self._records = list_comparisons()
        try:
            self._live_subjects = {p.subject_id for p in list_speaker_profiles()}
        except Exception:  # noqa: BLE001 - the history reads fine without this
            self._live_subjects = set()
        names = [ALL_SUBJECTS] + subjects_in(self._records)
        self.subject_combo.configure(values=names)
        if self.subject_var.get() not in names:
            self.subject_var.set(ALL_SUBJECTS)
        recordings = [ALL_RECORDINGS] + sorted(
            {r.questioned_filename for r in self._records if r.questioned_filename},
            key=str.casefold,
        )
        self.recording_combo.configure(values=recordings)
        if self.recording_var.get() not in recordings:
            self.recording_var.set(ALL_RECORDINGS)
        self._repaint()

    def _subject_cell(self, record: ComparisonRecord) -> str:
        """The subject, saying so when the profile behind it is gone.

        Deleting a profile must not quietly erase the record of what was run
        against it, and a record about a subject that no longer exists should
        not read as though it still does.
        """
        label = record.subject_label
        if (
            record.kind != KIND_GALLERY
            and record.reference_subject_id
            and record.reference_subject_id not in self._live_subjects
        ):
            return f"{label} (deleted)"
        return label

    def _repaint(self) -> None:
        wanted = self.subject_var.get()
        recording = self.recording_var.get()
        self._shown = [
            r
            for r in self._records
            if (wanted == ALL_SUBJECTS or r.subject_label == wanted)
            and (recording == ALL_RECORDINGS or r.questioned_filename == recording)
        ]
        self.tree.delete(*self.tree.get_children())
        for record in self._shown:
            self.tree.insert(
                "",
                "end",
                iid=record.record_id,
                values=(
                    _when(record.recorded_utc),
                    _kind(record),
                    self._subject_cell(record),
                    record.recording_label,
                    _score(record),
                    _band(record),
                ),
                tags=_tags(record),
            )
        if self._shown:
            self.empty.pack_forget()
            self._table.pack(fill="both", expand=True, before=self._actions)
        else:
            self._table.pack_forget()
            self.empty.pack(fill="both", expand=True, before=self._actions)
        self._select_none()
        total = len(self._records)
        if not total:
            self.status_var.set("No comparisons recorded yet.")
        elif len(self._shown) == total:
            self.status_var.set(f"{total} comparison(s) recorded.")
        else:
            self.status_var.set(
                f"Showing {len(self._shown)} of {total} recorded comparison(s)."
            )

    def _on_select(self) -> None:
        selection = self.tree.selection()
        record = next(
            (r for r in self._shown if selection and r.record_id == selection[0]),
            None,
        )
        self._selected = record
        if record is None:
            self._select_none()
            return
        self.detail_empty.pack_forget()
        self.detail_body.pack(fill="both", expand=True)
        self._delete_button.configure(state="normal")
        self._fill(record)

    def _select_none(self) -> None:
        self._selected = None
        self.detail_body.pack_forget()
        self.detail_empty.pack(fill="both", expand=True)
        self._delete_button.configure(state="disabled")

    def _fill(self, record: ComparisonRecord) -> None:
        self.headline_var.set(_band(record))
        self.headline.configure(style=_headline_style(record))
        self.summary_var.set(_summary(record))

        self._rows["when"].set(_when(record.recorded_utc, full=True))
        self._rows["subject"].set(self._subject_cell(record))
        self._rows["score"].set(_score(record))
        self._rows["threshold"].set(
            f"{record.operational_threshold:.2f} for high similarity"
        )
        windows = (
            f" across {record.questioned_window_count} window(s)"
            if record.questioned_window_count
            else ""
        )
        self._rows["questioned"].set(f"{record.questioned_seconds:.1f} sec{windows}")
        selection = record.questioned_selection or "not recorded"
        if record.questioned_spans:
            selection += f" — {_ranges(record.questioned_spans)}"
        self._rows["selection"].set(selection)
        self._rows["recording"].set(record.recording_label)
        self._rows["sha"].set(record.questioned_sha256 or "not recorded")
        where, where_style = _where(record)
        # Always pass a style: the row keeps whatever it was last given, so a
        # warning from one record would follow the next one into view.
        self._rows["where"].set(where, style_name=where_style)
        self._rows["reference"].set(
            f"{record.reference_seconds:.1f} sec"
            if record.reference_seconds
            else "not recorded"
        )
        quality = ", ".join(
            part
            for part in (
                f"questioned {record.questioned_quality.lower()}"
                if record.questioned_quality
                else "",
                f"reference {record.reference_quality.lower()}"
                if record.reference_quality
                else "",
            )
            if part
        )
        self._rows["snapshot"].set(_snapshot(record))
        self._rows["quality"].set(quality or "not recorded")
        self._rows["model"].set(record.embedding_model or "not recorded")
        self._rows["build"].set(record.app_version or "not recorded")
        if record.margin is not None:
            self._rows["margin"].set(
                f"{record.margin:.2f} over "
                f"{record.runner_up_name or 'the next best subject'}"
            )
            self._rows["margin"].pack(
                fill="x", pady=(0, SPACE_XS), before=self._rows["questioned"]
            )
        else:
            self._rows["margin"].pack_forget()
        # A gallery search is measured against every subject at once, so there
        # is no one reference profile to describe; those rows would otherwise
        # state facts about a subject that was never singled out.
        gallery = record.kind == KIND_GALLERY
        for key, following in (("reference", "snapshot"), ("snapshot", "quality")):
            if gallery:
                self._rows[key].pack_forget()
            else:
                self._rows[key].pack(
                    fill="x", pady=(0, SPACE_XS), before=self._rows[following]
                )
        self._rows["subject"].configure()
        self._fill_ranking(record)
        self._fill_caveats(record)

    def _fill_ranking(self, record: ComparisonRecord) -> None:
        self.ranking.delete(*self.ranking.get_children())
        if record.kind != KIND_GALLERY or not record.ranked:
            self.ranking_label.pack_forget()
            self.ranking.pack_forget()
            return
        for index, ranked in enumerate(record.ranked):
            self.ranking.insert(
                "",
                "end",
                iid=f"{record.record_id}-{index}",
                values=(
                    ranked.display_name,
                    f"{ranked.score:.2f}",
                    ranked.band or "—",
                ),
            )
        self.ranking_label.pack(
            anchor="w", pady=(SPACE_MD, SPACE_XS), before=self._disclaimer
        )
        self.ranking.pack(fill="x", before=self._disclaimer)

    def _fill_caveats(self, record: ComparisonRecord) -> None:
        text = "\n".join(f"• {w}" for w in record.warnings)
        if record.refusal_reason:
            text = f"• {record.refusal_reason}\n{text}".strip()
        if not text:
            self.caveats_label.pack_forget()
            self.caveats.pack_forget()
            return
        self.caveats.configure(text=text)
        self.caveats_label.pack(
            anchor="w", pady=(SPACE_MD, SPACE_XS), before=self._disclaimer
        )
        self.caveats.pack(anchor="w", before=self._disclaimer)

    # -- Actions -----------------------------------------------------------

    def _export(self) -> None:
        if not self._shown:
            self._status("There are no comparisons to export.")
            return
        path = filedialog.asksaveasfilename(
            title="Export comparison history",
            defaultextension=".csv",
            initialfile="comparison-history.csv",
            filetypes=[("Comma-separated values", "*.csv"), ("All files", "*.*")],
            parent=active_window(self.root),
        )
        if not path:
            return
        try:
            export_csv(self._shown, path)
        except OSError as exc:
            self.banner.show("error", friendly_error(exc))
            return
        self.banner.show(
            "success",
            f"Exported {len(self._shown)} comparison(s) to {Path(path).name}.",
        )
        self._status(f"Exported to {Path(path).name}.")

    def _delete(self) -> None:
        record = self._selected
        if record is None:
            self._status("Select a comparison first.")
            return
        if not messagebox.askyesno(
            "Delete this record",
            f"Delete the record of the comparison against "
            f"{record.subject_label} using {record.recording_label}?\n\n"
            "This removes the account of a comparison that was run. The "
            "recording and the speaker profile are not affected.",
            parent=active_window(self.root),
        ):
            return
        try:
            removed = delete_comparison(record)
        except ProfileError as exc:
            self.banner.show("error", friendly_error(exc))
            return
        if not removed:
            self.banner.show("warning", "That record could not be found on disk.")
        self.refresh()
        self._status("Record deleted.")

    def _status(self, message: str) -> None:
        self.root.after(0, lambda: self.status_var.set(message))

    # -- App plumbing ------------------------------------------------------

    def notify_cancelling(self) -> None:
        """Nothing here runs in the background; there is nothing to cancel."""

    def close(self) -> None:
        """No resources to release."""


# -- Presentation helpers ---------------------------------------------------


def _when(stamp: str, *, full: bool = False) -> str:
    """``2026-09-05T09:31:00+00:00`` -> ``2026-09-05 09:31`` (UTC when full)."""
    if not stamp:
        return "not recorded"
    text = stamp.replace("T", " ")
    for suffix in ("+00:00", "Z"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            return f"{text} UTC" if full else text[:16]
    return text if full else text[:16]


def _kind(record: ComparisonRecord) -> str:
    """A 1:1 comparison and a search over everyone answer different questions."""
    return "Gallery search" if record.kind == KIND_GALLERY else "Comparison"


def _snapshot(record: ComparisonRecord) -> str:
    """What the reference profile held when this was decided.

    A profile grows. The same recording measured against the same subject later
    can score differently for that reason alone, and this is what lets a record
    say so instead of looking like a contradiction.
    """
    if not record.reference_sample_count:
        return "not recorded"
    when = _when(record.reference_updated_utc) if record.reference_updated_utc else ""
    samples = f"{record.reference_sample_count} trusted sample(s)"
    return f"{samples}, last changed {when}" if when else samples


def _where(record: ComparisonRecord) -> "Tuple[str, str]":
    """Where the recording was, and whether it is still there.

    The recording itself is never copied into the history - that would quietly
    build a second store of operational audio - so this is a pointer, and the
    honest thing is to say when the pointer no longer resolves.
    """
    if not record.questioned_path:
        return "not recorded", Style.BODY
    try:
        present = Path(record.questioned_path).exists()
    except OSError:
        present = False
    if present:
        return record.questioned_path, Style.BODY
    return (
        f"{record.questioned_path} — not at this location now; the record of "
        "the comparison is unaffected",
        Style.WARNING,
    )


def _ranges(spans: "List[Tuple[float, float]]", limit: int = 4) -> str:
    """``0:10-0:18, 1:02-1:10`` - the audio the embedding was actually made from."""

    def clock(seconds: float) -> str:
        whole = max(0, int(seconds))
        return f"{whole // 60}:{whole % 60:02d}"

    shown = ", ".join(f"{clock(a)}-{clock(b)}" for a, b in spans[:limit])
    extra = len(spans) - limit
    return f"{shown} (+{extra} more)" if extra > 0 else shown


def _score(record: ComparisonRecord) -> str:
    """A score out of one, never a percentage."""
    if record.refused:
        return "—"
    return f"{record.score:.2f}"


def _band(record: ComparisonRecord) -> str:
    if record.refused:
        return "Not compared"
    return record.band or BAND_INSUFFICIENT


def _tags(record: ComparisonRecord) -> "Tuple[str, ...]":
    if record.refused:
        return ("refused",)
    if record.band == BAND_INSUFFICIENT:
        return ("insufficient",)
    if record.band == BAND_HIGH:
        return ("high",)
    return ()


def _headline_style(record: ComparisonRecord) -> str:
    if record.refused:
        return Style.DANGER
    if record.band == BAND_INSUFFICIENT:
        return Style.WARNING
    # High similarity is deliberately not green: green reads as "confirmed",
    # and a similarity score is never a confirmation of identity.
    if record.band == BAND_HIGH:
        return Style.ACCENT
    return Style.SECTION_TITLE


def _summary(record: ComparisonRecord) -> str:
    if record.refused:
        return "This comparison was not scored. " + (
            record.refusal_reason or "The reason was not recorded."
        )
    if record.kind == KIND_GALLERY:
        return (
            f"{record.recording_label} was searched against "
            f"{record.subjects_searched} subject profile(s)."
        )
    return (
        f"{record.questioned_label or 'The questioned speech'} in "
        f"{record.recording_label}, measured against the "
        f"{record.subject_label} reference profile."
    )
