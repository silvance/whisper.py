"""The Speaker Compare tab: one questioned speaker against a known subject.

A first-class workflow rather than a small dialog over two JSON files:

* **Reference** - pick a known subject; the tab shows the evidence behind that
  profile (sample count, total reference speech, embedding model and hash,
  quality warnings) so an operator can see what they are comparing against.
* **Questioned** - pick a new recording and say which speech to compare: the
  whole file, a diarized speaker, or explicit time ranges.
* **Result** - a similarity score out of 1.00, an assessment band, the
  operational threshold, both durations, both quality bands, model
  compatibility, and the investigative-only disclaimer.

Nothing here reports a percentage or a probability of identity, and an
insufficient or mismatched input produces an explicit refusal or
"Insufficient data" rather than a number that looks conclusive.
"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any, Callable, List, Optional, Tuple

from ..enrollment import (
    parse_time_ranges,
    prepare_source,
    spans_for_speaker,
    speaker_totals,
)
from ..matching import (
    ComparisonResult,
    GalleryResult,
    compare_questioned_to_profile,
    search_gallery_for_questioned,
)
from ..questioned import (
    SELECTION_DIARIZED,
    SELECTION_RANGES,
    SELECTION_WHOLE,
    QuestionedSpeaker,
    measure_from_wav,
)
from ..reports import write_analysis_report
from ..speaker_profiles import (
    SpeakerProfile,
    bundled_model_identity,
    display_labels,
    list_speaker_profiles,
)
from ..thresholds import (
    BAND_HIGH,
    BAND_INSUFFICIENT,
    DISCLAIMER,
    active,
    describe_active,
)
from ..transcription import AUDIO_EXTENSIONS
from ..voiceprints import SpeakerEmbedder
from .errors import friendly_error
from .theme import SPACE_LG, SPACE_MD, SPACE_SM, SPACE_XL, SPACE_XS, Style, theme
from .widgets import (
    Card,
    Disclosure,
    EmptyState,
    FileDropZone,
    KeyValueRow,
    PageHeader,
    StatusBanner,
    bind_wheel,
    primary_button,
    scrollable_body,
    secondary_button,
    set_readonly_text,
    style_text_widget,
    subtle_button,
)

MODE_WHOLE = "whole"
MODE_DIARIZE = "diarize"
MODE_RANGES = "ranges"


class SpeakerCompareTab:
    """Builds and drives the Speaker Compare tab inside ``parent``."""

    def __init__(
        self,
        parent: ttk.Frame,
        root: tk.Misc,
        cancel_event: threading.Event,
        on_cancel: Callable[[], None],
        *,
        get_analysis: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.parent = parent
        self.root = root
        # Supplies (result, speaker_names, provenance) from the Transcribe tab so
        # a report can carry the transcript and its traceability. Optional.
        self._get_analysis = get_analysis
        self._profiles: List[SpeakerProfile] = []
        # subject_id -> the unique label shown in the picker.
        self._labels: dict = {}
        # The diarized cluster the operator chose, for the selection record.
        self._picked_speaker: Optional[str] = None
        # Source hashes of every recording compared here, so a report only
        # attaches a transcript that is actually of the questioned audio.
        self._questioned_sources: set = set()
        self._comparisons: List[ComparisonResult] = []
        self._embedder: Optional[SpeakerEmbedder] = None
        self._last_result: Optional[ComparisonResult] = None

        self.reference_var = tk.StringVar(value="")
        self.reference_detail_var = tk.StringVar(value="No reference selected.")
        self.questioned_var = tk.StringVar(value="")
        self.mode_var = tk.StringVar(value=MODE_WHOLE)
        self.ranges_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Idle")
        self.allow_unverified_var = tk.BooleanVar(value=False)

        self._build()
        self.refresh()

    # -- UI ----------------------------------------------------------------

    def _build(self) -> None:
        canvas, container = scrollable_body(self.parent)

        PageHeader(
            container,
            "Compare Speakers",
            "Measure how similar a speaker in one recording is to a known "
            "person's reference voice.",
        ).pack(fill="x", pady=(0, SPACE_LG))

        self._build_reference_step(container)
        self._build_questioned_step(container)
        self._build_action_step(container)
        self._build_result_step(container)

        ttk.Label(
            container, textvariable=self.status_var, style=Style.PAGE_SUBTITLE
        ).pack(anchor="w", pady=(SPACE_MD, 0))
        bind_wheel(canvas, container)

    # -- Step 1: the known person -----------------------------------------

    def _build_reference_step(self, parent: tk.Misc) -> None:
        card = Card(parent, "1.  Reference speaker")
        card.pack(fill="x")
        row = ttk.Frame(card.body, style=Style.CARD_INNER)
        row.pack(fill="x")
        self.reference_combo = ttk.Combobox(
            row, textvariable=self.reference_var, state="readonly", width=34
        )
        self.reference_combo.pack(side="left")
        self.reference_combo.bind(
            "<<ComboboxSelected>>", lambda _e: self._render_reference()
        )
        subtle_button(row, "Refresh", self.refresh).pack(
            side="left", padx=(SPACE_SM, 0)
        )

        self._reference_facts = ttk.Frame(card.body, style=Style.CARD_INNER)
        self._reference_facts.pack(fill="x", pady=(SPACE_MD, 0))
        self._reference_rows = {
            "speech": KeyValueRow(self._reference_facts, "Reference speech"),
            "samples": KeyValueRow(self._reference_facts, "Trusted samples"),
            "model": KeyValueRow(self._reference_facts, "Voice model"),
        }
        for row_widget in self._reference_rows.values():
            row_widget.pack(fill="x", pady=(0, SPACE_XS))
        self._reference_empty = EmptyState(
            card.body,
            "No reference profile selected",
            "Create one under Speaker Profiles from a recording you know contains "
            "that person.",
        )

    # -- Step 2: the recording in question ---------------------------------

    def _build_questioned_step(self, parent: tk.Misc) -> None:
        card = Card(parent, "2.  Questioned recording")
        card.pack(fill="x", pady=(SPACE_MD, 0))
        self._questioned_zone = FileDropZone(
            card.body,
            self.questioned_var,
            self._choose_questioned,
            prompt="Drop the recording to check here",
            button_text="Choose recording",
            change_text="Change recording",
        )
        self._questioned_zone.pack(fill="x")

        ttk.Label(
            card.body, text="Whose voice should be measured?", style=Style.FIELD_LABEL
        ).pack(anchor="w", pady=(SPACE_MD, SPACE_XS))
        for text, value in (
            ("Only one person is speaking — use the whole recording", MODE_WHOLE),
            ("Several people — separate them and let me pick", MODE_DIARIZE),
            ("Only these parts of the recording", MODE_RANGES),
        ):
            ttk.Radiobutton(
                card.body,
                text=text,
                variable=self.mode_var,
                value=value,
                style=Style.RADIO,
            ).pack(anchor="w")
        ranges = ttk.Frame(card.body, style=Style.CARD_INNER)
        ranges.pack(anchor="w", padx=(SPACE_XL, 0))
        ttk.Entry(ranges, textvariable=self.ranges_var, width=40).pack(side="left")
        ttk.Label(ranges, text="e.g. 0:10-0:45, 1:20-2:00", style=Style.META).pack(
            side="left", padx=(SPACE_SM, 0)
        )

    # -- The comparison ----------------------------------------------------

    def _build_action_step(self, parent: tk.Misc) -> None:
        actions = ttk.Frame(parent, style=Style.PAGE)
        actions.pack(fill="x", pady=(SPACE_LG, 0))
        primary_button(actions, "Compare speakers", self._compare).pack(side="left")
        secondary_button(actions, "Search all profiles", self._search_gallery).pack(
            side="left", padx=(SPACE_SM, 0)
        )
        subtle_button(actions, "Thresholds…", self._show_thresholds).pack(side="right")

        advanced = Disclosure(parent, "Advanced")
        advanced.pack(fill="x", pady=(SPACE_SM, 0))
        ttk.Checkbutton(
            advanced.body,
            text="Allow a profile whose voice model cannot be verified",
            variable=self.allow_unverified_var,
        ).pack(anchor="w")
        ttk.Label(
            advanced.body,
            text=(
                "Only for an older profile that does not record which model made "
                "it. A profile from a different model is always refused."
            ),
            style=Style.META,
            wraplength=560,
            justify="left",
        ).pack(anchor="w", padx=(SPACE_XL, 0))

    # -- Step 3: what was found --------------------------------------------

    def _build_result_step(self, parent: tk.Misc) -> None:
        card = Card(parent, "3.  Result")
        card.pack(fill="both", expand=True, pady=(SPACE_LG, 0))
        self._result_card = card

        self._result_empty = EmptyState(
            card.body,
            "No comparison yet",
            "Choose a reference profile and a questioned recording, then select "
            "Compare speakers.",
        )
        self._result_empty.pack(fill="x")

        self._result_body = ttk.Frame(card.body, style=Style.CARD_INNER)

        # The band, not the number, is the finding. It is stated in words, in
        # the largest type on the card, and never as a percentage.
        self.band_var = tk.StringVar(value="")
        self.band_label = ttk.Label(
            self._result_body, textvariable=self.band_var, style=Style.SECTION_TITLE
        )
        self.band_label.pack(anchor="w")
        self.lead_var = tk.StringVar(value="")
        ttk.Label(
            self._result_body,
            textvariable=self.lead_var,
            style=Style.BODY,
            wraplength=680,
            justify="left",
        ).pack(anchor="w", pady=(SPACE_XS, 0))

        facts = ttk.Frame(self._result_body, style=Style.CARD_INNER)
        facts.pack(fill="x", pady=(SPACE_MD, 0))
        self._result_rows = {
            "score": KeyValueRow(facts, "Similarity score"),
            "threshold": KeyValueRow(facts, "Operational threshold"),
            "questioned": KeyValueRow(facts, "Questioned speech"),
            "reference": KeyValueRow(facts, "Reference speech"),
            "quality": KeyValueRow(facts, "Audio quality"),
            "model": KeyValueRow(facts, "Voice model"),
            "source": KeyValueRow(facts, "Recording"),
        }
        for row_widget in self._result_rows.values():
            row_widget.pack(fill="x", pady=(0, SPACE_XS))

        self.result_warnings = StatusBanner(self._result_body, wraplength=680)

        # The full text block stays: it is what an analyst copies into a case
        # file, and it holds the detail the summary above deliberately omits.
        self._detail_disclosure = Disclosure(self._result_body, "Full result text")
        self.result_warnings.insert_before(self._detail_disclosure)
        self._detail_disclosure.pack(fill="x", pady=(SPACE_MD, 0))
        self.result_text = ScrolledText(
            self._detail_disclosure.body, wrap="word", height=12, state="disabled"
        )
        style_text_widget(self.result_text)
        self.result_text.configure(font=theme().mono)
        self.result_text.pack(fill="both", expand=True)

        # Kept as the last thing built but remembered, so the result can be
        # inserted above it rather than after it.
        self._disclaimer_label = ttk.Label(
            card.body,
            text=DISCLAIMER,
            style=Style.META,
            wraplength=680,
            justify="left",
        )
        self._disclaimer_label.pack(anchor="w", pady=(SPACE_MD, 0))

        result_actions = ttk.Frame(card.body, style=Style.CARD_INNER)
        result_actions.pack(fill="x", pady=(SPACE_MD, 0))
        secondary_button(result_actions, "Export report…", self._export_report).pack(
            side="left"
        )
        secondary_button(result_actions, "Copy result", self._copy_result).pack(
            side="left", padx=(SPACE_SM, 0)
        )
        subtle_button(
            result_actions, "Clear recorded comparisons", self._clear_comparisons
        ).pack(side="left", padx=(SPACE_SM, 0))

    # -- Reference ----------------------------------------------------------

    def refresh(self) -> None:
        self._profiles = list_speaker_profiles()
        # Two subjects can share a display name; picking by name alone would let
        # an operator compare against one while believing they chose the other.
        self._labels = display_labels(self._profiles)
        labels = [self._labels[p.subject_id] for p in self._profiles]
        self.reference_combo.configure(values=labels)
        if labels and self.reference_var.get() not in labels:
            self.reference_var.set(labels[0])
        if not labels:
            self.reference_var.set("")
        self._render_reference()

    def _selected_profile(self) -> Optional[SpeakerProfile]:
        """The subject behind the chosen label - by subject id, never by name."""
        label = self.reference_var.get()
        for profile in self._profiles:
            if self._labels.get(profile.subject_id) == label:
                return profile
        return None

    def _render_reference(self) -> None:
        profile = self._selected_profile()
        if profile is None:
            self._reference_facts.pack_forget()
            self._reference_empty.pack(fill="x")
            self.reference_detail_var.set(
                "No reference selected. Create one under Speaker Profiles."
            )
            return
        self._reference_empty.pack_forget()
        self._reference_facts.pack(fill="x", pady=(SPACE_MD, 0))

        summary = profile.summary()
        minimum = active().min_reference_seconds
        thin = profile.total_reference_seconds < minimum
        self._reference_rows["speech"].set(
            f"{profile.total_reference_seconds:.1f} sec"
            + (f"  —  less than the recommended {minimum:.0f} sec" if thin else ""),
            style_name=Style.WARNING if thin else Style.BODY,
        )
        self._reference_rows["samples"].set(str(summary["trusted_sample_count"]))
        if profile.embedding_model:
            self._reference_rows["model"].set(
                profile.embedding_model.describe(), style_name=Style.BODY
            )
        else:
            self._reference_rows["model"].set(
                "unknown (older profile) — a comparison needs explicit "
                "confirmation under Advanced",
                style_name=Style.WARNING,
            )
        # Still maintained for anything that reads the plain string.
        self.reference_detail_var.set(
            f"{profile.display_name} — {summary['trusted_sample_count']} trusted "
            f"samples, {profile.total_reference_seconds:.1f}s reference speech"
        )

    # -- Questioned ---------------------------------------------------------

    def _choose_questioned(self) -> None:
        patterns = " ".join(f"*{ext}" for ext in AUDIO_EXTENSIONS)
        path = filedialog.askopenfilename(
            title="Questioned recording",
            filetypes=[("Audio/Video", patterns), ("All files", "*.*")],
        )
        if path:
            self.questioned_var.set(path)

    def _get_embedder(self) -> SpeakerEmbedder:
        """Load the speaker-embedding model once (sherpa loads on first use)."""
        if self._embedder is None:
            self._embedder = SpeakerEmbedder()
        return self._embedder

    def _prepare_spans(self, wav: Path) -> Optional[List[Tuple[float, float]]]:
        """Resolve the questioned speech according to the selected mode."""
        mode = self.mode_var.get()
        if mode == MODE_RANGES:
            return parse_time_ranges(self.ranges_var.get())
        if mode == MODE_DIARIZE:
            from ..diarization import diarize

            self._status("Separating speakers…")
            segments = diarize(wav, progress=self._status)
            totals = speaker_totals(segments)
            if not totals:
                return None
            picked: dict = {"value": None}
            done = threading.Event()

            def _ask() -> None:
                try:
                    picked["value"] = self._choose_cluster(totals)
                finally:
                    done.set()

            self.root.after(0, _ask)
            done.wait(timeout=300)
            if not picked["value"]:
                return None
            self._picked_speaker = str(picked["value"])
            return spans_for_speaker(segments, picked["value"])
        return [(0.0, _duration(wav))]

    def _choose_cluster(self, totals) -> Optional[str]:
        win = tk.Toplevel(self.root)
        win.title("Which speaker is the questioned speaker?")
        win.transient(self.root)  # type: ignore[call-overload]
        win.grab_set()
        labels = [f"{name} — {seconds:.0f}s of speech" for name, seconds in totals]
        var = tk.StringVar(value=labels[0])
        chosen: dict = {"value": None}
        frame = ttk.Frame(win, padding=12)
        frame.pack(fill="both", expand=True)
        ttk.Label(
            frame,
            text="Pick the speaker to compare against the reference profile.",
            wraplength=380,
            justify="left",
        ).pack(anchor="w", pady=(0, 8))
        ttk.Combobox(
            frame, textvariable=var, values=labels, state="readonly", width=42
        ).pack(anchor="w")

        def _ok() -> None:
            chosen["value"] = totals[labels.index(var.get())][0]
            win.destroy()

        row = ttk.Frame(frame)
        row.pack(fill="x", pady=(10, 0))
        ttk.Button(row, text="Cancel", command=win.destroy).pack(side="right")
        ttk.Button(row, text="Use this speaker", command=_ok).pack(
            side="right", padx=(0, 6)
        )
        win.wait_window()
        return chosen["value"]

    # -- Run ----------------------------------------------------------------

    def _compare(self) -> None:
        profile = self._selected_profile()
        if profile is None:
            self._status("Select a reference subject first.")
            return
        source = self.questioned_var.get().strip()
        if not source:
            self._status("Choose a questioned recording first.")
            return
        threading.Thread(
            target=self._compare_worker, args=(profile, Path(source)), daemon=True
        ).start()

    def _measure_questioned(self, source: Path) -> Optional[QuestionedSpeaker]:
        """Measure the questioned speaker, keeping the audio and the source hash.

        The selection is resolved first (which may need diarization), then every
        usable window of it is embedded and averaged - so the duration and
        quality reported are the ones behind the embedding, and the recording
        identifies itself in the result and any report.
        """
        embedder = self._get_embedder()
        wav, digest, temporary = prepare_source(source, progress=self._status)
        try:
            spans = self._prepare_spans(wav)
            if not spans:
                return None
            measured = measure_from_wav(
                wav,
                spans,
                embedder,
                selection_mode=self._selection_description(spans),
                progress=self._status,
            )
        finally:
            if temporary:
                try:
                    Path(wav).unlink()
                except OSError:
                    pass
        measured.source_filename = source.name
        measured.source_sha256 = digest
        try:
            measured.source_size = source.stat().st_size
        except OSError:
            measured.source_size = None
        return measured

    def _selection_description(self, spans) -> str:
        """How the questioned speech was chosen, in words, for the record."""
        mode = self.mode_var.get()
        if mode == MODE_RANGES:
            return f"{SELECTION_RANGES} ({self.ranges_var.get().strip()})"
        if mode == MODE_DIARIZE:
            speaker = self._picked_speaker or "selected speaker"
            return f"{SELECTION_DIARIZED} ({speaker}, {len(spans)} turn(s))"
        return SELECTION_WHOLE

    def _compare_worker(self, profile: SpeakerProfile, source: Path) -> None:
        try:
            measured = self._measure_questioned(source)
            if measured is None or not measured.usable:
                self._report_unusable(measured)
                return
            result = compare_questioned_to_profile(
                measured,
                profile,
                questioned_model=bundled_model_identity(),
                allow_unverified_model=self.allow_unverified_var.get(),
            )
            self._last_result = result
            self._comparisons.append(result)
            self._questioned_sources.add(measured.source_sha256 or "")
            self.root.after(0, lambda: self._show_comparison(result))
            self._status("Comparison complete.")
        except Exception as exc:  # noqa: BLE001 - surfaced to the operator
            self._status(friendly_error(exc))

    def _search_gallery(self) -> None:
        source = self.questioned_var.get().strip()
        if not source:
            self._status("Choose a questioned recording first.")
            return
        threading.Thread(
            target=self._gallery_worker, args=(Path(source),), daemon=True
        ).start()

    def _gallery_worker(self, source: Path) -> None:
        try:
            measured = self._measure_questioned(source)
            if measured is None or not measured.usable:
                self._report_unusable(measured)
                return
            result = search_gallery_for_questioned(
                measured,
                self._profiles,
                questioned_model=bundled_model_identity(),
            )
            self.root.after(0, lambda: self._show_gallery(measured, result))
            self._status("Search complete.")
        except Exception as exc:  # noqa: BLE001 - surfaced to the operator
            self._status(friendly_error(exc))

    # -- Output -------------------------------------------------------------

    def _show(self, lines: List[str]) -> None:
        """The full text block - what an analyst copies into a case file."""
        set_readonly_text(self.result_text, "\n".join(lines) + "\n\n" + DISCLAIMER)
        self.root.after(0, self._reveal_result)

    def _reveal_result(self) -> None:
        self._result_empty.pack_forget()
        self._result_body.pack(fill="both", expand=True, before=self._disclaimer_label)

    def _show_comparison(self, result: ComparisonResult) -> None:
        """Present a comparison as a finding, with the numbers supporting it.

        The band leads, in words. The score is shown out of 1.00 beside it -
        never as a percentage, which reads as a probability that these are the
        same person, which is not what a cosine similarity is. A result the
        audio cannot support outranks whatever the number happens to be.
        """
        self._show(result.format_lines())

        if result.refused:
            self._set_band("Comparison refused", Style.DANGER)
            self.lead_var.set(result.refusal_reason)
        elif result.band == BAND_INSUFFICIENT:
            self._set_band(BAND_INSUFFICIENT, Style.WARNING)
            self.lead_var.set(
                "There is not enough usable speech to support any assessment. "
                "The score below is not meaningful."
            )
        elif result.band == BAND_HIGH:
            # Blue, not green. Green is this application's "success / ready",
            # and a green speaker result reads as "we got the right person" -
            # the exact conclusion every word on this card is written to avoid.
            self._set_band(result.band, Style.ACCENT)
            self.lead_var.set(
                f"The questioned speaker produced high similarity to the "
                f"{result.reference_name} reference profile. Further review is "
                "warranted."
            )
        else:
            self._set_band(result.band, Style.BODY)
            self.lead_var.set(
                f"The questioned speech is not strongly similar to the "
                f"{result.reference_name} reference profile."
            )

        self._result_rows["score"].set(
            "—" if result.refused else f"{result.score:.2f} / 1.00"
        )
        self._result_rows["threshold"].set(
            f"{result.operational_threshold:.2f} for high similarity"
        )
        measured = (
            f" across {result.questioned_window_count} window(s)"
            if result.questioned_window_count
            else ""
        )
        self._result_rows["questioned"].set(
            f"{result.questioned_seconds:.1f} sec{measured}"
        )
        self._result_rows["reference"].set(f"{result.reference_seconds:.1f} sec")
        self._result_rows["quality"].set(
            f"questioned {result.questioned_quality.lower()}, "
            f"reference {result.reference_quality.lower()}"
        )
        self._result_rows["model"].set(result.embedding_model)
        self._result_rows["source"].set(
            result.questioned_source_filename or "not recorded"
        )

        if result.warnings:
            self.result_warnings.show("warning", "  ".join(result.warnings))
        else:
            self.result_warnings.hide()

    def _report_unusable(self, measured: object) -> None:
        """Not enough speech to measure is a result, and it has to look like one."""
        message = "No usable speech was found in the questioned selection."
        self._status(message)
        if measured is None:
            return
        describe = getattr(measured, "describe", None)
        warnings = list(getattr(measured, "warnings", []))
        self._show((list(describe()) if callable(describe) else []) + [""] + warnings)

        def _paint() -> None:
            self._set_band(BAND_INSUFFICIENT, Style.WARNING)
            self.lead_var.set(
                message + " Select more of the speaker's audio, or a recording "
                "with more of them speaking."
            )
            for row in self._result_rows.values():
                row.set("—")
            self._result_rows["source"].set(
                str(getattr(measured, "source_filename", "") or "not recorded")
            )
            if warnings:
                self.result_warnings.show("warning", "  ".join(warnings))

        self.root.after(0, _paint)

    def _set_band(self, text: str, style_name: str) -> None:
        self.band_var.set(text)
        self.band_label.configure(style=style_name)

    def _show_gallery(self, measured: object, result: GalleryResult) -> None:
        """A ranked search is context, not an identification."""
        describe = getattr(measured, "describe", None)
        lines = list(describe()) if callable(describe) else []
        self._show(lines + [""] + result.summary_lines())
        if result.inadequate_reason:
            self._set_band(BAND_INSUFFICIENT, Style.WARNING)
            self.lead_var.set(
                f"{result.inadequate_reason} The ranking below supports no "
                "conclusion about any subject."
            )
        elif result.accepted_name:
            self._set_band("Possible lead", Style.ACCENT)
            self.lead_var.set(
                f"The questioned speaker produced high similarity to the "
                f"{result.accepted_name} reference profile. Further review is "
                "warranted."
            )
        else:
            self._set_band("No sufficiently strong match", Style.BODY)
            self.lead_var.set("No known profile produced a sufficiently strong match.")
        top = result.matches[0] if result.matches else None
        self._result_rows["score"].set(
            f"{top.score:.2f} / 1.00 ({top.display_name})" if top else "—"
        )
        self._result_rows["threshold"].set(
            f"{result.thresholds.recognition_acceptance:.2f} to accept, "
            f"{result.thresholds.recognition_margin:.2f} clear of the next"
        )
        self._result_rows["questioned"].set(
            f"{getattr(measured, 'speech_seconds', 0.0):.1f} sec"
        )
        self._result_rows["reference"].set(f"{result.searched} profile(s) searched")
        self._result_rows["quality"].set(str(getattr(measured, "quality", "—")))
        self._result_rows["model"].set("—")
        self._result_rows["source"].set(
            str(getattr(measured, "source_filename", "") or "not recorded")
        )
        if result.skipped:
            self.result_warnings.show(
                "warning",
                f"{len(result.skipped)} profile(s) skipped: "
                + "; ".join(result.skipped[:2]),
            )
        else:
            self.result_warnings.hide()

    def _show_thresholds(self) -> None:
        """Show the decision thresholds in force (read-only)."""
        window = tk.Toplevel(self.root)
        window.title("Whispers - active thresholds")
        window.transient(self.root)  # type: ignore[call-overload]
        text = ScrolledText(
            window, wrap="word", width=76, height=22, font="TkFixedFont"
        )
        text.pack(fill="both", expand=True, padx=8, pady=8)
        text.insert("end", "\n".join(describe_active()))
        text.configure(state="disabled")
        ttk.Button(window, text="Close", command=window.destroy).pack(
            side="right", padx=8, pady=(0, 8)
        )

    def _copy_result(self) -> None:
        text = self.result_text.get("1.0", "end-1c")
        if not text.strip():
            self._status("Nothing to copy yet.")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self._status("Result copied to clipboard.")

    def _analysis_matches(self, provenance) -> bool:
        """True when the open analysis is of a recording these comparisons used.

        Compared by source SHA-256: a matching name proves nothing, and an
        analysis with no recorded hash cannot be shown to be the right one.
        """
        digest = None
        if provenance is not None and getattr(provenance, "source", None):
            digest = provenance.source.sha256
        if not digest:
            return False
        return digest in self._questioned_sources

    def _export_report(self) -> None:
        """Write a report of the comparisons made, with the transcript if there is one."""
        if not self._comparisons:
            self._status("Run a comparison first.")
            return
        result = names = provenance = None
        if self._get_analysis is not None:
            try:
                result, names, provenance = self._get_analysis()
            except Exception:  # noqa: BLE001 - a report without the transcript is fine
                result = names = provenance = None
        # A transcript from a *different* recording must not be presented as the
        # source of these comparisons. Attach it only when its source hash is
        # one of the recordings actually compared here.
        if result is not None and not self._analysis_matches(provenance):
            result = names = provenance = None
            self._status(
                "The open transcript is of a different recording, so it was left "
                "out of this report."
            )
        path = filedialog.asksaveasfilename(
            title="Export analysis report",
            defaultextension=".docx",
            initialfile="speaker-analysis-report.docx",
            filetypes=[("Word document", "*.docx"), ("Text file", "*.txt")],
        )
        if not path:
            return
        try:
            write_analysis_report(
                path,
                result=result,
                speaker_names=names,
                provenance=provenance,
                comparisons=self._comparisons,
            )
            self._status(f"Saved {Path(path).name}")
        except Exception as exc:  # noqa: BLE001 - surfaced to the operator
            self._status(friendly_error(exc))

    def _clear_comparisons(self) -> None:
        self._comparisons = []
        self._status("Cleared the recorded comparisons.")

    def _status(self, message: str) -> None:
        self.root.after(0, lambda: self.status_var.set(message))

    # -- Shared hooks -------------------------------------------------------

    def notify_cancelling(self) -> None:
        return None

    def get_settings(self) -> dict:
        return {}

    def apply_settings(self, data: dict) -> None:
        return None


def _duration(wav_path: Path) -> float:
    import wave

    with wave.open(str(wav_path), "rb") as wav:
        return wav.getnframes() / float(wav.getframerate() or 1)
