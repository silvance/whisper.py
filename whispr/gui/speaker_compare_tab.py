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
from ..matching import ComparisonResult, compare_embedding_to_profile, search_gallery
from ..quality import analyse_span, combine
from ..reports import write_analysis_report
from ..speaker_profiles import (
    SpeakerProfile,
    bundled_model_identity,
    list_speaker_profiles,
)
from ..thresholds import DISCLAIMER, active, describe_active
from ..transcription import AUDIO_EXTENSIONS
from ..voiceprints import SpeakerEmbedder
from .errors import friendly_error
from .widgets import bind_wheel, scrollable_body, set_readonly_text

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

        ttk.Label(container, text="Speaker comparison", font=("", 12, "bold")).pack(
            anchor="w"
        )
        ttk.Label(
            container,
            text=(
                "Compare a speaker in a questioned recording against a known "
                "subject's reference voice. The result is an investigative lead, "
                "not an identification."
            ),
            wraplength=620,
            justify="left",
            font=("", 8),
        ).pack(anchor="w", pady=(0, 8))

        # --- Reference -----------------------------------------------------
        reference = ttk.LabelFrame(
            container, text="Reference (known subject)", padding=8
        )
        reference.pack(fill="x")
        row = ttk.Frame(reference)
        row.pack(fill="x")
        self.reference_combo = ttk.Combobox(
            row, textvariable=self.reference_var, state="readonly", width=36
        )
        self.reference_combo.pack(side="left")
        self.reference_combo.bind(
            "<<ComboboxSelected>>", lambda _e: self._render_reference()
        )
        ttk.Button(row, text="Refresh", command=self.refresh).pack(
            side="left", padx=(6, 0)
        )
        ttk.Label(
            reference,
            textvariable=self.reference_detail_var,
            wraplength=620,
            justify="left",
        ).pack(anchor="w", pady=(6, 0))

        # --- Questioned ----------------------------------------------------
        questioned = ttk.LabelFrame(container, text="Questioned recording", padding=8)
        questioned.pack(fill="x", pady=(10, 0))
        pick = ttk.Frame(questioned)
        pick.pack(fill="x")
        ttk.Entry(pick, textvariable=self.questioned_var).pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(pick, text="Browse…", command=self._choose_questioned).pack(
            side="left", padx=(6, 0)
        )
        ttk.Radiobutton(
            questioned,
            text="One speaker only — compare the whole recording",
            variable=self.mode_var,
            value=MODE_WHOLE,
        ).pack(anchor="w", pady=(6, 0))
        ttk.Radiobutton(
            questioned,
            text="Several people — separate speakers and let me pick",
            variable=self.mode_var,
            value=MODE_DIARIZE,
        ).pack(anchor="w")
        ttk.Radiobutton(
            questioned,
            text="Compare only these time ranges",
            variable=self.mode_var,
            value=MODE_RANGES,
        ).pack(anchor="w")
        ttk.Entry(questioned, textvariable=self.ranges_var, width=44).pack(
            anchor="w", padx=(24, 0)
        )
        ttk.Label(questioned, text="e.g. 0:10-0:45, 1:20-2:00", font=("", 8)).pack(
            anchor="w", padx=(24, 0)
        )

        # --- Actions -------------------------------------------------------
        actions = ttk.Frame(container)
        actions.pack(fill="x", pady=(10, 0))
        ttk.Button(actions, text="Compare", command=self._compare).pack(side="left")
        ttk.Button(
            actions, text="Search all subjects", command=self._search_gallery
        ).pack(side="left", padx=(6, 0))
        ttk.Checkbutton(
            actions,
            text="Allow a profile whose embedding model can't be verified",
            variable=self.allow_unverified_var,
        ).pack(side="left", padx=(12, 0))
        # Read-only: the thresholds behind every verdict on this tab. Shown so an
        # analyst can state them, not offered as a control - retuning them
        # changes how every result should be read, and belongs to a validation
        # run rather than a click.
        ttk.Button(actions, text="Thresholds…", command=self._show_thresholds).pack(
            side="right"
        )

        # --- Result --------------------------------------------------------
        result_frame = ttk.LabelFrame(container, text="Result", padding=8)
        result_frame.pack(fill="both", expand=True, pady=(10, 0))
        self.result_text = ScrolledText(
            result_frame, wrap="word", height=14, state="disabled", font="TkFixedFont"
        )
        self.result_text.pack(fill="both", expand=True)
        ttk.Label(
            result_frame,
            text=DISCLAIMER,
            wraplength=620,
            justify="left",
            font=("", 8),
        ).pack(anchor="w", pady=(6, 0))
        ttk.Button(result_frame, text="Copy result", command=self._copy_result).pack(
            anchor="w", pady=(6, 0)
        )

        ttk.Label(container, textvariable=self.status_var, wraplength=620).pack(
            anchor="w", pady=(8, 0)
        )
        bind_wheel(canvas, container)

    # -- Reference ----------------------------------------------------------

    def refresh(self) -> None:
        self._profiles = list_speaker_profiles()
        labels = [p.display_name for p in self._profiles]
        self.reference_combo.configure(values=labels)
        if labels and self.reference_var.get() not in labels:
            self.reference_var.set(labels[0])
        if not labels:
            self.reference_var.set("")
        self._render_reference()

    def _selected_profile(self) -> Optional[SpeakerProfile]:
        name = self.reference_var.get()
        for profile in self._profiles:
            if profile.display_name == name:
                return profile
        return None

    def _render_reference(self) -> None:
        profile = self._selected_profile()
        if profile is None:
            self.reference_detail_var.set(
                "No reference selected. Create one in the Speaker profiles tab."
            )
            return
        summary = profile.summary()
        lines = [
            f"Subject: {profile.display_name}",
            f"Enrolment samples: {summary['trusted_sample_count']} trusted",
            f"Total reference speech: {profile.total_reference_seconds:.1f} sec",
        ]
        if profile.embedding_model:
            lines.append(f"Embedding model: {profile.embedding_model.describe()}")
        else:
            lines.append(
                "Embedding model: unknown (legacy import) — comparison needs "
                "explicit confirmation."
            )
        minimum = active().min_reference_seconds
        if profile.total_reference_seconds < minimum:
            lines.append(
                "Warning: this profile holds less reference speech than the "
                f"recommended {minimum:.0f}s."
            )
        self.reference_detail_var.set("\n".join(lines))

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

    def _questioned_embedding(self, source: Path):
        """Embed the selected questioned speech; returns (vector, seconds, quality)."""
        embedder = self._get_embedder()
        wav, _digest, temporary = prepare_source(source, progress=self._status)
        try:
            spans = self._prepare_spans(wav)
            if not spans:
                return None, 0.0, None
            reports = [analyse_span(wav, start, end) for start, end in spans]
            merged = combine(reports)
            # Embed the longest usable span: one clean stretch beats an average
            # over everything the speaker said, including their weakest audio.
            usable = [
                (start, end)
                for (start, end), report in zip(spans, reports)
                if report.usable
            ]
            target = max(usable or spans, key=lambda s: s[1] - s[0])
            self._status("Measuring the questioned voice…")
            vector = embedder.embed_span(wav, target[0], target[1])
            return vector, merged.voiced_seconds, merged
        finally:
            if temporary:
                try:
                    Path(wav).unlink()
                except OSError:
                    pass

    def _compare_worker(self, profile: SpeakerProfile, source: Path) -> None:
        try:
            vector, seconds, merged = self._questioned_embedding(source)
            if not vector:
                self._status("No usable speech was found in the questioned selection.")
                return
            result = compare_embedding_to_profile(
                vector,
                profile,
                questioned_seconds=seconds,
                questioned_quality=merged.assessment if merged else "Insufficient",
                questioned_warnings=merged.warnings if merged else None,
                questioned_label=source.name,
                questioned_model=bundled_model_identity(),
                allow_unverified_model=self.allow_unverified_var.get(),
            )
            self._last_result = result
            self._comparisons.append(result)
            self._show(result.format_lines())
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
            vector, seconds, merged = self._questioned_embedding(source)
            if not vector:
                self._status("No usable speech was found in the questioned selection.")
                return
            result = search_gallery(
                vector,
                self._profiles,
                questioned_seconds=seconds,
                questioned_label=source.name,
                questioned_model=bundled_model_identity(),
            )
            lines = [
                f"Questioned speaker: {source.name}",
                f"Questioned speech: {seconds:.1f} sec "
                f"(quality: {merged.assessment if merged else 'unknown'})",
                "",
            ] + result.summary_lines()
            self._show(lines)
            self._status("Gallery search complete.")
        except Exception as exc:  # noqa: BLE001 - surfaced to the operator
            self._status(friendly_error(exc))

    # -- Output -------------------------------------------------------------

    def _show(self, lines: List[str]) -> None:
        set_readonly_text(self.result_text, "\n".join(lines) + "\n\n" + DISCLAIMER)

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
