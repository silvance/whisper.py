"""The Speaker profiles tab: build known-subject reference voices from audio.

The deliberate enrolment workflow: create a subject, point at historical
recordings that contain them, choose which speech is theirs (the whole file, a
diarized speaker, or explicit time ranges), and store trusted reference samples.
No transcription pass is required to get from *historical recordings* to a
*known-actor profile*.

Automatically learned samples (from transcript corrections elsewhere in the app)
appear here too, but as pending items an operator reviews and promotes - they are
never silently folded into a trusted reference.
"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Callable, List, Optional, Sequence, Tuple

from ..enrollment import (
    enroll_from_wav,
    parse_time_ranges,
    prepare_source,
    spans_for_speaker,
    speaker_totals,
)
from ..hashing import short
from ..speaker_profiles import (
    ProfileError,
    SpeakerProfile,
    delete_speaker_profile,
    list_speaker_profiles,
    load_profile_file,
    save_speaker_profile,
)
from ..transcription import AUDIO_EXTENSIONS
from ..voiceprints import SpeakerEmbedder
from .errors import friendly_error
from .widgets import bind_wheel, scrollable_body

# Enrolment modes offered when a recording is added.
MODE_WHOLE = "whole"
MODE_DIARIZE = "diarize"
MODE_RANGES = "ranges"


class SpeakerProfilesTab:
    """Builds and drives the Speaker profiles tab inside ``parent``."""

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
        self._profiles: List[SpeakerProfile] = []
        self._selected: Optional[SpeakerProfile] = None
        self._embedder: Optional[SpeakerEmbedder] = None
        self.status_var = tk.StringVar(value="Idle")
        self.summary_var = tk.StringVar(value="No subject selected.")
        self._build()
        self.refresh()

    # -- UI ----------------------------------------------------------------

    def _build(self) -> None:
        canvas, container = scrollable_body(self.parent)

        ttk.Label(
            container,
            text="Known-subject reference voices",
            font=("", 12, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            container,
            text=(
                "Create a subject, then add historical recordings you know contain "
                "them. Whispers enrols several trusted samples per recording and "
                "records where each came from."
            ),
            wraplength=620,
            justify="left",
            font=("", 8),
        ).pack(anchor="w", pady=(0, 8))

        # --- Subjects ------------------------------------------------------
        subjects = ttk.LabelFrame(container, text="Subjects", padding=8)
        subjects.pack(fill="x")
        self.subject_tree = ttk.Treeview(
            subjects,
            columns=("samples", "speech", "model"),
            show="tree headings",
            height=6,
        )
        self.subject_tree.heading("#0", text="Subject")
        self.subject_tree.heading("samples", text="Samples")
        self.subject_tree.heading("speech", text="Reference speech")
        self.subject_tree.heading("model", text="Embedding model")
        self.subject_tree.column("#0", width=180)
        self.subject_tree.column("samples", width=110, anchor="center")
        self.subject_tree.column("speech", width=120, anchor="center")
        self.subject_tree.column("model", width=220)
        self.subject_tree.pack(fill="x")
        self.subject_tree.bind("<<TreeviewSelect>>", lambda _e: self._on_select())

        buttons = ttk.Frame(subjects)
        buttons.pack(fill="x", pady=(6, 0))
        ttk.Button(buttons, text="New subject…", command=self._new_subject).pack(
            side="left"
        )
        ttk.Button(buttons, text="Import…", command=self._import_profile).pack(
            side="left", padx=(6, 0)
        )
        ttk.Button(buttons, text="Export…", command=self._export_profile).pack(
            side="left", padx=(6, 0)
        )
        ttk.Button(buttons, text="Delete", command=self._delete_profile).pack(
            side="left", padx=(6, 0)
        )
        ttk.Button(buttons, text="Refresh", command=self.refresh).pack(
            side="left", padx=(6, 0)
        )

        # --- Selected subject ---------------------------------------------
        detail = ttk.LabelFrame(container, text="Selected subject", padding=8)
        detail.pack(fill="both", expand=True, pady=(10, 0))
        ttk.Label(
            detail, textvariable=self.summary_var, wraplength=620, justify="left"
        ).pack(anchor="w")

        add_row = ttk.Frame(detail)
        add_row.pack(fill="x", pady=(8, 4))
        ttk.Button(
            add_row,
            text="Add historical recording…",
            command=self._add_recording,
        ).pack(side="left")
        ttk.Button(add_row, text="Approve sample", command=self._approve).pack(
            side="left", padx=(6, 0)
        )
        ttk.Button(add_row, text="Remove sample", command=self._remove).pack(
            side="left", padx=(6, 0)
        )

        self.sample_tree = ttk.Treeview(
            detail,
            columns=("type", "state", "source", "span", "speech", "quality"),
            show="headings",
            height=8,
        )
        for column, heading, width in (
            ("type", "Class", 90),
            ("state", "State", 90),
            ("source", "Source", 190),
            ("span", "Span", 110),
            ("speech", "Speech", 80),
            ("quality", "Quality", 90),
        ):
            self.sample_tree.heading(column, text=heading)
            self.sample_tree.column(column, width=width, anchor="w")
        self.sample_tree.pack(fill="both", expand=True)

        ttk.Label(container, textvariable=self.status_var, wraplength=620).pack(
            anchor="w", pady=(8, 0)
        )
        bind_wheel(canvas, container)

    # -- Data --------------------------------------------------------------

    def refresh(self) -> None:
        """Reload the stored subjects and repaint the list."""
        self._profiles = list_speaker_profiles()
        self.subject_tree.delete(*self.subject_tree.get_children())
        for profile in self._profiles:
            summary = profile.summary()
            model = (
                profile.embedding_model.describe()
                if profile.embedding_model
                else "unknown (legacy)"
            )
            self.subject_tree.insert(
                "",
                "end",
                iid=profile.subject_id,
                text=profile.display_name,
                values=(
                    f"{summary['trusted_sample_count']} trusted / "
                    f"{summary['pending_sample_count']} pending",
                    f"{profile.total_reference_seconds:.1f}s",
                    model,
                ),
            )
        self._on_select()

    def _on_select(self) -> None:
        selection = self.subject_tree.selection()
        self._selected = None
        if selection:
            for profile in self._profiles:
                if profile.subject_id == selection[0]:
                    self._selected = profile
                    break
        self._render_detail()

    def _render_detail(self) -> None:
        self.sample_tree.delete(*self.sample_tree.get_children())
        profile = self._selected
        if profile is None:
            self.summary_var.set("No subject selected.")
            return
        summary = profile.summary()
        lines = [
            f"{profile.display_name}  ({profile.subject_id})",
            f"Trusted samples: {summary['trusted_sample_count']}   "
            f"Pending review: {summary['pending_sample_count']}   "
            f"Reference speech: {profile.total_reference_seconds:.1f}s",
            f"Sources: {', '.join(profile.source_files()) or 'none recorded'}",
        ]
        if profile.embedding_model:
            lines.append(
                f"Embedding model: {profile.embedding_model.name} "
                f"(sha256 {short(profile.embedding_model.sha256)}, "
                f"dim {profile.embedding_model.vector_dimension})"
            )
        else:
            lines.append(
                "Embedding model: unknown — imported from an older format, so "
                "model compatibility cannot be proven."
            )
        self.summary_var.set("\n".join(lines))

        for sample in profile.samples:
            span = (
                f"{sample.source_start:.0f}-{sample.source_end:.0f}s"
                if sample.source_start is not None and sample.source_end is not None
                else "—"
            )
            self.sample_tree.insert(
                "",
                "end",
                iid=sample.sample_id,
                values=(
                    sample.sample_type,
                    "trusted" if sample.is_trusted else "needs review",
                    sample.source_filename or "—",
                    span,
                    f"{sample.speech_duration:.1f}s",
                    str(sample.quality.get("assessment", "—")),
                ),
            )

    # -- Subject management -------------------------------------------------

    def _new_subject(self) -> None:
        name = simpledialog.askstring(
            "New subject", "Subject / display name:", parent=self.root
        )
        if not name or not name.strip():
            return
        profile = SpeakerProfile(display_name=name.strip())
        try:
            save_speaker_profile(profile)
        except ProfileError as exc:
            self._status(friendly_error(exc))
            return
        self.refresh()
        self.subject_tree.selection_set(profile.subject_id)
        self._status(f"Created subject '{profile.display_name}'.")

    def _delete_profile(self) -> None:
        profile = self._selected
        if profile is None:
            self._status("Select a subject first.")
            return
        if not messagebox.askyesno(
            "Delete subject",
            f"Delete '{profile.display_name}' and its enrolled voice samples?",
            parent=self.root,
        ):
            return
        delete_speaker_profile(profile)
        self.refresh()
        self._status("Subject deleted.")

    def _export_profile(self) -> None:
        profile = self._selected
        if profile is None:
            self._status("Select a subject first.")
            return
        from ..speaker_profiles import SPEAKER_PROFILE_SUFFIX

        path = filedialog.asksaveasfilename(
            title="Export speaker profile",
            defaultextension=SPEAKER_PROFILE_SUFFIX,
            initialfile=f"{profile.display_name}{SPEAKER_PROFILE_SUFFIX}",
            filetypes=[("Whispers speaker profile", f"*{SPEAKER_PROFILE_SUFFIX}")],
        )
        if not path:
            return
        try:
            save_speaker_profile(profile, path)
            self._status(f"Exported to {Path(path).name}")
        except ProfileError as exc:
            self._status(friendly_error(exc))

    def _import_profile(self) -> None:
        path = filedialog.askopenfilename(
            title="Import speaker profile",
            filetypes=[("Whispers profile", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            imported = load_profile_file(path)
            # import_warnings describe *this* read and are not persisted, so
            # collect them before saving - after the reload they are gone, and
            # an operator would never learn that samples were dropped.
            issues = [
                (profile.display_name, warning)
                for profile in imported
                for warning in profile.import_warnings
            ]
            for profile in imported:
                save_speaker_profile(profile)
        except ProfileError as exc:
            self._status(friendly_error(exc))
            return
        self.refresh()
        legacy = sum(1 for p in imported if p.is_legacy)
        note = (
            f" {legacy} imported from an older format (embedding model unknown)."
            if legacy
            else ""
        )
        self._status(f"Imported {len(imported)} subject(s).{note}")
        if issues:
            self._report_import_issues(Path(path).name, issues)

    def _report_import_issues(
        self, filename: str, issues: Sequence[Tuple[str, str]]
    ) -> None:
        """Tell the operator exactly what the import dropped or demoted.

        Data being safely quarantined is not the same as the operator knowing
        it happened: a profile that arrives two samples lighter, or with
        reference material demoted to pending, changes what any later
        comparison means.
        """
        lines = [
            f"{filename} was imported, but not everything in it could be "
            "trusted as it stood:",
            "",
        ]
        for subject, warning in issues:
            lines.append(f"  {subject}: {warning}")
        lines += [
            "",
            "Dropped samples held vectors that could not be compared. Demoted "
            "samples are kept as pending and can be approved under "
            "'Selected subject' after review.",
        ]
        messagebox.showwarning(
            "Imported with changes", "\n".join(lines), parent=self.root
        )

    # -- Sample review ------------------------------------------------------

    def _selected_sample_id(self) -> Optional[str]:
        selection = self.sample_tree.selection()
        return selection[0] if selection else None

    def _approve(self) -> None:
        profile, sample_id = self._selected, self._selected_sample_id()
        if profile is None or sample_id is None:
            self._status("Select a sample to approve.")
            return
        if profile.approve_sample(sample_id) and self._save(profile):
            self._render_detail()
            self._status("Sample promoted into the trusted reference.")

    def _remove(self) -> None:
        profile, sample_id = self._selected, self._selected_sample_id()
        if profile is None or sample_id is None:
            self._status("Select a sample to remove.")
            return
        if profile.remove_sample(sample_id) and self._save(profile):
            self.refresh()
            self._status("Sample removed.")

    def _save(self, profile: SpeakerProfile) -> bool:
        try:
            save_speaker_profile(profile)
            return True
        except ProfileError as exc:
            # Losing operational work silently is not acceptable; surface it.
            messagebox.showerror("Could not save", str(exc), parent=self.root)
            self._status(friendly_error(exc))
            return False

    # -- Enrolment ----------------------------------------------------------

    def _add_recording(self) -> None:
        profile = self._selected
        if profile is None:
            self._status("Select or create a subject first.")
            return
        patterns = " ".join(f"*{ext}" for ext in AUDIO_EXTENSIONS)
        path = filedialog.askopenfilename(
            title="Historical recording containing this subject",
            filetypes=[("Audio/Video", patterns), ("All files", "*.*")],
        )
        if not path:
            return
        choice = self._ask_mode()
        if choice is None:
            return
        mode, ranges_text = choice
        spans: Optional[List[Tuple[float, float]]] = None
        if mode == MODE_RANGES:
            try:
                spans = parse_time_ranges(ranges_text)
            except ValueError as exc:
                self._status(f"Couldn't read the time ranges: {exc}")
                return
        threading.Thread(
            target=self._enroll_worker,
            args=(profile, Path(path), mode, spans),
            daemon=True,
        ).start()

    def _ask_mode(self) -> Optional[Tuple[str, str]]:
        """Ask which speech in the recording belongs to the subject."""
        win = tk.Toplevel(self.root)
        win.title("Which speech is the subject?")
        win.transient(self.root)  # type: ignore[call-overload]
        win.grab_set()
        mode_var = tk.StringVar(value=MODE_WHOLE)
        ranges_var = tk.StringVar(value="")
        chosen: dict = {"value": None}

        frame = ttk.Frame(win, padding=12)
        frame.pack(fill="both", expand=True)
        ttk.Radiobutton(
            frame,
            text="The whole recording is this subject",
            variable=mode_var,
            value=MODE_WHOLE,
        ).pack(anchor="w")
        ttk.Radiobutton(
            frame,
            text="Several people — separate speakers and let me pick",
            variable=mode_var,
            value=MODE_DIARIZE,
        ).pack(anchor="w")
        ttk.Radiobutton(
            frame,
            text="I'll give the time ranges",
            variable=mode_var,
            value=MODE_RANGES,
        ).pack(anchor="w")
        ttk.Entry(frame, textvariable=ranges_var, width=44).pack(
            anchor="w", padx=(24, 0), pady=(2, 0)
        )
        ttk.Label(
            frame,
            text="e.g. 0:10-0:45, 1:20-2:00",
            font=("", 8),
        ).pack(anchor="w", padx=(24, 0))

        def _ok() -> None:
            chosen["value"] = (mode_var.get(), ranges_var.get())
            win.destroy()

        row = ttk.Frame(frame)
        row.pack(fill="x", pady=(10, 0))
        ttk.Button(row, text="Cancel", command=win.destroy).pack(side="right")
        ttk.Button(row, text="Continue", command=_ok).pack(side="right", padx=(0, 6))
        win.wait_window()
        return chosen["value"]

    def _get_embedder(self) -> SpeakerEmbedder:
        """Load the speaker-embedding model once (sherpa loads on first use)."""
        if self._embedder is None:
            self._embedder = SpeakerEmbedder()
        return self._embedder

    def _enroll_worker(
        self,
        profile: SpeakerProfile,
        source: Path,
        mode: str,
        spans: Optional[List[Tuple[float, float]]],
    ) -> None:
        wav = None
        temporary = False
        try:
            self._status(f"Preparing {source.name}…")
            embedder = self._get_embedder()
            wav, digest, temporary = prepare_source(source, progress=self._status)
            if mode == MODE_DIARIZE:
                spans = self._diarize_and_pick(wav)
                if spans is None:
                    self._status("Enrolment cancelled.")
                    return
            if spans is None:
                spans = [(0.0, _duration(wav))]
            result = enroll_from_wav(
                profile,
                wav,
                spans,
                embedder,
                source_filename=source.name,
                source_sha256=digest,
                progress=self._status,
            )
            if result.added:
                self._save(profile)
            self.root.after(0, self.refresh)
            quality = result.quality.assessment if result.quality else "n/a"
            message = (
                f"Enrolled {result.added_count} sample(s), "
                f"{result.added_seconds:.1f}s of speech (quality: {quality})."
            )
            if result.skipped:
                message += f" Skipped {len(result.skipped)}: {result.skipped[0]}"
            self._status(message)
        except Exception as exc:  # noqa: BLE001 - surfaced to the operator
            self._status(friendly_error(exc))
        finally:
            if temporary and wav is not None:
                try:
                    Path(wav).unlink()
                except OSError:
                    pass

    def _diarize_and_pick(self, wav: Path) -> Optional[List[Tuple[float, float]]]:
        """Diarize the recording and let the operator choose the subject's cluster."""
        from ..diarization import diarize

        self._status("Separating speakers…")
        segments = diarize(wav, progress=self._status)
        totals = speaker_totals(segments)
        if not totals:
            self._status("No speakers were found in that recording.")
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
        speaker = picked["value"]
        if not speaker:
            return None
        return spans_for_speaker(segments, speaker)

    def _choose_cluster(self, totals: Sequence[Tuple[str, float]]) -> Optional[str]:
        """Modal picker listing each detected speaker by how long they talk."""
        win = tk.Toplevel(self.root)
        win.title("Which speaker is the subject?")
        win.transient(self.root)  # type: ignore[call-overload]
        win.grab_set()
        labels = [f"{name} — {seconds:.0f}s of speech" for name, seconds in totals]
        var = tk.StringVar(value=labels[0])
        chosen: dict = {"value": None}
        frame = ttk.Frame(win, padding=12)
        frame.pack(fill="both", expand=True)
        ttk.Label(
            frame,
            text=(
                "Pick the speaker that is the subject. The longest talker is "
                "usually the subject of a targeted recording, but listen first "
                "if you are unsure."
            ),
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

    # -- Shared hooks -------------------------------------------------------

    def _status(self, message: str) -> None:
        self.root.after(0, lambda: self.status_var.set(message))

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
