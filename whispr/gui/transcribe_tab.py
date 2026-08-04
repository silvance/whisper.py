"""The Transcribe tab: settings, the transcription run, and the transcript pane.

Wraps :mod:`whispr.transcription` (faster-whisper) and
:mod:`whispr.diarization`. The interactive transcript (speaker/word corrections)
is delegated to :class:`whispr.gui.transcript_view.TranscriptView`; this module
owns the settings UI and drives the background run.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, simpledialog, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Callable, Dict, List, Optional, Tuple

from ..diarization import assign_speakers, diarize
from ..export import transcript_to_docx
from ..playback import PlaybackError, SegmentPlayer, playback_available
from ..profiles import (
    Profile,
    delete_profile,
    list_profiles,
    load_profile,
    save_profile,
)
from ..project import PROJECT_SUFFIX, load_project, save_project
from ..resources import bundled_models
from ..transcription import (
    AUDIO_EXTENSIONS,
    MODEL_SIZES,
    CancelledError,
    TranscriptionResult,
    convert_to_wav,
    is_video,
    transcribe_audio,
)
from ..voiceprints import SpeakerEmbedder, enroll_spans, recognize
from .errors import friendly_error
from .transcript_view import TranscriptView
from .widgets import (
    CollapsibleSection,
    append_line,
    bind_wheel,
    register_drop,
    scrollable_body,
)

# A handful of common languages for the dropdown; "Auto" lets Whisper detect.
COMMON_LANGUAGES = [
    "Auto",
    "en",
    "es",
    "fr",
    "de",
    "it",
    "pt",
    "nl",
    "ru",
    "ar",
    "zh",
    "ja",
    "ko",
    "hi",
]

# Diarization engine choices: dropdown label -> diarize() backend. "Auto" uses
# pyannote when it's available, else sherpa. pyannote is best on hard/overlapping
# audio; sherpa is lighter and faster and fine for clean audio.
ENGINE_CHOICES = {
    "Auto (pyannote if available)": "auto",
    "pyannote - most accurate": "pyannote",
    "sherpa - faster, for clean audio": "sherpa",
}
ENGINE_LABELS = list(ENGINE_CHOICES)


class TranscribeTab:
    """Builds and drives the Transcribe tab inside ``parent``."""

    def __init__(
        self,
        parent: ttk.Frame,
        root: tk.Misc,
        cancel_event: threading.Event,
        on_cancel: Callable[[], None],
        *,
        dnd_ok: bool,
    ) -> None:
        self.parent = parent
        self.root = root
        self._cancel_event = cancel_event
        self._on_cancel = on_cancel
        self._dnd_ok = dnd_ok

        # Bundled (offline) models take priority so the app works air-gapped.
        # Prefer the fast English base.en, then small, else the first bundled.
        self._bundled_models = bundled_models()
        default_model = "base.en"
        if self._bundled_models:
            for preferred in ("base.en", "small"):
                if preferred in self._bundled_models:
                    default_model = preferred
                    break
            else:
                default_model = next(iter(self._bundled_models))

        self.input_file_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()
        self.write_output_var = tk.BooleanVar(value=True)
        self.model_var = tk.StringVar(value=default_model)
        # Live note under the Model box: whether the current pick is in this build.
        self.model_status_var = tk.StringVar(value="")
        self.task_var = tk.StringVar(value="transcribe")
        self.language_var = tk.StringVar(value="Auto")
        self.vad_var = tk.BooleanVar(value=True)
        self.convert_video_var = tk.BooleanVar(value=True)
        self.diarize_var = tk.BooleanVar(value=False)
        self.engine_var = tk.StringVar(value=ENGINE_LABELS[0])
        self.num_speakers_var = tk.StringVar(value="")
        # Optional per-speaker names, created to match the speaker count and
        # applied to the diarized transcript (Speaker 1 -> first speaker, etc.).
        self.speaker_name_vars: List[tk.StringVar] = []
        self.sensitivity_var = tk.StringVar(value="0.5")
        self.srt_var = tk.BooleanVar(value=False)
        # Put a blank line between segments in the transcript (and saved .txt).
        self.blank_lines_var = tk.BooleanVar(value=True)
        # Optional vocabulary hint (names/jargon) to prime the decoder.
        self.vocab_var = tk.StringVar(value="")
        # Colour low-confidence words/segments so they can be verified.
        self.highlight_conf_var = tk.BooleanVar(value=False)
        self.progress_label_var = tk.StringVar(value="Idle")

        # State for the last result, so speakers can be renamed after a run. The
        # transcript view shares these (same objects) and mutates them in place.
        self._result: Optional[TranscriptionResult] = None
        self._result_source: Optional[Path] = None
        self._result_outdir: Optional[Path] = None
        self._speaker_names: Dict[str, str] = {}

        # Optional batch queue; when non-empty, Run transcribes all of these
        # instead of the single "Audio / video file" above.
        self._batch_files: List[Path] = []
        self.batch_files_var = tk.StringVar(value="")

        # Active operation profile (saved settings + learned speaker voiceprints),
        # and whether to recognise/learn voices for it. See whispr.profiles /
        # whispr.voiceprints.
        self.profile_var = tk.StringVar(value="")
        self.learn_var = tk.BooleanVar(value=True)
        self._profile: Optional[Profile] = None
        # Names recognised from voiceprints on the last diarized run (voice::Name).
        self._recognized_names: Dict[str, str] = {}
        # A 16 kHz mono copy of the last diarized audio, kept so corrections can
        # enrol voiceprints after the run; removed on the next run and on close.
        self._session_wav: Optional[Path] = None
        # Lazily-built speaker-embedding extractor; False once it has failed to
        # load (e.g. sherpa-onnx/model missing) so we don't retry every edit.
        self._embedder: "Optional[SpeakerEmbedder] | bool" = None

        # Offline segment playback (click a line to re-listen). Disabled cleanly
        # when neither ffmpeg nor an OS player is available.
        self._player = SegmentPlayer()
        self._playback_ok = playback_available()

        self._build()

    # -- UI construction ---------------------------------------------------

    def _build(self) -> None:
        transcribe_canvas, container = scrollable_body(self.parent)

        # Collapsible settings sections (hidden as a group when a run starts).
        self._setting_sections: List[CollapsibleSection] = []

        # --- Input & output ------------------------------------------------
        io_section = CollapsibleSection(container, "Input & output")
        io_section.pack(fill="x")
        self._setting_sections.append(io_section)
        io_frame = io_section.body
        io_frame.columnconfigure(1, weight=1)

        ttk.Label(io_frame, text="Audio / video file").grid(
            row=0, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Entry(io_frame, textvariable=self.input_file_var).grid(
            row=0, column=1, sticky="ew", pady=4
        )
        ttk.Button(io_frame, text="Browse…", command=self.choose_file).grid(
            row=0, column=2, padx=(8, 0), pady=4
        )

        ttk.Checkbutton(
            io_frame,
            text="Save transcript to a folder",
            variable=self.write_output_var,
            command=self._update_output_state,
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 2))

        ttk.Label(io_frame, text="Output folder").grid(
            row=2, column=0, sticky="w", padx=(0, 8), pady=4
        )
        self.output_dir_entry = ttk.Entry(io_frame, textvariable=self.output_dir_var)
        self.output_dir_entry.grid(row=2, column=1, sticky="ew", pady=4)
        self.output_dir_button = ttk.Button(
            io_frame, text="Select…", command=self.choose_output_dir
        )
        self.output_dir_button.grid(row=2, column=2, padx=(8, 0), pady=4)

        # Batch queue (optional): Run transcribes all of these instead of the
        # single file above. Outputs go to the chosen folder, else beside each
        # source so a multi-file run never loses results.
        ttk.Label(io_frame, text="Batch (optional)").grid(
            row=3, column=0, sticky="w", padx=(0, 8), pady=4
        )
        batch_row = ttk.Frame(io_frame)
        batch_row.grid(row=3, column=1, columnspan=2, sticky="w", pady=4)
        ttk.Button(batch_row, text="Add files…", command=self._add_batch_files).pack(
            side="left"
        )
        ttk.Button(batch_row, text="Clear", command=self._clear_batch_files).pack(
            side="left", padx=(8, 0)
        )
        ttk.Label(
            io_frame,
            textvariable=self.batch_files_var,
            wraplength=420,
            justify="left",
        ).grid(row=4, column=1, columnspan=2, sticky="w")

        # --- Model & language ---------------------------------------------
        model_section = CollapsibleSection(container, "Model & language")
        model_section.pack(fill="x", pady=(8, 0))
        self._setting_sections.append(model_section)
        model_frame = model_section.body
        model_frame.columnconfigure(1, weight=1)

        # A bundled model name, a size name, or a path to a local CTranslate2
        # model. Bundled (offline) models are listed first.
        model_values = list(self._bundled_models) + [
            size for size in MODEL_SIZES if size not in self._bundled_models
        ]
        ttk.Label(model_frame, text="Model").grid(
            row=0, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Combobox(
            model_frame, textvariable=self.model_var, values=model_values
        ).grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Button(model_frame, text="Browse…", command=self.choose_model_dir).grid(
            row=0, column=2, padx=(8, 0), pady=4
        )
        # A live note telling the operator whether the chosen model is actually in
        # this (offline) build - the dropdown lists every size, but only bundled
        # ones work air-gapped, so this flags a pick that would fail before Run.
        ttk.Label(model_frame, textvariable=self.model_status_var, font=("", 8)).grid(
            row=1, column=1, columnspan=2, sticky="w"
        )
        self.model_var.trace_add("write", self._update_model_status)
        self._update_model_status()

        ttk.Label(model_frame, text="Task").grid(
            row=2, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Combobox(
            model_frame,
            textvariable=self.task_var,
            values=["transcribe", "translate"],
            state="readonly",
            width=16,
        ).grid(row=2, column=1, sticky="w", pady=4)

        ttk.Label(model_frame, text="Language").grid(
            row=3, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Combobox(
            model_frame,
            textvariable=self.language_var,
            values=COMMON_LANGUAGES,
            width=16,
        ).grid(row=3, column=1, sticky="w", pady=4)

        ttk.Label(model_frame, text="Custom words").grid(
            row=4, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Entry(model_frame, textvariable=self.vocab_var).grid(
            row=4, column=1, columnspan=2, sticky="ew", pady=4
        )
        ttk.Label(
            model_frame,
            text="Names, places, jargon or callsigns to expect (improves accuracy).",
            font=("", 8),
        ).grid(row=5, column=1, columnspan=2, sticky="w")

        # --- Options -------------------------------------------------------
        opt_section = CollapsibleSection(container, "Options")
        opt_section.pack(fill="x", pady=(8, 0))
        self._setting_sections.append(opt_section)
        opt_frame = opt_section.body
        ttk.Checkbutton(
            opt_frame,
            text="Skip silence (voice activity detection)",
            variable=self.vad_var,
        ).grid(row=0, column=0, sticky="w", pady=2)
        ttk.Checkbutton(
            opt_frame,
            text="Convert video to WAV first (ffmpeg)",
            variable=self.convert_video_var,
        ).grid(row=1, column=0, sticky="w", pady=2)
        ttk.Checkbutton(
            opt_frame, text="Also save .srt subtitles", variable=self.srt_var
        ).grid(row=2, column=0, sticky="w", pady=2)
        ttk.Checkbutton(
            opt_frame,
            text="Blank line between segments (easier to read / paste)",
            variable=self.blank_lines_var,
            command=self._rerender_transcript,
        ).grid(row=3, column=0, sticky="w", pady=2)
        ttk.Checkbutton(
            opt_frame,
            text="Highlight low-confidence words (verify these)",
            variable=self.highlight_conf_var,
            command=self._rerender_transcript,
        ).grid(row=4, column=0, sticky="w", pady=2)

        # --- Speakers ------------------------------------------------------
        spk_section = CollapsibleSection(container, "Speakers")
        spk_section.pack(fill="x", pady=(8, 0))
        self._setting_sections.append(spk_section)
        spk_frame = spk_section.body
        spk_frame.columnconfigure(1, weight=1)
        ttk.Checkbutton(
            spk_frame,
            text="Identify speakers (diarization)",
            variable=self.diarize_var,
            command=self._update_speaker_state,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))

        ttk.Label(spk_frame, text="Engine").grid(
            row=1, column=0, sticky="w", padx=(0, 8), pady=4
        )
        self.engine_combo = ttk.Combobox(
            spk_frame,
            textvariable=self.engine_var,
            values=ENGINE_LABELS,
            state="readonly",
            width=32,
        )
        self.engine_combo.grid(row=1, column=1, sticky="w", pady=4)

        ttk.Label(spk_frame, text="Number of speakers (blank = auto)").grid(
            row=2, column=0, sticky="w", padx=(0, 8), pady=4
        )
        self.num_speakers_entry = ttk.Entry(
            spk_frame, textvariable=self.num_speakers_var, width=10
        )
        self.num_speakers_entry.grid(row=2, column=1, sticky="w", pady=4)
        # Entering a count reveals a name field per speaker (filled in below).
        self.num_speakers_var.trace_add("write", self._on_num_speakers_changed)

        ttk.Label(
            spk_frame, text="Sensitivity (higher = fewer speakers; sherpa only)"
        ).grid(row=3, column=0, sticky="w", padx=(0, 8), pady=4)
        self.sensitivity_entry = ttk.Entry(
            spk_frame, textvariable=self.sensitivity_var, width=10
        )
        self.sensitivity_entry.grid(row=3, column=1, sticky="w", pady=4)

        # Dynamic per-speaker name fields, rebuilt when the count changes.
        self.speaker_names_frame = ttk.Frame(spk_frame)
        self.speaker_names_frame.grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=(2, 0)
        )

        ttk.Label(
            spk_frame,
            text=(
                "Tip: if you know how many people are in the recording, enter it "
                "above and (optionally) name them. In the Transcript, click a "
                "[speaker] tag to rename or move the whole line, click a single "
                "word to move just that word (or from it onward), or highlight a "
                "run of words and drag it onto another speaker's line to move it "
                "there (right-click still offers the same as a menu)."
            ),
            wraplength=420,
            justify="left",
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(6, 0))

        # --- Profile (learn speakers) -------------------------------------
        profile_section = CollapsibleSection(container, "Profile (learn speakers)")
        profile_section.pack(fill="x", pady=(8, 0))
        self._setting_sections.append(profile_section)
        prof_frame = profile_section.body
        prof_frame.columnconfigure(1, weight=1)

        ttk.Label(prof_frame, text="Profile").grid(
            row=0, column=0, sticky="w", padx=(0, 8), pady=4
        )
        self.profile_combo = ttk.Combobox(
            prof_frame,
            textvariable=self.profile_var,
            values=[""] + list_profiles(),
            state="readonly",
            width=26,
        )
        self.profile_combo.grid(row=0, column=1, sticky="w", pady=4)
        self.profile_combo.bind(
            "<<ComboboxSelected>>", lambda _e: self._on_profile_selected()
        )
        prof_buttons = ttk.Frame(prof_frame)
        prof_buttons.grid(row=0, column=2, sticky="e", pady=4)
        ttk.Button(prof_buttons, text="New…", command=self._new_profile).pack(
            side="left"
        )
        ttk.Button(prof_buttons, text="Save", command=self._save_profile_settings).pack(
            side="left", padx=(6, 0)
        )
        ttk.Button(prof_buttons, text="Delete", command=self._delete_profile).pack(
            side="left", padx=(6, 0)
        )

        ttk.Checkbutton(
            prof_frame,
            text="Recognise & learn speaker voices for this profile",
            variable=self.learn_var,
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 0))

        ttk.Label(
            prof_frame,
            text=(
                "Pick a profile per operation. With diarization on, the app "
                "matches each speaker against the voices this profile has learned "
                "and labels them automatically — and every correction you make "
                "(rename or move a line to a named speaker) teaches it that voice, "
                "so the next recording gets better. Voiceprints only sharpen "
                "labelling; they don't retrain the transcription model."
            ),
            wraplength=460,
            justify="left",
            font=("", 8),
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 0))

        # --- Run + progress -----------------------------------------------
        run_frame = ttk.Frame(container)
        run_frame.pack(fill="x", pady=(12, 0))
        run_frame.columnconfigure(2, weight=1)
        self.run_button = ttk.Button(run_frame, text="Run", command=self.run_in_thread)
        self.run_button.grid(row=0, column=0, sticky="w")
        self.cancel_button = ttk.Button(
            run_frame, text="Cancel", command=self._on_cancel, state="disabled"
        )
        self.cancel_button.grid(row=0, column=1, sticky="w", padx=(8, 0))
        self.progress_bar = ttk.Progressbar(run_frame, mode="indeterminate")
        self.progress_bar.grid(row=0, column=2, sticky="ew", padx=(10, 0))
        self.toggle_settings_button = ttk.Button(
            run_frame, text="Hide settings", command=self._toggle_all_settings
        )
        self.toggle_settings_button.grid(row=0, column=3, padx=(10, 0))
        ttk.Label(run_frame, textvariable=self.progress_label_var).grid(
            row=1, column=0, columnspan=4, sticky="w", pady=(4, 0)
        )

        # --- Output tabs ---------------------------------------------------
        tabs = ttk.Notebook(container)
        tabs.pack(fill="both", expand=True, pady=(12, 0))
        self.transcript_view = TranscriptView(
            tabs,
            self.root,
            self.blank_lines_var,
            self._save_outputs_if_possible,
            highlight_var=self.highlight_conf_var,
            on_play=self._play_segment if self._playback_ok else None,
            on_enroll=self._enroll_voice,
        )
        self.status = ScrolledText(
            tabs, wrap="word", state="disabled", height=14, font="TkFixedFont"
        )
        tabs.add(self.transcript_view.widget, text="Transcript")
        tabs.add(self.status, text="Status")

        # Find within the transcript (Enter = next, Shift+Enter = previous).
        find_row = ttk.Frame(container)
        find_row.pack(fill="x", pady=(6, 0))
        ttk.Label(find_row, text="Find").pack(side="left")
        self.find_var = tk.StringVar()
        find_entry = ttk.Entry(find_row, textvariable=self.find_var, width=30)
        find_entry.pack(side="left", padx=(6, 0))
        find_entry.bind("<Return>", lambda _e: self._find_next())
        find_entry.bind("<Shift-Return>", lambda _e: self._find_prev())
        ttk.Button(find_row, text="Next", command=self._find_next).pack(
            side="left", padx=(6, 0)
        )
        ttk.Button(find_row, text="Prev", command=self._find_prev).pack(
            side="left", padx=(4, 0)
        )
        self.find_status_var = tk.StringVar()
        ttk.Label(find_row, textvariable=self.find_status_var).pack(
            side="left", padx=(8, 0)
        )

        # Copy / export the transcript (handy for pasting into Word).
        export_row = ttk.Frame(container)
        export_row.pack(fill="x", pady=(6, 0))
        ttk.Button(
            export_row, text="Copy transcript", command=self._copy_transcript
        ).pack(side="left")
        ttk.Button(
            export_row, text="Save as Word…", command=self._save_transcript_docx
        ).pack(side="left", padx=(8, 0))
        ttk.Button(export_row, text="Save project…", command=self._save_project).pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(export_row, text="Open project…", command=self._open_project).pack(
            side="left", padx=(8, 0)
        )
        if self._playback_ok:
            ttk.Button(export_row, text="⏹ Stop audio", command=self._stop_audio).pack(
                side="left", padx=(8, 0)
            )
            ttk.Label(
                export_row,
                text="Ctrl-click a line (or a word) to play its audio.",
                font=("", 8),
            ).pack(side="left", padx=(10, 0))

        # Initialise the enabled/disabled state of dependent fields.
        self._update_output_state()
        self._update_speaker_state()
        # Mouse-wheel scrolls the page (the scrollbar always works regardless).
        bind_wheel(transcribe_canvas, container)
        # Drag an audio/video file onto the transcript pane to load it.
        register_drop(
            self.root, self._dnd_ok, self.transcript_view.widget, self._on_drop_media
        )
        register_drop(self.root, self._dnd_ok, self.status, self._on_drop_media)

    # -- Cancellation ------------------------------------------------------

    def notify_cancelling(self) -> None:
        self.cancel_button.configure(state="disabled")
        append_line(self.status, "Cancelling… (will stop at the next checkpoint)")
        self.progress_label_var.set("Cancelling…")

    # -- Settings state ----------------------------------------------------

    def _toggle_all_settings(self) -> None:
        expand = not any(section.expanded for section in self._setting_sections)
        for section in self._setting_sections:
            section.set_expanded(expand)
        self.toggle_settings_button.configure(
            text="Hide settings" if expand else "Show settings"
        )

    def _collapse_all_settings(self) -> None:
        for section in self._setting_sections:
            section.set_expanded(False)
        self.toggle_settings_button.configure(text="Show settings")

    def _update_output_state(self) -> None:
        state = "normal" if self.write_output_var.get() else "disabled"
        self.output_dir_entry.configure(state=state)
        self.output_dir_button.configure(state=state)

    def _update_speaker_state(self) -> None:
        enabled = self.diarize_var.get()
        state = "normal" if enabled else "disabled"
        self.num_speakers_entry.configure(state=state)
        self.sensitivity_entry.configure(state=state)
        # Comboboxes use "readonly" (selectable but not free-text) when enabled.
        self.engine_combo.configure(state="readonly" if enabled else "disabled")
        self._rebuild_speaker_name_fields()

    def _on_num_speakers_changed(self, *_args: object) -> None:
        self._rebuild_speaker_name_fields()

    def _rebuild_speaker_name_fields(self) -> None:
        """Show one name entry per speaker, matching the requested count.

        Existing names are preserved across rebuilds. Fields only appear while
        diarization is enabled and a positive count is given (capped at 10).
        """
        try:
            count = int(self.num_speakers_var.get().strip())
        except ValueError:
            count = 0
        count = max(0, min(count, 10))

        existing = [var.get() for var in self.speaker_name_vars]
        for child in self.speaker_names_frame.winfo_children():
            child.destroy()
        self.speaker_name_vars = []

        if count <= 0 or not self.diarize_var.get():
            return

        self.speaker_names_frame.columnconfigure(1, weight=1)
        ttk.Label(self.speaker_names_frame, text="Speaker names (optional):").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(4, 2)
        )
        for i in range(count):
            var = tk.StringVar(value=existing[i] if i < len(existing) else "")
            self.speaker_name_vars.append(var)
            ttk.Label(self.speaker_names_frame, text=f"Speaker {i + 1}").grid(
                row=i + 1, column=0, sticky="w", padx=(0, 8), pady=2
            )
            ttk.Entry(self.speaker_names_frame, textvariable=var, width=24).grid(
                row=i + 1, column=1, sticky="w", pady=2
            )

    # -- Thread-safe progress helpers --------------------------------------

    def _set_busy(self, busy: bool, message: Optional[str] = None) -> None:
        def _do() -> None:
            if busy:
                self.run_button.configure(state="disabled")
                self.cancel_button.configure(state="normal")
                # Indeterminate while we don't yet have a measurable fraction
                # (setup, ffmpeg conversion, model loading).
                self.progress_bar.configure(mode="indeterminate")
                self.progress_bar.start(12)
                self.progress_label_var.set(message or "Processing...")
            else:
                self.progress_bar.stop()
                self.progress_bar.configure(mode="determinate")
                self.progress_bar["value"] = 0
                self.progress_label_var.set(message or "Idle")
                self.run_button.configure(state="normal")
                self.cancel_button.configure(state="disabled")

        self.root.after(0, _do)

    def _set_progress(self, fraction: float, message: str) -> None:
        """Show real progress on a determinate bar (fraction is 0..1)."""
        pct = max(0.0, min(1.0, fraction)) * 100.0

        def _do() -> None:
            self.progress_bar.stop()
            self.progress_bar.configure(mode="determinate")
            self.progress_bar["value"] = pct
            self.progress_label_var.set(f"{message} {pct:.0f}%")

        self.root.after(0, _do)

    # -- File pickers ------------------------------------------------------

    def choose_file(self) -> None:
        patterns = " ".join(f"*{ext}" for ext in AUDIO_EXTENSIONS)
        path = filedialog.askopenfilename(
            filetypes=[("Audio/Video", patterns), ("All files", "*.*")]
        )
        if path:
            self.input_file_var.set(path)

    def choose_output_dir(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.output_dir_var.set(path)

    def choose_model_dir(self) -> None:
        path = filedialog.askdirectory(title="Select a CTranslate2 model directory")
        if path:
            self.model_var.set(path)

    # -- Model selection ---------------------------------------------------

    def _selected_model(self) -> str:
        """The chosen model name or path, trimmed of surrounding whitespace."""
        return self.model_var.get().strip()

    def _model_is_available(self, model: str) -> bool:
        """True if ``model`` is bundled in this build or is a local model folder."""
        return model in self._bundled_models or (bool(model) and Path(model).is_dir())

    def _update_model_status(self, *_args: object) -> None:
        """Reflect whether the chosen model is actually usable in this build."""
        model = self._selected_model()
        if not model:
            self.model_status_var.set("")
        elif model in self._bundled_models:
            self.model_status_var.set("✓ in this build (works offline)")
        elif Path(model).is_dir():
            self.model_status_var.set("✓ local model folder")
        elif self._bundled_models:
            # A real bundle, but this size wasn't included - it can't be fetched
            # on an air-gapped machine, so say so before Run rather than after.
            self.model_status_var.set(
                "⚠ not in this build — rebuild including it, or pick a ✓ model "
                "(see Self-test…)"
            )
        else:
            # Running from source with nothing bundled: faster-whisper will fetch it.
            self.model_status_var.set("will be downloaded on first use")

    # -- Run ---------------------------------------------------------------

    def run_in_thread(self) -> None:
        # Collapse the settings so the transcript and progress get the space.
        self._collapse_all_settings()
        self._cancel_event.clear()
        threading.Thread(target=self._run, daemon=True).start()

    def _collect_jobs(self) -> List[Path]:
        """The files to transcribe: the batch queue, else the single input file."""
        if self._batch_files:
            return list(self._batch_files)
        path = self.input_file_var.get().strip()
        return [Path(path)] if path else []

    def _run(self) -> None:
        task = self.task_var.get()
        self._set_busy(
            True, "Translating..." if task == "translate" else "Transcribing..."
        )
        final_status = "Finished"
        # Drop any prior run's kept audio so a correction can only ever enrol
        # against audio from this run's diarized file(s).
        self._clear_session_wav()
        try:
            self.transcript_view.set_result(None, {})
            jobs = self._collect_jobs()
            if not jobs:
                append_line(
                    self.status,
                    "Couldn't find a file. Pick an audio/video file with Browse… "
                    "or add files to the batch.",
                )
                final_status = "No input file"
                return

            outdir = self.output_dir_var.get() if self.write_output_var.get() else None
            total = len(jobs)
            done = 0
            for index, src in enumerate(jobs, start=1):
                if self._cancel_event.is_set():
                    raise CancelledError("Transcription cancelled.")
                if not src.exists():
                    append_line(self.status, f"Skipped (file not found): {src}")
                    continue
                prefix = f"({index}/{total}) " if total > 1 else ""
                # Where to write: the chosen folder, else beside the source for a
                # batch (so a multi-file run never silently drops output). A single
                # file with no output folder keeps the old "don't write" behaviour.
                if outdir:
                    save_dir: Optional[Path] = Path(outdir)
                elif total > 1:
                    save_dir = src.parent
                else:
                    save_dir = None
                self._transcribe_one(
                    src, task, save_dir, prefix, set_view=(index == total)
                )
                done += 1
            final_status = f"Finished {done} file(s)" if total > 1 else "Finished"
        except CancelledError:
            append_line(self.status, "Cancelled.")
            final_status = "Cancelled"
        except Exception as exc:
            append_line(self.status, friendly_error(exc))
            # Keep the full traceback in the log for troubleshooting.
            append_line(self.status, traceback.format_exc())
            final_status = "Error"
        finally:
            self._set_busy(False, final_status)

    def _transcribe_one(
        self,
        src: Path,
        task: str,
        save_dir: Optional[Path],
        prefix: str,
        *,
        set_view: bool,
    ) -> None:
        """Transcribe one file: convert, transcribe, diarize, save, show.

        ``set_view`` loads the result into the transcript pane (used for the last
        file of a batch, or the only file). ``save_dir`` writes outputs there when
        set; ``prefix`` is the ``(i/n)`` batch marker for status lines.
        """
        temp_wav: Optional[Path] = None
        try:
            language = self.language_var.get().strip()
            language_arg = None if language in ("", "Auto") else language
            append_line(self.status, f"{prefix}Processing: {src}")

            # Optionally pre-convert video to WAV with ffmpeg before transcribing.
            media_path = src
            media_is_normalized = False  # True when media_path is our 16 kHz mono WAV
            if self.convert_video_var.get() and is_video(media_path):
                if save_dir and save_dir.is_dir():
                    wav_dest: Optional[Path] = save_dir / (media_path.stem + ".wav")
                else:
                    wav_dest = None  # convert to a temp file we clean up afterwards
                media_path = convert_to_wav(
                    media_path,
                    wav_dest,
                    progress=lambda msg: append_line(self.status, msg),
                )
                media_is_normalized = True
                if wav_dest is None:
                    temp_wav = media_path
                append_line(self.status, f"Converted to {media_path}")

            # Resolve a bundled model name to its local directory so we never
            # try to download on an air-gapped machine.
            model_sel = self._selected_model()
            # This build has models, but the chosen one isn't among them and isn't
            # a local folder: it can't be fetched offline, so fail with a clear
            # message instead of a cryptic download error. (When nothing is bundled
            # - running from source - fall through and let faster-whisper fetch it.)
            if self._bundled_models and not self._model_is_available(model_sel):
                raise RuntimeError(
                    f"The model '{model_sel}' isn't in this build. Open Self-test… "
                    f"to see the bundled models and pick one of those, or rebuild "
                    f"the bundle with '{model_sel}' included."
                )
            model = str(self._bundled_models.get(model_sel, model_sel))

            verb = "Translating" if task == "translate" else "Transcribing"
            transcribe_label = f"{prefix}{verb}"
            # Word timestamps power word-level speaker assignment (diarization) and
            # word-level confidence highlighting; skip the extra alignment pass when
            # neither is needed so plain transcription stays fast.
            need_words = self.diarize_var.get() or self.highlight_conf_var.get()
            result = transcribe_audio(
                media_path,
                model_size=model,
                task=task,
                language=language_arg,
                vad_filter=self.vad_var.get(),
                word_timestamps=need_words,
                initial_prompt=self.vocab_var.get().strip() or None,
                progress=lambda msg: append_line(self.status, msg),
                on_progress=lambda f: self._set_progress(f, transcribe_label),
                cancelled=self._cancel_event.is_set,
            )

            append_line(
                self.status,
                f"Detected language: {result.language} "
                f"({result.language_probability:.0%}), "
                f"duration: {result.duration:.1f}s",
            )

            if self.diarize_var.get():
                self._diarize_into(result, src, media_path, media_is_normalized)

            names = self._preset_names_for(result)
            if set_view:
                # Remember the result so speakers can be renamed afterwards.
                self._result = result
                self._result_source = src
                self._result_outdir = save_dir
                self._speaker_names = names
                self.transcript_view.set_result(result, names)

            if save_dir is not None:
                self._save_outputs(result, src, save_dir, names)
        finally:
            if temp_wav is not None:
                try:
                    temp_wav.unlink()
                except OSError:
                    pass

    def _parse_num_speakers(self) -> Optional[int]:
        raw = self.num_speakers_var.get().strip()
        if not raw:
            return None
        try:
            value = int(raw)
        except ValueError:
            append_line(self.status, f"Ignoring invalid speaker count: {raw!r}")
            return None
        return value if value > 0 else None

    def _parse_threshold(self) -> float:
        raw = self.sensitivity_var.get().strip()
        if not raw:
            return 0.5
        try:
            value = float(raw)
        except ValueError:
            append_line(self.status, f"Ignoring invalid sensitivity: {raw!r}")
            return 0.5
        return min(max(value, 0.05), 1.0)

    def _diarize_into(
        self,
        result: TranscriptionResult,
        source: Path,
        media_path: Path,
        media_is_normalized: bool,
    ) -> None:
        # Diarization needs a 16 kHz mono WAV; convert to a temp file unless the
        # media we already have is one we normalized.
        diar_wav = media_path
        diar_temp: Optional[Path] = None
        if not media_is_normalized:
            append_line(self.status, "Preparing audio for diarization...")
            diar_wav = convert_to_wav(
                source, progress=lambda msg: append_line(self.status, msg)
            )
            diar_temp = diar_wav
        try:
            speaker_segments = diarize(
                diar_wav,
                backend=ENGINE_CHOICES.get(self.engine_var.get(), "auto"),
                num_speakers=self._parse_num_speakers(),
                threshold=self._parse_threshold(),
                progress=lambda msg: append_line(self.status, msg),
                on_progress=lambda f: self._set_progress(f, "Identifying speakers"),
                cancelled=self._cancel_event.is_set,
            )
            count = len({seg.speaker for seg in speaker_segments})
            append_line(self.status, f"Identified {count} speaker(s).")
            # Recognise enrolled voices (relabel turns to known speakers) and keep
            # a copy of the audio so post-run corrections can teach new voices.
            self._recognized_names = {}
            if self._profile is not None and self.learn_var.get():
                speaker_segments = self._recognize_speakers(speaker_segments, diar_wav)
                self._set_session_wav(diar_wav)
            result.segments = assign_speakers(result.segments, speaker_segments)
        finally:
            if diar_temp is not None:
                try:
                    diar_temp.unlink()
                except OSError:
                    pass

    # -- Output / export ---------------------------------------------------

    def _save_outputs(
        self,
        result: TranscriptionResult,
        source: Path,
        outdir: Path,
        names: Optional[Dict[str, str]] = None,
    ) -> None:
        if not outdir.is_dir():
            append_line(self.status, f"Output folder does not exist: {outdir}")
            return
        names = self._speaker_names if names is None else names
        txt_path = outdir / (source.name + ".txt")
        txt_path.write_text(
            result.to_txt(names, blank_lines=self.blank_lines_var.get()),
            encoding="utf-8",
        )
        append_line(self.status, f"Wrote transcript to {txt_path}")
        if self.srt_var.get():
            srt_path = outdir / (source.name + ".srt")
            srt_path.write_text(result.to_srt(names), encoding="utf-8")
            append_line(self.status, f"Wrote subtitles to {srt_path}")

    def _save_outputs_if_possible(self) -> None:
        """Re-save after a transcript edit, when an output folder is in use."""
        if self._result is not None and self._result_source and self._result_outdir:
            self._save_outputs(self._result, self._result_source, self._result_outdir)

    def _rerender_transcript(self) -> None:
        self.transcript_view.render()

    def _find_next(self) -> None:
        self._find(backwards=False)

    def _find_prev(self) -> None:
        self._find(backwards=True)

    def _find(self, *, backwards: bool) -> None:
        query = self.find_var.get()
        if not query.strip():
            self.transcript_view.find("")
            self.find_status_var.set("")
            return
        count = self.transcript_view.find(query, backwards=backwards)
        self.find_status_var.set(f"{count} match(es)" if count else "No matches")

    def _copy_transcript(self) -> None:
        """Copy the rendered transcript text to the clipboard."""
        text = self.transcript_view.get_text()
        if not text.strip():
            self.progress_label_var.set("Nothing to copy yet.")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.progress_label_var.set("Transcript copied to clipboard.")

    def _save_transcript_docx(self) -> None:
        """Save the current transcript as a Word document."""
        result = self._result
        if result is None:
            self.progress_label_var.set("Run a transcription first.")
            return
        default = (
            f"{self._result_source.stem}.docx"
            if self._result_source
            else "transcript.docx"
        )
        path = filedialog.asksaveasfilename(
            title="Save transcript as Word",
            defaultextension=".docx",
            initialfile=default,
            filetypes=[("Word document", "*.docx")],
        )
        if not path:
            return
        try:
            transcript_to_docx(
                result,
                path,
                self._speaker_names,
                blank_lines=self.blank_lines_var.get(),
            )
            self.progress_label_var.set(f"Saved {Path(path).name}")
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            self.progress_label_var.set(friendly_error(exc))

    def _save_project(self) -> None:
        """Save the result + speaker edits to a reloadable project file."""
        result = self._result
        if result is None:
            self.progress_label_var.set("Run a transcription first.")
            return
        stem = self._result_source.stem if self._result_source else "transcript"
        path = filedialog.asksaveasfilename(
            title="Save project",
            defaultextension=PROJECT_SUFFIX,
            initialfile=f"{stem}{PROJECT_SUFFIX}",
            filetypes=[
                ("Whispers project", f"*{PROJECT_SUFFIX}"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            save_project(path, result, self._speaker_names, self._result_source)
            self.progress_label_var.set(f"Saved {Path(path).name}")
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            self.progress_label_var.set(friendly_error(exc))

    def _open_project(self) -> None:
        """Load a saved project so its transcript can be reviewed/edited again."""
        path = filedialog.askopenfilename(
            title="Open project",
            filetypes=[
                ("Whispers project", f"*{PROJECT_SUFFIX}"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            result, speaker_names, source = load_project(path)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            self.progress_label_var.set(friendly_error(exc))
            return
        self._result = result
        self._speaker_names = speaker_names
        # Restore the source path so audio playback still works; outputs are no
        # longer auto-saved to a folder until the next run (re-save the project).
        self._result_source = Path(source) if source else None
        self._result_outdir = None
        # This transcript is not from the current run's audio, so voiceprint
        # enrolment must not attach its corrections to a stale recording.
        self._clear_session_wav()
        self._recognized_names = {}
        self.transcript_view.set_result(result, self._speaker_names)
        self.progress_label_var.set(f"Opened {Path(path).name}")

    def _on_drop_media(self, paths: List[Path]) -> None:
        """A dropped file loads as input; several dropped files fill the batch."""
        if not paths:
            return
        if len(paths) == 1:
            self.input_file_var.set(str(paths[0]))
            self.progress_label_var.set(f"Loaded {paths[0].name}")
        else:
            self._add_batch_paths(paths)

    # -- Batch queue -------------------------------------------------------

    def _add_batch_files(self) -> None:
        patterns = " ".join(f"*{ext}" for ext in AUDIO_EXTENSIONS)
        paths = filedialog.askopenfilenames(
            filetypes=[("Audio/Video", patterns), ("All files", "*.*")]
        )
        self._add_batch_paths([Path(p) for p in paths if p])

    def _add_batch_paths(self, paths: List[Path]) -> None:
        for path in paths:
            if path not in self._batch_files:
                self._batch_files.append(path)
        self._update_batch_label()

    def _clear_batch_files(self) -> None:
        self._batch_files = []
        self._update_batch_label()

    def _update_batch_label(self) -> None:
        count = len(self._batch_files)
        if not count:
            self.batch_files_var.set("")
            return
        names = ", ".join(p.name for p in self._batch_files[:4])
        more = "" if count <= 4 else f" (+{count - 4} more)"
        self.batch_files_var.set(f"Batch: {count} file(s) — {names}{more}")

    # -- Audio playback ----------------------------------------------------

    def _play_segment(self, start: float, end: float) -> None:
        """Play the source audio between ``start`` and ``end`` (off the UI thread)."""
        source = self._result_source
        if source is None:
            self.progress_label_var.set("Run a transcription first.")
            return

        def _worker() -> None:
            try:
                self._player.play_segment(source, start, end)
                self.progress_label_var.set(
                    f"Playing {self._clock(start)}–{self._clock(end)}…"
                )
            except PlaybackError as exc:
                self.progress_label_var.set(friendly_error(exc))

        # ffmpeg extraction is quick but still I/O; keep the click responsive.
        threading.Thread(target=_worker, daemon=True).start()

    def _stop_audio(self) -> None:
        self._player.stop()
        self.progress_label_var.set("Stopped.")

    @staticmethod
    def _clock(seconds: float) -> str:
        seconds = max(0, int(seconds))
        return f"{seconds // 60}:{seconds % 60:02d}"

    def _preset_names_for(self, result: TranscriptionResult) -> Dict[str, str]:
        """Build a speaker-id -> display-name map for this result.

        Voices recognised from the active profile are applied first (their ids
        already carry the person's name). Any remaining, unrecognised speakers are
        matched to the Speaker N fields in label order (SPEAKER_00 -> "Speaker 1",
        ...); the labelling the diarizer assigns is arbitrary, so the operator may
        still need to swap two names - one click per [speaker] tag.
        """
        ids = sorted({seg.speaker for seg in result.segments if seg.speaker})
        # Recognised ids (voice::Name) already map to a display name; keep only
        # those present in this result.
        names: Dict[str, str] = {
            sid: self._recognized_names[sid]
            for sid in ids
            if sid in self._recognized_names
        }
        unrecognised = [sid for sid in ids if sid not in names]
        for sid, var in zip(unrecognised, self.speaker_name_vars):
            name = var.get().strip()
            if name:
                names[sid] = name
        return names

    # -- Voiceprint recognition / enrolment --------------------------------

    def _get_embedder(self) -> Optional[SpeakerEmbedder]:
        """The speaker-embedding extractor, built once; ``None`` if unavailable."""
        if self._embedder is None:
            try:
                self._embedder = SpeakerEmbedder()
            except Exception as exc:  # noqa: BLE001 - degrade to no voiceprints
                self._embedder = False
                append_line(
                    self.status,
                    "Voiceprints unavailable (speaker-embedding model not "
                    f"loaded): {exc}",
                )
        return self._embedder if isinstance(self._embedder, SpeakerEmbedder) else None

    def _recognize_speakers(self, speaker_segments, diar_wav: Path):
        """Relabel diarizer turns to enrolled voices; fills ``_recognized_names``."""
        profile = self._profile
        if profile is None or not profile.voiceprints:
            return speaker_segments
        embedder = self._get_embedder()
        if embedder is None:
            return speaker_segments
        try:
            relabeled, name_map = recognize(
                diar_wav,
                speaker_segments,
                list(profile.voiceprints.values()),
                embedder,
            )
        except Exception as exc:  # noqa: BLE001 - never fail a run on recognition
            append_line(self.status, f"Voice recognition skipped: {exc}")
            return speaker_segments
        self._recognized_names = name_map
        if name_map:
            recognised = ", ".join(sorted(set(name_map.values())))
            append_line(self.status, f"Recognised voice(s): {recognised}.")
        return relabeled

    def _set_session_wav(self, diar_wav: Path) -> None:
        """Keep a 16 kHz copy of the current audio for post-run enrolment."""
        self._clear_session_wav()
        try:
            handle, tmp = tempfile.mkstemp(suffix=".wav")
            os.close(handle)
            shutil.copyfile(diar_wav, tmp)
            self._session_wav = Path(tmp)
        except OSError:
            self._session_wav = None

    def _clear_session_wav(self) -> None:
        if self._session_wav is not None:
            try:
                self._session_wav.unlink()
            except OSError:
                pass
            self._session_wav = None

    def _enroll_voice(self, name: str, spans: List[Tuple[float, float]]) -> None:
        """Fold the audio of ``spans`` into ``name``'s voiceprint (off the UI thread).

        Triggered by a correction in the transcript. A no-op unless a profile is
        active, learning is on, and we kept this run's audio.
        """
        profile = self._profile
        if profile is None or not self.learn_var.get():
            return
        wav = self._session_wav
        if wav is None or not wav.exists():
            return
        embedder = self._get_embedder()
        if embedder is None:
            return

        def _worker() -> None:
            try:
                voiceprint = profile.voiceprint_for(name)
                added = enroll_spans(voiceprint, wav, spans, embedder)
                if added:
                    save_profile(profile)
                    self.progress_label_var.set(
                        f"Learned {name}'s voice (+{added} sample(s))."
                    )
            except Exception as exc:  # noqa: BLE001 - surfaced, never fatal
                self.progress_label_var.set(friendly_error(exc))

        threading.Thread(target=_worker, daemon=True).start()

    # -- Profile management ------------------------------------------------

    def _refresh_profiles(self) -> None:
        self.profile_combo.configure(values=[""] + list_profiles())

    def _on_profile_selected(self) -> None:
        self._select_profile(self.profile_var.get())

    def _select_profile(self, name: str, *, apply: bool = True) -> None:
        """Make ``name`` the active profile, optionally applying its settings."""
        name = (name or "").strip()
        if not name:
            self._profile = None
            self.profile_var.set("")
            return
        profile = load_profile(name)
        if profile is None:
            self._profile = None
            self.profile_var.set("")
            return
        self._profile = profile
        self.profile_var.set(profile.name)
        if apply and profile.settings:
            self._apply_profile_settings(profile.settings)
        vp_count = len(profile.voiceprints)
        self.progress_label_var.set(
            f"Profile '{profile.name}' loaded ({vp_count} learned voice(s))."
        )

    def _new_profile(self) -> None:
        name = simpledialog.askstring(
            "New profile", "Name this operation/profile:", parent=self.root
        )
        if not name or not name.strip():
            return
        profile = Profile(name=name.strip(), settings=self._profile_settings())
        save_profile(profile)
        self._profile = profile
        self._refresh_profiles()
        self.profile_var.set(profile.name)
        self.progress_label_var.set(f"Created profile '{profile.name}'.")

    def _save_profile_settings(self) -> None:
        """Store the current settings into the active profile (voiceprints kept)."""
        profile = self._profile
        if profile is None:
            self.progress_label_var.set("Pick or create a profile first.")
            return
        profile.settings = self._profile_settings()
        save_profile(profile)
        self.progress_label_var.set(f"Saved profile '{profile.name}'.")

    def _delete_profile(self) -> None:
        profile = self._profile
        if profile is None:
            self.progress_label_var.set("No profile selected.")
            return
        delete_profile(profile.name)
        self._profile = None
        self.profile_var.set("")
        self._refresh_profiles()
        self.progress_label_var.set("Profile deleted.")

    def _profile_settings(self) -> Dict[str, object]:
        """The settings a profile remembers (the persisted set plus custom words).

        The active-profile name is dropped: a profile does not store which profile
        it is (and keeping it would re-trigger selection when the profile's own
        settings are applied).
        """
        data = self.get_settings()
        data.pop("profile", None)
        data["vocab"] = self.vocab_var.get()
        return data

    def _apply_profile_settings(self, data: Dict[str, object]) -> None:
        self.apply_settings(data)
        if "vocab" in data:
            try:
                self.vocab_var.set(str(data["vocab"]))
            except Exception:  # noqa: BLE001 - ignore a stale value
                pass

    def close(self) -> None:
        """Release session resources (called when the app closes)."""
        self._clear_session_wav()

    # -- Persisted preferences ---------------------------------------------

    def _settings_vars(self) -> "Dict[str, tk.Variable]":
        """The settings persisted across launches (recording-specific fields like
        the input file, batch queue and custom words are intentionally excluded)."""
        return {
            "model": self.model_var,
            "language": self.language_var,
            "task": self.task_var,
            "vad": self.vad_var,
            "convert_video": self.convert_video_var,
            "srt": self.srt_var,
            "blank_lines": self.blank_lines_var,
            "highlight_conf": self.highlight_conf_var,
            "diarize": self.diarize_var,
            "engine": self.engine_var,
            "num_speakers": self.num_speakers_var,
            "sensitivity": self.sensitivity_var,
            "write_output": self.write_output_var,
            "output_dir": self.output_dir_var,
        }

    def get_settings(self) -> Dict[str, object]:
        data: Dict[str, object] = {
            key: var.get() for key, var in self._settings_vars().items()
        }
        # Remember the active profile + learning toggle across launches.
        data["profile"] = self.profile_var.get()
        data["learn_voices"] = self.learn_var.get()
        return data

    def apply_settings(self, data: Dict[str, object]) -> None:
        if not data:
            return
        for key, var in self._settings_vars().items():
            if key in data:
                try:
                    var.set(data[key])
                except Exception:  # noqa: BLE001 - ignore a stale/invalid value
                    pass
        if "learn_voices" in data:
            try:
                self.learn_var.set(bool(data["learn_voices"]))
            except Exception:  # noqa: BLE001 - ignore a stale value
                pass
        # Reflect any changes to the enable/disable + speaker-name state.
        self._update_output_state()
        self._update_speaker_state()
        # Re-select the last-used profile (which applies its own settings on top).
        # profile.settings carries no "profile" key, so this doesn't recurse.
        profile_name = data.get("profile")
        if profile_name:
            self._select_profile(str(profile_name))
