"""The Transcribe tab: settings, the transcription run, and the transcript pane.

Wraps :mod:`whispr.transcription` (faster-whisper) and
:mod:`whispr.diarization`. The interactive transcript (speaker/word corrections)
is delegated to :class:`whispr.gui.transcript_view.TranscriptView`; this module
owns the settings UI and drives the background run.
"""

from __future__ import annotations

import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Callable, Dict, List, Optional

from ..diarization import assign_speakers, diarize
from ..export import transcript_to_docx
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

        ttk.Label(model_frame, text="Task").grid(
            row=1, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Combobox(
            model_frame,
            textvariable=self.task_var,
            values=["transcribe", "translate"],
            state="readonly",
            width=16,
        ).grid(row=1, column=1, sticky="w", pady=4)

        ttk.Label(model_frame, text="Language").grid(
            row=2, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Combobox(
            model_frame,
            textvariable=self.language_var,
            values=COMMON_LANGUAGES,
            width=16,
        ).grid(row=2, column=1, sticky="w", pady=4)

        ttk.Label(model_frame, text="Custom words").grid(
            row=3, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Entry(model_frame, textvariable=self.vocab_var).grid(
            row=3, column=1, columnspan=2, sticky="ew", pady=4
        )
        ttk.Label(
            model_frame,
            text="Names, places, jargon or callsigns to expect (improves accuracy).",
            font=("", 8),
        ).grid(row=4, column=1, columnspan=2, sticky="w")

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
                "[speaker] tag to rename or move the whole line, or click a single "
                "word to move just that word (or from it onward) to another speaker."
            ),
            wraplength=420,
            justify="left",
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(6, 0))

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
        )
        self.status = ScrolledText(
            tabs, wrap="word", state="disabled", height=14, font="TkFixedFont"
        )
        tabs.add(self.transcript_view.widget, text="Transcript")
        tabs.add(self.status, text="Status")

        # Copy / export the transcript (handy for pasting into Word).
        export_row = ttk.Frame(container)
        export_row.pack(fill="x", pady=(6, 0))
        ttk.Button(
            export_row, text="Copy transcript", command=self._copy_transcript
        ).pack(side="left")
        ttk.Button(
            export_row, text="Save as Word…", command=self._save_transcript_docx
        ).pack(side="left", padx=(8, 0))

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

    # -- Run ---------------------------------------------------------------

    def run_in_thread(self) -> None:
        # Collapse the settings so the transcript and progress get the space.
        self._collapse_all_settings()
        self._cancel_event.clear()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        task = self.task_var.get()
        self._set_busy(
            True, "Translating..." if task == "translate" else "Transcribing..."
        )
        final_status = "Finished"
        temp_wav: Optional[Path] = None
        try:
            self.transcript_view.set_result(None, {})
            path = self.input_file_var.get()
            if not path or not Path(path).exists():
                append_line(
                    self.status,
                    "Couldn't find that file. Pick an audio or video file with "
                    f"Browse… (got: {path or 'nothing selected'}).",
                )
                final_status = "No input file"
                return

            language = self.language_var.get().strip()
            language_arg = None if language in ("", "Auto") else language
            outdir = self.output_dir_var.get() if self.write_output_var.get() else None

            append_line(self.status, f"Processing: {path}")

            # Optionally pre-convert video to WAV with ffmpeg before transcribing.
            media_path = Path(path)
            media_is_normalized = False  # True when media_path is our 16 kHz mono WAV
            if self.convert_video_var.get() and is_video(media_path):
                if outdir and Path(outdir).is_dir():
                    wav_dest: Optional[Path] = Path(outdir) / (media_path.stem + ".wav")
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
            model_sel = self.model_var.get()
            model = str(self._bundled_models.get(model_sel, model_sel))

            transcribe_label = "Translating" if task == "translate" else "Transcribing"
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
                self._diarize_into(result, Path(path), media_path, media_is_normalized)

            # Remember the result so speakers can be renamed afterwards.
            self._result = result
            self._result_source = Path(path)
            self._result_outdir = Path(outdir) if outdir else None
            self._speaker_names = {}
            self._apply_preset_speaker_names()
            self.transcript_view.set_result(result, self._speaker_names)

            if outdir:
                self._save_outputs(result, Path(path), Path(outdir))

            append_line(self.status, "Finished.")
        except CancelledError:
            append_line(self.status, "Cancelled.")
            final_status = "Cancelled"
        except Exception as exc:
            append_line(self.status, friendly_error(exc))
            # Keep the full traceback in the log for troubleshooting.
            append_line(self.status, traceback.format_exc())
            final_status = "Error"
        finally:
            if temp_wav is not None:
                try:
                    temp_wav.unlink()
                except OSError:
                    pass
            self._set_busy(False, final_status)

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
            result.segments = assign_speakers(result.segments, speaker_segments)
            count = len({seg.speaker for seg in speaker_segments})
            append_line(self.status, f"Identified {count} speaker(s).")
        finally:
            if diar_temp is not None:
                try:
                    diar_temp.unlink()
                except OSError:
                    pass

    # -- Output / export ---------------------------------------------------

    def _save_outputs(
        self, result: TranscriptionResult, source: Path, outdir: Path
    ) -> None:
        if not outdir.is_dir():
            append_line(self.status, f"Output folder does not exist: {outdir}")
            return
        names = self._speaker_names
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

    def _on_drop_media(self, paths: List[Path]) -> None:
        """Handle a file dropped on the transcript pane: load it as input."""
        if paths:
            self.input_file_var.set(str(paths[0]))
            self.progress_label_var.set(f"Loaded {paths[0].name}")

    def _apply_preset_speaker_names(self) -> None:
        """Seed display names from the Speaker N fields onto the diarized result.

        Speakers are matched in label order (SPEAKER_00 -> "Speaker 1", ...).
        The labelling pyannote assigns is arbitrary, so the operator may still
        need to swap two names - one click per [speaker] tag in the transcript.
        """
        result = self._result
        if result is None:
            return
        ids = sorted({seg.speaker for seg in result.segments if seg.speaker})
        for sid, var in zip(ids, self.speaker_name_vars):
            name = var.get().strip()
            if name:
                self._speaker_names[sid] = name
