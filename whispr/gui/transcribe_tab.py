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
from tkinter import filedialog, messagebox, simpledialog, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Callable, Dict, List, Optional, Tuple

from ..acceleration import (
    DEFAULT_MODE,
    MODE_LABELS,
    Device,
    cuda_available,
    describe_hardware,
    fallback_to_cpu,
)
from ..acceleration import resolve as resolve_device
from ..diarization import assign_speakers, diarize
from ..enrollment import enroll_from_wav
from ..export import transcript_to_docx
from ..playback import PlaybackError, SegmentPlayer, playback_available
from ..profiles import (
    PROFILE_SUFFIX,
    Profile,
    delete_profile,
    export_profile,
    list_profiles,
    load_profile,
    read_profile_file,
    save_profile,
)
from ..project import PROJECT_SUFFIX, load_project_record, save_project
from ..provenance import (
    AnalysisProvenance,
    DiarizationProvenance,
    SourceRecord,
    TranscriptionProvenance,
    transcription_model_sha256,
)
from ..reports import write_analysis_report
from ..resources import bundled_models
from ..speaker_profiles import (
    SAMPLE_LEARNED,
    find_speaker_profile_by_name,
    save_speaker_profile,
)
from ..thresholds import active as active_thresholds
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
from . import speaker_compare
from .errors import friendly_error
from .theme import (
    SPACE_LG,
    SPACE_MD,
    SPACE_SM,
    SPACE_XL,
    SPACE_XS,
    Style,
    theme,
)
from .transcript_view import TranscriptView
from .widgets import (
    Card,
    Disclosure,
    FileDropZone,
    PageHeader,
    StatusBanner,
    append_line,
    bind_wheel,
    danger_button,
    primary_button,
    register_drop,
    scrollable_body,
    secondary_button,
    style_text_widget,
    subtle_button,
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


# What each model size means to someone choosing one. The size stays visible -
# an operator who knows the names still needs to see them, and a report has to
# name the model that ran - but "small.en" is not a description of anything.
# Deliberately relative words: none of these promises an accuracy figure.
MODEL_DESCRIPTIONS = {
    "tiny": "Fastest",
    "tiny.en": "Fastest",
    "base": "Fast",
    "base.en": "Fast",
    "small": "Balanced",
    "small.en": "Balanced",
    "medium": "More thorough",
    "medium.en": "More thorough",
    "large": "Most thorough, slowest",
    "large-v3": "Most thorough, slowest",
    "turbo": "Thorough and quick",
}


def _model_label(model: str) -> str:
    """``small.en`` -> ``Balanced — small.en``; a folder path shows as itself."""
    description = MODEL_DESCRIPTIONS.get(model)
    return f"{description} — {model}" if description else model


# The hardware dropdown shows friendly labels; the run needs the mode behind one.
_DEVICE_MODES = {label: mode for mode, label in MODE_LABELS}


def _device_label(mode: str) -> str:
    for candidate, label in MODE_LABELS:
        if candidate == mode:
            return label
    return MODE_LABELS[0][1]


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
        # What the dropdown shows. model_var stays the raw model name, because
        # that is what a run, a saved profile and a report all record.
        self.model_choice_var = tk.StringVar(value=_model_label(default_model))
        self._syncing_model = False
        # Live note under the Model box: whether the current pick is in this build.
        self.model_status_var = tk.StringVar(value="")
        self.task_var = tk.StringVar(value="transcribe")
        self.language_var = tk.StringVar(value="Auto")
        self.vad_var = tk.BooleanVar(value=True)
        # Compute device for transcription. CPU is the supported
        # baseline; a GPU only makes the same work finish sooner.
        self.device_var = tk.StringVar(value=_device_label(DEFAULT_MODE))
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
        # Traceability for the displayed result: source hash, models, settings.
        self._result_provenance: Optional[AnalysisProvenance] = None

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
        self._page = container

        PageHeader(
            container,
            "Transcribe",
            "Turn an audio or video recording into searchable text.",
        ).pack(fill="x", pady=(0, SPACE_LG))

        # Everything the operator sets before running, in one region that can be
        # put away once there is a transcript to read.
        self._settings_area = ttk.Frame(container, style=Style.PAGE)
        self._settings_area.pack(fill="x")

        self._build_recording_card(self._settings_area)
        self._build_basic_options(self._settings_area)
        self._build_advanced_options(self._settings_area)
        self._build_action_row(container)
        self._build_results(container)

        # A model can also arrive from a saved profile or last session's
        # settings, so the dropdown follows the value rather than owning it.
        self.model_var.trace_add("write", self._sync_model_choice)
        self._update_model_status()

        # Initialise the enabled/disabled state of dependent fields.
        self._update_output_state()
        self._update_speaker_state()
        # Mouse-wheel scrolls the page (the scrollbar always works regardless).
        bind_wheel(transcribe_canvas, container)
        # Drop a recording anywhere it would plausibly be dropped.
        for target in (self._drop_zone, self.transcript_view.widget, self.status):
            register_drop(self.root, self._dnd_ok, target, self._on_drop_media)

    # -- Recording ---------------------------------------------------------

    def _build_recording_card(self, parent: tk.Misc) -> None:
        card = Card(parent, "Recording")
        card.pack(fill="x")
        self._drop_zone = FileDropZone(card.body, self.input_file_var, self.choose_file)
        self._drop_zone.pack(fill="x")

        # Batch stays available, but a queue is the exception: it sits under the
        # single file it would otherwise compete with.
        batch = ttk.Frame(card.body, style=Style.CARD_INNER)
        batch.pack(fill="x", pady=(SPACE_MD, 0))
        subtle_button(batch, "Add several files…", self._add_batch_files).pack(
            side="left"
        )
        self._batch_clear = subtle_button(batch, "Clear list", self._clear_batch_files)
        self._batch_row = batch
        ttk.Label(
            card.body,
            textvariable=self.batch_files_var,
            style=Style.META,
            wraplength=560,
            justify="left",
        ).pack(anchor="w", pady=(SPACE_XS, 0))
        self._update_batch_label()

    # -- Basic options -----------------------------------------------------

    def _build_basic_options(self, parent: tk.Misc) -> None:
        """Only what most operators need, in the words they would use."""
        card = Card(parent, "Options")
        card.pack(fill="x", pady=(SPACE_MD, 0))
        body = card.body
        body.columnconfigure(1, weight=1)

        ttk.Label(body, text="Language", style=Style.FIELD_LABEL).grid(
            row=0, column=0, sticky="w", padx=(0, SPACE_MD), pady=SPACE_XS
        )
        ttk.Combobox(
            body,
            textvariable=self.language_var,
            values=COMMON_LANGUAGES,
            width=18,
        ).grid(row=0, column=1, sticky="w", pady=SPACE_XS)

        ttk.Label(body, text="Quality", style=Style.FIELD_LABEL).grid(
            row=1, column=0, sticky="w", padx=(0, SPACE_MD), pady=SPACE_XS
        )
        model_row = ttk.Frame(body, style=Style.CARD_INNER)
        model_row.grid(row=1, column=1, sticky="ew", pady=SPACE_XS)
        self.model_combo = ttk.Combobox(
            model_row,
            textvariable=self.model_choice_var,
            values=self._model_labels(),
            state="readonly",
            width=34,
        )
        self.model_combo.pack(side="left")
        self.model_combo.bind(
            "<<ComboboxSelected>>", lambda _e: self._on_model_choice()
        )
        subtle_button(model_row, "Use a model folder…", self.choose_model_dir).pack(
            side="left", padx=(SPACE_SM, 0)
        )
        ttk.Label(body, textvariable=self.model_status_var, style=Style.META).grid(
            row=2, column=1, sticky="w"
        )

        ttk.Checkbutton(
            body,
            text="Identify who is speaking",
            variable=self.diarize_var,
            command=self._update_speaker_state,
            style=Style.CHECK,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(SPACE_MD, 0))
        ttk.Label(
            body,
            text="Splits the transcript by speaker, so each line says who said it.",
            style=Style.META,
        ).grid(row=4, column=0, columnspan=2, sticky="w", padx=(SPACE_XL, 0))

        ttk.Checkbutton(
            body,
            text="Save a copy to a folder",
            variable=self.write_output_var,
            command=self._update_output_state,
            style=Style.CHECK,
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(SPACE_MD, 0))
        output_row = ttk.Frame(body, style=Style.CARD_INNER)
        output_row.grid(row=6, column=0, columnspan=2, sticky="ew", padx=(SPACE_XL, 0))
        output_row.columnconfigure(0, weight=1)
        self.output_dir_entry = ttk.Entry(output_row, textvariable=self.output_dir_var)
        self.output_dir_entry.grid(row=0, column=0, sticky="ew")
        self.output_dir_button = secondary_button(
            output_row, "Choose folder…", self.choose_output_dir
        )
        self.output_dir_button.grid(row=0, column=1, padx=(SPACE_SM, 0))

    # -- Advanced ----------------------------------------------------------

    def _build_advanced_options(self, parent: tk.Misc) -> None:
        """Everything that assumes you already know what it means.

        Nothing is removed - an operator who understands diarization engines or
        compute devices still has them - but nobody needs to walk past them to
        transcribe a recording.
        """
        self._advanced = Disclosure(parent, "Advanced options")
        self._advanced.pack(fill="x", pady=(SPACE_MD, 0))
        body = self._advanced.body

        audio = Card(body, "Audio and output")
        audio.pack(fill="x")
        ttk.Checkbutton(
            audio.body,
            text="Skip silence (voice activity detection)",
            variable=self.vad_var,
            style=Style.CHECK,
        ).pack(anchor="w")
        ttk.Checkbutton(
            audio.body,
            text="Convert video to audio first (ffmpeg)",
            variable=self.convert_video_var,
            style=Style.CHECK,
        ).pack(anchor="w", pady=(SPACE_XS, 0))
        ttk.Checkbutton(
            audio.body,
            text="Also save .srt subtitles",
            variable=self.srt_var,
            style=Style.CHECK,
        ).pack(anchor="w", pady=(SPACE_XS, 0))
        ttk.Checkbutton(
            audio.body,
            text="Blank line between segments (easier to read and paste)",
            variable=self.blank_lines_var,
            command=self._rerender_transcript,
            style=Style.CHECK,
        ).pack(anchor="w", pady=(SPACE_XS, 0))
        ttk.Checkbutton(
            audio.body,
            text="Highlight low-confidence words (verify these)",
            variable=self.highlight_conf_var,
            command=self._rerender_transcript,
            style=Style.CHECK,
        ).pack(anchor="w", pady=(SPACE_XS, 0))

        language = Card(body, "Language and vocabulary")
        language.pack(fill="x", pady=(SPACE_MD, 0))
        language.body.columnconfigure(1, weight=1)
        ttk.Label(language.body, text="Task", style=Style.FIELD_LABEL).grid(
            row=0, column=0, sticky="w", padx=(0, SPACE_MD), pady=SPACE_XS
        )
        ttk.Combobox(
            language.body,
            textvariable=self.task_var,
            values=["transcribe", "translate"],
            state="readonly",
            width=18,
        ).grid(row=0, column=1, sticky="w", pady=SPACE_XS)
        ttk.Label(language.body, text="Expected words", style=Style.FIELD_LABEL).grid(
            row=1, column=0, sticky="w", padx=(0, SPACE_MD), pady=SPACE_XS
        )
        ttk.Entry(language.body, textvariable=self.vocab_var).grid(
            row=1, column=1, sticky="ew", pady=SPACE_XS
        )
        ttk.Label(
            language.body,
            text="Names, places, jargon or callsigns to expect, so they are spelled right.",
            style=Style.META,
        ).grid(row=2, column=1, sticky="w")

        hardware = Card(body, "Processing hardware")
        hardware.pack(fill="x", pady=(SPACE_MD, 0))
        row = ttk.Frame(hardware.body, style=Style.CARD_INNER)
        row.pack(fill="x")
        ttk.Combobox(
            row,
            textvariable=self.device_var,
            values=[label for _, label in MODE_LABELS],
            state="readonly",
            width=34,
        ).pack(side="left")
        ttk.Label(hardware.body, text=describe_hardware(), style=Style.META).pack(
            anchor="w", pady=(SPACE_XS, 0)
        )
        if not cuda_available():
            ttk.Label(
                hardware.body,
                text="The processor runs everything; a graphics card only makes it faster.",
                style=Style.META,
            ).pack(anchor="w")

        self._build_speaker_advanced(body)
        self._build_profile_card(body)

    def _build_speaker_advanced(self, parent: tk.Misc) -> None:
        card = Card(
            parent,
            "Speaker separation",
            "Used when “Identify who is speaking” is on.",
        )
        card.pack(fill="x", pady=(SPACE_MD, 0))
        frame = card.body
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Method", style=Style.FIELD_LABEL).grid(
            row=0, column=0, sticky="w", padx=(0, SPACE_MD), pady=SPACE_XS
        )
        self.engine_combo = ttk.Combobox(
            frame,
            textvariable=self.engine_var,
            values=ENGINE_LABELS,
            state="readonly",
            width=34,
        )
        self.engine_combo.grid(row=0, column=1, sticky="w", pady=SPACE_XS)

        ttk.Label(
            frame, text="How many people (blank = work it out)", style=Style.FIELD_LABEL
        ).grid(row=1, column=0, sticky="w", padx=(0, SPACE_MD), pady=SPACE_XS)
        self.num_speakers_entry = ttk.Entry(
            frame, textvariable=self.num_speakers_var, width=8
        )
        self.num_speakers_entry.grid(row=1, column=1, sticky="w", pady=SPACE_XS)
        # Entering a count reveals a name field per speaker (filled in below).
        self.num_speakers_var.trace_add("write", self._on_num_speakers_changed)

        ttk.Label(
            frame,
            text="Grouping sensitivity (higher = fewer speakers)",
            style=Style.FIELD_LABEL,
        ).grid(row=2, column=0, sticky="w", padx=(0, SPACE_MD), pady=SPACE_XS)
        self.sensitivity_entry = ttk.Entry(
            frame, textvariable=self.sensitivity_var, width=8
        )
        self.sensitivity_entry.grid(row=2, column=1, sticky="w", pady=SPACE_XS)

        # Dynamic per-speaker name fields, rebuilt when the count changes.
        self.speaker_names_frame = ttk.Frame(frame, style=Style.CARD_INNER)
        self.speaker_names_frame.grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(SPACE_SM, 0)
        )
        ttk.Label(
            frame,
            text=(
                "In the transcript you can click a speaker tag to rename or move a "
                "whole line, click a word to move it, or drag a highlighted run of "
                "words onto another speaker."
            ),
            style=Style.META,
            wraplength=560,
            justify="left",
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(SPACE_SM, 0))

    def _build_profile_card(self, parent: tk.Misc) -> None:
        card = Card(
            parent,
            "Operation profile",
            "Remembers these settings and the voices you correct, per operation.",
        )
        card.pack(fill="x", pady=(SPACE_MD, 0))
        frame = card.body
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Profile", style=Style.FIELD_LABEL).grid(
            row=0, column=0, sticky="w", padx=(0, SPACE_MD), pady=SPACE_XS
        )
        self.profile_combo = ttk.Combobox(
            frame,
            textvariable=self.profile_var,
            values=[""] + list_profiles(),
            state="readonly",
            width=26,
        )
        self.profile_combo.grid(row=0, column=1, sticky="w", pady=SPACE_XS)
        self.profile_combo.bind(
            "<<ComboboxSelected>>", lambda _e: self._on_profile_selected()
        )

        buttons = ttk.Frame(frame, style=Style.CARD_INNER)
        buttons.grid(row=1, column=0, columnspan=2, sticky="w", pady=(SPACE_SM, 0))
        for text, command in (
            ("New…", self._new_profile),
            ("Save", self._save_profile_settings),
            ("Export…", self._export_profile),
            ("Import…", self._import_profile),
            ("Export speaker…", self._export_speaker),
            ("Compare voices…", self._compare_voices),
        ):
            subtle_button(buttons, text, command).pack(side="left", padx=(0, SPACE_SM))
        danger_button(buttons, "Delete", self._delete_profile).pack(side="left")

        ttk.Checkbutton(
            frame,
            text="Recognise and learn speaker voices for this profile",
            variable=self.learn_var,
            style=Style.CHECK,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(SPACE_MD, 0))
        ttk.Label(
            frame,
            text=(
                "With speaker identification on, each speaker is matched against "
                "the voices this profile has learned. Correcting a name teaches it "
                "that voice for next time; it does not change the words themselves."
            ),
            style=Style.META,
            wraplength=560,
            justify="left",
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(SPACE_XS, 0))

    # -- The action ---------------------------------------------------------

    def _build_action_row(self, parent: tk.Misc) -> None:
        """One obvious thing to do, and an honest account of it while it runs."""
        actions = ttk.Frame(parent, style=Style.PAGE)
        actions.pack(fill="x", pady=(SPACE_LG, 0))
        self._action_row = actions
        self.run_button = primary_button(
            actions, "Transcribe recording", self.run_in_thread
        )
        self.run_button.pack(side="left")
        self.cancel_button = secondary_button(actions, "Cancel", self._on_cancel)
        self.cancel_button.configure(state="disabled")
        self.cancel_button.pack(side="left", padx=(SPACE_SM, 0))
        self.toggle_settings_button = subtle_button(
            actions, "Hide settings", self._toggle_all_settings
        )
        self.toggle_settings_button.pack(side="right")

        progress = ttk.Frame(parent, style=Style.PAGE)
        progress.pack(fill="x", pady=(SPACE_SM, 0))
        self.progress_bar = ttk.Progressbar(progress, mode="indeterminate")
        self.progress_bar.pack(fill="x")
        ttk.Label(
            progress, textvariable=self.progress_label_var, style=Style.PAGE_SUBTITLE
        ).pack(anchor="w", pady=(SPACE_XS, 0))
        # Inserted above the results, so a failure is read before the
        # empty transcript that follows it.
        self.banner = StatusBanner(parent)

    # -- Results ------------------------------------------------------------

    def _build_results(self, parent: tk.Misc) -> None:
        """The transcript is the product; after a run it gets the page."""
        results = Card(parent, "Transcript")
        results.pack(fill="both", expand=True, pady=(SPACE_XL, 0))
        self.banner.insert_before(results)
        self._results_card = results

        self.result_summary_var = tk.StringVar(value="")
        ttk.Label(
            results.body, textvariable=self.result_summary_var, style=Style.MUTED
        ).pack(anchor="w")

        self._output_tabs = ttk.Notebook(results.body)
        tabs = self._output_tabs
        tabs.pack(fill="both", expand=True, pady=(SPACE_SM, 0))
        self.transcript_view = TranscriptView(
            tabs,
            self.root,
            self.blank_lines_var,
            self._save_outputs_if_possible,
            highlight_var=self.highlight_conf_var,
            on_play=self._play_segment if self._playback_ok else None,
            on_enroll=self._enroll_voice,
            placeholder=(
                "Choose a recording above, then select Transcribe recording.\n\n"
                "Your transcript appears here. Drag a file onto this window to "
                "load it.\n\n"
                "If something goes wrong, the Status tab explains what happened, "
                "and System status (top right) shows what this copy can do."
            ),
        )
        style_text_widget(self.transcript_view.widget)
        self.status = ScrolledText(tabs, wrap="word", state="disabled", height=12)
        style_text_widget(self.status)
        self.status.configure(font=theme().mono)
        tabs.add(self.transcript_view.widget, text="Transcript")
        tabs.add(self.status, text="Status")

        # Find within the transcript (Enter = next, Shift+Enter = previous).
        tools = ttk.Frame(results.body, style=Style.CARD_INNER)
        tools.pack(fill="x", pady=(SPACE_MD, 0))
        ttk.Label(tools, text="Find", style=Style.FIELD_LABEL, width=6).pack(
            side="left"
        )
        self.find_var = tk.StringVar()
        find_entry = ttk.Entry(tools, textvariable=self.find_var, width=26)
        find_entry.pack(side="left")
        find_entry.bind("<Return>", lambda _e: self._find_next())
        find_entry.bind("<Shift-Return>", lambda _e: self._find_prev())
        subtle_button(tools, "Next", self._find_next).pack(
            side="left", padx=(SPACE_SM, 0)
        )
        subtle_button(tools, "Previous", self._find_prev).pack(side="left")
        self.find_status_var = tk.StringVar()
        ttk.Label(tools, textvariable=self.find_status_var, style=Style.META).pack(
            side="left", padx=(SPACE_SM, 0)
        )

        exports = ttk.Frame(results.body, style=Style.CARD_INNER)
        exports.pack(fill="x", pady=(SPACE_SM, 0))
        for text, command in (
            ("Copy transcript", self._copy_transcript),
            ("Save as Word…", self._save_transcript_docx),
            ("Analysis report…", self._save_analysis_report),
            ("Save project…", self._save_project),
            ("Open project…", self._open_project),
        ):
            secondary_button(exports, text, command).pack(
                side="left", padx=(0, SPACE_SM)
            )
        if self._playback_ok:
            subtle_button(exports, "Stop audio", self._stop_audio).pack(side="left")
            ttk.Label(
                results.body,
                text="Ctrl-click a line or a word to play its audio.",
                style=Style.META,
            ).pack(anchor="w", pady=(SPACE_XS, 0))

    # -- Cancellation ------------------------------------------------------

    def notify_cancelling(self) -> None:
        self.cancel_button.configure(state="disabled")
        append_line(self.status, "Cancelling… (will stop at the next checkpoint)")
        self.progress_label_var.set("Cancelling…")

    # -- Settings state ----------------------------------------------------

    def _settings_visible(self) -> bool:
        return bool(self._settings_area.winfo_manager())

    def _toggle_all_settings(self) -> None:
        self._show_settings(not self._settings_visible())

    def _collapse_all_settings(self) -> None:
        """Put the settings away so the transcript has the page."""
        self._show_settings(False)

    def _show_settings(self, visible: bool) -> None:
        if visible:
            # Re-inserted above the action row so the page order is preserved.
            self._settings_area.pack(fill="x", before=self._action_row)
        else:
            self._settings_area.pack_forget()
        self.toggle_settings_button.configure(
            text="Hide settings" if visible else "Show settings"
        )

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
                self.progress_label_var.set(message or "Working…")
                self.banner.hide()
            else:
                self.progress_bar.stop()
                self.progress_bar.configure(mode="determinate")
                self.progress_bar["value"] = 0
                self.progress_label_var.set(message or "Ready")
                self.run_button.configure(state="normal")
                self.cancel_button.configure(state="disabled")

        self.root.after(0, _do)

    def _announce(self, kind: str, message: str) -> None:
        """Put a result or a failure where it cannot be scrolled past."""
        self.root.after(0, lambda: self.banner.show(kind, message))

    def _describe_result(self) -> None:
        """The one-line account of what was produced, above the transcript."""
        result = self._result
        if result is None:
            self.result_summary_var.set("")
            return
        parts = []
        if self._result_source is not None:
            parts.append(self._result_source.name)
        if result.language:
            parts.append(f"language {result.language}")
        if result.duration:
            parts.append(f"{result.duration / 60:.1f} min")
        speakers = {seg.speaker for seg in result.segments if seg.speaker}
        if speakers:
            parts.append(f"{len(speakers)} speaker(s)")
        self.result_summary_var.set("  ·  ".join(parts))

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

    def _model_labels(self) -> List[str]:
        """The models this build can actually offer, described."""
        if self._bundled_models:
            names = list(self._bundled_models)
        else:
            names = list(MODEL_SIZES)
        return [_model_label(name) for name in names]

    def _on_model_choice(self) -> None:
        """Dropdown -> the raw model name the rest of the application uses."""
        label = self.model_choice_var.get()
        for name in list(self._bundled_models) or list(MODEL_SIZES):
            if _model_label(name) == label:
                self._set_model(name)
                return
        self._set_model(label)

    def _set_model(self, name: str) -> None:
        self._syncing_model = True
        try:
            self.model_var.set(name)
        finally:
            self._syncing_model = False
        self._update_model_status()

    def _sync_model_choice(self, *_args: object) -> None:
        """Raw model name -> dropdown, for a profile or a remembered setting."""
        if self._syncing_model:
            return
        self.model_choice_var.set(_model_label(self._selected_model()))
        self._update_model_status()

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
        self._set_busy(True, "Preparing audio…")
        final_status = "Finished"
        # Drop any prior run's kept audio so a correction can only ever enrol
        # against audio from this run's diarized file(s).
        self._clear_session_wav()
        try:
            self.transcript_view.set_result(None, {})
            jobs = self._collect_jobs()
            if not jobs:
                message = (
                    "Choose a recording first — drop one onto the window, or "
                    "select Choose file."
                )
                append_line(self.status, message)
                self._announce("warning", message)
                final_status = "No recording chosen"
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
            self._announce(
                "success",
                f"Transcription complete — {done} recording(s)."
                if total > 1
                else "Transcription complete.",
            )
            self.root.after(0, self._describe_result)
        except CancelledError:
            append_line(self.status, "Cancelled.")
            self._announce("info", "Cancelled before finishing.")
            final_status = "Cancelled"
        except Exception as exc:
            friendly = friendly_error(exc)
            append_line(self.status, friendly)
            # Keep the full traceback in the log for troubleshooting - never in
            # front of an operator who cannot act on it.
            append_line(self.status, traceback.format_exc())
            final_status = "Error"
            self._announce(
                "error", f"{friendly}  (Status tab has the technical details.)"
            )
            # Surface the reason: a non-technical user won't think to open the
            # Status tab on their own, so bring it to the front.
            self._show_status_tab()
        finally:
            self._set_busy(False, final_status)

    def _show_status_tab(self) -> None:
        """Bring the Status tab to the front (so an error/notice is seen)."""

        def _do() -> None:
            try:
                self._output_tabs.select(self.status)
            except tk.TclError:
                pass

        self.root.after(0, _do)

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
            device = resolve_device(
                _DEVICE_MODES.get(self.device_var.get(), DEFAULT_MODE)
            )
            if device.note:
                append_line(self.status, device.note)
            append_line(self.status, f"Processing on: {device.describe()}")

            def _run(on: Device):
                return transcribe_audio(
                    media_path,
                    model_size=model,
                    device=on.device,
                    compute_type=on.compute_type,
                    task=task,
                    language=language_arg,
                    vad_filter=self.vad_var.get(),
                    word_timestamps=need_words,
                    initial_prompt=self.vocab_var.get().strip() or None,
                    progress=lambda msg: append_line(self.status, msg),
                    on_progress=lambda f: self._set_progress(f, transcribe_label),
                    cancelled=self._cancel_event.is_set,
                )

            try:
                result = _run(device)
            except Exception as exc:  # noqa: BLE001 - see fallback_to_cpu
                # A GPU can be visible and still fail (VRAM, driver, a busy
                # card). Losing the job to that is worse than running it slowly.
                retry = fallback_to_cpu(device, exc)
                if retry is None or self._cancel_event.is_set():
                    raise
                append_line(self.status, retry.note)
                device = retry
                result = _run(device)

            # Record what produced this result while the settings are in hand.
            provenance = AnalysisProvenance(
                source=SourceRecord.from_path(src),
                transcription=TranscriptionProvenance(
                    model_name=model_sel,
                    model_sha256=transcription_model_sha256(model),
                    device=device.device,
                    compute_type=device.compute_type,
                    language_setting=language or "auto",
                    detected_language=result.language,
                    vad=self.vad_var.get(),
                    initial_prompt=self.vocab_var.get().strip(),
                ),
            )

            append_line(
                self.status,
                f"Detected language: {result.language} "
                f"({result.language_probability:.0%}), "
                f"duration: {result.duration:.1f}s",
            )

            if self.diarize_var.get():
                self._diarize_into(result, src, media_path, media_is_normalized)
                provenance.diarization = DiarizationProvenance.from_bundle(
                    engine=ENGINE_CHOICES.get(self.engine_var.get(), "auto"),
                    expected_speaker_count=self._parse_num_speakers(),
                    clustering_threshold=self._parse_threshold(),
                )

            names = self._preset_names_for(result)
            if set_view:
                # Remember the result so speakers can be renamed afterwards.
                self._result = result
                self._result_source = src
                self._result_outdir = save_dir
                self._speaker_names = names
                self._result_provenance = provenance
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

    def current_analysis(
        self,
    ) -> "Tuple[Optional[TranscriptionResult], Dict[str, str], Optional[AnalysisProvenance]]":
        """The displayed result, its speaker names and its provenance.

        Lets the Speaker Compare tab put the transcript and its traceability into
        an analysis report without reaching into this tab's internals.
        """
        return self._result, self._speaker_names, self._result_provenance

    def _save_analysis_report(self) -> None:
        """Write a report covering the transcript and how it was produced."""
        result = self._result
        if result is None:
            self.progress_label_var.set("Run a transcription first.")
            return
        stem = self._result_source.stem if self._result_source else "analysis"
        path = filedialog.asksaveasfilename(
            title="Save analysis report",
            defaultextension=".docx",
            initialfile=f"{stem}-report.docx",
            filetypes=[
                ("Word document", "*.docx"),
                ("Text file", "*.txt"),
            ],
        )
        if not path:
            return
        try:
            write_analysis_report(
                path,
                result=result,
                speaker_names=self._speaker_names,
                provenance=self._result_provenance,
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
            save_project(
                path,
                result,
                self._speaker_names,
                self._result_source,
                self._result_provenance,
            )
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
            record = load_project_record(path)
            result = record.result
            speaker_names = record.speaker_names
            source = record.source
            self._result_provenance = record.provenance
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
            # Nothing to clear, so nothing offering to.
            self._batch_clear.pack_forget()
            return
        names = ", ".join(p.name for p in self._batch_files[:4])
        more = "" if count <= 4 else f" (+{count - 4} more)"
        self.batch_files_var.set(
            f"{count} recording(s) queued — {names}{more}. "
            "Transcribe recording will work through all of them."
        )
        self._batch_clear.pack(side="left", padx=(SPACE_SM, 0))

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
            active = active_thresholds()
            relabeled, name_map = recognize(
                diar_wav,
                speaker_segments,
                list(profile.voiceprints.values()),
                embedder,
                threshold=active.recognition_acceptance,
                margin=active.recognition_margin,
                # min_seconds stays voiceprints.MIN_TURN_SECONDS: it is the
                # shortest turn worth embedding at all, not the comparison
                # tool's minimum for making an assessment.
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
            self._propose_to_speaker_profile(name, wav, spans, embedder)

        threading.Thread(target=_worker, daemon=True).start()

    def _propose_to_speaker_profile(
        self, name: str, wav: Path, spans: List[Tuple[float, float]], embedder
    ) -> None:
        """Offer this correction to a known subject's profile - unapproved.

        A correction can be wrong, so it never joins a known actor's trusted
        reference material on its own: the sample is stored pending review in
        *Speaker profiles*, where an analyst approves or removes it. The
        operation profile's own voiceprints (above) are a separate, disposable
        who-said-what aid and are updated regardless.
        """
        try:
            subject = find_speaker_profile_by_name(name)
            if subject is None:
                return
            source = self._result_source
            result = enroll_from_wav(
                subject,
                wav,
                spans,
                embedder,
                source_filename=source.name if source else None,
                source_sha256=(
                    self._result_provenance.source.sha256
                    if self._result_provenance and self._result_provenance.source
                    else None
                ),
                sample_type=SAMPLE_LEARNED,
                notes="Proposed automatically from a transcript correction.",
            )
            if not result.added:
                return
            save_speaker_profile(subject)
            self.progress_label_var.set(
                f"{result.added_count} sample(s) proposed for {subject.display_name} - "
                "approve them in Speaker profiles before they are used."
            )
        except Exception as exc:  # noqa: BLE001 - never fail a correction on this
            append_line(self.status, f"Could not propose a speaker sample: {exc}")

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

    def _export_profile(self) -> None:
        """Write the active profile (settings + voiceprints) to a shareable file."""
        profile = self._profile
        if profile is None:
            self.progress_label_var.set("Pick or create a profile first.")
            return
        path = filedialog.asksaveasfilename(
            title="Export profile",
            defaultextension=PROFILE_SUFFIX,
            initialfile=f"{profile.name}{PROFILE_SUFFIX}",
            filetypes=[
                ("Whispers profile", f"*{PROFILE_SUFFIX}"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            export_profile(profile, path)
            self.progress_label_var.set(
                f"Exported '{profile.name}' to {Path(path).name}"
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            self.progress_label_var.set(friendly_error(exc))

    def _import_profile(self) -> None:
        """Load a profile exported from another instance and add it here."""
        path = filedialog.askopenfilename(
            title="Import profile",
            filetypes=[
                ("Whispers profile", f"*{PROFILE_SUFFIX}"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            profile = read_profile_file(path)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            self.progress_label_var.set(friendly_error(exc))
            return
        # Don't silently clobber a same-named profile already on this machine.
        if load_profile(profile.name) is not None and not messagebox.askyesno(
            "Import profile",
            f"A profile named '{profile.name}' already exists here. Overwrite it "
            "with the imported one?",
            parent=self.root,
        ):
            return
        save_profile(profile)
        self._profile = profile
        self._refresh_profiles()
        self.profile_var.set(profile.name)
        if profile.settings:
            self._apply_profile_settings(profile.settings)
        vp_count = len(profile.voiceprints)
        self.progress_label_var.set(
            f"Imported profile '{profile.name}' ({vp_count} learned voice(s))."
        )

    # -- Single-speaker export + voice comparison --------------------------
    # The dialogs live in whispr.gui.speaker_compare (self-contained); these
    # thin wrappers keep the button commands and pass the active profile + a
    # status callback.

    def _export_speaker(self) -> None:
        speaker_compare.export_speaker(
            self.root, self._profile, self.progress_label_var.set
        )

    def _compare_voices(self) -> None:
        speaker_compare.open_compare_dialog(self.root)

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
            "device": self.device_var,
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
        # A label from an older build (or a hand-edited settings file) would
        # otherwise sit in the dropdown and resolve to Auto silently.
        if self.device_var.get() not in _DEVICE_MODES:
            self.device_var.set(_device_label(DEFAULT_MODE))
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
