"""Whispers - desktop GUI for audio/video transcription and text translation.

A small Tkinter application that wraps :mod:`whispr.transcription` (faster-whisper)
and :mod:`whispr.translation` (Argos Translate) so an operator can transcribe a
recording or batch-translate text without touching the command line. Work runs on
a background thread so the UI stays responsive, and output is streamed as produced.

Run with ``python -m whispr`` or the ``whispr`` console script.
"""

from __future__ import annotations

import importlib.util
import os
import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, simpledialog, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Callable, Dict, List, Optional

from .diarization import assign_speakers, diarize
from .export import text_to_docx, transcript_to_docx
from .ocr import (
    OCR_EXTENSIONS,
    extract_text,
    is_ocr_file,
    ocr_available,
    tesseract_lang,
)
from .resources import (
    bundled_argos_data_dir,
    bundled_models,
    configure_offline_hf_cache,
    configure_offline_ocr,
    configure_offline_translation,
)
from .transcription import (
    AUDIO_EXTENSIONS,
    MODEL_SIZES,
    CancelledError,
    Segment,
    TranscriptionResult,
    Word,
    convert_to_wav,
    is_video,
    transcribe_audio,
)
from .translation import detect_language

# Sentinel shown in the "From" dropdown for automatic language detection.
AUTO_DETECT_LABEL = "Auto-detect language"
AUTO_DETECT_CODE = "__auto__"
# Lets users OCR an English document by picking English explicitly. There is no
# English->English translation, so this is extract-only (the OCR'd text is the
# output). Mapped to the target code "en".
ENGLISH_LABEL = "English (OCR only, no translation)"
# OCR needs a script up front; when "From" is Auto-detect we default to Latin/
# English so the common case (English/Latin documents) works without a manual pick.
DEFAULT_OCR_LANG = "eng"

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


def _translation_available() -> bool:
    """True if the translation engine is usable (bundled packs or installed lib).

    Lets a lean transcriber-only build (no argostranslate / no packs) present as a
    single-purpose app, without importing the heavy library just to check.
    """
    if bundled_argos_data_dir() is not None:
        return True
    try:
        return importlib.util.find_spec("argostranslate") is not None
    except ModuleNotFoundError:
        return False


class CollapsibleSection(ttk.Frame):
    """A titled section whose body can be collapsed to a single header row.

    Clicking the header toggles the body. Collapsing the settings frees vertical
    space for the transcript and lets the window be resized down without clipping
    controls. Put child widgets in ``.body``.
    """

    def __init__(
        self,
        parent: tk.Misc,
        title: str,
        *,
        expanded: bool = True,
        body_padding: tuple = (10, 6, 10, 10),
    ) -> None:
        super().__init__(parent)
        self.title = title
        self.expanded = expanded
        self.header = ttk.Button(self, command=self.toggle)
        self.header.pack(fill="x")
        self.body = ttk.Frame(self, padding=body_padding)
        if expanded:
            self.body.pack(fill="both", expand=True)
        self._refresh_header()

    def _refresh_header(self) -> None:
        arrow = "▼" if self.expanded else "▶"
        self.header.configure(text=f"{arrow}  {self.title}")

    def toggle(self) -> None:
        self.set_expanded(not self.expanded)

    def set_expanded(self, value: bool) -> None:
        if value == self.expanded:
            return
        self.expanded = value
        if value:
            self.body.pack(fill="both", expand=True)
        else:
            self.body.forget()
        self._refresh_header()


class WhisprApp:
    """The main application window."""

    def __init__(self, root: tk.Tk, *, drag_and_drop: bool = False) -> None:
        self.root = root
        self.root.title("Whispers")
        # Whether tkdnd was loaded (see _enable_drag_and_drop) so widgets can
        # register file-drop targets.
        self._dnd_ok = drag_and_drop

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
        self.progress_label_var = tk.StringVar(value="Idle")

        # State for the last result, so speakers can be renamed after a run.
        self._result: Optional[TranscriptionResult] = None
        self._result_source: Optional[Path] = None
        self._result_outdir: Optional[Path] = None
        self._speaker_names: Dict[str, str] = {}

        # Set while a job runs so the Cancel button can stop it cooperatively.
        self._cancel_event = threading.Event()

        self._build_ui()

    # -- UI construction ---------------------------------------------------

    def _build_ui(self) -> None:
        self.root.minsize(680, 480)
        try:
            self.root.geometry("860x760")
        except tk.TclError:
            pass

        # Translation is shown only when its engine/packs are bundled (so a lean
        # transcriber-only build is a clean single-purpose app), and can be force-
        # hidden with WHISPR_MODE=transcribe even on a full bundle.
        self._show_translate = (
            _translation_available()
            and os.environ.get("WHISPR_MODE", "").lower() != "transcribe"
        )
        # OCR (image/PDF -> text) is offered in the Translate tab when a Tesseract
        # engine is available; auto-detect needs langdetect. Both degrade cleanly.
        self._ocr_available = ocr_available()
        self._detect_available = importlib.util.find_spec("langdetect") is not None

        # App header.
        subtitle = (
            "Offline transcription & translation"
            if self._show_translate
            else "Offline transcription"
        )
        header = ttk.Frame(self.root, padding=(14, 10, 14, 4))
        header.pack(fill="x")
        ttk.Label(header, text="Whispers", font=("", 17, "bold")).pack(side="left")
        ttk.Label(header, text=subtitle, font=("", 9)).pack(
            side="left", padx=(10, 0), pady=(7, 0)
        )

        # With translation, a top-level Transcribe/Translate notebook; without it,
        # the transcribe UI fills the window directly (no redundant tab chrome).
        if self._show_translate:
            self.main_nb: Optional[ttk.Notebook] = ttk.Notebook(self.root)
            self.main_nb.pack(fill="both", expand=True, padx=8, pady=(0, 8))
            transcribe_root = ttk.Frame(self.main_nb)
            self.main_nb.add(transcribe_root, text="Transcribe")
        else:
            self.main_nb = None
            transcribe_root = ttk.Frame(self.root)
            transcribe_root.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        transcribe_canvas, container = self._scrollable_body(transcribe_root)

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
            command=self._render_transcript,
        ).grid(row=3, column=0, sticky="w", pady=2)

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
            run_frame, text="Cancel", command=self.cancel, state="disabled"
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
        self.output = ScrolledText(
            tabs, wrap="word", state="disabled", height=14, font="TkFixedFont"
        )
        self.status = ScrolledText(
            tabs, wrap="word", state="disabled", height=14, font="TkFixedFont"
        )
        tabs.add(self.output, text="Transcript")
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
        self._bind_wheel(transcribe_canvas, container)
        # Drag an audio/video file onto the transcript pane to load it.
        self._register_drop(self.output, self._on_drop_media)
        self._register_drop(self.status, self._on_drop_media)

        # --- Translate tab (only when translation is available) ------------
        if self._show_translate:
            self._build_translate_tab()

    def _scrollable_body(self, parent: tk.Misc) -> tuple[tk.Canvas, ttk.Frame]:
        """Wrap a scrollable region in ``parent`` and return ``(canvas, inner)``.

        Settings sections can stack taller than the window (especially on small
        screens), which previously pushed the Run button and transcript off the
        bottom with no way to reach them. This puts everything in a vertically
        scrollable canvas: the scrollbar always works, and the mouse wheel is
        wired up by :meth:`_bind_wheel`.
        """
        canvas = tk.Canvas(parent, highlightthickness=0)
        vsb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = ttk.Frame(canvas, padding=12)
        window = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _sync() -> None:
            # Match the inner frame's width to the canvas, and let it stretch to
            # fill the viewport when the content is shorter than it (so widgets
            # that expand look right) while still allowing it to overflow+scroll.
            canvas.itemconfigure(window, width=canvas.winfo_width())
            canvas.itemconfigure(
                window, height=max(inner.winfo_reqheight(), canvas.winfo_height())
            )
            canvas.configure(scrollregion=canvas.bbox("all"))

        # Re-sync both when the viewport resizes and when the content grows or
        # shrinks (e.g. as settings sections are expanded/collapsed).
        canvas.bind("<Configure>", lambda _e: _sync())
        inner.bind("<Configure>", lambda _e: _sync())
        return canvas, inner

    def _bind_wheel(self, canvas: tk.Canvas, root_widget: tk.Misc) -> None:
        """Make the mouse wheel scroll ``canvas`` while over its content.

        Bound recursively to every widget except ``tk.Text`` (and its
        ``ScrolledText`` subclass), which keep their own native scrolling so the
        transcript/status panes don't fight the page scroll.
        """

        def _on_wheel(event: "tk.Event[tk.Misc]") -> None:
            if getattr(event, "num", None) == 4:
                canvas.yview_scroll(-1, "units")
            elif getattr(event, "num", None) == 5:
                canvas.yview_scroll(1, "units")
            else:
                canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

        def _bind(widget: tk.Misc) -> None:
            if not isinstance(widget, tk.Text):
                widget.bind("<MouseWheel>", _on_wheel, add="+")  # Windows / macOS
                widget.bind("<Button-4>", _on_wheel, add="+")  # Linux scroll up
                widget.bind("<Button-5>", _on_wheel, add="+")  # Linux scroll down
            for child in widget.winfo_children():
                _bind(child)

        _bind(canvas)
        _bind(root_widget)

    def _register_drop(
        self, widget: tk.Misc, handler: Callable[[List[Path]], None]
    ) -> None:
        """Register ``widget`` as a file-drop target calling ``handler(paths)``.

        No-op when tkdnd isn't loaded. Uses tkinterdnd2's wrapper methods directly
        on the widget (the root is a themed ttkbootstrap window, not a
        ``TkinterDnD.Tk``), and parses the platform-specific drop payload into
        clean ``Path`` objects.
        """
        if not self._dnd_ok:
            return
        try:
            from tkinterdnd2 import DND_FILES, TkinterDnD
        except Exception:  # noqa: BLE001 - convenience feature only
            return

        def _on_drop(event: object) -> None:
            data = getattr(event, "data", "")
            try:
                raw = self.root.tk.splitlist(data)
            except Exception:  # noqa: BLE001 - fall back to a naive split
                raw = str(data).split()
            paths = [Path(item) for item in raw if item]
            if paths:
                handler(paths)

        try:
            TkinterDnD.DnDWrapper.drop_target_register(widget, DND_FILES)
            TkinterDnD.DnDWrapper.dnd_bind(widget, "<<Drop>>", _on_drop)
        except Exception:  # noqa: BLE001 - never let DnD wiring break the UI
            pass

    def _build_translate_tab(self) -> None:
        """Build the text-translation tab (paste box + batch files, foreign->EN)."""
        assert self.main_nb is not None  # only called when the notebook exists
        tab = ttk.Frame(self.main_nb)
        self.main_nb.add(tab, text="Translate")
        translate_canvas, container = self._scrollable_body(tab)

        # --- Languages -----------------------------------------------------
        lang_frame = ttk.LabelFrame(container, text="Languages", padding=10)
        lang_frame.pack(fill="x")
        lang_frame.columnconfigure(1, weight=1)
        ttk.Label(lang_frame, text="From").grid(
            row=0, column=0, sticky="w", padx=(0, 8), pady=4
        )
        self.translate_from_var = tk.StringVar()
        self.translate_from_combo = ttk.Combobox(
            lang_frame, textvariable=self.translate_from_var, state="readonly", width=28
        )
        self.translate_from_combo.grid(row=0, column=1, sticky="w", pady=4)
        ttk.Button(lang_frame, text="Refresh", command=self._refresh_languages).grid(
            row=0, column=2, padx=(8, 0), pady=4
        )
        ttk.Label(lang_frame, text="To").grid(
            row=1, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Label(lang_frame, text="English").grid(row=1, column=1, sticky="w", pady=4)
        self.translate_hint_var = tk.StringVar()
        ttk.Label(
            lang_frame,
            textvariable=self.translate_hint_var,
            wraplength=460,
            justify="left",
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 0))

        # --- Paste box -----------------------------------------------------
        paste_frame = ttk.LabelFrame(container, text="Translate text", padding=10)
        paste_frame.pack(fill="both", expand=True, pady=(10, 0))
        ttk.Label(paste_frame, text="Paste text to translate:").pack(anchor="w")
        self.translate_input = ScrolledText(
            paste_frame, wrap="word", height=6, font="TkFixedFont"
        )
        self.translate_input.pack(fill="both", expand=True, pady=(2, 6))
        paste_buttons = ttk.Frame(paste_frame)
        paste_buttons.pack(fill="x")
        ttk.Button(
            paste_buttons, text="Translate", command=self._translate_paste_in_thread
        ).pack(side="left")
        if self._ocr_available:
            # OCR a single image/PDF into the box so its text can be reviewed and
            # corrected before translating (OCR is rarely perfect).
            ttk.Button(
                paste_buttons,
                text="Extract from image/PDF…",
                command=self._extract_to_paste,
            ).pack(side="left", padx=(8, 0))
        result_header = ttk.Frame(paste_frame)
        result_header.pack(fill="x", pady=(6, 0))
        ttk.Label(result_header, text="Result:").pack(side="left")
        ttk.Button(result_header, text="Copy", command=self._copy_translation).pack(
            side="right"
        )
        ttk.Button(
            result_header, text="Save as Word…", command=self._save_translation_docx
        ).pack(side="right", padx=(0, 8))
        self.translate_output = ScrolledText(
            paste_frame, wrap="word", height=6, state="disabled", font="TkFixedFont"
        )
        self.translate_output.pack(fill="both", expand=True, pady=(2, 0))
        # Drop an image/PDF (or text file) onto the paste box to extract its text.
        self._register_drop(self.translate_input, self._on_drop_translate_source)

        # --- Batch files ---------------------------------------------------
        batch_frame = ttk.LabelFrame(
            container, text="Translate files (batch)", padding=10
        )
        batch_frame.pack(fill="x", pady=(10, 0))
        self._translate_files: List[Path] = []
        self.translate_files_var = tk.StringVar(value="No files selected.")
        row = ttk.Frame(batch_frame)
        row.pack(fill="x")
        ttk.Button(row, text="Add files…", command=self._add_translate_files).pack(
            side="left"
        )
        ttk.Button(row, text="Clear", command=self._clear_translate_files).pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(
            row, text="Translate files", command=self._translate_files_in_thread
        ).pack(side="left", padx=(8, 0))
        ttk.Label(
            batch_frame,
            textvariable=self.translate_files_var,
            wraplength=460,
            justify="left",
        ).pack(anchor="w", pady=(6, 0))
        batch_help = "Each file is translated to <name>.en.<ext> next to the original."
        if self._ocr_available:
            batch_help += (
                " Images and PDFs are OCR'd first (the extracted text is also saved "
                "as <name>.ocr.txt)."
            )
        ttk.Label(
            batch_frame,
            text=batch_help,
            wraplength=460,
            justify="left",
        ).pack(anchor="w")
        # Drop files onto the batch box to add them to the queue.
        self._register_drop(batch_frame, self._on_drop_translate_files)

        # --- Run controls + progress --------------------------------------
        run_frame = ttk.Frame(container)
        run_frame.pack(fill="x", pady=(12, 0))
        run_frame.columnconfigure(1, weight=1)
        self.translate_cancel_button = ttk.Button(
            run_frame, text="Cancel", command=self.cancel, state="disabled"
        )
        self.translate_cancel_button.grid(row=0, column=0, sticky="w")
        self.translate_progress = ttk.Progressbar(run_frame, mode="determinate")
        self.translate_progress.grid(row=0, column=1, sticky="ew", padx=(10, 0))
        self.translate_status_var = tk.StringVar(value="Idle")
        ttk.Label(run_frame, textvariable=self.translate_status_var).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )

        self._refresh_languages()
        self._bind_wheel(translate_canvas, container)

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

    # -- Thread-safe widget helpers ---------------------------------------

    def _append(self, widget: ScrolledText, text: str) -> None:
        def _do() -> None:
            widget.configure(state="normal")
            widget.insert("end", str(text) + "\n")
            widget.see("end")
            widget.configure(state="disabled")

        widget.after(0, _do)

    def _clear(self, widget: ScrolledText) -> None:
        def _do() -> None:
            widget.configure(state="normal")
            widget.delete("1.0", "end")
            widget.configure(state="disabled")

        widget.after(0, _do)

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

    # -- Callbacks ---------------------------------------------------------

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

    def run_in_thread(self) -> None:
        # Collapse the settings so the transcript and progress get the space.
        self._collapse_all_settings()
        self._cancel_event.clear()
        threading.Thread(target=self._run, daemon=True).start()

    def cancel(self) -> None:
        """Request the running job (transcribe or translate) stop at its next
        cancellation checkpoint."""
        self._cancel_event.set()
        for button in (
            getattr(self, "cancel_button", None),
            getattr(self, "translate_cancel_button", None),
        ):
            if button is not None:
                button.configure(state="disabled")
        self._append(self.status, "Cancelling… (will stop at the next checkpoint)")
        self.progress_label_var.set("Cancelling…")
        if hasattr(self, "translate_status_var"):
            self.translate_status_var.set("Cancelling…")

    # -- Translation -------------------------------------------------------

    def _refresh_languages(self) -> None:
        self.translate_hint_var.set("Loading languages…")
        threading.Thread(target=self._load_languages, daemon=True).start()

    def _load_languages(self) -> None:
        try:
            from .translation import available_source_languages

            langs = available_source_languages()
            err: Optional[Exception] = None
        except Exception as exc:  # noqa: BLE001 - surfaced to the user as a hint
            langs, err = [], exc

        def _apply() -> None:
            self._translate_lang_codes = {name: code for code, name in langs}
            names = list(self._translate_lang_codes)
            # Offer automatic detection (langdetect) as the first choice.
            if names and self._detect_available:
                self._translate_lang_codes[AUTO_DETECT_LABEL] = AUTO_DETECT_CODE
                names = [AUTO_DETECT_LABEL] + names
            # Offer English explicitly so English images/PDFs can be OCR'd
            # (extract-only; there's no English->English translation).
            if names and self._ocr_available:
                self._translate_lang_codes[ENGLISH_LABEL] = "en"
                names = names + [ENGLISH_LABEL]
            self.translate_from_combo.configure(values=names)
            if names:
                if self.translate_from_var.get() not in names:
                    self.translate_from_var.set(names[0])
                self.translate_hint_var.set("")
            elif err is not None:
                self.translate_hint_var.set(self._friendly_error(err))
            else:
                self.translate_hint_var.set(
                    "No language packs found. Use a build with bundled packs, or "
                    "install Argos packs."
                )

        self.root.after(0, _apply)

    def _selected_from_code(self) -> Optional[str]:
        codes = getattr(self, "_translate_lang_codes", {})
        return codes.get(self.translate_from_var.get())

    def _installed_from_codes(self) -> set[str]:
        """Set of source language codes that have an installed pack to English."""
        codes = getattr(self, "_translate_lang_codes", {})
        # Exclude the auto sentinel and the target ("en"): there's no en->en pack.
        return {code for code in codes.values() if code not in (AUTO_DETECT_CODE, "en")}

    def _resolve_from_code(self, text: str) -> Optional[str]:
        """Resolve the translation source language for ``text``.

        Returns a concrete foreign language code with an installed pack, or
        ``None`` when there's nothing to translate (no selection, English source,
        auto-detect failed, or the detected language has no bundled pack) - the
        caller reports a friendly message or treats it as extract-only.
        """
        selected = self._selected_from_code()
        if selected == AUTO_DETECT_CODE:
            detected = detect_language(text)
            if detected and detected in self._installed_from_codes():
                return detected
            return None
        # A concrete foreign language; English ("en") is extract-only (no en->en).
        if selected and selected != "en":
            return selected
        return None

    def _ocr_lang_code(self) -> str:
        """Tesseract language for OCR, from the selected 'From' language.

        When a specific 'From' language is chosen we use it; otherwise (Auto-detect)
        we default to Latin/English so the common case - English/Latin-script
        documents - just works. For a non-Latin scan, pick that language first.
        """
        selected = self._selected_from_code()
        if not selected or selected == AUTO_DETECT_CODE:
            return DEFAULT_OCR_LANG
        return tesseract_lang(selected)

    def _add_translate_files(self) -> None:
        if self._ocr_available:
            ocr_patterns = " ".join(f"*{ext}" for ext in OCR_EXTENSIONS)
            filetypes = [
                ("Text, images & PDFs", f"*.txt {ocr_patterns}"),
                ("All files", "*.*"),
            ]
        else:
            filetypes = [("Text files", "*.txt"), ("All files", "*.*")]
        paths = filedialog.askopenfilenames(filetypes=filetypes)
        self._add_translate_paths([Path(raw) for raw in paths if raw])

    def _add_translate_paths(self, paths: List[Path]) -> None:
        for path in paths:
            if path not in self._translate_files:
                self._translate_files.append(path)
        self._update_translate_files_label()

    def _clear_translate_files(self) -> None:
        self._translate_files = []
        self._update_translate_files_label()

    def _update_translate_files_label(self) -> None:
        count = len(self._translate_files)
        if not count:
            self.translate_files_var.set("No files selected.")
            return
        names = ", ".join(p.name for p in self._translate_files[:5])
        more = "" if count <= 5 else f" (+{count - 5} more)"
        self.translate_files_var.set(f"{count} file(s): {names}{more}")

    def _set_translate_busy(self, busy: bool, message: Optional[str] = None) -> None:
        def _do() -> None:
            self.translate_cancel_button.configure(
                state="normal" if busy else "disabled"
            )
            if not busy:
                self.translate_progress["value"] = 0
            self.translate_status_var.set(message or ("Working…" if busy else "Idle"))

        self.root.after(0, _do)

    def _set_translate_progress(self, fraction: float) -> None:
        pct = max(0.0, min(1.0, fraction)) * 100.0
        self.root.after(0, lambda: self.translate_progress.configure(value=pct))

    def _set_translate_status(self, message: str) -> None:
        self.root.after(0, lambda: self.translate_status_var.set(message))

    def _set_translate_output(self, text: str) -> None:
        def _do() -> None:
            self.translate_output.configure(state="normal")
            self.translate_output.delete("1.0", "end")
            self.translate_output.insert("end", text)
            self.translate_output.configure(state="disabled")

        self.root.after(0, _do)

    def _translate_paste_in_thread(self) -> None:
        if not self._selected_from_code():
            self.translate_status_var.set("Pick a 'From' language first.")
            return
        text = self.translate_input.get("1.0", "end-1c")
        if not text.strip():
            self.translate_status_var.set("Nothing to translate.")
            return
        self._cancel_event.clear()
        threading.Thread(
            target=self._translate_paste, args=(text,), daemon=True
        ).start()

    def _translate_paste(self, text: str) -> None:
        from .translation import translate_text

        self._set_translate_busy(True, "Translating…")
        final = "Done"
        try:
            from_code = self._resolve_from_code(text)
            if not from_code:
                if self._selected_from_code() == "en":
                    self._set_translate_output(
                        "That's English already — there's nothing to translate. To "
                        "pull text out of an image or PDF, use 'Extract from "
                        "image/PDF…'."
                    )
                else:
                    self._set_translate_output(
                        "Couldn't determine the source language. Pick a specific "
                        "'From' language (auto-detect found no bundled pack for this "
                        "text)."
                    )
                return
            result = translate_text(
                text,
                from_code=from_code,
                to_code="en",
                on_progress=self._set_translate_progress,
                cancelled=self._cancel_event.is_set,
            )
            self._set_translate_output(result)
        except CancelledError:
            final = "Cancelled"
        except Exception as exc:  # noqa: BLE001
            self._set_translate_output(self._friendly_error(exc))
            final = "Error"
        finally:
            self._set_translate_busy(False, final)

    def _translate_files_in_thread(self) -> None:
        if not self._selected_from_code():
            self.translate_status_var.set("Pick a 'From' language first.")
            return
        if not self._translate_files:
            self.translate_status_var.set("Add files first.")
            return
        self._cancel_event.clear()
        files = list(self._translate_files)
        threading.Thread(
            target=self._translate_files_worker, args=(files,), daemon=True
        ).start()

    def _translate_files_worker(self, files: List[Path]) -> None:
        from .translation import translate_text

        self._set_translate_busy(True, "Translating files…")
        final = "Done"
        translated_count = 0
        extracted_count = 0
        skipped = 0
        try:
            total = max(1, len(files))
            for index, src in enumerate(files):
                if self._cancel_event.is_set():
                    raise CancelledError("Translation cancelled.")
                # Get the foreign text: read text files directly, OCR images/PDFs.
                is_ocr = is_ocr_file(src)
                if is_ocr:
                    self._set_translate_status(f"Reading {src.name} (OCR)…")
                    text = extract_text(
                        src,
                        lang=self._ocr_lang_code(),
                        progress=self._set_translate_status,
                        cancelled=self._cancel_event.is_set,
                    )
                    # Always keep the extracted text, even if we can't translate it.
                    ocr_dest = src.with_name(f"{src.stem}.ocr.txt")
                    ocr_dest.write_text(text, encoding="utf-8")
                    extracted_count += 1
                    self._set_translate_status(f"Extracted {ocr_dest.name}")
                    dest = src.with_name(f"{src.stem}.en.txt")
                else:
                    text = src.read_text(encoding="utf-8", errors="replace")
                    dest = src.with_name(f"{src.stem}.en{src.suffix}")

                from_code = self._resolve_from_code(text)
                if not from_code:
                    # OCR files already produced .ocr.txt; text files yield nothing.
                    if is_ocr:
                        self._set_translate_status(
                            f"{src.name}: extracted text only (no foreign language "
                            "to translate)."
                        )
                    else:
                        self._set_translate_status(
                            f"Skipped {src.name}: couldn't determine its language."
                        )
                        skipped += 1
                    self._set_translate_progress((index + 1) / total)
                    continue
                self._set_translate_status(f"Translating {src.name}…")
                translated = translate_text(
                    text,
                    from_code=from_code,
                    to_code="en",
                    cancelled=self._cancel_event.is_set,
                )
                dest.write_text(translated, encoding="utf-8")
                translated_count += 1
                self._set_translate_status(f"Wrote {dest.name}")
                self._set_translate_progress((index + 1) / total)
            parts = []
            if translated_count:
                parts.append(f"translated {translated_count}")
            if extracted_count:
                parts.append(f"extracted {extracted_count}")
            if skipped:
                parts.append(f"skipped {skipped}")
            final = "Done — " + (", ".join(parts) if parts else "nothing to do")
        except CancelledError:
            final = "Cancelled"
        except Exception as exc:  # noqa: BLE001
            # Show the cause in the Result box; the status line is too short and
            # gets overwritten by the final "busy off" message below.
            self._set_translate_output(self._friendly_error(exc))
            final = "Error — see Result box"
        finally:
            self._set_translate_busy(False, final)

    def _set_translate_input(self, text: str) -> None:
        def _do() -> None:
            self.translate_input.delete("1.0", "end")
            self.translate_input.insert("end", text)

        self.root.after(0, _do)

    def _extract_to_paste(self) -> None:
        """OCR a single image/PDF into the paste box for review before translating."""
        lang = self._ocr_lang_code()
        patterns = " ".join(f"*{ext}" for ext in OCR_EXTENSIONS)
        path = filedialog.askopenfilename(
            title="Choose an image or PDF",
            filetypes=[("Images & PDFs", patterns), ("All files", "*.*")],
        )
        if not path:
            return
        self._cancel_event.clear()
        threading.Thread(
            target=self._extract_to_paste_worker, args=(Path(path), lang), daemon=True
        ).start()

    def _extract_to_paste_worker(self, path: Path, lang: str) -> None:
        self._set_translate_busy(True, f"Reading {path.name} (OCR)…")
        final = "Extracted — review, then Translate"
        try:
            text = extract_text(
                path,
                lang=lang,
                progress=self._set_translate_status,
                cancelled=self._cancel_event.is_set,
            )
            self._set_translate_input(text)
            if not text.strip():
                final = "No text found in that file."
        except CancelledError:
            final = "Cancelled"
        except Exception as exc:  # noqa: BLE001
            self._set_translate_output(self._friendly_error(exc))
            final = "Error — see Result box"
        finally:
            self._set_translate_busy(False, final)

    def _copy_translation(self) -> None:
        text = self.translate_output.get("1.0", "end-1c")
        if not text.strip():
            self.translate_status_var.set("Nothing to copy yet.")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.translate_status_var.set("Translation copied to clipboard.")

    def _save_translation_docx(self) -> None:
        text = self.translate_output.get("1.0", "end-1c")
        if not text.strip():
            self.translate_status_var.set("Nothing to save yet.")
            return
        path = filedialog.asksaveasfilename(
            title="Save translation as Word",
            defaultextension=".docx",
            initialfile="translation.docx",
            filetypes=[("Word document", "*.docx")],
        )
        if not path:
            return
        try:
            text_to_docx(text, path)
            self.translate_status_var.set(f"Saved {Path(path).name}")
        except Exception as exc:  # noqa: BLE001
            self.translate_status_var.set(self._friendly_error(exc))

    def _on_drop_translate_source(self, paths: List[Path]) -> None:
        """A file dropped on the paste box: OCR an image/PDF, or load a text file."""
        path = paths[0]
        if is_ocr_file(path):
            lang = self._ocr_lang_code()
            self._cancel_event.clear()
            threading.Thread(
                target=self._extract_to_paste_worker, args=(path, lang), daemon=True
            ).start()
        else:
            try:
                self._set_translate_input(
                    path.read_text(encoding="utf-8", errors="replace")
                )
            except OSError as exc:
                self.translate_status_var.set(self._friendly_error(exc))

    def _on_drop_translate_files(self, paths: List[Path]) -> None:
        """Files dropped on the batch box: add them to the queue."""
        self._add_translate_paths(paths)
        self.translate_status_var.set(f"Added {len(paths)} file(s) to the batch.")

    def _run(self) -> None:
        task = self.task_var.get()
        self._set_busy(
            True, "Translating..." if task == "translate" else "Transcribing..."
        )
        final_status = "Finished"
        temp_wav: Optional[Path] = None
        try:
            self._clear(self.output)
            path = self.input_file_var.get()
            if not path or not Path(path).exists():
                self._append(
                    self.status,
                    "Couldn't find that file. Pick an audio or video file with "
                    f"Browse… (got: {path or 'nothing selected'}).",
                )
                final_status = "No input file"
                return

            language = self.language_var.get().strip()
            language_arg = None if language in ("", "Auto") else language
            outdir = self.output_dir_var.get() if self.write_output_var.get() else None

            self._append(self.status, f"Processing: {path}")

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
                    progress=lambda msg: self._append(self.status, msg),
                )
                media_is_normalized = True
                if wav_dest is None:
                    temp_wav = media_path
                self._append(self.status, f"Converted to {media_path}")

            # Resolve a bundled model name to its local directory so we never
            # try to download on an air-gapped machine.
            model_sel = self.model_var.get()
            model = str(self._bundled_models.get(model_sel, model_sel))

            transcribe_label = "Translating" if task == "translate" else "Transcribing"
            result = transcribe_audio(
                media_path,
                model_size=model,
                task=task,
                language=language_arg,
                vad_filter=self.vad_var.get(),
                # Word timestamps are only needed for diarization; skipping them
                # when not diarizing makes plain transcription noticeably faster.
                word_timestamps=self.diarize_var.get(),
                progress=lambda msg: self._append(self.status, msg),
                on_progress=lambda f: self._set_progress(f, transcribe_label),
                cancelled=self._cancel_event.is_set,
            )

            self._append(
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
            self._render_transcript()

            if outdir:
                self._save_outputs(result, Path(path), Path(outdir))

            self._append(self.status, "Finished.")
        except CancelledError:
            self._append(self.status, "Cancelled.")
            final_status = "Cancelled"
        except Exception as exc:
            self._append(self.status, self._friendly_error(exc))
            # Keep the full traceback in the log for troubleshooting.
            self._append(self.status, traceback.format_exc())
            final_status = "Error"
        finally:
            if temp_wav is not None:
                try:
                    temp_wav.unlink()
                except OSError:
                    pass
            self._set_busy(False, final_status)

    def _friendly_error(self, exc: Exception) -> str:
        """Map a raw exception to a plain-English line for non-technical users.

        The full traceback is still written to the Status log right after this.
        """
        name = type(exc).__name__
        msg = str(exc)
        low = msg.lower()
        if isinstance(exc, FileNotFoundError) or "no such file" in low:
            return f"A required file was missing: {msg}"
        if "ffmpeg" in low:
            return (
                "Couldn't run ffmpeg, which is needed to read this file. The "
                "packaged app includes ffmpeg; if running from source, install "
                "ffmpeg and make sure it's on your PATH."
            )
        if "faster-whisper is not installed" in low or "faster_whisper" in low:
            return (
                "The transcription engine isn't installed "
                "(pip install 'silvance-whisper[gui]')."
            )
        if "sherpa-onnx is not installed" in low:
            return (
                "The sherpa speaker engine isn't installed. Switch Engine to "
                "pyannote, or install sherpa-onnx."
            )
        if "pyannote.audio is not installed" in low:
            return (
                "The pyannote speaker engine isn't installed. Switch Engine to "
                "sherpa, or use a build that includes pyannote."
            )
        if (
            "no diarization models" in low
            or "offline mode" in low
            or "localentrynotfound" in name.lower()
        ):
            return (
                "Couldn't load the speaker models. Use a build that bundles them, "
                "switch the Engine, or untick 'Identify speakers' to transcribe only."
            )
        if "memoryerror" in name.lower() or "out of memory" in low:
            return (
                "Ran out of memory. Try a smaller model (e.g. base.en) or a "
                "shorter recording."
            )
        if "tesseract" in low:
            return (
                "OCR needs the Tesseract engine, which this build can't find. Use a "
                "build that bundles it (set 'ocr_langs' in the release workflow), or "
                "install Tesseract."
            )
        if "pytesseract" in low or "pypdfium2" in low or "pillow" in low:
            return (
                "The OCR components aren't installed. Use a build with OCR bundled "
                "(pip install 'silvance-whisper[ocr]')."
            )
        short = msg.splitlines()[0] if msg else name
        return f"Something went wrong ({name}): {short}"

    def _parse_num_speakers(self) -> Optional[int]:
        raw = self.num_speakers_var.get().strip()
        if not raw:
            return None
        try:
            value = int(raw)
        except ValueError:
            self._append(self.status, f"Ignoring invalid speaker count: {raw!r}")
            return None
        return value if value > 0 else None

    def _parse_threshold(self) -> float:
        raw = self.sensitivity_var.get().strip()
        if not raw:
            return 0.5
        try:
            value = float(raw)
        except ValueError:
            self._append(self.status, f"Ignoring invalid sensitivity: {raw!r}")
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
            self._append(self.status, "Preparing audio for diarization...")
            diar_wav = convert_to_wav(
                source, progress=lambda msg: self._append(self.status, msg)
            )
            diar_temp = diar_wav
        try:
            speaker_segments = diarize(
                diar_wav,
                backend=ENGINE_CHOICES.get(self.engine_var.get(), "auto"),
                num_speakers=self._parse_num_speakers(),
                threshold=self._parse_threshold(),
                progress=lambda msg: self._append(self.status, msg),
                on_progress=lambda f: self._set_progress(f, "Identifying speakers"),
                cancelled=self._cancel_event.is_set,
            )
            result.segments = assign_speakers(result.segments, speaker_segments)
            count = len({seg.speaker for seg in speaker_segments})
            self._append(self.status, f"Identified {count} speaker(s).")
        finally:
            if diar_temp is not None:
                try:
                    diar_temp.unlink()
                except OSError:
                    pass

    def _save_outputs(
        self, result: TranscriptionResult, source: Path, outdir: Path
    ) -> None:
        if not outdir.is_dir():
            self._append(self.status, f"Output folder does not exist: {outdir}")
            return
        names = self._speaker_names
        txt_path = outdir / (source.name + ".txt")
        txt_path.write_text(
            result.to_txt(names, blank_lines=self.blank_lines_var.get()),
            encoding="utf-8",
        )
        self._append(self.status, f"Wrote transcript to {txt_path}")
        if self.srt_var.get():
            srt_path = outdir / (source.name + ".srt")
            srt_path.write_text(result.to_srt(names), encoding="utf-8")
            self._append(self.status, f"Wrote subtitles to {srt_path}")

    # -- Transcript copy / export -----------------------------------------

    def _copy_transcript(self) -> None:
        """Copy the rendered transcript text to the clipboard."""
        text = self.output.get("1.0", "end-1c")
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
            self.progress_label_var.set(self._friendly_error(exc))

    def _on_drop_media(self, paths: List[Path]) -> None:
        """Handle a file dropped on the transcript pane: load it as input."""
        if paths:
            self.input_file_var.set(str(paths[0]))
            self.progress_label_var.set(f"Loaded {paths[0].name}")

    # -- Transcript rendering + speaker renaming ---------------------------

    def _render_transcript(self) -> None:
        """Render the current result into the Transcript tab. When diarized,
        each speaker tag is clickable to rename that speaker."""

        def _do() -> None:
            result = self._result
            self.output.configure(state="normal")
            self.output.delete("1.0", "end")
            if result is None:
                self.output.configure(state="disabled")
                return
            # Blank line between segments/turns when enabled (easier to read/paste).
            line_end = "\n\n" if self.blank_lines_var.get() else "\n"
            if not result.has_speakers:
                if result.segments:
                    self.output.insert(
                        "end", line_end.join(s.text for s in result.segments) + "\n"
                    )
                else:
                    self.output.insert("end", result.text + "\n")
            else:
                bound: set[str] = set()
                for index, segment in enumerate(result.segments):
                    sid = segment.speaker or "UNKNOWN"
                    name = self._speaker_names.get(sid, sid)
                    spk_tag = f"spk::{sid}"
                    line_tag = f"line::{index}"
                    if sid not in bound:
                        bound.add(sid)
                        self.output.tag_config(spk_tag, underline=True)
                    # Bind on a per-line tag so a click knows which segment it
                    # hit: the menu can both fix this one line and rename globally.
                    self.output.tag_bind(
                        line_tag, "<Button-1>", self._speaker_menu_handler(index)
                    )
                    self.output.tag_bind(
                        line_tag, "<Enter>", self._cursor_handler("hand2")
                    )
                    self.output.tag_bind(line_tag, "<Leave>", self._cursor_handler(""))
                    self.output.insert("end", f"[{name}]", (spk_tag, line_tag))
                    if segment.words:
                        # Render words individually so a single misattributed
                        # word can be clicked and moved to another speaker.
                        for w_index, word in enumerate(segment.words):
                            text = word.word
                            if w_index == 0 and not text[:1].isspace():
                                text = " " + text
                            wtag = f"word::{index}::{w_index}"
                            self.output.tag_bind(
                                wtag,
                                "<Button-1>",
                                self._word_menu_handler(index, w_index),
                            )
                            self.output.tag_bind(
                                wtag, "<Enter>", self._cursor_handler("hand2")
                            )
                            self.output.tag_bind(
                                wtag, "<Leave>", self._cursor_handler("")
                            )
                            self.output.insert("end", text, (wtag,))
                        self.output.insert("end", line_end)
                    else:
                        self.output.insert("end", f" {segment.text}{line_end}")
            self.output.configure(state="disabled")

        self.root.after(0, _do)

    def _cursor_handler(self, cursor: str) -> Callable[[object], None]:
        def handler(_event: object) -> None:
            self.output.config(cursor=cursor)

        return handler

    def _ordered_speaker_ids(self) -> List[str]:
        """Distinct speaker ids in first-appearance order across the result."""
        ids: List[str] = []
        seen: set[str] = set()
        if self._result is not None:
            for segment in self._result.segments:
                sid = segment.speaker or "UNKNOWN"
                if sid not in seen:
                    seen.add(sid)
                    ids.append(sid)
        return ids

    def _speaker_menu_handler(self, index: int) -> Callable[[object], None]:
        def handler(event: object) -> None:
            result = self._result
            if result is None or index >= len(result.segments):
                return
            current = result.segments[index].speaker or "UNKNOWN"
            menu = tk.Menu(self.root, tearoff=0)
            # Reassign just this line to the correct speaker - the fix for the
            # boundary/overlap errors diarization can't get right on its own.
            for sid in self._ordered_speaker_ids():
                name = self._speaker_names.get(sid, sid)
                mark = "  ✓" if sid == current else ""
                menu.add_command(
                    label=f"This line is {name}{mark}",
                    command=self._reassign_command(index, sid),
                )
            menu.add_separator()
            cur_name = self._speaker_names.get(current, current)
            menu.add_command(
                label=f"Rename '{cur_name}' everywhere…",
                command=lambda: self._rename_speaker(current),
            )
            try:
                menu.tk_popup(event.x_root, event.y_root)  # type: ignore[attr-defined]
            finally:
                menu.grab_release()

        return handler

    def _reassign_command(self, index: int, speaker_id: str) -> Callable[[], None]:
        def command() -> None:
            self._reassign_segment(index, speaker_id)

        return command

    def _reassign_segment(self, index: int, speaker_id: str) -> None:
        result = self._result
        if result is None or index >= len(result.segments):
            return
        result.segments[index].speaker = speaker_id
        self._render_transcript()
        if self._result_source and self._result_outdir:
            self._save_outputs(result, self._result_source, self._result_outdir)

    def _word_menu_handler(
        self, seg_index: int, word_index: int
    ) -> Callable[[object], None]:
        def handler(event: object) -> None:
            result = self._result
            if result is None or seg_index >= len(result.segments):
                return
            segment = result.segments[seg_index]
            if word_index >= len(segment.words):
                return
            current = segment.speaker or "UNKNOWN"
            word_text = segment.words[word_index].word.strip()
            menu = tk.Menu(self.root, tearoff=0)
            # Move just this word (splits the segment around it).
            for sid in self._ordered_speaker_ids():
                if sid == current:
                    continue
                name = self._speaker_names.get(sid, sid)
                menu.add_command(
                    label=f"Move “{word_text}” → {name}",
                    command=self._word_command(seg_index, word_index, sid, False),
                )
            menu.add_separator()
            # Reassign from this word to the end of the line (the common case: a
            # new speaker's turn actually starts mid-line).
            for sid in self._ordered_speaker_ids():
                if sid == current:
                    continue
                name = self._speaker_names.get(sid, sid)
                menu.add_command(
                    label=f"From “{word_text}” onward → {name}",
                    command=self._word_command(seg_index, word_index, sid, True),
                )
            try:
                menu.tk_popup(event.x_root, event.y_root)  # type: ignore[attr-defined]
            finally:
                menu.grab_release()

        return handler

    def _word_command(
        self, seg_index: int, word_index: int, speaker_id: str, to_end: bool
    ) -> Callable[[], None]:
        def command() -> None:
            self._reassign_word_span(seg_index, word_index, speaker_id, to_end)

        return command

    def _reassign_word_span(
        self, seg_index: int, word_index: int, speaker_id: str, to_end: bool
    ) -> None:
        result = self._result
        if result is None or seg_index >= len(result.segments):
            return
        segment = result.segments[seg_index]
        words = segment.words
        if word_index >= len(words):
            return
        end = len(words) if to_end else word_index + 1
        before, middle, after = (
            words[:word_index],
            words[word_index:end],
            words[end:],
        )

        def _make(chunk: List[Word], spk: Optional[str]) -> Optional[Segment]:
            if not chunk:
                return None
            return Segment(
                start=chunk[0].start,
                end=chunk[-1].end,
                text="".join(w.word for w in chunk).strip(),
                speaker=spk,
                words=list(chunk),
            )

        parts = [
            seg
            for seg in (
                _make(before, segment.speaker),
                _make(middle, speaker_id),
                _make(after, segment.speaker),
            )
            if seg is not None
        ]
        result.segments[seg_index : seg_index + 1] = parts
        result.segments = self._coalesce_segments(result.segments)
        self._render_transcript()
        if self._result_source and self._result_outdir:
            self._save_outputs(result, self._result_source, self._result_outdir)

    @staticmethod
    def _coalesce_segments(segments: List[Segment]) -> List[Segment]:
        """Merge adjacent segments that share a speaker (rebuilding text/words)."""
        out: List[Segment] = []
        for seg in segments:
            if out and out[-1].speaker == seg.speaker and out[-1].words and seg.words:
                prev = out[-1]
                prev.words = prev.words + seg.words
                prev.end = seg.end
                prev.text = "".join(w.word for w in prev.words).strip()
            else:
                out.append(seg)
        return out

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

    def _rename_speaker(self, speaker_id: str) -> None:
        current = self._speaker_names.get(speaker_id, speaker_id)
        new_name = simpledialog.askstring(
            "Rename speaker",
            f"New name for {current}:",
            initialvalue=current,
            parent=self.root,
        )
        if not new_name or not new_name.strip():
            return
        self._speaker_names[speaker_id] = new_name.strip()
        self._render_transcript()
        # Keep saved files in sync if an output folder was used.
        if self._result is not None and self._result_source and self._result_outdir:
            self._save_outputs(self._result, self._result_source, self._result_outdir)


def main() -> None:
    """Launch the Whispers GUI."""
    # Must happen before faster-whisper / pyannote / argostranslate are imported
    # (which only occurs once a job runs), so configuring at startup is early
    # enough. Points HF and Argos at the bundled offline caches when present;
    # both are no-ops otherwise.
    configure_offline_hf_cache()
    configure_offline_translation()
    configure_offline_ocr()

    try:
        # ttkbootstrap gives a modern theme; fall back to stock Tk if absent.
        import ttkbootstrap as tb

        root = tb.Window(themename="darkly")
    except ImportError:
        root = tk.Tk()

    # Best-effort: load the tkdnd extension so files can be dragged onto the
    # window. No-op (and the app works normally) when tkinterdnd2 isn't bundled.
    dnd_ok = _enable_drag_and_drop(root)

    WhisprApp(root, drag_and_drop=dnd_ok)
    root.mainloop()


def _enable_drag_and_drop(root: tk.Misc) -> bool:
    """Initialise tkinterdnd2's tkdnd on ``root``; return True on success.

    We load tkdnd onto the existing (ttkbootstrap-themed) root rather than using
    ``TkinterDnD.Tk()`` so the theme is preserved. Individual widgets opt in via
    :meth:`WhisprApp._register_drop`.
    """
    try:
        from tkinterdnd2 import TkinterDnD

        TkinterDnD._require(root)
        return True
    except Exception:  # noqa: BLE001 - DnD is a convenience; never block startup
        return False


if __name__ == "__main__":
    main()
