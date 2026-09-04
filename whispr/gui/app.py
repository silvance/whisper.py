"""The Whispers application shell: window, header, tabs, and entry point.

Holds the shared cancel signal and assembles the tabs this build can actually
support: Transcribe always, then Speaker profiles and Speaker Compare when a
speaker-embedding model is bundled, Live when ffmpeg is present, and Translate
when the translation packs are. ``WHISPR_MODE=transcribe`` forces the lean
single-purpose window. ``whispr.app`` re-exports :func:`main` so the console
script, ``python -m whispr`` and the PyInstaller bundle entry keep working
unchanged.
"""

from __future__ import annotations

import importlib.util
import os
import threading
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from ..diagnostics import format_report
from ..ocr import ocr_available
from ..resources import (
    bundled_argos_data_dir,
    bundled_embedding_model,
    configure_offline_hf_cache,
    configure_offline_ocr,
    configure_offline_translation,
    find_ffmpeg,
)
from ..settings import load_settings, save_settings
from .live_tab import LiveTab
from .speaker_compare_tab import SpeakerCompareTab
from .speaker_profiles_tab import SpeakerProfilesTab
from .transcribe_tab import TranscribeTab
from .translate_tab import TranslateTab


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


class WhisprApp:
    """The main application window."""

    def __init__(self, root: tk.Tk, *, drag_and_drop: bool = False) -> None:
        self.root = root
        self.root.title("Whispers")
        # Whether tkdnd was loaded (see _enable_drag_and_drop) so tabs can register
        # file-drop targets.
        self._dnd_ok = drag_and_drop
        # One shared cancel signal; either tab's Cancel button stops the running
        # job (only one runs at a time).
        self.cancel_event = threading.Event()
        self._tabs: list = []
        # Remembered preferences from the last session (best-effort).
        self._settings = load_settings()
        self._build_ui()
        # Save preferences on close.
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        self.root.minsize(680, 480)
        try:
            self.root.geometry("860x760")
        except tk.TclError:
            pass

        # Translation and live capture are shown as extra tabs when available.
        # WHISPR_MODE=transcribe forces the lean single-purpose (file transcriber)
        # window even on a full bundle.
        full_mode = os.environ.get("WHISPR_MODE", "").lower() != "transcribe"
        show_translate = _translation_available() and full_mode
        # Live (stream) transcription needs ffmpeg to read the stream.
        show_live = find_ffmpeg() is not None and full_mode
        # Speaker enrolment/comparison need the bundled speaker-embedding model.
        show_speakers = bundled_embedding_model() is not None and full_mode

        # App header.
        subtitle = (
            "Offline transcription & translation"
            if show_translate
            else "Offline transcription"
        )
        header = ttk.Frame(self.root, padding=(14, 10, 14, 4))
        header.pack(fill="x")
        ttk.Label(header, text="Whispers", font=("", 17, "bold")).pack(side="left")
        ttk.Label(header, text=subtitle, font=("", 9)).pack(
            side="left", padx=(10, 0), pady=(7, 0)
        )
        # Build self-test: confirm which engines/models this bundle contains.
        ttk.Button(header, text="Self-test…", command=self._show_diagnostics).pack(
            side="right"
        )
        # Plain-language getting-started guide for first-time / non-technical users.
        ttk.Button(header, text="Help", command=self._show_help).pack(
            side="right", padx=(0, 8)
        )

        # With any extra tab (Translate and/or Live), a top-level notebook; with
        # only Transcribe, that UI fills the window directly (no redundant chrome).
        use_notebook = show_translate or show_live or show_speakers
        if use_notebook:
            notebook = ttk.Notebook(self.root)
            notebook.pack(fill="both", expand=True, padx=8, pady=(0, 8))
            transcribe_root = ttk.Frame(notebook)
            notebook.add(transcribe_root, text="Transcribe")
        else:
            transcribe_root = ttk.Frame(self.root)
            transcribe_root.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self.transcribe = TranscribeTab(
            transcribe_root,
            self.root,
            self.cancel_event,
            self.cancel,
            dnd_ok=self._dnd_ok,
        )
        self.transcribe.apply_settings(self._settings.get("transcribe", {}))
        self._tabs.append(self.transcribe)

        # Known-subject reference voices, and comparing a questioned speaker
        # against them. Both need the speaker-embedding model, so they appear
        # only when this build actually bundles one.
        if show_speakers:
            profiles_root = ttk.Frame(notebook)
            notebook.add(profiles_root, text="Speaker profiles")
            self.speaker_profiles = SpeakerProfilesTab(
                profiles_root, self.root, self.cancel_event, self.cancel
            )
            self._tabs.append(self.speaker_profiles)

            compare_root = ttk.Frame(notebook)
            notebook.add(compare_root, text="Speaker Compare")
            self.speaker_compare = SpeakerCompareTab(
                compare_root,
                self.root,
                self.cancel_event,
                self.cancel,
                # Lets a comparison report carry the current transcript and the
                # provenance of the run that produced it.
                get_analysis=self.transcribe.current_analysis,
            )
            self._tabs.append(self.speaker_compare)

        # Live (stream) transcription, next to Transcribe since it's closely related.
        if show_live:
            live_root = ttk.Frame(notebook)
            notebook.add(live_root, text="Live")
            self.live = LiveTab(live_root, self.root, self.cancel_event, self.cancel)
            self._tabs.append(self.live)

        if show_translate:
            translate_root = ttk.Frame(notebook)
            notebook.add(translate_root, text="Translate")
            self.translate = TranslateTab(
                translate_root,
                self.root,
                self.cancel_event,
                self.cancel,
                ocr_available=ocr_available(),
                detect_available=importlib.util.find_spec("langdetect") is not None,
                dnd_ok=self._dnd_ok,
            )
            self._tabs.append(self.translate)

    def cancel(self) -> None:
        """Request the running job (transcribe or translate) stop at its next
        cancellation checkpoint, and reflect that in each tab."""
        self.cancel_event.set()
        for tab in self._tabs:
            tab.notify_cancelling()

    def _show_help(self) -> None:
        """Open a short, plain-language guide for first-time / non-technical users."""
        guide = (
            "Whispers — quick start\n"
            "======================\n"
            "\n"
            "Transcribe an audio or video file\n"
            "---------------------------------\n"
            "1. Under “Input & output”, click Browse… and choose your file\n"
            "   (or just drag the file onto the window).\n"
            "2. To save a copy, tick “Save transcript to a folder” and pick one.\n"
            "3. Click Run. The text appears in the Transcript tab when it finishes.\n"
            "\n"
            "Tips\n"
            "----\n"
            "• Bigger model = more accurate words but slower. Start with the\n"
            "  default; switch under “Model & language” if you need more accuracy.\n"
            "• “Custom words” (under Model & language): type names, places or\n"
            "  callsigns you expect, so they’re spelled right.\n"
            "• Copy transcript / Save as Word… are below the transcript.\n"
            "\n"
            "Who said what (speakers)\n"
            "------------------------\n"
            "• Open “Speakers”, tick “Identify speakers”, and (if you know it)\n"
            "  enter how many people are talking.\n"
            "• In the transcript you can rename a speaker, or drag a run of words\n"
            "  onto another speaker’s line to reassign it.\n"
            "\n"
            "If something goes wrong\n"
            "-----------------------\n"
            "• The Status tab shows what happened, in plain language.\n"
            "• “Self-test…” (top-right) lists exactly what this copy can do —\n"
            "  which models, speaker engines and languages are built in.\n"
        )
        win = tk.Toplevel(self.root)
        win.title("Whispers — Help")
        win.transient(self.root)
        text = ScrolledText(win, wrap="word", width=74, height=26)
        text.pack(fill="both", expand=True, padx=8, pady=8)
        text.insert("end", guide)
        text.configure(state="disabled")
        ttk.Button(win, text="Close", command=win.destroy).pack(
            side="right", padx=8, pady=(0, 8)
        )

    def _show_diagnostics(self) -> None:
        """Open a window listing which engines/models this build actually has."""
        report = format_report()
        win = tk.Toplevel(self.root)
        win.title("Whispers — build self-test")
        win.transient(self.root)
        text = ScrolledText(win, wrap="none", width=72, height=22, font="TkFixedFont")
        text.pack(fill="both", expand=True, padx=8, pady=8)
        text.insert("end", report)
        text.configure(state="disabled")

        def _copy() -> None:
            self.root.clipboard_clear()
            self.root.clipboard_append(report)

        buttons = ttk.Frame(win)
        buttons.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(buttons, text="Copy", command=_copy).pack(side="right")
        ttk.Button(buttons, text="Close", command=win.destroy).pack(
            side="right", padx=(0, 8)
        )

    def _on_close(self) -> None:
        """Persist preferences, then close the window."""
        try:
            save_settings({"transcribe": self.transcribe.get_settings()})
        except Exception:  # noqa: BLE001 - never block closing on a save failure
            pass
        # Release any tab resources (e.g. a running live ffmpeg session).
        for tab in self._tabs:
            close = getattr(tab, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 - never block closing on cleanup
                    pass
        self.root.destroy()


def _enable_drag_and_drop(root: tk.Misc) -> bool:
    """Initialise tkinterdnd2's tkdnd on ``root``; return True on success.

    We load tkdnd onto the existing (ttkbootstrap-themed) root rather than using
    ``TkinterDnD.Tk()`` so the theme is preserved. Individual widgets opt in via
    :func:`whispr.gui.widgets.register_drop`.
    """
    try:
        from tkinterdnd2 import TkinterDnD

        TkinterDnD._require(root)
        return True
    except Exception:  # noqa: BLE001 - DnD is a convenience; never block startup
        return False


def main() -> None:
    """Launch the Whispers GUI."""
    # Must happen before faster-whisper / pyannote / argostranslate are imported
    # (which only occurs once a job runs), so configuring at startup is early
    # enough. Points HF, Argos and Tesseract at the bundled offline caches when
    # present; all are no-ops otherwise.
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


if __name__ == "__main__":
    main()
