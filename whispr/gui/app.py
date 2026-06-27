"""The Whispers application shell: window, header, tabs, and entry point.

Holds the shared cancel signal and assembles the Transcribe and (when available)
Translate tabs. ``whispr.app`` re-exports :func:`main` so the console script,
``python -m whispr`` and the PyInstaller bundle entry keep working unchanged.
"""

from __future__ import annotations

import importlib.util
import os
import threading
import tkinter as tk
from tkinter import ttk

from ..ocr import ocr_available
from ..resources import (
    bundled_argos_data_dir,
    configure_offline_hf_cache,
    configure_offline_ocr,
    configure_offline_translation,
)
from ..settings import load_settings, save_settings
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

        # Translation is shown only when its engine/packs are bundled (so a lean
        # transcriber-only build is a clean single-purpose app), and can be force-
        # hidden with WHISPR_MODE=transcribe even on a full bundle.
        show_translate = (
            _translation_available()
            and os.environ.get("WHISPR_MODE", "").lower() != "transcribe"
        )

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

        # With translation, a top-level Transcribe/Translate notebook; without it,
        # the transcribe UI fills the window directly (no redundant tab chrome).
        if show_translate:
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

    def _on_close(self) -> None:
        """Persist preferences, then close the window."""
        try:
            save_settings({"transcribe": self.transcribe.get_settings()})
        except Exception:  # noqa: BLE001 - never block closing on a save failure
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
