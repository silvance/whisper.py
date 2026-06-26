"""Tkinter GUI for Whispers, split into focused modules.

- :mod:`whispr.gui.app` - the application shell and ``main`` entry point
- :mod:`whispr.gui.transcribe_tab` - the Transcribe tab (settings + run)
- :mod:`whispr.gui.translate_tab` - the Translate tab (text/OCR translation)
- :mod:`whispr.gui.transcript_view` - the interactive transcript pane
- :mod:`whispr.gui.widgets` / :mod:`whispr.gui.errors` - shared helpers

``whispr.app`` re-exports :func:`main` for the console script, ``python -m
whispr`` and the PyInstaller bundle entry.
"""

from .app import WhisprApp, main

__all__ = ["WhisprApp", "main"]
