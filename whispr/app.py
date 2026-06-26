"""Backwards-compatible entry point for the Whispers GUI.

The GUI was split into the :mod:`whispr.gui` package; this thin shim keeps
``whispr.app:main`` importable so the console script, ``python -m whispr`` and the
PyInstaller bundle entry (``packaging/whispr_entry.py``) keep working unchanged.
"""

from __future__ import annotations

from .gui.app import WhisprApp, main

__all__ = ["WhisprApp", "main"]


if __name__ == "__main__":
    main()
