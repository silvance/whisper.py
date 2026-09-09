"""Entry point for Whispers: the GUI, or a headless self-test.

The GUI itself lives in :mod:`whispr.gui`; this is the thin front door that
``whispr.app:main`` names, so the console script, ``python -m whispr`` and the
PyInstaller bundle entry (``packaging/whispr_entry.py``) all go through one
place.

It deliberately does **not** import the GUI at module level. ``--self-test``
has to work on a machine with no display and has to keep working when the
window toolkit is exactly what is broken - a build missing Tk should report
that, not crash on the way to saying it.
"""

from __future__ import annotations

import sys
from typing import Any, List, Optional

# WhisprApp is resolved lazily by __getattr__ below, so it is not listed here:
# naming it would make a static check demand a module-level import, which is the
# one thing this module exists to avoid.
__all__ = ["main", "self_test", "launch"]

USAGE = """Whispers - offline audio analysis

  whispr               open the application
  whispr --self-test   print what this copy can do, and exit
  whispr --help        show this message
"""


def self_test() -> int:
    """Print the build self-test and return an exit code.

    The same report the application shows under System status, on stdout. Two
    uses: a build can prove the executable it just made actually starts, and an
    operator on an air-gapped machine can capture what their copy can do
    (``whispr --self-test > report.txt``) without opening a window.

    A build that deliberately leaves translation or OCR out is not ready for
    everything and is not broken for it, so the report is printed and this
    returns 0 either way. Only an application that cannot run at all fails -
    by raising on the way here, long before this line.
    """
    from .diagnostics import format_report

    print(format_report())
    return 0


def main(argv: "Optional[List[str]]" = None) -> None:
    """Run Whispers: the self-test if asked for it, otherwise the window."""
    args = sys.argv[1:] if argv is None else list(argv)
    if "--self-test" in args:
        raise SystemExit(self_test())
    if "--help" in args or "-h" in args:
        print(USAGE)
        raise SystemExit(0)
    launch()


def launch() -> None:
    """Open the main window. Imported here so --self-test never needs Tk."""
    from .gui.app import main as gui_main

    gui_main()


def __getattr__(name: str) -> Any:
    """Keep ``from whispr.app import WhisprApp`` working, without importing Tk."""
    if name == "WhisprApp":
        from .gui.app import WhisprApp

        return WhisprApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    main()
