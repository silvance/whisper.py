"""Persist a few GUI preferences between launches (best-effort, never fatal).

Stores a small JSON file in the user's config location so the last-used model,
output folder and options are remembered. All operations swallow errors: a
read-only or missing config dir just means settings aren't remembered, never a
crash.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


def settings_path() -> Path:
    """Per-user settings file location (``%APPDATA%`` on Windows, XDG elsewhere)."""
    if os.name == "nt":
        base = os.environ.get("APPDATA") or str(Path.home())
        return Path(base) / "Whispers" / "settings.json"
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "whispr" / "settings.json"


def load_settings() -> Dict[str, Any]:
    """Return the saved settings, or an empty dict if none/unreadable."""
    try:
        data = json.loads(settings_path().read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - missing/corrupt file -> no saved settings
        return {}
    return data if isinstance(data, dict) else {}


def save_settings(data: Dict[str, Any]) -> None:
    """Write ``data`` as JSON to the settings file (best-effort)."""
    try:
        path = settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError:
        pass
