"""Per-operation profiles: saved settings + learned speaker voiceprints.

A *profile* bundles everything that is specific to one recurring operation or
team so an operator can pick it once and get consistent, improving results:

* the transcription settings to use (model, diarization engine, expected speaker
  count, custom vocabulary, ...);
* the roster of speaker voiceprints learned from that operation's corrections
  (see :mod:`whispr.voiceprints`).

Profiles live as small JSON files under the same per-user config location as the
GUI settings, so they persist between launches and survive app updates. As with
:mod:`whispr.settings`, all disk operations are best-effort: a missing or
read-only config dir just means profiles can't be saved, never a crash.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .settings import settings_path
from .voiceprints import Voiceprint


def profiles_dir() -> Path:
    """Directory holding the per-user profile files (beside ``settings.json``)."""
    return settings_path().parent / "profiles"


def _slug(name: str) -> str:
    """A filesystem-safe stem for ``name`` (keeps letters/digits/-/_)."""
    cleaned = "".join(
        ch if (ch.isalnum() or ch in "-_") else "_" for ch in name.strip()
    ).strip("_")
    return cleaned.lower() or "profile"


@dataclass
class Profile:
    """One operation's saved settings and learned speaker voiceprints."""

    name: str
    settings: Dict[str, object] = field(default_factory=dict)
    voiceprints: Dict[str, Voiceprint] = field(default_factory=dict)

    def voiceprint_for(self, speaker_name: str) -> Voiceprint:
        """The voiceprint for ``speaker_name``, created (empty) if new."""
        vp = self.voiceprints.get(speaker_name)
        if vp is None:
            vp = Voiceprint(name=speaker_name)
            self.voiceprints[speaker_name] = vp
        return vp

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "settings": self.settings,
            "voiceprints": [vp.to_dict() for vp in self.voiceprints.values()],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "Profile":
        name = str(data.get("name", "")) or "Profile"
        settings = data.get("settings")
        settings = settings if isinstance(settings, dict) else {}
        voiceprints: Dict[str, Voiceprint] = {}
        raw = data.get("voiceprints") or []
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    vp = Voiceprint.from_dict(item)
                    if vp.name:
                        voiceprints[vp.name] = vp
        return cls(name=name, settings=settings, voiceprints=voiceprints)


def list_profiles() -> List[str]:
    """Names of the saved profiles, sorted, or ``[]`` if none/unreadable."""
    names: List[str] = []
    try:
        entries = sorted(profiles_dir().glob("*.json"))
    except OSError:
        return names
    for path in entries:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - skip a corrupt/unreadable profile
            continue
        if isinstance(data, dict) and data.get("name"):
            names.append(str(data["name"]))
    return sorted(names)


def load_profile(name: str) -> Optional[Profile]:
    """Load the profile called ``name``, or ``None`` if missing/unreadable."""
    path = profiles_dir() / f"{_slug(name)}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - missing/corrupt -> no profile
        return None
    if not isinstance(data, dict):
        return None
    return Profile.from_dict(data)


def save_profile(profile: Profile) -> None:
    """Write ``profile`` to its JSON file (best-effort; never raises on OSError)."""
    try:
        directory = profiles_dir()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{_slug(profile.name)}.json"
        path.write_text(
            json.dumps(profile.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass


def delete_profile(name: str) -> None:
    """Remove the profile called ``name`` (best-effort)."""
    try:
        (profiles_dir() / f"{_slug(name)}.json").unlink()
    except OSError:
        pass
