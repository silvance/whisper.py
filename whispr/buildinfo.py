"""Identity of this application build, for provenance and the self-test.

An analysis record has to name the software that produced it. Where a bundle was
assembled by CI, ``whispr_assets/build_manifest.json`` carries the build id, git
commit, pinned dependency versions and the SHA-256 of every bundled model; from a
source checkout most of that is simply unknown, and is reported as such rather
than invented.

Reading is local and best-effort - nothing here touches the network.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .resources import asset_dirs

MANIFEST_NAME = "build_manifest.json"
UNKNOWN = "unknown"


def manifest_path() -> Optional[Path]:
    """Location of the bundled build manifest, if this build has one."""
    for base in asset_dirs():
        candidate = base / MANIFEST_NAME
        if candidate.is_file():
            return candidate
    return None


def load_manifest() -> Dict[str, Any]:
    """The build manifest as a dict, or ``{}`` when absent/unreadable."""
    path = manifest_path()
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def application_version() -> str:
    """Installed package version, without importing the heavy ``whisper`` package."""
    try:
        from importlib.metadata import version

        return version("silvance-whisper")
    except Exception:  # noqa: BLE001 - not installed as a distribution
        return UNKNOWN


@dataclass
class BuildInfo:
    """What produced this run: build, commit, platform and bundled models."""

    build_id: str = UNKNOWN
    git_commit: str = UNKNOWN
    build_timestamp: str = UNKNOWN
    application_version: str = UNKNOWN
    platform: str = ""
    python_version: str = ""
    dependencies: Dict[str, str] = field(default_factory=dict)
    models: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def from_manifest(self) -> bool:
        """True when a CI-produced manifest supplied the identity."""
        return self.build_id != UNKNOWN

    def model_sha256(self, friendly_name: str) -> Optional[str]:
        """SHA-256 recorded for a bundled model, if the manifest lists it."""
        for entry in self.models:
            if entry.get("friendly_name") == friendly_name:
                digest = entry.get("sha256")
                return str(digest) if digest else None
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "build_id": self.build_id,
            "git_commit": self.git_commit,
            "build_timestamp": self.build_timestamp,
            "application_version": self.application_version,
            "platform": self.platform,
            "python_version": self.python_version,
            "dependencies": dict(self.dependencies),
            "models": list(self.models),
        }

    def describe(self) -> List[str]:
        """Human-readable lines for the self-test and report headers."""
        lines = [
            f"Application version: {self.application_version}",
            f"Build ID: {self.build_id}",
            f"Git commit: {self.git_commit}",
            f"Built: {self.build_timestamp}",
            f"Platform: {self.platform}",
            f"Python: {self.python_version}",
        ]
        if not self.from_manifest:
            lines.append(
                "No build manifest found - this is a source checkout or an "
                "unmanifested build, so build identity cannot be confirmed."
            )
        return lines


def build_info() -> BuildInfo:
    """Collect this build's identity from the manifest plus the live runtime."""
    manifest = load_manifest()
    models = manifest.get("models")
    dependencies = manifest.get("dependencies")
    return BuildInfo(
        build_id=str(manifest.get("build_id") or UNKNOWN),
        git_commit=str(manifest.get("git_commit") or UNKNOWN),
        build_timestamp=str(manifest.get("build_timestamp") or UNKNOWN),
        application_version=str(
            manifest.get("application_version") or application_version()
        ),
        platform=str(manifest.get("platform") or platform.platform()),
        python_version=str(manifest.get("python_version") or platform.python_version()),
        dependencies=(
            {str(k): str(v) for k, v in dependencies.items()}
            if isinstance(dependencies, dict)
            else {}
        ),
        models=[m for m in (models or []) if isinstance(m, dict)],
    )


def runtime_summary() -> Dict[str, str]:
    """Minimal runtime facts recorded alongside every analysis."""
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "executable": sys.executable,
    }
