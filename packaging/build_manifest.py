"""Write ``whispr_assets/build_manifest.json`` describing an assembled bundle.

Run at build time, after the assets have been fetched and before PyInstaller, so
the shipped application can state exactly what it is: which commit produced it,
which dependency versions went in, and the SHA-256 of every bundled model.

Without this file the application reports its build identity as "unknown" rather
than guessing - see :mod:`whispr.buildinfo`.

Usage::

    python packaging/build_manifest.py [--build-id ID] [--commit SHA]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whispr.hashing import sha256_file_or_none  # noqa: E402

ASSETS = Path("whispr_assets")
MANIFEST = ASSETS / "build_manifest.json"

# Production dependencies whose versions materially affect behaviour and so are
# worth recording with the build.
TRACKED_PACKAGES = (
    "silvance-whisper",
    "faster-whisper",
    "ctranslate2",
    "sherpa-onnx",
    "onnxruntime",
    "pyannote.audio",
    "torch",
    "torchaudio",
    "numpy",
    "ttkbootstrap",
    "python-docx",
    "argostranslate",
    "pytesseract",
    "pypdfium2",
    "pillow",
    "pyinstaller",
)


def _package_versions() -> Dict[str, str]:
    """Installed versions of the tracked packages (absent ones are omitted)."""
    from importlib.metadata import version

    out: Dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = version(name)
        except Exception:  # noqa: BLE001 - not installed in this build
            continue
    return out


def _git_commit(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    for variable in ("GITHUB_SHA", "GIT_COMMIT"):
        value = os.environ.get(variable)
        if value:
            return value
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except OSError:
        pass
    return "unknown"


def _model_entries() -> List[Dict[str, Any]]:
    """Every bundled model file, with its size and SHA-256."""
    entries: List[Dict[str, Any]] = []

    def _add(friendly_name: str, path: Path, kind: str) -> None:
        if not path.is_file():
            return
        entries.append(
            {
                "friendly_name": friendly_name,
                "kind": kind,
                "filename": str(path.relative_to(ASSETS)),
                "size": path.stat().st_size,
                "sha256": sha256_file_or_none(path),
            }
        )

    models_dir = ASSETS / "models"
    if models_dir.is_dir():
        for model in sorted(p for p in models_dir.iterdir() if p.is_dir()):
            _add(model.name, model / "model.bin", "transcription")

    diarization = ASSETS / "diarization"
    _add("segmentation", diarization / "segmentation.onnx", "diarization")
    embedding_name = "embedding"
    note = diarization / "embedding_model.txt"
    if note.is_file():
        try:
            embedding_name = note.read_text(encoding="utf-8").strip() or "embedding"
        except OSError:
            pass
    _add(embedding_name, diarization / "embedding.onnx", "speaker-embedding")

    for name in ("ffmpeg", "ffmpeg.exe"):
        _add("ffmpeg", ASSETS / "ffmpeg" / name, "tool")
    return entries


def build_manifest(
    *, build_id: Optional[str] = None, commit: Optional[str] = None
) -> Dict[str, Any]:
    """Assemble the manifest contents for the current working tree."""
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    versions = _package_versions()
    return {
        "build_id": build_id
        or os.environ.get("WHISPR_BUILD_ID")
        or timestamp.replace(":", "").replace("-", ""),
        "git_commit": _git_commit(commit),
        "build_timestamp": timestamp,
        "application_version": versions.get("silvance-whisper", "unknown"),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "dependencies": versions,
        "models": _model_entries(),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Write the bundle build manifest.")
    parser.add_argument("--build-id", default=None)
    parser.add_argument("--commit", default=None)
    args = parser.parse_args(argv)

    if not ASSETS.is_dir():
        raise SystemExit(
            f"{ASSETS} does not exist - fetch the assets before writing the manifest."
        )
    manifest = build_manifest(build_id=args.build_id, commit=args.commit)
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"build manifest -> {MANIFEST}")
    print(f"  build_id: {manifest['build_id']}")
    print(f"  commit:   {manifest['git_commit']}")
    print(f"  models:   {len(manifest['models'])}")
    for entry in manifest["models"]:
        digest = (entry.get("sha256") or "unknown")[:12]
        print(f"    {entry['friendly_name']:<18} {entry['kind']:<18} {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
