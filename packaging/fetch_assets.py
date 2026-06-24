"""Download offline assets (ffmpeg + faster-whisper models) into ``whispr_assets/``.

Run at build time on a connected machine (or in CI) before invoking PyInstaller.
The downloaded assets are bundled into the executable so the target machine never
needs network access.

Usage::

    python packaging/fetch_assets.py ffmpeg
    python packaging/fetch_assets.py models small,medium,large-v3

Requires the build extras: ``pip install "silvance-whisper[bundle]"``.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import List

ASSETS = Path("whispr_assets")

# faster-whisper's official CTranslate2 model repositories on the Hugging Face Hub.
MODEL_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
}

# sherpa-onnx diarization models (non-gated). (repo_id, filename) -> saved as
# whispr_assets/diarization/<segmentation|embedding>.onnx.
DIARIZATION_MODELS = {
    "segmentation": (
        "csukuangfj/sherpa-onnx-pyannote-segmentation-3-0",
        "model.onnx",
    ),
    "embedding": (
        "csukuangfj/speaker-embedding-models",
        "nemo_en_titanet_small.onnx",
    ),
}


def fetch_ffmpeg() -> None:
    """Copy a platform-appropriate ffmpeg binary into ``whispr_assets/ffmpeg``."""
    import imageio_ffmpeg

    src = Path(imageio_ffmpeg.get_ffmpeg_exe())
    dest_dir = ASSETS / "ffmpeg"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / ("ffmpeg.exe" if src.suffix.lower() == ".exe" else "ffmpeg")
    shutil.copy2(src, dest)
    if dest.suffix.lower() != ".exe":
        dest.chmod(0o755)
    print(f"ffmpeg -> {dest}")


def fetch_models(names: List[str]) -> None:
    """Download each named CTranslate2 model into ``whispr_assets/models/<name>``."""
    from huggingface_hub import snapshot_download

    for name in names:
        if name not in MODEL_REPOS:
            raise SystemExit(
                f"unknown model '{name}'; choose from {', '.join(MODEL_REPOS)}"
            )
        out = ASSETS / "models" / name
        out.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=MODEL_REPOS[name],
            local_dir=str(out),
            allow_patterns=["*.bin", "*.json", "*.txt"],
        )
        print(f"model {name} -> {out}")


def fetch_diarization() -> None:
    """Download the sherpa-onnx diarization models into whispr_assets/diarization."""
    from huggingface_hub import hf_hub_download

    out = ASSETS / "diarization"
    out.mkdir(parents=True, exist_ok=True)
    for local_name, (repo_id, filename) in DIARIZATION_MODELS.items():
        downloaded = hf_hub_download(repo_id=repo_id, filename=filename)
        dest = out / f"{local_name}.onnx"
        shutil.copy2(downloaded, dest)
        print(f"diarization {local_name} -> {dest}")


def main(argv: List[str]) -> None:
    if not argv:
        raise SystemExit(
            "usage: fetch_assets.py [ffmpeg | models <names> | diarization]"
        )
    command = argv[0]
    if command == "ffmpeg":
        fetch_ffmpeg()
    elif command == "models":
        names = argv[1].split(",") if len(argv) > 1 else ["small", "medium", "large-v3"]
        fetch_models([n.strip() for n in names if n.strip()])
    elif command == "diarization":
        fetch_diarization()
    else:
        raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    main(sys.argv[1:])
