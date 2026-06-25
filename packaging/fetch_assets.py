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
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import List

ASSETS = Path("whispr_assets")

# faster-whisper's official CTranslate2 model repositories on the Hugging Face Hub.
MODEL_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "base": "Systran/faster-whisper-base",
    "base.en": "Systran/faster-whisper-base.en",
    "small": "Systran/faster-whisper-small",
    "small.en": "Systran/faster-whisper-small.en",
    "medium": "Systran/faster-whisper-medium",
    "medium.en": "Systran/faster-whisper-medium.en",
    "large-v3": "Systran/faster-whisper-large-v3",
}

# sherpa-onnx diarization models, downloaded from the official k2-fsa GitHub
# release assets (stable, non-gated URLs). The segmentation model ships as a
# .tar.bz2 containing model.onnx; the embedding model is a single .onnx.
# We use NeMo TitaNet-large (English) for the speaker embeddings - it separates
# English voices far better than the Chinese-trained demo model.
DIARIZATION_SEGMENTATION_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
DIARIZATION_EMBEDDING_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/nemo_en_titanet_large.onnx"
)


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
    """Download the sherpa-onnx diarization models into whispr_assets/diarization.

    Saves whispr_assets/diarization/segmentation.onnx and embedding.onnx.
    """
    out = ASSETS / "diarization"
    out.mkdir(parents=True, exist_ok=True)

    embedding_dest = out / "embedding.onnx"
    print(f"downloading {DIARIZATION_EMBEDDING_URL}")
    urllib.request.urlretrieve(DIARIZATION_EMBEDDING_URL, embedding_dest)
    print(f"diarization embedding -> {embedding_dest}")

    segmentation_dest = out / "segmentation.onnx"
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "segmentation.tar.bz2"
        print(f"downloading {DIARIZATION_SEGMENTATION_URL}")
        urllib.request.urlretrieve(DIARIZATION_SEGMENTATION_URL, archive)
        with tarfile.open(archive, "r:bz2") as tar:
            member = next(
                (m for m in tar.getmembers() if m.name.endswith("model.onnx")), None
            )
            if member is None:
                raise SystemExit("model.onnx not found in segmentation archive")
            member.name = Path(member.name).name  # flatten any leading directory
            tar.extract(member, out)
        (out / "model.onnx").replace(segmentation_dest)
    print(f"diarization segmentation -> {segmentation_dest}")


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
