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

import os
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
# voices much better than the small variant (which merged distinct male/female
# speakers), and with multi-threaded inference it is fast enough on CPU.
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


# pyannote.audio's speaker-diarization-3.1 pipeline and the gated models it pulls
# in: segmentation, speaker embedding, and the PLDA from speaker-diarization-
# community-1 (current pyannote 3.1 loads its PLDA transform from there). All are
# gated on the Hugging Face Hub: the account behind HF_TOKEN must have accepted
# each model's license.
PYANNOTE_REPOS = [
    "pyannote/speaker-diarization-3.1",
    "pyannote/segmentation-3.0",
    "pyannote/wespeaker-voxceleb-resnet34-LM",
    "pyannote/speaker-diarization-community-1",
]


def fetch_pyannote() -> None:
    """Download the gated pyannote models into an offline HF cache.

    Saves them under ``whispr_assets/pyannote/hub`` in the standard Hugging Face
    cache layout. At runtime the app points ``HF_HOME`` at ``whispr_assets/pyannote``
    and sets ``HF_HUB_OFFLINE=1`` (see ``whispr.resources.pyannote_cache_dir``), so
    the air-gapped machine needs neither network nor token.

    Requires ``HF_TOKEN`` (a Hugging Face token whose account has accepted the
    licenses for each repo in ``PYANNOTE_REPOS``).
    """
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN is not set. Create a Hugging Face token, accept the licenses "
            "for:\n  " + "\n  ".join(PYANNOTE_REPOS) + "\nthen set HF_TOKEN."
        )

    # Materialise real files (not symlinks) so the cache survives being copied
    # into the PyInstaller bundle on every platform.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    cache = ASSETS / "pyannote" / "hub"
    cache.mkdir(parents=True, exist_ok=True)
    for repo in PYANNOTE_REPOS:
        print(f"downloading {repo}")
        path = snapshot_download(repo_id=repo, cache_dir=str(cache), token=token)
        print(f"pyannote {repo} -> {path}")


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
            "usage: fetch_assets.py [ffmpeg | models <names> | diarization | pyannote]"
        )
    command = argv[0]
    if command == "ffmpeg":
        fetch_ffmpeg()
    elif command == "models":
        names = argv[1].split(",") if len(argv) > 1 else ["small", "medium", "large-v3"]
        fetch_models([n.strip() for n in names if n.strip()])
    elif command == "diarization":
        fetch_diarization()
    elif command == "pyannote":
        fetch_pyannote()
    else:
        raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    main(sys.argv[1:])
