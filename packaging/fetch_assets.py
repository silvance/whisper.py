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

import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whispr.hashing import sha256_file_or_none  # noqa: E402

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
DIARIZATION_SEGMENTATION_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)

# Speaker-embedding models power both sherpa diarization and the speaker profile
# / comparison features, so the choice governs how well voices separate and how
# comparable two recordings are. Selectable at build time; all are single .onnx
# assets from the sherpa-onnx release below, and a direct .onnx URL may be passed
# instead of an alias.
#
# Each entry is described by the file that is actually downloaded rather than by
# a marketing label. In particular the CAM++ and ERes2Net assets published there
# are the 3D-Speaker *zh-cn 16k-common* checkpoints: they are trained on Mandarin
# data, not on a broad multilingual corpus, and are recorded as such. The same
# architecture trained elsewhere would behave differently, so the architecture
# name alone says little.
#
# Verification performance varies substantially with language, channel,
# microphone, background noise, the amount of usable speech, and any mismatch
# between the enrolment and questioned recordings. No entry here is validated for
# a particular operational setting; see whispr/validation.py for measuring one.
EMBEDDING_RELEASE_BASE = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/"
)
DEFAULT_EMBEDDING = "titanet-large"
EMBEDDING_MODELS: Dict[str, Dict[str, str]] = {
    "titanet-large": {
        "filename": "nemo_en_titanet_large.onnx",
        "label": "NeMo TitaNet Large",
        "architecture": "TitaNet (ECAPA-style, NVIDIA NeMo)",
        "training_data": "English speaker-verification corpora (NeMo release)",
    },
    "titanet-small": {
        "filename": "nemo_en_titanet_small.onnx",
        "label": "NeMo TitaNet Small",
        "architecture": "TitaNet (NVIDIA NeMo)",
        "training_data": "English speaker-verification corpora (NeMo release)",
    },
    "wespeaker-en": {
        "filename": "wespeaker_en_voxceleb_resnet34.onnx",
        "label": "WeSpeaker ResNet34 (VoxCeleb)",
        "architecture": "ResNet34 (WeSpeaker)",
        "training_data": "VoxCeleb (predominantly English interview speech)",
    },
    "campplus": {
        "filename": "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx",
        "label": "CAM++ (3D-Speaker, zh-cn 16k common)",
        "architecture": "CAM++ (3D-Speaker)",
        "training_data": "Mandarin (zh-cn) 16 kHz common-domain data",
    },
    "eres2net": {
        "filename": "3dspeaker_speech_eres2net_sv_zh-cn_16k-common.onnx",
        "label": "ERes2Net (3D-Speaker, zh-cn 16k common)",
        "architecture": "ERes2Net (3D-Speaker)",
        "training_data": "Mandarin (zh-cn) 16 kHz common-domain data",
    },
}

# Recorded with every bundle so the caveat travels with the model, not just with
# this build script.
EMBEDDING_CAVEAT = (
    "Speaker-embedding performance varies with language, channel, microphone, "
    "background noise, the amount of usable speech, and any mismatch between "
    "the enrolment and questioned recordings. This model has not been validated "
    "for any particular operational setting."
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


# pyannote.audio 3.1.1's speaker-diarization-3.1 pipeline and the two gated models
# it pulls in (segmentation + speaker embedding). These are gated on the Hugging
# Face Hub: the account behind HF_TOKEN must have accepted each model's license.
PYANNOTE_REPOS = [
    "pyannote/speaker-diarization-3.1",
    "pyannote/segmentation-3.0",
    "pyannote/wespeaker-voxceleb-resnet34-LM",
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


def _resolve_embedding(embedding: str) -> "tuple[str, str]":
    """Resolve an embedding alias (or a direct .onnx URL) to ``(name, url)``."""
    embedding = embedding.strip()
    if embedding.startswith(("http://", "https://")):
        return Path(embedding).stem, embedding
    if embedding not in EMBEDDING_MODELS:
        raise SystemExit(
            f"unknown embedding model '{embedding}'; choose from "
            f"{', '.join(EMBEDDING_MODELS)}, or pass a direct .onnx URL"
        )
    return embedding, EMBEDDING_RELEASE_BASE + EMBEDDING_MODELS[embedding]["filename"]


def embedding_metadata(name: str, url: str, path: Optional[Path] = None) -> Dict:
    """Describe the embedding model that was actually downloaded.

    Only what is known is recorded. For a direct URL nothing is known beyond the
    file itself, so the descriptive fields stay empty rather than being guessed
    from the file name.
    """
    known = EMBEDDING_MODELS.get(name, {})
    meta: Dict[str, object] = {
        "name": name,
        "label": known.get("label") or name,
        "architecture": known.get("architecture", ""),
        "training_data": known.get("training_data", ""),
        "filename": known.get("filename") or Path(url).name,
        "source_url": url,
        "backend": "sherpa-onnx",
        "caveat": EMBEDDING_CAVEAT,
    }
    if path is not None and path.is_file():
        meta["size"] = path.stat().st_size
        meta["sha256"] = sha256_file_or_none(path)
    return meta


def fetch_embedding(embedding: str = DEFAULT_EMBEDDING) -> None:
    """Download the speaker-embedding model into whispr_assets/diarization.

    Saves ``embedding.onnx`` plus two sidecars: ``embedding_model.txt`` (the
    alias, read by older builds) and ``embedding_model.json``, which describes
    the exact file - architecture, what it was trained on, source URL, size and
    SHA-256 - so the app can say what it is using rather than a broad label, and
    an operator can confirm two builds share a model before comparing voices.
    Needed for speaker profiles/comparison on *every* build, so it is fetched
    independently of the sherpa segmentation model.
    """
    out = ASSETS / "diarization"
    out.mkdir(parents=True, exist_ok=True)
    name, url = _resolve_embedding(embedding)
    embedding_dest = out / "embedding.onnx"
    print(f"downloading embedding '{name}' from {url}")
    urllib.request.urlretrieve(url, embedding_dest)
    (out / "embedding_model.txt").write_text(name + "\n", encoding="utf-8")
    meta = embedding_metadata(name, url, embedding_dest)
    (out / "embedding_model.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"diarization embedding ({meta['label']}) -> {embedding_dest}")
    if meta.get("training_data"):
        print(f"  trained on: {meta['training_data']}")


def fetch_segmentation() -> None:
    """Download the sherpa-onnx segmentation model into whispr_assets/diarization."""
    out = ASSETS / "diarization"
    out.mkdir(parents=True, exist_ok=True)
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


def fetch_diarization(embedding: str = DEFAULT_EMBEDDING) -> None:
    """Download both sherpa-onnx diarization models (segmentation + embedding)."""
    fetch_embedding(embedding)
    fetch_segmentation()


# Argos Translate source languages to bundle (each translates -> English). The
# intel-leaning set; all are present in the Argos index (Pashto has no Argos pack).
# Override on the command line / via the workflow input.
ARGOS_DEFAULT_LANGS = ["ar", "ru", "zh", "fa", "uk", "he", "ko"]


def fetch_argos(codes: List[str]) -> None:
    """Install Argos packs (<code> -> en) and warm the offline sentence splitter.

    Everything lives under Argos's data dir, which it derives from XDG_DATA_HOME;
    pointing that at ``whispr_assets/argos`` puts the packs and the MiniSBD model
    cache inside the bundle. A sample translation per language is then run, which
    downloads the small per-language MiniSBD onnx into the cache - so at runtime,
    air-gapped, no model download is attempted. Unavailable codes are skipped.
    """
    argos_home = ASSETS / "argos"
    (argos_home / "argos-translate").mkdir(parents=True, exist_ok=True)
    # Must be set before argostranslate is imported (read into settings at import).
    os.environ["XDG_DATA_HOME"] = str(argos_home)
    os.environ["ARGOS_CHUNK_TYPE"] = "MINISBD"

    import argostranslate.package as package
    import argostranslate.translate as translate

    print("updating Argos package index")
    package.update_package_index()
    available = package.get_available_packages()

    installed_codes: List[str] = []
    for code in codes:
        match = next(
            (p for p in available if p.from_code == code and p.to_code == "en"), None
        )
        if match is None:
            print(f"WARNING: no Argos pack for {code} -> en; skipping")
            continue
        print(f"downloading Argos pack {code} -> en")
        package.install_from_path(match.download())
        installed_codes.append(code)
        print(f"argos {code} -> en installed")

    if not installed_codes:
        raise SystemExit("no Argos packs were installed (check the language codes)")

    # Warm the MiniSBD cache (and prove the pack works) with a sample translation.
    for code in installed_codes:
        try:
            sample = translate.translate("Test sentence. Another one.", code, "en")
            print(f"warmup {code} -> en ok: {sample!r}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"WARNING: warmup {code} -> en failed: {exc}")

    print(f"argos data -> {argos_home / 'argos-translate'}")


# Tesseract OCR language data. tessdata_fast gives the best speed/size trade-off
# for CPU OCR. We always add eng (often present in mixed docs) and osd (orientation
# + script detection). The intel-leaning default mirrors the Argos set.
TESSDATA_BASE_URL = "https://github.com/tesseract-ocr/tessdata_fast/raw/main"
OCR_DEFAULT_LANGS = ["ar", "ru", "zh", "fa", "uk", "he", "ko"]

# Shared libraries we must NOT relocate when bundling a Linux Tesseract: the C
# runtime and dynamic loader belong to the target system, not the build runner.
_LINUX_LIB_DENYLIST = (
    "libc.so",
    "libm.so",
    "libpthread.so",
    "libdl.so",
    "librt.so",
    "libresolv.so",
    "ld-linux",
)


def _copy_linux_tesseract_libs(binary: Path, dest_dir: Path) -> None:
    """Copy the Tesseract binary's shared-lib dependencies beside it (Linux).

    Uses ``ldd`` and skips the C runtime / loader (kept from the target system).
    At runtime the app prepends this directory to ``LD_LIBRARY_PATH`` so the
    bundled binary resolves libtesseract/leptonica/etc. from here.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["ldd", str(binary)], capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        print("WARNING: ldd not available; not bundling Tesseract libraries")
        return
    for line in out.stdout.splitlines():
        if "=>" not in line:
            continue
        rhs = line.split("=>", 1)[1].strip()
        lib_path = rhs.split(" (")[0].strip()
        if not lib_path or not os.path.exists(lib_path):
            continue
        base = os.path.basename(lib_path)
        if any(base.startswith(name) for name in _LINUX_LIB_DENYLIST):
            continue
        shutil.copy2(lib_path, dest_dir / base)
        print(f"  lib {base}")


def fetch_tesseract(codes: List[str]) -> None:
    """Bundle Tesseract OCR: language data plus the binary and its libraries.

    Downloads ``<lang>.traineddata`` into ``whispr_assets/tesseract/tessdata`` and
    copies a system Tesseract binary (installed by the build step) plus its
    dependent libraries next to it, so the air-gapped target needs no Tesseract
    install. Unknown/unavailable language codes are skipped with a warning.
    """
    from whispr.ocr import tesseract_lang

    tess_dir = ASSETS / "tesseract"
    tessdata = tess_dir / "tessdata"
    tessdata.mkdir(parents=True, exist_ok=True)

    wanted = ["eng", "osd"] + [tesseract_lang(c) for c in codes]
    seen = set()
    for lang in wanted:
        if lang in seen:
            continue
        seen.add(lang)
        url = f"{TESSDATA_BASE_URL}/{lang}.traineddata"
        dest = tessdata / f"{lang}.traineddata"
        print(f"downloading tessdata {lang}")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"tessdata {lang} -> {dest}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"WARNING: could not fetch tessdata {lang!r}: {exc}")
            if dest.exists():
                dest.unlink()

    binary = shutil.which("tesseract")
    if binary is None:
        # choco/apt update the machine PATH, but the current step's process often
        # doesn't see it via `which` (e.g. choco on a Windows runner). Check the
        # standard install locations directly before giving up.
        for candidate in (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            "/usr/bin/tesseract",
            "/usr/local/bin/tesseract",
        ):
            if os.path.exists(candidate):
                binary = candidate
                break
    if binary is None:
        print(
            "WARNING: no 'tesseract' binary found to bundle. Install it in the "
            "build step (apt-get install tesseract-ocr / choco install tesseract); "
            "at runtime the app will otherwise need a system Tesseract."
        )
        return

    src = Path(binary)
    dest_name = "tesseract.exe" if src.suffix.lower() == ".exe" else "tesseract"
    dest = tess_dir / dest_name
    shutil.copy2(src, dest)
    if dest.suffix.lower() != ".exe":
        dest.chmod(0o755)
    print(f"tesseract binary -> {dest}")

    if os.name == "nt":
        # Windows DLLs live next to tesseract.exe in the install dir.
        for dll in src.parent.glob("*.dll"):
            shutil.copy2(dll, tess_dir / dll.name)
            print(f"  dll {dll.name}")
    else:
        _copy_linux_tesseract_libs(src, tess_dir)


def main(argv: List[str]) -> None:
    if not argv:
        raise SystemExit(
            "usage: fetch_assets.py [ffmpeg | models <names> | embedding <model> | "
            "segmentation | diarization <embedding> | pyannote | argos <codes> | "
            "tesseract <codes>]"
        )
    command = argv[0]
    if command == "ffmpeg":
        fetch_ffmpeg()
    elif command == "models":
        names = argv[1].split(",") if len(argv) > 1 else ["small", "medium", "large-v3"]
        fetch_models([n.strip() for n in names if n.strip()])
    elif command == "embedding":
        embedding = argv[1].strip() if len(argv) > 1 and argv[1].strip() else None
        fetch_embedding(embedding or DEFAULT_EMBEDDING)
    elif command == "segmentation":
        fetch_segmentation()
    elif command == "diarization":
        embedding = argv[1].strip() if len(argv) > 1 and argv[1].strip() else None
        fetch_diarization(embedding or DEFAULT_EMBEDDING)
    elif command == "pyannote":
        fetch_pyannote()
    elif command == "argos":
        codes = argv[1].split(",") if len(argv) > 1 else ARGOS_DEFAULT_LANGS
        fetch_argos([c.strip() for c in codes if c.strip()])
    elif command == "tesseract":
        codes = argv[1].split(",") if len(argv) > 1 else OCR_DEFAULT_LANGS
        fetch_tesseract([c.strip() for c in codes if c.strip()])
    else:
        raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    main(sys.argv[1:])
