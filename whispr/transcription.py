"""Audio/video transcription backend for the W.H.I.S.P.R. GUI.

This wraps `faster-whisper <https://github.com/SYSTRAN/faster-whisper>`_
(a CTranslate2 reimplementation of Whisper) which is substantially faster and
lighter than the reference PyTorch implementation on CPU-only hardware.

The ``faster_whisper`` import is deferred to call time so that importing this
module - and therefore the rest of the package - does not require the optional
transcription/GUI dependencies to be installed.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Union

from .resources import find_ffmpeg

PathLike = Union[str, Path]
ProgressCallback = Callable[[str], None]

# Media container extensions the GUI offers for transcription. faster-whisper
# decodes via ffmpeg, so anything ffmpeg can read will work; this list just
# drives the file picker filter and ``is_supported_media``.
AUDIO_EXTENSIONS = (
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".ogg",
    ".opus",
    ".aac",
    ".wma",
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".webm",
)

# Subset of AUDIO_EXTENSIONS that are video containers. When the GUI's
# "convert video to WAV" option is on, these are pre-converted with ffmpeg.
VIDEO_EXTENSIONS = (
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".webm",
)

# Sample rate Whisper operates at; converting to it keeps the WAV small.
WHISPER_SAMPLE_RATE = 16000

# Model sizes accepted by faster-whisper's ``WhisperModel``.
MODEL_SIZES = (
    "tiny",
    "base",
    "small",
    "medium",
    "large-v3",
    "turbo",
)


@dataclass
class Segment:
    """A single timestamped transcript segment."""

    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    """The full result of transcribing one media file."""

    text: str
    language: str
    language_probability: float
    duration: float
    segments: List[Segment] = field(default_factory=list)

    def to_txt(self) -> str:
        return self.text

    def to_srt(self) -> str:
        """Render the segments as SubRip (``.srt``) subtitles."""
        blocks = []
        for index, segment in enumerate(self.segments, start=1):
            start = _format_timestamp(segment.start)
            end = _format_timestamp(segment.end)
            blocks.append(f"{index}\n{start} --> {end}\n{segment.text}\n")
        return "\n".join(blocks)


def _format_timestamp(seconds: float) -> str:
    """Format ``seconds`` as an ``HH:MM:SS,mmm`` SubRip timestamp."""
    if seconds < 0:
        seconds = 0.0
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def is_supported_media(path: PathLike) -> bool:
    """Return True if ``path`` has a known audio/video extension."""
    return Path(path).suffix.lower() in AUDIO_EXTENSIONS


def is_video(path: PathLike) -> bool:
    """Return True if ``path`` has a known video container extension."""
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def convert_to_wav(
    path: PathLike,
    dest: Optional[PathLike] = None,
    *,
    sample_rate: int = WHISPER_SAMPLE_RATE,
    progress: Optional[ProgressCallback] = None,
) -> Path:
    """Extract/convert a media file's audio to a mono PCM WAV with ffmpeg.

    Parameters
    ----------
    path
        Source audio or video file.
    dest
        Destination ``.wav`` path. If ``None``, a temporary file is created and
        returned; the caller is responsible for deleting it.
    sample_rate
        Output sample rate in Hz (defaults to Whisper's 16 kHz).
    progress
        Optional status callback.

    Returns
    -------
    Path
        The path to the written WAV file.
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Input file does not exist: {src}")

    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg was not found (neither bundled in whispr_assets nor on PATH). "
            "Install it to convert video files (e.g. `sudo apt install ffmpeg`, "
            "`brew install ffmpeg`)."
        )

    if dest is None:
        handle, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(handle)
        out = Path(tmp)
    else:
        out = Path(dest)

    if progress is not None:
        progress(f"Converting {src.name} to WAV (ffmpeg)...")

    result = subprocess.run(
        [
            str(ffmpeg),
            "-y",
            "-i",
            str(src),
            "-vn",  # drop any video stream
            "-ac",
            "1",  # mono
            "-ar",
            str(sample_rate),
            "-acodec",
            "pcm_s16le",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed:\n{result.stderr.strip()}")
    return out


def transcribe_audio(
    path: PathLike,
    *,
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
    task: str = "transcribe",
    language: Optional[str] = None,
    beam_size: int = 5,
    vad_filter: bool = True,
    progress: Optional[ProgressCallback] = None,
) -> TranscriptionResult:
    """Transcribe a single audio/video file with faster-whisper.

    Parameters
    ----------
    path
        Path to the audio or video file.
    model_size
        A faster-whisper model size (see ``MODEL_SIZES``) or a path to a
        local CTranslate2 model directory.
    device
        ``"cpu"`` (default, recommended for CPU-only hosts), ``"cuda"`` or
        ``"auto"``.
    compute_type
        Quantization, e.g. ``"int8"`` (default, best for CPU), ``"int8_float16"``
        or ``"float16"`` (GPU).
    task
        ``"transcribe"`` (default) keeps the source language; ``"translate"``
        translates speech into English.
    language
        ISO language code (e.g. ``"en"``) or ``None`` to auto-detect.
    beam_size
        Beam search width.
    vad_filter
        Drop non-speech with the Silero VAD before transcription.
    progress
        Optional callback invoked with human-readable status/segment strings;
        the GUI uses this to stream output as it is produced.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Install the GUI/transcription "
            "extras with:  pip install 'silvance-whisper[gui]'"
        ) from exc

    def _report(message: str) -> None:
        if progress is not None:
            progress(message)

    _report(f"Loading model '{model_size}' ({device}/{compute_type})...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    verb = "Translating" if task == "translate" else "Transcribing"
    _report(f"{verb} {path.name}...")
    segments_iter, info = model.transcribe(
        str(path),
        task=task,
        language=language,
        beam_size=beam_size,
        vad_filter=vad_filter,
    )

    segments: List[Segment] = []
    texts: List[str] = []
    for raw in segments_iter:
        text = raw.text.strip()
        segments.append(Segment(start=raw.start, end=raw.end, text=text))
        texts.append(text)
        _report(f"[{_format_timestamp(raw.end)}] {text}")

    return TranscriptionResult(
        text="\n".join(texts).strip(),
        language=info.language,
        language_probability=info.language_probability,
        duration=info.duration,
        segments=segments,
    )
