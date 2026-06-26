"""Audio/video transcription backend for the Whispers GUI.

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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .resources import find_ffmpeg

PathLike = Union[str, Path]
ProgressCallback = Callable[[str], None]
# A predicate the GUI can pass so long-running work can be stopped cooperatively.
CancelCallback = Callable[[], bool]


class CancelledError(Exception):
    """Raised to unwind a transcription/diarization that the user cancelled."""


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

# Model sizes accepted by faster-whisper's ``WhisperModel``. The English-only
# ".en" models are smaller/faster and usually as accurate on English audio
# (base.en mirrors whisper.cpp's ggml-base.en). The plain names are multilingual
# and required for non-English audio or the translate task.
MODEL_SIZES = (
    "base.en",
    "small.en",
    "medium.en",
    "tiny.en",
    "tiny",
    "base",
    "small",
    "medium",
    "large-v3",
    "turbo",
)


@dataclass
class Word:
    """A single timestamped word (from faster-whisper word timestamps).

    ``probability`` is the model's confidence for this word (0..1), when word
    timestamps were computed; used to highlight shaky words in the GUI.
    """

    start: float
    end: float
    word: str
    probability: Optional[float] = None


@dataclass
class Segment:
    """A single timestamped transcript segment.

    ``speaker`` is set when speaker diarization has been applied (see
    :func:`whispr.diarization.assign_speakers`); otherwise it is ``None``.
    ``words`` carries per-word timestamps when available, enabling word-level
    speaker assignment.
    """

    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    words: List[Word] = field(default_factory=list)
    # Average token log-probability for the segment (higher is more confident);
    # None when not provided by the backend. Used for confidence highlighting.
    avg_logprob: Optional[float] = None


# Thresholds for the GUI's low-confidence highlighting. Word ``probability`` is
# 0..1 (from word timestamps); segment ``avg_logprob`` is a log-prob (roughly
# -0.1 is excellent, below ~-0.7 is shaky). Below these, text is flagged so an
# analyst knows which parts to re-check.
LOW_WORD_PROBABILITY = 0.55
LOW_SEGMENT_LOGPROB = -0.7


def is_low_confidence_word(word: "Word") -> bool:
    """True if ``word`` has a probability below the low-confidence threshold."""
    return word.probability is not None and word.probability < LOW_WORD_PROBABILITY


def is_low_confidence_segment(segment: "Segment") -> bool:
    """True if ``segment``'s average log-prob is below the low-confidence cut."""
    return segment.avg_logprob is not None and segment.avg_logprob < LOW_SEGMENT_LOGPROB


@dataclass
class TranscriptionResult:
    """The full result of transcribing one media file."""

    text: str
    language: str
    language_probability: float
    duration: float
    segments: List[Segment] = field(default_factory=list)

    @property
    def has_speakers(self) -> bool:
        return any(segment.speaker for segment in self.segments)

    def _speaker_label(
        self, speaker: Optional[str], speaker_names: Optional[Dict[str, str]]
    ) -> str:
        label = speaker or "UNKNOWN"
        if speaker_names:
            return speaker_names.get(label, label)
        return label

    def to_txt(
        self,
        speaker_names: Optional[Dict[str, str]] = None,
        *,
        blank_lines: bool = False,
    ) -> str:
        """Plain-text transcript. When diarized, lines are prefixed with the
        speaker (optionally remapped via ``speaker_names``: id -> display name).

        ``blank_lines`` puts an empty line between segments (between speaker turns
        when diarized) - easier to read and to paste into a document.
        """
        sep = "\n\n" if blank_lines else "\n"
        if not self.has_speakers:
            if self.segments:
                return sep.join(segment.text for segment in self.segments)
            return self.text
        return sep.join(
            f"[{self._speaker_label(segment.speaker, speaker_names)}] {segment.text}"
            for segment in self.segments
        )

    def to_srt(self, speaker_names: Optional[Dict[str, str]] = None) -> str:
        """Render the segments as SubRip (``.srt``) subtitles."""
        blocks = []
        for index, segment in enumerate(self.segments, start=1):
            start = _format_timestamp(segment.start)
            end = _format_timestamp(segment.end)
            if segment.speaker:
                label = self._speaker_label(segment.speaker, speaker_names)
                text = f"[{label}] {segment.text}"
            else:
                text = segment.text
            blocks.append(f"{index}\n{start} --> {end}\n{text}\n")
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

    created_temp = dest is None
    if dest is None:
        handle, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(handle)
        out = Path(tmp)
    else:
        out = Path(dest)

    if progress is not None:
        progress(f"Converting {src.name} to WAV (ffmpeg)...")

    try:
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
    except BaseException:
        # Don't leak the temp file we created if ffmpeg fails (or is interrupted).
        if created_temp:
            try:
                out.unlink()
            except OSError:
                pass
        raise
    return out


# Loaded faster-whisper models, keyed by (model_size, device, compute_type), so
# repeated runs (different files, retry after cancel) don't pay the load cost each
# time. Models are read-only at inference time, so sharing one is safe.
_MODEL_CACHE: Dict[Tuple[str, str, str], Any] = {}


def _load_model(
    model_size: str,
    device: str,
    compute_type: str,
    *,
    report: Optional[ProgressCallback] = None,
) -> Any:
    key = (model_size, device, compute_type)
    model = _MODEL_CACHE.get(key)
    if model is not None:
        return model
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Install the GUI/transcription "
            "extras with:  pip install 'silvance-whisper[gui]'"
        ) from exc
    if report is not None:
        report(f"Loading model '{model_size}' ({device}/{compute_type})...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    _MODEL_CACHE[key] = model
    return model


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
    word_timestamps: bool = False,
    initial_prompt: Optional[str] = None,
    progress: Optional[ProgressCallback] = None,
    on_progress: Optional[Callable[[float], None]] = None,
    cancelled: Optional[CancelCallback] = None,
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
    initial_prompt
        Optional text to prime the decoder - e.g. names, places, jargon or
        callsigns likely in the audio - which biases recognition toward that
        vocabulary. ``None`` (default) disables it.
    progress
        Optional callback invoked with human-readable status/segment strings;
        the GUI uses this to stream output as it is produced.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    def _report(message: str) -> None:
        if progress is not None:
            progress(message)

    model = _load_model(model_size, device, compute_type, report=_report)

    verb = "Translating" if task == "translate" else "Transcribing"
    _report(f"{verb} {path.name}...")
    segments_iter, info = model.transcribe(
        str(path),
        task=task,
        language=language,
        beam_size=beam_size,
        vad_filter=vad_filter,
        # Word timestamps add an alignment pass; only compute them when needed
        # (diarization uses them for word-level speaker assignment).
        word_timestamps=word_timestamps,
        # None is the faster-whisper default (no priming).
        initial_prompt=initial_prompt or None,
    )

    segments: List[Segment] = []
    texts: List[str] = []
    for raw in segments_iter:
        if cancelled is not None and cancelled():
            raise CancelledError("Transcription cancelled.")
        text = raw.text.strip()
        words = [
            Word(
                start=w.start,
                end=w.end,
                word=w.word,
                probability=getattr(w, "probability", None),
            )
            for w in (raw.words or [])
        ]
        segments.append(
            Segment(
                start=raw.start,
                end=raw.end,
                text=text,
                words=words,
                avg_logprob=getattr(raw, "avg_logprob", None),
            )
        )
        texts.append(text)
        _report(f"[{_format_timestamp(raw.end)}] {text}")
        # Real progress: how far the latest segment's end is through the audio.
        if on_progress is not None and info.duration:
            on_progress(min(raw.end / info.duration, 1.0))
    if on_progress is not None:
        on_progress(1.0)

    return TranscriptionResult(
        text="\n".join(texts).strip(),
        language=info.language,
        language_probability=info.language_probability,
        duration=info.duration,
        segments=segments,
    )
