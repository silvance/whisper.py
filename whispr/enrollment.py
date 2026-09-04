"""Build a known subject's reference voice profile from historical recordings.

This is the deliberate counterpart to the incidental learning that happens when
an operator corrects a transcript: an analyst picks a subject, points at one or
more recordings they know contain that person, and enrols trusted reference
samples - without running a transcription first.

Three ways to say *which* speech belongs to the subject:

* the whole recording, when the source contains only them;
* a diarized cluster, when the recording has several people;
* explicit time ranges, as the always-available fallback.

Each usable span is split into several windows so a profile holds **multiple
embeddings** rather than one averaged vector for an entire recording, and every
sample carries its provenance (source file, SHA-256, time span, duration) and its
measured quality. Spans without enough usable speech are skipped rather than
silently enrolled as weak evidence.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

from .hashing import sha256_file_or_none
from .quality import QualityReport, analyse_span, combine
from .speaker_profiles import (
    SAMPLE_REFERENCE,
    EmbeddingModelIdentity,
    EnrollmentSample,
    ProfileError,
    SpeakerProfile,
    bundled_model_identity,
    check_compatibility,
)
from .transcription import WHISPER_SAMPLE_RATE

PathLike = Union[str, Path]
ProgressFn = Callable[[str], None]

# Enrolment window: long enough for a stable embedding, short enough that one
# recording yields several independent samples.
WINDOW_SECONDS = 8.0
# A window shorter than this is not embedded at all.
MIN_WINDOW_SECONDS = 3.0


@dataclass
class EnrollmentResult:
    """What one enrolment run added, and why anything was left out."""

    profile: SpeakerProfile
    added: List[EnrollmentSample] = field(default_factory=list)
    quality: Optional[QualityReport] = None
    skipped: List[str] = field(default_factory=list)

    @property
    def added_count(self) -> int:
        return len(self.added)

    @property
    def added_seconds(self) -> float:
        return round(sum(s.speech_duration for s in self.added), 2)


def windows(
    start: float,
    end: float,
    *,
    window: float = WINDOW_SECONDS,
    minimum: float = MIN_WINDOW_SECONDS,
) -> List[Tuple[float, float]]:
    """Split ``[start, end]`` into consecutive enrolment windows.

    A trailing remainder shorter than ``minimum`` is dropped (it would only add a
    noisy embedding); a span shorter than ``window`` but at least ``minimum`` is
    kept whole. Returns ``[]`` when the span is too short to use.
    """
    span = end - start
    if span < minimum:
        return []
    if span <= window:
        return [(start, end)]
    out: List[Tuple[float, float]] = []
    cursor = start
    while cursor < end:
        stop = min(cursor + window, end)
        if stop - cursor >= minimum:
            out.append((cursor, stop))
        cursor = stop
    return out


def _resolve_model_identity(
    profile: SpeakerProfile, vector_dimension: int
) -> EmbeddingModelIdentity:
    """Stamp (or verify) the embedding model this profile's vectors come from.

    A profile must never mix models: if it already names one, this build's model
    has to match it, otherwise the existing samples and the new ones would not be
    comparable.
    """
    current = bundled_model_identity()
    if current is None:
        raise ProfileError(
            "No speaker-embedding model is bundled in this build, so a reference "
            "profile cannot be enrolled."
        )
    current.vector_dimension = vector_dimension
    existing = profile.embedding_model
    if existing is None:
        return current
    if existing.vector_dimension is None:
        existing.vector_dimension = vector_dimension
    verdict = check_compatibility(existing, current)
    if not verdict.ok and not verdict.needs_confirmation:
        raise ProfileError(
            "This profile was enrolled with a different speaker-embedding model "
            f"({existing.describe()}). {verdict.reason}"
        )
    return existing


def _already_enrolled(
    profile: SpeakerProfile, source_sha256: Optional[str], start: float, end: float
) -> bool:
    """True if this exact source span is already a sample (avoids double weight)."""
    if not source_sha256:
        return False
    for sample in profile.samples:
        if (
            sample.source_sha256 == source_sha256
            and sample.source_start is not None
            and sample.source_end is not None
            and abs(sample.source_start - start) < 0.01
            and abs(sample.source_end - end) < 0.01
        ):
            return True
    return False


def enroll_from_wav(
    profile: SpeakerProfile,
    wav_path: PathLike,
    spans: Sequence[Tuple[float, float]],
    embedder: Any,
    *,
    source_filename: Optional[str] = None,
    source_sha256: Optional[str] = None,
    sample_type: str = SAMPLE_REFERENCE,
    notes: str = "",
    allow_duplicates: bool = False,
    progress: Optional[ProgressFn] = None,
) -> EnrollmentResult:
    """Enrol ``spans`` of an already-prepared 16 kHz mono WAV into ``profile``.

    The audio-independent half of enrolment: callers that already have a
    converted WAV (and the original file's hash) use this directly.
    """
    result = EnrollmentResult(profile=profile)
    reports: List[QualityReport] = []
    # Overlapping or repeated ranges describe one stretch of speech; enrolling
    # each independently would weight those seconds twice in the centroid.
    distinct = normalize_spans(spans)
    if len(distinct) < len(list(spans)):
        result.skipped.append(
            f"{len(list(spans))} given range(s) cover {len(distinct)} distinct "
            "stretch(es) of audio; ranges that overlap, repeat or run straight "
            "on are enrolled once."
        )
    for span_start, span_end in distinct:
        pieces = windows(span_start, span_end)
        if not pieces:
            result.skipped.append(
                f"{span_start:.1f}-{span_end:.1f}s: shorter than "
                f"{MIN_WINDOW_SECONDS:.0f}s of speech."
            )
            continue
        for start, end in pieces:
            if not allow_duplicates and _already_enrolled(
                profile, source_sha256, start, end
            ):
                result.skipped.append(
                    f"{start:.1f}-{end:.1f}s: already enrolled from this recording."
                )
                continue
            report = analyse_span(wav_path, start, end)
            if not report.usable:
                result.skipped.append(
                    f"{start:.1f}-{end:.1f}s: {report.assessment.lower()} audio "
                    f"({report.voiced_seconds:.1f}s of speech)."
                )
                continue
            if progress is not None:
                progress(f"Embedding {start:.0f}-{end:.0f}s…")
            vector = embedder.embed_span(wav_path, start, end)
            if not vector:
                result.skipped.append(f"{start:.1f}-{end:.1f}s: no embedding produced.")
                continue
            profile.embedding_model = _resolve_model_identity(profile, len(vector))
            sample = EnrollmentSample(
                embedding=list(vector),
                sample_type=sample_type,
                approved=(sample_type == SAMPLE_REFERENCE),
                source_filename=source_filename,
                source_sha256=source_sha256,
                source_start=start,
                source_end=end,
                # Usable speech, not wall-clock: a window that is half silence
                # should not be credited as a full window of reference material.
                speech_duration=round(report.voiced_seconds, 2),
                quality=report.to_dict(),
                notes=notes,
            )
            if sample_type == SAMPLE_REFERENCE:
                profile.add_reference_sample(sample)
            else:
                profile.propose_learned_sample(sample)
            result.added.append(sample)
            reports.append(report)
    result.quality = combine(reports) if reports else None
    return result


def _is_analysis_wav(path: Path) -> bool:
    """True when ``path`` is already 16 kHz mono 16-bit PCM WAV.

    Anything else - a different rate, stereo, 8- or 24-bit, or a non-WAV
    container - goes through ffmpeg, because the embedder and the quality
    metrics both assume this exact format.
    """
    if path.suffix.lower() != ".wav":
        return False
    try:
        with wave.open(str(path), "rb") as handle:
            return (
                handle.getnchannels() == 1
                and handle.getsampwidth() == 2
                and handle.getframerate() == WHISPER_SAMPLE_RATE
            )
    except (OSError, wave.Error, EOFError):
        # A truncated file, or one that is not a WAV despite the extension,
        # raises rather than reporting a format - either way, convert it.
        return False


def prepare_source(
    source: PathLike, *, progress: Optional[ProgressFn] = None
) -> Tuple[Path, Optional[str], bool]:
    """Convert ``source`` to a 16 kHz mono WAV and hash the *original* file.

    Returns ``(wav_path, source_sha256, is_temporary)``. The hash is of the
    source the operator selected - the artefact a report must be traceable to -
    not of the converted copy. The source itself is never modified.
    """
    from .transcription import convert_to_wav

    source_path = Path(source)
    if progress is not None:
        progress(f"Hashing {source_path.name}…")
    digest = sha256_file_or_none(source_path)
    if _is_analysis_wav(source_path):
        # Already exactly what the embedder and the quality metrics want, so
        # there is nothing to convert - and no reason to require ffmpeg for a
        # corpus that was prepared in advance.
        return source_path, digest, False
    if progress is not None:
        progress("Preparing audio…")
    wav = convert_to_wav(source_path, progress=progress)
    return wav, digest, wav != source_path


def enroll_from_media(
    profile: SpeakerProfile,
    source: PathLike,
    embedder: Any,
    *,
    spans: Optional[Sequence[Tuple[float, float]]] = None,
    sample_type: str = SAMPLE_REFERENCE,
    notes: str = "",
    allow_duplicates: bool = False,
    progress: Optional[ProgressFn] = None,
) -> EnrollmentResult:
    """Enrol from any media file, converting it first.

    ``spans`` selects the subject's speech; omit it when the recording contains
    only the subject, in which case the whole recording is used.
    """
    wav, digest, temporary = prepare_source(source, progress=progress)
    try:
        if spans is None:
            duration = _wav_duration(wav)
            spans = [(0.0, duration)]
        return enroll_from_wav(
            profile,
            wav,
            spans,
            embedder,
            source_filename=Path(source).name,
            source_sha256=digest,
            sample_type=sample_type,
            notes=notes,
            allow_duplicates=allow_duplicates,
            progress=progress,
        )
    finally:
        if temporary:
            try:
                Path(wav).unlink()
            except OSError:
                pass


def _wav_duration(wav_path: PathLike) -> float:
    import wave

    with wave.open(str(wav_path), "rb") as wav:
        rate = wav.getframerate() or 1
        return wav.getnframes() / float(rate)


def spans_for_speaker(
    speaker_segments: Sequence[Any], speaker_id: str
) -> List[Tuple[float, float]]:
    """The time ranges a diarized cluster attributes to ``speaker_id``."""
    return [
        (seg.start, seg.end)
        for seg in speaker_segments
        if getattr(seg, "speaker", None) == speaker_id
    ]


def speaker_totals(speaker_segments: Sequence[Any]) -> List[Tuple[str, float]]:
    """``(speaker_id, total_seconds)`` per cluster, longest first.

    Lets the operator pick the right cluster ("the one that talks for 4 minutes")
    without guessing from raw labels.
    """
    totals: dict = {}
    for seg in speaker_segments:
        speaker = getattr(seg, "speaker", None)
        if speaker is None:
            continue
        totals[speaker] = totals.get(speaker, 0.0) + (seg.end - seg.start)
    return sorted(
        ((name, round(seconds, 2)) for name, seconds in totals.items()),
        key=lambda item: item[1],
        reverse=True,
    )


def normalize_spans(
    spans: Sequence[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Sort, drop empties and union overlaps, so no audio is counted twice.

    An operator typing ``0:10-0:20, 0:10-0:20`` (or ``0:10-0:20, 0:15-0:25``)
    means one stretch of speech, not two. Measuring each range independently
    would embed the same seconds twice, weight them twice in the centroid, and
    report twice the speech actually behind the result - the same overstatement
    the windowed measurement exists to prevent, in a narrower form.

    The operator's original ranges are kept separately for the record; this is
    what the audio is read from.
    """
    ordered = sorted((float(start), float(end)) for start, end in spans if end > start)
    merged: List[Tuple[float, float]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def parse_time_ranges(text: str) -> List[Tuple[float, float]]:
    """Parse operator-typed time ranges into ``[(start, end), ...]``.

    Accepts ``mm:ss`` / ``h:mm:ss`` / plain seconds, ranges separated by ``-``,
    and several ranges separated by commas, semicolons or newlines::

        "0:10-0:45, 1:20-2:00"   ->  [(10.0, 45.0), (80.0, 120.0)]

    Raises ``ValueError`` naming the offending piece, so the operator can see
    exactly what failed instead of silently enrolling the wrong audio.
    """
    ranges: List[Tuple[float, float]] = []
    cleaned = text.replace(";", ",").replace("\n", ",")
    for chunk in cleaned.split(","):
        piece = chunk.strip()
        if not piece:
            continue
        if "-" not in piece:
            raise ValueError(f"'{piece}' is not a range (expected start-end).")
        raw_start, _, raw_end = piece.partition("-")
        start = _parse_timestamp(raw_start.strip(), piece)
        end = _parse_timestamp(raw_end.strip(), piece)
        if end <= start:
            raise ValueError(f"'{piece}' ends before it starts.")
        ranges.append((start, end))
    if not ranges:
        raise ValueError("No time ranges given.")
    return ranges


def _parse_timestamp(value: str, context: str) -> float:
    """Seconds from ``ss``, ``mm:ss`` or ``h:mm:ss``."""
    if not value:
        raise ValueError(f"'{context}' is missing a time.")
    parts = value.split(":")
    if len(parts) > 3:
        raise ValueError(f"'{context}' has too many ':' parts.")
    try:
        numbers = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"'{context}' is not a valid time.") from exc
    seconds = 0.0
    for number in numbers:
        seconds = seconds * 60.0 + number
    if seconds < 0:
        raise ValueError(f"'{context}' is negative.")
    return seconds
