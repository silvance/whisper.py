"""Measure a questioned speaker from a recording, and record what was measured.

A comparison is only as honest as its account of the audio behind it. Two things
this module exists to keep straight:

**The embedding must represent the speech the result claims.** An earlier
version embedded the single longest usable stretch but reported the voiced
duration of *everything* selected, so a speaker with 24 seconds spread over many
short turns could be measured from 2 seconds of audio and still clear a
minimum-duration check written to prevent exactly that. Here the selection is
split into windows, every usable window is embedded, the embeddings are
averaged, and the duration reported is the speech of the windows that actually
went into the average - the same construction the reference side uses, so the
two are comparable.

**The questioned recording identifies itself.** The result carries the source
file's name, size and SHA-256, the operator's selection, and the exact spans
embedded, so a report describes the recording that was compared rather than
whatever happened to be open in another tab.

Local only: it reads audio, hashes a file and runs the bundled embedder.
"""

from __future__ import annotations

import os
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

from .enrollment import (
    MIN_WINDOW_SECONDS,
    normalize_spans,
    prepare_source,
    windows,
)
from .hashing import short
from .quality import (
    INSUFFICIENT,
    QualityReport,
    analyse_span,
    combine,
    read_wav_span,
)
from .voiceprints import centroid

PathLike = Union[str, Path]
ProgressFn = Callable[[str], None]

# How the operator picked the questioned speech, for the record.
SELECTION_WHOLE = "whole recording"
SELECTION_DIARIZED = "diarized speaker"
SELECTION_RANGES = "time ranges"


@dataclass
class QuestionedSpeaker:
    """A questioned speaker's embedding, plus the audio and source behind it."""

    embedding: List[float] = field(default_factory=list)
    # Speech in the windows that were actually embedded - never the whole
    # selection, which may include audio no embedding was made from.
    speech_seconds: float = 0.0
    quality: str = INSUFFICIENT
    warnings: List[str] = field(default_factory=list)
    # Every span that went into the embedding, and everything the operator
    # selected, kept separately so the difference is visible.
    embedded_spans: List[Tuple[float, float]] = field(default_factory=list)
    # How many windows went into the average, and whether several turns had to
    # be joined to make windows at all.
    window_count: int = 0
    aggregated: bool = False
    selected_spans: List[Tuple[float, float]] = field(default_factory=list)
    selected_seconds: float = 0.0
    selection_mode: str = SELECTION_WHOLE
    skipped: List[str] = field(default_factory=list)
    source_filename: str = ""
    source_sha256: Optional[str] = None
    source_size: Optional[int] = None

    @property
    def usable(self) -> bool:
        return bool(self.embedding)

    @property
    def label(self) -> str:
        return self.source_filename or "Questioned speaker"

    def describe(self) -> List[str]:
        """Operator-facing lines naming the recording and what was measured."""
        lines = [
            f"Questioned recording: {self.source_filename or 'unknown'}",
            f"Source SHA-256: {self.source_sha256 or 'not recorded'}",
            f"Selection: {self.selection_mode}"
            + (
                f" ({len(self.selected_spans)} span(s), "
                f"{self.selected_seconds:.1f} sec of speech)"
                if self.selected_spans
                else ""
            ),
            f"Speech measured: {self.speech_seconds:.1f} sec across "
            f"{self.window_count} window(s) (quality: {self.quality})",
        ]
        if self.skipped:
            lines.append(f"Windows skipped: {len(self.skipped)} ({self.skipped[0]})")
        return lines

    def to_dict(self) -> dict:
        return {
            "source_filename": self.source_filename,
            "source_sha256": self.source_sha256,
            "source_size": self.source_size,
            "selection_mode": self.selection_mode,
            "selected_spans": [list(s) for s in self.selected_spans],
            "selected_seconds": round(self.selected_seconds, 2),
            "embedded_spans": [list(s) for s in self.embedded_spans],
            "window_count": self.window_count,
            "turns_joined": self.aggregated,
            "speech_seconds": round(self.speech_seconds, 2),
            "quality": self.quality,
            "warnings": list(self.warnings),
            "skipped": list(self.skipped),
        }


# A join between two turns is a discontinuity; a short ramp either side keeps it
# from becoming a click the embedder has to account for.
_JOIN_FADE_SECONDS = 0.01


def concatenate_spans(
    wav_path: PathLike, spans: Sequence[Tuple[float, float]]
) -> Tuple[Path, List[Tuple[float, float, float, float]]]:
    """Write just ``spans`` to a temporary speaker-only WAV.

    Returns the file and a mapping of ``(start, end, source_start, source_end)``
    tying every stretch of the new file back to where it came from, so a
    measurement over the concatenation can still be reported in the source
    recording's own timeline.

    This is what makes a conversational speaker measurable. Their speech
    arrives as short turns between other people's; taken one at a time most
    turns are too brief to characterise a voice, and 20 seconds of a target
    spread over ten turns would yield nothing. Joined, it is 20 seconds of that
    speaker. The audio is only ever *their* selected speech - nothing from
    another speaker is included, and the joins are not presented as continuous
    speech in the source.
    """
    import numpy as np

    pieces: List[Tuple[float, float, float, float]] = []
    chunks: List[Any] = []
    position = 0.0
    rate = 16000
    for source_start, source_end in spans:
        samples, rate = read_wav_span(wav_path, source_start, source_end)
        if samples.size == 0:
            continue
        samples = samples.copy()
        fade = min(int(_JOIN_FADE_SECONDS * rate), samples.size // 2)
        if fade > 0:
            ramp = np.linspace(0.0, 1.0, fade, dtype=samples.dtype)
            samples[:fade] *= ramp
            samples[-fade:] *= ramp[::-1]
        duration = samples.size / float(rate)
        pieces.append((position, position + duration, source_start, source_end))
        chunks.append(samples)
        position += duration

    handle, temp_path = tempfile.mkstemp(suffix=".wav", prefix="whispr-questioned-")
    os.close(handle)
    out = Path(temp_path)
    joined = np.concatenate(chunks) if chunks else np.zeros(0, dtype="float32")
    with wave.open(str(out), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes((joined * 32767.0).astype("<i2").tobytes())
    return out, pieces


def source_intervals(
    pieces: Sequence[Tuple[float, float, float, float]],
    start: float,
    end: float,
) -> List[Tuple[float, float]]:
    """Where ``[start, end)`` of a concatenation came from in the source."""
    out: List[Tuple[float, float]] = []
    for piece_start, piece_end, source_start, _source_end in pieces:
        overlap_start = max(start, piece_start)
        overlap_end = min(end, piece_end)
        if overlap_end <= overlap_start:
            continue
        out.append(
            (
                source_start + (overlap_start - piece_start),
                source_start + (overlap_end - piece_start),
            )
        )
    return out


def measure_from_wav(
    wav_path: PathLike,
    spans: Sequence[Tuple[float, float]],
    embedder: Any,
    *,
    selection_mode: str = SELECTION_WHOLE,
    progress: Optional[ProgressFn] = None,
) -> QuestionedSpeaker:
    """Embed every usable window of ``spans`` and average them.

    Averaging several windows, rather than picking the longest one, uses all of
    the speech the operator selected and matches how a reference profile's
    centroid is built - so the two sides of a comparison are the same kind of
    quantity.

    Repeated or overlapping ranges are unioned first, so the same seconds
    cannot be measured, weighted and counted twice. Where the selection is
    several turns, they are joined into one speaker-only stretch before being
    windowed: a conversational target speaks in bursts too short to
    characterise individually, and measuring turn by turn would throw away
    speech that is perfectly usable once assembled. Windows are reported in
    the source recording's own timeline either way.
    """
    selected = [(float(a), float(b)) for a, b in spans]
    out = QuestionedSpeaker(
        selection_mode=selection_mode, selected_spans=list(selected)
    )
    measured_spans = normalize_spans(selected)
    if len(measured_spans) < len(selected):
        out.warnings.append(
            f"{len(selected)} selected range(s) cover "
            f"{len(measured_spans)} distinct stretch(es) of audio; ranges that "
            "overlap, repeat or run straight on are measured once."
        )
    if not measured_spans:
        out.warnings.append("The questioned selection is empty.")
        return out

    # One stretch is measured where it lies; several are joined first, so short
    # turns accumulate into windows long enough to characterise a voice.
    aggregated = len(measured_spans) > 1
    working: PathLike = wav_path
    pieces: List[Tuple[float, float, float, float]] = [
        (a, b, a, b) for a, b in measured_spans
    ]
    if aggregated:
        if progress is not None:
            progress(f"Assembling {len(measured_spans)} turn(s) of speech…")
        working, pieces = concatenate_spans(wav_path, measured_spans)
    out.aggregated = aggregated

    # Measured on the same audio the windows are drawn from, so "measured X of
    # the Y selected" compares like with like rather than two different
    # voiced-speech estimates.
    first_start = pieces[0][0] if pieces else 0.0
    last_end = pieces[-1][1] if pieces else 0.0
    out.selected_seconds = round(
        analyse_span(working, first_start, last_end).voiced_seconds, 2
    )

    try:
        _measure_windows(out, working, pieces, embedder, progress)
    finally:
        if aggregated:
            try:
                Path(working).unlink()
            except OSError:
                pass
    return out


def _measure_windows(
    out: QuestionedSpeaker,
    working: PathLike,
    pieces: Sequence[Tuple[float, float, float, float]],
    embedder: Any,
    progress: Optional[ProgressFn],
) -> None:
    """Window ``working``, embed what is usable, and record it in source time."""
    total = pieces[-1][1] if pieces else 0.0
    first = pieces[0][0] if pieces else 0.0
    used_reports: List[QualityReport] = []
    vectors: List[List[float]] = []

    slots = windows(first, total)
    if not slots:
        out.skipped.append(
            f"{total - first:.1f}s of speech: shorter than the "
            f"{MIN_WINDOW_SECONDS:.0f}s needed to measure a voice."
        )
    for start, end in slots:
        report = analyse_span(working, start, end)
        origin = source_intervals(pieces, start, end)
        where = _describe_intervals(origin)
        if not report.usable:
            out.skipped.append(
                f"{where}: {report.assessment.lower()} audio "
                f"({report.voiced_seconds:.1f}s of speech)."
            )
            continue
        if progress is not None:
            progress(f"Measuring {where}…")
        vector = embedder.embed_span(working, start, end)
        if not vector:
            out.skipped.append(f"{where}: no embedding produced.")
            continue
        vectors.append(list(vector))
        used_reports.append(report)
        out.embedded_spans.extend(origin)

    if not vectors:
        out.warnings.append(
            "No window of the questioned selection held enough usable speech to "
            "measure."
        )
        return

    out.embedding = centroid(vectors)
    # The duration behind the embedding, not the duration of the selection.
    out.speech_seconds = round(sum(r.voiced_seconds for r in used_reports), 2)
    out.window_count = len(vectors)
    merged = combine(used_reports)
    out.quality = merged.assessment
    out.warnings.extend(merged.warnings)
    if out.speech_seconds < out.selected_seconds - 0.05:
        out.warnings.append(
            f"Measured {out.speech_seconds:.1f}s of the "
            f"{out.selected_seconds:.1f}s selected; the rest was not usable."
        )


def _describe_intervals(intervals: Sequence[Tuple[float, float]]) -> str:
    """Source spans as an operator-readable list, e.g. ``12.0-20.0, 31.5-35.0s``."""
    if not intervals:
        return "no source interval"
    return ", ".join(f"{start:.1f}-{end:.1f}" for start, end in intervals) + "s"


def measure(
    source: PathLike,
    spans: Optional[Sequence[Tuple[float, float]]],
    embedder: Any,
    *,
    selection_mode: str = SELECTION_WHOLE,
    progress: Optional[ProgressFn] = None,
) -> QuestionedSpeaker:
    """Measure a questioned speaker from any media file, recording its identity.

    ``spans`` selects the questioned speech; omit it for the whole recording.
    The SHA-256 is of the file the operator chose - the artefact a report must
    be traceable to - taken before any conversion, and the source is never
    modified.
    """
    source_path = Path(source)
    wav, digest, temporary = prepare_source(source_path, progress=progress)
    try:
        if spans is None:
            spans = [(0.0, _wav_duration(wav))]
            selection_mode = SELECTION_WHOLE
        result = measure_from_wav(
            wav,
            spans,
            embedder,
            selection_mode=selection_mode,
            progress=progress,
        )
    finally:
        if temporary:
            try:
                Path(wav).unlink()
            except OSError:
                pass
    result.source_filename = source_path.name
    result.source_sha256 = digest
    try:
        result.source_size = source_path.stat().st_size
    except OSError:
        result.source_size = None
    return result


def _wav_duration(wav_path: PathLike) -> float:
    import wave

    with wave.open(str(wav_path), "rb") as handle:
        rate = handle.getframerate() or 1
        return handle.getnframes() / float(rate)


def short_digest(value: Optional[str]) -> str:
    """Abbreviated source hash for a status line (the full value stays on record)."""
    return short(value)
