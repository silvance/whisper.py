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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

from .enrollment import prepare_source, windows
from .hashing import short
from .quality import INSUFFICIENT, QualityReport, analyse_span, combine
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
            f"{len(self.embedded_spans)} window(s) (quality: {self.quality})",
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
            "speech_seconds": round(self.speech_seconds, 2),
            "quality": self.quality,
            "warnings": list(self.warnings),
            "skipped": list(self.skipped),
        }


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
    """
    out = QuestionedSpeaker(
        selection_mode=selection_mode,
        selected_spans=[(float(a), float(b)) for a, b in spans],
    )
    selected_reports: List[QualityReport] = []
    used_reports: List[QualityReport] = []
    vectors: List[List[float]] = []

    for span_start, span_end in spans:
        selected_reports.append(analyse_span(wav_path, span_start, span_end))
        pieces = windows(span_start, span_end)
        if not pieces:
            out.skipped.append(
                f"{span_start:.1f}-{span_end:.1f}s: too short to measure."
            )
            continue
        for start, end in pieces:
            report = analyse_span(wav_path, start, end)
            if not report.usable:
                out.skipped.append(
                    f"{start:.1f}-{end:.1f}s: {report.assessment.lower()} audio "
                    f"({report.voiced_seconds:.1f}s of speech)."
                )
                continue
            if progress is not None:
                progress(f"Measuring {start:.0f}-{end:.0f}s…")
            vector = embedder.embed_span(wav_path, start, end)
            if not vector:
                out.skipped.append(f"{start:.1f}-{end:.1f}s: no embedding produced.")
                continue
            vectors.append(list(vector))
            used_reports.append(report)
            out.embedded_spans.append((start, end))

    out.selected_seconds = round(sum(r.voiced_seconds for r in selected_reports), 2)
    if not vectors:
        out.warnings.append(
            "No window of the questioned selection held enough usable speech to "
            "measure."
        )
        return out

    out.embedding = centroid(vectors)
    # The duration behind the embedding, not the duration of the selection.
    out.speech_seconds = round(sum(r.voiced_seconds for r in used_reports), 2)
    merged = combine(used_reports)
    out.quality = merged.assessment
    out.warnings.extend(merged.warnings)
    if out.speech_seconds < out.selected_seconds - 0.05:
        out.warnings.append(
            f"Measured {out.speech_seconds:.1f}s of the "
            f"{out.selected_seconds:.1f}s selected; the rest was not usable."
        )
    return out


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
