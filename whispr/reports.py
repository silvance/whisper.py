"""Operator-facing analysis report: what was analysed, by what, and what it means.

Produces a hand-off document covering the case/source information (including the
source recording's SHA-256), the transcript, every speaker comparison performed,
and a prominent statement of what a similarity result does and does not
establish.

The *content* is assembled as plain sections by :func:`build_report_sections`,
which needs no optional dependency and is unit-tested; :func:`write_docx` renders
those sections through python-docx. Keeping them apart means the wording - which
is the part that matters for how a result is read - is verified by tests even
where python-docx is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from .matching import ComparisonResult
from .provenance import AnalysisProvenance
from .thresholds import DISCLAIMER, Thresholds, active, describe
from .transcription import TranscriptionResult

PathLike = Union[str, Path]

REPORT_TITLE = "Whispers analysis report"
DISCLAIMER_HEADING = "Interpretation and limitations"


@dataclass
class ReportSection:
    """One heading plus its body lines."""

    heading: str
    lines: List[str] = field(default_factory=list)


def _clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _transcript_lines(
    result: Optional[TranscriptionResult],
    speaker_names: Optional[Dict[str, str]] = None,
) -> List[str]:
    """The transcript with timestamps and speaker labels, one line per segment."""
    if result is None:
        return ["No transcript is attached to this analysis."]
    if not result.segments:
        return [result.text or "(empty transcript)"]
    lines: List[str] = []
    for segment in result.segments:
        stamp = f"[{_clock(segment.start)}-{_clock(segment.end)}]"
        if segment.speaker:
            label = result._speaker_label(segment.speaker, speaker_names)
            lines.append(f"{stamp} [{label}] {segment.text}")
        else:
            lines.append(f"{stamp} {segment.text}")
    return lines


def _comparison_lines(comparison: ComparisonResult) -> List[str]:
    """One comparison, stated in similarity terms with its supporting numbers."""
    lines = [
        f"Reference subject: {comparison.reference_name}",
        f"Questioned speaker: {comparison.questioned_label}",
    ]
    if comparison.refused:
        lines += [
            "Result: comparison refused.",
            comparison.refusal_reason,
        ]
        return lines
    lines += [
        f"Similarity score: {comparison.score:.2f} / 1.00",
        f"Assessment: {comparison.band}",
        f"Operational threshold: {comparison.operational_threshold:.2f}",
    ]
    if comparison.margin is not None:
        lines.append(
            f"Margin over next best ({comparison.runner_up_name or 'n/a'}): "
            f"{comparison.margin:.2f}"
        )
    lines += [
        f"Reference speech: {comparison.reference_seconds:.1f} sec "
        f"({comparison.reference_quality})",
        f"Questioned speech: {comparison.questioned_seconds:.1f} sec "
        f"({comparison.questioned_quality})",
        f"Speaker embedding model: {comparison.embedding_model}",
    ]
    for warning in comparison.warnings:
        lines.append(f"Warning: {warning}")
    if comparison.conclusive and comparison.band.startswith("High"):
        # Deliberately phrased as a lead, never as an identification.
        lines.append(
            f"The questioned speaker produced high similarity to the "
            f"{comparison.reference_name} reference profile. Further review is "
            "warranted."
        )
    return lines


def build_report_sections(
    *,
    result: Optional[TranscriptionResult] = None,
    speaker_names: Optional[Dict[str, str]] = None,
    provenance: Optional[AnalysisProvenance] = None,
    comparisons: Sequence[ComparisonResult] = (),
    thresholds: Optional[Thresholds] = None,
    case_notes: str = "",
) -> List[ReportSection]:
    """Assemble the report's content, in order, as headed sections."""
    # A result can only be interpreted against the numbers that produced it, so
    # the report records the set actually in force, not the shipped defaults.
    thresholds = thresholds or active()
    generated = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    case_lines: List[str] = [f"Report generated (UTC): {generated}"]
    if provenance is not None:
        case_lines += provenance.describe()
    else:
        case_lines += [
            "Source file: not recorded",
            "Source SHA-256: not recorded",
            "No provenance was stored with this analysis, so the source and the "
            "models that produced it cannot be confirmed from this report.",
        ]
    if case_notes.strip():
        case_lines += ["", f"Case notes: {case_notes.strip()}"]

    sections = [
        ReportSection("Case and source information", case_lines),
        ReportSection(
            "Transcription",
            (
                [
                    f"Detected language: "
                    f"{result.language or 'unknown'} "
                    f"({result.language_probability:.0%} language-ID confidence)",
                    f"Duration: {result.duration:.1f} sec",
                    "",
                ]
                if result is not None
                else []
            )
            + _transcript_lines(result, speaker_names),
        ),
    ]

    comparison_lines: List[str] = []
    if comparisons:
        for index, comparison in enumerate(comparisons, 1):
            if index > 1:
                comparison_lines.append("")
            comparison_lines.append(f"Comparison {index}")
            comparison_lines += _comparison_lines(comparison)
    else:
        comparison_lines.append("No speaker comparisons were performed.")
    comparison_lines += ["", "Active thresholds:"] + describe(thresholds)
    sections.append(ReportSection("Speaker analysis", comparison_lines))

    sections.append(
        ReportSection(
            DISCLAIMER_HEADING,
            [
                DISCLAIMER,
                "",
                "Similarity is affected by recording quality, channel and "
                "microphone differences, background noise, and how much speech "
                "each side provides. Thresholds in this build are conservative "
                "defaults and require validation against representative "
                "operational recordings before any operational weight is placed "
                "on a specific score.",
            ],
        )
    )
    return sections


def render_text(sections: Sequence[ReportSection], *, title: str = REPORT_TITLE) -> str:
    """Plain-text rendering - useful for previews, logs and tests."""
    out = [title, "=" * len(title), ""]
    for section in sections:
        out.append(section.heading)
        out.append("-" * len(section.heading))
        out.extend(section.lines)
        out.append("")
    return "\n".join(out)


def write_docx(
    sections: Sequence[ReportSection],
    path: PathLike,
    *,
    title: str = REPORT_TITLE,
) -> Path:
    """Render ``sections`` to a ``.docx`` at ``path``."""
    from .export import _document

    document = _document()
    document.add_heading(title, level=0)
    for section in sections:
        document.add_heading(section.heading, level=1)
        for line in section.lines:
            document.add_paragraph(line)
    out = Path(path)
    document.save(str(out))
    return out


def write_analysis_report(
    path: PathLike,
    *,
    result: Optional[TranscriptionResult] = None,
    speaker_names: Optional[Dict[str, str]] = None,
    provenance: Optional[AnalysisProvenance] = None,
    comparisons: Sequence[ComparisonResult] = (),
    thresholds: Optional[Thresholds] = None,
    case_notes: str = "",
    title: str = REPORT_TITLE,
) -> Path:
    """Build and write the full analysis report (``.docx``, or ``.txt`` by suffix)."""
    sections = build_report_sections(
        result=result,
        speaker_names=speaker_names,
        provenance=provenance,
        comparisons=comparisons,
        thresholds=thresholds,
        case_notes=case_notes,
    )
    out = Path(path)
    if out.suffix.lower() == ".txt":
        out.write_text(render_text(sections, title=title), encoding="utf-8")
        return out
    return write_docx(sections, out, title=title)
