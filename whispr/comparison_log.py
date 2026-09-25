"""A durable record of every speaker comparison this copy has run.

A comparison is an investigative act about a named person, so the fact that it
happened is part of the record - not just the answer it gave. Without this, the
only trace of a comparison is whatever the operator happened to export before
running the next one, and a question asked six weeks later ("what did we
actually run against this subject, and what came back?") has no answer.

What is kept is the *account* of a comparison: which subject, which recording
(by name and SHA-256), which speech was selected, the score, the band, the
thresholds in force, the audio quality on both sides and any warnings. What is
never kept here is the evidence itself - no audio, no transcript, and no
embeddings. A voice embedding is the sensitive artefact in this system; a log of
activity should not become a second copy of it.

Records are individual files, written atomically and never rewritten, so an
entry cannot be lost or truncated by a later write. They are read back
newest-first and an unreadable one is skipped rather than taken as an empty log.

The bands and wording here are the same investigative ones used everywhere else:
a similarity score with a stated threshold, never a percentage, a probability of
identity, or a claim that two recordings are the same person.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

from .buildinfo import UNKNOWN as UNKNOWN_BUILD
from .settings import settings_path
from .speaker_profiles import ProfileError, SpeakerProfile, write_json_atomic

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .matching import ComparisonResult, GalleryResult

PathLike = Union[str, Path]

SCHEMA_VERSION = 2
RECORD_SUFFIX = ".whispr-comparison.json"

# The two things that get logged.
KIND_PROFILE = "profile"
KIND_GALLERY = "gallery"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _stamp() -> str:
    """A filename-safe, sortable timestamp, so the directory lists in order."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _spans(raw: Any) -> List[Tuple[float, float]]:
    """Read back time ranges, dropping anything that is not a usable pair."""
    out: List[Tuple[float, float]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            try:
                out.append((float(item[0]), float(item[1])))
            except (TypeError, ValueError):
                continue
    return out


def _thresholds(raw: Any) -> Dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, float] = {}
    for key, value in raw.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


@dataclass
class RankedSubject:
    """One subject's place in a gallery search."""

    display_name: str
    subject_id: str = ""
    score: float = 0.0
    band: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "display_name": self.display_name,
            "subject_id": self.subject_id,
            "score": round(self.score, 4),
            "band": self.band,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "RankedSubject":
        if not isinstance(data, dict):
            return cls(display_name="")
        return cls(
            display_name=str(data.get("display_name") or ""),
            subject_id=str(data.get("subject_id") or ""),
            score=float(data.get("score") or 0.0),
            band=str(data.get("band") or ""),
        )


@dataclass
class ComparisonRecord:
    """One logged comparison: what was run, against whom, and what came back."""

    record_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    recorded_utc: str = field(default_factory=_utc_now)
    kind: str = KIND_PROFILE

    # Who it was measured against. Empty for a gallery search, which is measured
    # against every subject at once.
    reference_name: str = ""
    reference_subject_id: str = ""
    reference_seconds: float = 0.0
    reference_quality: str = ""

    # What the reference profile *was* at the time. A profile grows: the same
    # recording measured against the same subject next month can legitimately
    # score differently, and without this the record could not say why.
    reference_sample_count: int = 0
    reference_updated_utc: str = ""

    # What was measured.
    questioned_filename: str = ""
    questioned_sha256: str = ""
    # Where the recording was when it was compared. A pointer for finding it
    # again, never a claim that it is still there - and never a copy of it.
    questioned_path: str = ""
    questioned_selection: str = ""
    questioned_label: str = ""
    questioned_seconds: float = 0.0
    questioned_quality: str = ""
    questioned_window_count: int = 0
    # The stretches of the recording the embedding was actually made from.
    questioned_spans: List[Tuple[float, float]] = field(default_factory=list)

    # What came back.
    score: float = 0.0
    band: str = ""
    operational_threshold: float = 0.0
    refused: bool = False
    refusal_reason: str = ""
    # A gallery search ranks every subject; the 1:1 comparison leaves this empty.
    ranked: List[RankedSubject] = field(default_factory=list)
    subjects_searched: int = 0
    margin: Optional[float] = None
    runner_up_name: str = ""
    embedding_model: str = ""
    # Every threshold in force when this was decided, not just the one the band
    # turned on: a record read later has to be readable against its own rules.
    thresholds: Dict[str, float] = field(default_factory=dict)
    app_version: str = ""
    # Which build decided this. The package version is inherited from upstream
    # Whisper and is the same string across every Whispers release, so on its
    # own it cannot tell one operational build from another. The build id and
    # commit can, and they are what a later question - "which build produced
    # this score?" - actually needs answering with.
    build_id: str = ""
    git_commit: str = ""
    warnings: List[str] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    @property
    def subject_label(self) -> str:
        """How this record is filed under a subject in the history."""
        if self.kind == KIND_GALLERY:
            top = self.ranked[0].display_name if self.ranked else ""
            return top or "(all subjects)"
        return self.reference_name or "(unnamed subject)"

    @property
    def recording_label(self) -> str:
        return self.questioned_filename or "(recording not named)"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "record_id": self.record_id,
            "recorded_utc": self.recorded_utc,
            "kind": self.kind,
            "reference_name": self.reference_name,
            "reference_subject_id": self.reference_subject_id,
            "reference_seconds": round(self.reference_seconds, 2),
            "reference_quality": self.reference_quality,
            "reference_sample_count": self.reference_sample_count,
            "reference_updated_utc": self.reference_updated_utc,
            "questioned_filename": self.questioned_filename,
            "questioned_sha256": self.questioned_sha256,
            "questioned_path": self.questioned_path,
            "questioned_selection": self.questioned_selection,
            "questioned_label": self.questioned_label,
            "questioned_seconds": round(self.questioned_seconds, 2),
            "questioned_quality": self.questioned_quality,
            "questioned_window_count": self.questioned_window_count,
            "questioned_spans": [
                [round(a, 2), round(b, 2)] for a, b in self.questioned_spans
            ],
            "score": round(self.score, 4),
            "band": self.band,
            "operational_threshold": round(self.operational_threshold, 4),
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "ranked": [r.to_dict() for r in self.ranked],
            "subjects_searched": self.subjects_searched,
            "margin": self.margin,
            "runner_up_name": self.runner_up_name,
            "embedding_model": self.embedding_model,
            "thresholds": dict(self.thresholds),
            "app_version": self.app_version,
            "build_id": self.build_id,
            "git_commit": self.git_commit,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComparisonRecord":
        version = data.get("schema_version")
        if isinstance(version, int) and version > SCHEMA_VERSION:
            raise ProfileError(
                f"This comparison record was written by a newer version of "
                f"Whispers (schema {version}, this build understands "
                f"{SCHEMA_VERSION})."
            )
        raw_ranked = data.get("ranked")
        ranked = (
            [RankedSubject.from_dict(item) for item in raw_ranked]
            if isinstance(raw_ranked, list)
            else []
        )
        warnings = data.get("warnings")
        return cls(
            record_id=str(data.get("record_id") or uuid.uuid4().hex[:12]),
            recorded_utc=str(data.get("recorded_utc") or ""),
            kind=str(data.get("kind") or KIND_PROFILE),
            reference_name=str(data.get("reference_name") or ""),
            reference_subject_id=str(data.get("reference_subject_id") or ""),
            reference_seconds=float(data.get("reference_seconds") or 0.0),
            reference_quality=str(data.get("reference_quality") or ""),
            reference_sample_count=int(data.get("reference_sample_count") or 0),
            reference_updated_utc=str(data.get("reference_updated_utc") or ""),
            questioned_filename=str(data.get("questioned_filename") or ""),
            questioned_sha256=str(data.get("questioned_sha256") or ""),
            questioned_path=str(data.get("questioned_path") or ""),
            questioned_selection=str(data.get("questioned_selection") or ""),
            questioned_label=str(data.get("questioned_label") or ""),
            questioned_seconds=float(data.get("questioned_seconds") or 0.0),
            questioned_quality=str(data.get("questioned_quality") or ""),
            questioned_window_count=int(data.get("questioned_window_count") or 0),
            questioned_spans=_spans(data.get("questioned_spans")),
            score=float(data.get("score") or 0.0),
            band=str(data.get("band") or ""),
            operational_threshold=float(data.get("operational_threshold") or 0.0),
            refused=bool(data.get("refused")),
            refusal_reason=str(data.get("refusal_reason") or ""),
            ranked=ranked,
            subjects_searched=int(data.get("subjects_searched") or 0),
            margin=(float(data["margin"]) if data.get("margin") is not None else None),
            runner_up_name=str(data.get("runner_up_name") or ""),
            embedding_model=str(data.get("embedding_model") or ""),
            thresholds=_thresholds(data.get("thresholds")),
            app_version=str(data.get("app_version") or ""),
            # Absent from schema 1 records, which is the honest answer for
            # them: the build was not recorded, so it is not known now.
            build_id=str(data.get("build_id") or ""),
            git_commit=str(data.get("git_commit") or ""),
            warnings=[str(w) for w in warnings] if isinstance(warnings, list) else [],
        )


# -- Building a record from a result ---------------------------------------


def _identity() -> "Tuple[str, str, str]":
    """``(application version, build id, git commit)`` for this copy.

    Best-effort: a record is worth keeping even when the build cannot say what
    it is, and an empty string is the truthful answer in that case. Nothing
    here invents an identity it does not have.
    """
    try:
        from .buildinfo import build_info

        info = build_info()
        return (
            info.application_version,
            "" if info.build_id == UNKNOWN_BUILD else info.build_id,
            "" if info.git_commit == UNKNOWN_BUILD else info.git_commit,
        )
    except Exception:  # noqa: BLE001 - provenance is nice to have, not required
        return "", "", ""


def record_from_comparison(
    result: "ComparisonResult",
    *,
    profile: "Optional[SpeakerProfile]" = None,
    questioned_path: str = "",
    questioned_spans: Optional[Sequence[Tuple[float, float]]] = None,
) -> ComparisonRecord:
    """The account of a 1:1 comparison, without the evidence behind it.

    ``profile`` is read only for the state it was in - how many trusted samples
    it held and when it last changed - so a record can later say whether a
    different score today means a different recording or simply a profile that
    has grown since.
    """
    return ComparisonRecord(
        kind=KIND_PROFILE,
        reference_name=result.reference_name,
        reference_subject_id=result.reference_subject_id,
        reference_seconds=result.reference_seconds,
        reference_quality=result.reference_quality,
        reference_sample_count=(
            len(profile.trusted_samples()) if profile is not None else 0
        ),
        reference_updated_utc=(profile.updated_utc if profile is not None else ""),
        questioned_filename=result.questioned_source_filename or "",
        questioned_sha256=result.questioned_source_sha256 or "",
        questioned_path=questioned_path,
        questioned_selection=result.questioned_selection,
        questioned_label=result.questioned_label,
        questioned_seconds=result.questioned_seconds,
        questioned_quality=result.questioned_quality,
        questioned_window_count=result.questioned_window_count,
        questioned_spans=list(questioned_spans or []),
        score=result.score,
        band=result.band,
        operational_threshold=result.operational_threshold,
        refused=result.refused,
        refusal_reason=result.refusal_reason,
        margin=result.margin,
        runner_up_name=result.runner_up_name or "",
        embedding_model=result.embedding_model,
        thresholds=result.thresholds.to_dict(),
        app_version=_identity()[0],
        build_id=_identity()[1],
        git_commit=_identity()[2],
        warnings=list(result.warnings),
    )


def record_from_gallery(
    result: "GalleryResult",
    *,
    questioned_seconds: float = 0.0,
    questioned_quality: str = "",
    questioned_window_count: int = 0,
    questioned_path: str = "",
    questioned_spans: Optional[Sequence[Tuple[float, float]]] = None,
    embedding_model: str = "",
) -> ComparisonRecord:
    """The account of a search across every subject.

    The whole ranking is kept, not just the top row: which subjects were *not*
    close is as much a part of the record as which one was, and a lead that
    rested on a thin margin should still read that way months later.
    """
    ranked = [
        RankedSubject(
            display_name=match.display_name,
            subject_id=match.subject_id,
            score=match.score,
            band=match.band,
        )
        for match in result.matches
    ]
    warnings = list(result.skipped)
    if result.inadequate_reason:
        warnings.insert(0, result.inadequate_reason)
    return ComparisonRecord(
        kind=KIND_GALLERY,
        questioned_filename=result.questioned_source_filename or "",
        questioned_sha256=result.questioned_source_sha256 or "",
        questioned_path=questioned_path,
        questioned_selection=result.questioned_selection,
        questioned_label=result.questioned_label,
        questioned_seconds=questioned_seconds,
        questioned_quality=questioned_quality,
        questioned_window_count=questioned_window_count,
        questioned_spans=list(questioned_spans or []),
        score=ranked[0].score if ranked else 0.0,
        band=ranked[0].band if ranked else "",
        operational_threshold=result.thresholds.recognition_acceptance,
        # A search over a gallery with nothing usable in it is still a record of
        # having looked; "refused" is reserved for a comparison that could not
        # lawfully be scored at all.
        refused=False,
        ranked=ranked,
        subjects_searched=result.searched,
        margin=result.decision.margin if result.decision is not None else None,
        runner_up_name=(
            (result.decision.second_name or "") if result.decision is not None else ""
        ),
        embedding_model=embedding_model,
        thresholds=result.thresholds.to_dict(),
        app_version=_identity()[0],
        build_id=_identity()[1],
        git_commit=_identity()[2],
        warnings=warnings,
    )


# -- Storage ---------------------------------------------------------------


def comparisons_dir() -> Path:
    """Directory holding the comparison history, beside the subject profiles."""
    return settings_path().parent / "comparisons"


def record_path(record: ComparisonRecord) -> Path:
    """Where ``record`` lives: timestamp first, so the directory sorts by time."""
    return comparisons_dir() / f"{_stamp()}-{record.record_id}{RECORD_SUFFIX}"


def save_comparison(record: ComparisonRecord) -> Path:
    """Write ``record`` atomically. Raises :class:`ProfileError` on failure."""
    target = record_path(record)
    try:
        return write_json_atomic(target, record.to_dict())
    except OSError as exc:
        raise ProfileError(
            f"Couldn't write the comparison record to {target}: {exc}"
        ) from exc


def load_comparison(path: PathLike) -> ComparisonRecord:
    """Read one record. Raises :class:`ProfileError` if it cannot be trusted."""
    import json

    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProfileError(f"Couldn't read {Path(path).name}: {exc}") from exc
    if not isinstance(data, dict):
        raise ProfileError(f"{Path(path).name} is not a comparison record.")
    return ComparisonRecord.from_dict(data)


def list_comparisons(limit: Optional[int] = None) -> List[ComparisonRecord]:
    """Every stored record, newest first; unreadable ones are skipped.

    A record that cannot be parsed is left on disk rather than deleted - it is
    still evidence that something was run, and an operator may want to look at
    the file itself.
    """
    try:
        entries = sorted(comparisons_dir().glob(f"*{RECORD_SUFFIX}"), reverse=True)
    except OSError:
        return []
    out: List[ComparisonRecord] = []
    for entry in entries:
        try:
            out.append(load_comparison(entry))
        except ProfileError:
            continue
        if limit is not None and len(out) >= limit:
            break
    out.sort(key=lambda r: r.recorded_utc, reverse=True)
    return out


def delete_comparison(record: ComparisonRecord) -> bool:
    """Remove one record. Returns True if a file was actually deleted."""
    try:
        entries = list(comparisons_dir().glob(f"*{record.record_id}{RECORD_SUFFIX}"))
    except OSError:
        return False
    removed = False
    for entry in entries:
        try:
            entry.unlink()
            removed = True
        except OSError:
            continue
    return removed


def subjects_in(records: Sequence[ComparisonRecord]) -> List[str]:
    """The subject names present in ``records``, in alphabetical order."""
    names = {r.subject_label for r in records if r.subject_label}
    return sorted(names, key=str.casefold)


def export_csv(records: Sequence[ComparisonRecord], path: PathLike) -> Path:
    """Write ``records`` as a spreadsheet-readable table.

    One row per comparison, in the same terms the screen uses. This is also the
    shape the threshold work wants: every trial the office has actually run,
    with the audio conditions beside the score.
    """
    import csv

    columns = [
        "recorded_utc",
        "kind",
        "subject",
        "questioned_filename",
        "questioned_sha256",
        "questioned_path",
        "questioned_selection",
        "questioned_seconds",
        "questioned_quality",
        "questioned_windows",
        "reference_seconds",
        "reference_quality",
        "reference_samples_at_the_time",
        "reference_updated_at_the_time",
        "score",
        "operational_threshold",
        "assessment",
        "margin_over_runner_up",
        "runner_up",
        "refused",
        "refusal_reason",
        "subjects_searched",
        "embedding_model",
        "app_version",
        "build_id",
        "git_commit",
        "warnings",
    ]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for record in records:
            writer.writerow(
                [
                    record.recorded_utc,
                    record.kind,
                    record.subject_label,
                    record.questioned_filename,
                    record.questioned_sha256,
                    record.questioned_path,
                    record.questioned_selection,
                    f"{record.questioned_seconds:.2f}",
                    record.questioned_quality,
                    record.questioned_window_count,
                    f"{record.reference_seconds:.2f}",
                    record.reference_quality,
                    record.reference_sample_count,
                    record.reference_updated_utc,
                    f"{record.score:.4f}",
                    f"{record.operational_threshold:.2f}",
                    record.band,
                    "" if record.margin is None else f"{record.margin:.4f}",
                    record.runner_up_name,
                    "yes" if record.refused else "no",
                    record.refusal_reason,
                    record.subjects_searched,
                    record.embedding_model,
                    record.app_version,
                    record.build_id,
                    record.git_commit,
                    " | ".join(record.warnings),
                ]
            )
    return out
