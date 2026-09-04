"""Speaker comparison and open-set identification against known-subject profiles.

Two operations sit here, both deliberately conservative:

* **1:1 comparison** - one questioned speaker against one known subject's
  reference profile. Produces a similarity *score*, a qualitative band, the
  thresholds and durations behind it, and the quality of both sides.
* **Gallery search** - one questioned speaker against every known subject. Ranks
  them, but applies the same acceptance-and-margin rule before calling anything a
  lead, so a "best" match is not automatically an identification.

Three rules run through all of it:

* A cosine similarity is **not** a probability of identity. Nothing here reports
  a percentage, a confidence of identity, or a likelihood that two recordings
  contain the same person.
* Insufficient or poor audio outranks a good-looking number: with too little
  usable speech the assessment is ``Insufficient data``, whatever the score.
* Vectors from different embedding models are not comparable, so a comparison is
  refused outright unless the models match (or the operator explicitly accepts an
  unverifiable legacy profile).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from .quality import INSUFFICIENT
from .speaker_profiles import (
    Compatibility,
    SpeakerProfile,
    check_compatibility,
    display_labels,
)
from .thresholds import (
    BAND_INSUFFICIENT,
    DISCLAIMER,
    Thresholds,
    active,
)
from .voiceprints import (
    MatchDecision,
    cosine_similarity,
    decide_identity,
    similarity_band,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .questioned import QuestionedSpeaker


@dataclass
class ComparisonResult:
    """One questioned speaker measured against one reference profile."""

    reference_name: str = ""
    reference_subject_id: str = ""
    questioned_label: str = "Questioned speaker"
    score: float = 0.0
    band: str = BAND_INSUFFICIENT
    reference_seconds: float = 0.0
    questioned_seconds: float = 0.0
    reference_quality: str = INSUFFICIENT
    questioned_quality: str = INSUFFICIENT
    embedding_model: str = "unknown"
    # The questioned recording, so a report describes what was compared rather
    # than whatever analysis happened to be open elsewhere in the application.
    questioned_source_filename: Optional[str] = None
    questioned_source_sha256: Optional[str] = None
    questioned_selection: str = ""
    questioned_window_count: int = 0
    thresholds: Thresholds = field(default_factory=active)
    compatibility: Optional[Compatibility] = None
    refused: bool = False
    refusal_reason: str = ""
    warnings: List[str] = field(default_factory=list)
    # Set for gallery results, where the runner-up matters.
    margin: Optional[float] = None
    runner_up_name: Optional[str] = None
    runner_up_score: Optional[float] = None

    @property
    def operational_threshold(self) -> float:
        """The band edge a score must clear to read as 'High similarity'."""
        return self.thresholds.comparison_high

    @property
    def conclusive(self) -> bool:
        """False when the audio could not support any assessment."""
        return not self.refused and self.band != BAND_INSUFFICIENT

    def source_lines(self) -> List[str]:
        """The questioned recording's identity, when it was recorded."""
        if not self.questioned_source_filename:
            return []
        lines = [f"Questioned recording: {self.questioned_source_filename}"]
        lines.append(
            f"Source SHA-256: {self.questioned_source_sha256 or 'not recorded'}"
        )
        if self.questioned_selection:
            lines.append(f"Selection: {self.questioned_selection}")
        return lines

    def format_lines(self) -> List[str]:
        """The operator-facing result block - never a percentage or probability."""
        if self.refused:
            return [
                f"Reference subject: {self.reference_name}",
                f"Questioned speaker: {self.questioned_label}",
                *self.source_lines(),
                "",
                "Comparison refused.",
                self.refusal_reason,
            ]
        lines = [
            f"Reference subject: {self.reference_name}",
            f"Questioned speaker: {self.questioned_label}",
            *self.source_lines(),
            "",
            f"Similarity score: {self.score:.2f} / 1.00",
            f"Assessment: {self.band}",
            f"Operational threshold: {self.operational_threshold:.2f}",
        ]
        if self.margin is not None:
            runner = self.runner_up_name or "next best"
            lines.append(
                f"Margin over {runner}: {self.margin:.2f} "
                f"(minimum {self.thresholds.recognition_margin:.2f})"
            )
        measured = (
            f" (measured across {self.questioned_window_count} window(s))"
            if self.questioned_window_count
            else ""
        )
        lines += [
            "",
            f"Reference speech: {self.reference_seconds:.1f} sec",
            f"Questioned speech: {self.questioned_seconds:.1f} sec{measured}",
            "",
            f"Reference quality: {self.reference_quality}",
            f"Questioned quality: {self.questioned_quality}",
            f"Embedding model: {self.embedding_model}",
        ]
        if self.warnings:
            lines.append("")
            lines += [f"Warning: {w}" for w in self.warnings]
        return lines

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reference_name": self.reference_name,
            "reference_subject_id": self.reference_subject_id,
            "questioned_label": self.questioned_label,
            "score": round(self.score, 4),
            "assessment": self.band,
            "operational_threshold": self.operational_threshold,
            "margin": self.margin,
            "runner_up_name": self.runner_up_name,
            "runner_up_score": self.runner_up_score,
            "reference_seconds": round(self.reference_seconds, 2),
            "questioned_seconds": round(self.questioned_seconds, 2),
            "reference_quality": self.reference_quality,
            "questioned_quality": self.questioned_quality,
            "embedding_model": self.embedding_model,
            "questioned_source_filename": self.questioned_source_filename,
            "questioned_source_sha256": self.questioned_source_sha256,
            "questioned_selection": self.questioned_selection,
            "questioned_window_count": self.questioned_window_count,
            "thresholds": self.thresholds.to_dict(),
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "warnings": list(self.warnings),
            "disclaimer": DISCLAIMER,
        }


def _reference_view(profile: SpeakerProfile) -> Tuple[float, str, List[str]]:
    """Total trusted speech, its quality band, and any profile-level warnings."""
    from .quality import QualityReport, combine

    reports: List[QualityReport] = []
    for sample in profile.trusted_samples():
        data = sample.quality or {}
        report = QualityReport(
            duration_seconds=float(data.get("duration_seconds") or 0.0),
            voiced_seconds=float(
                data.get("voiced_seconds") or sample.speech_duration or 0.0
            ),
            rms=float(data.get("rms") or 0.0),
            peak=float(data.get("peak") or 0.0),
            clipping_percent=float(data.get("clipping_percent") or 0.0),
            level_margin_db=float(data.get("level_margin_db") or 0.0),
            assessment=str(data.get("assessment") or INSUFFICIENT),
            warnings=list(data.get("warnings") or []),
        )
        report.voiced_ratio = float(data.get("voiced_ratio") or 0.0)
        reports.append(report)
    if not reports:
        return 0.0, INSUFFICIENT, ["This profile has no trusted reference samples."]
    merged = combine(reports)
    return merged.voiced_seconds, merged.assessment, list(merged.warnings)


def questioned_adequacy(
    *,
    speech_seconds: float,
    quality: str,
    thresholds: Thresholds,
) -> Optional[str]:
    """Why the questioned audio cannot support an assessment, or ``None``.

    The *single* place this is decided. 1:1 comparison and gallery search used
    to gate differently - the gallery checked only duration - so the same clip
    could be "Insufficient data" against one subject and a lead against a
    gallery containing that subject. Both now ask this function.
    """
    if speech_seconds < thresholds.min_questioned_seconds:
        return (
            f"Questioned speech is only {speech_seconds:.1f}s; at least "
            f"{thresholds.min_questioned_seconds:.1f}s is needed for an assessment."
        )
    if quality == INSUFFICIENT:
        return (
            "The questioned audio holds too little usable speech for an "
            "assessment, whatever the score."
        )
    return None


def compare_embedding_to_profile(
    embedding: Sequence[float],
    profile: SpeakerProfile,
    *,
    questioned_seconds: float,
    questioned_quality: str = INSUFFICIENT,
    questioned_warnings: Optional[Sequence[str]] = None,
    questioned_label: str = "Questioned speaker",
    questioned_model: Optional[Any] = None,
    thresholds: Optional[Thresholds] = None,
    allow_unverified_model: bool = False,
) -> ComparisonResult:
    """Compare one questioned embedding against a known subject's reference.

    ``questioned_model`` is the embedding-model identity that produced
    ``embedding`` (normally this build's). When it cannot be shown to match the
    profile's model the comparison is refused, unless the caller passes
    ``allow_unverified_model`` after an operator has accepted the risk - and even
    then only for missing provenance, never for a proven mismatch.
    """
    thresholds = thresholds or active()
    reference_seconds, reference_quality, warnings = _reference_view(profile)
    result = ComparisonResult(
        reference_name=profile.display_name,
        reference_subject_id=profile.subject_id,
        questioned_label=questioned_label,
        questioned_seconds=round(questioned_seconds, 2),
        questioned_quality=questioned_quality,
        reference_seconds=round(reference_seconds, 2),
        reference_quality=reference_quality,
        thresholds=thresholds,
        embedding_model=(
            profile.embedding_model.describe()
            if profile.embedding_model
            else "unknown (legacy profile)"
        ),
        warnings=list(warnings) + list(questioned_warnings or []),
    )

    verdict = check_compatibility(profile.embedding_model, questioned_model)
    result.compatibility = verdict
    if not verdict.ok:
        if not (verdict.needs_confirmation and allow_unverified_model):
            result.refused = True
            result.refusal_reason = verdict.reason
            return result
        result.warnings.append(
            "Model compatibility could not be verified; this comparison was run "
            "at the operator's discretion and the score may be meaningless."
        )

    result.score = cosine_similarity(embedding, profile.centroid())

    # Audio adequacy overrides the number: too little usable speech on either
    # side means no assessment, however high the score looks.
    if not profile.trusted_samples():
        result.band = BAND_INSUFFICIENT
        result.warnings.append("The reference profile has no trusted samples.")
        return result
    inadequate = questioned_adequacy(
        speech_seconds=questioned_seconds,
        quality=questioned_quality,
        thresholds=thresholds,
    )
    if inadequate:
        result.band = BAND_INSUFFICIENT
        result.warnings.append(inadequate)
        return result
    if reference_seconds < thresholds.min_reference_seconds:
        result.band = BAND_INSUFFICIENT
        result.warnings.append(
            f"The reference profile holds only {reference_seconds:.1f}s of speech; "
            f"at least {thresholds.min_reference_seconds:.1f}s is recommended."
        )
        return result
    if reference_quality == INSUFFICIENT:
        result.band = BAND_INSUFFICIENT
        result.warnings.append(
            "The reference profile's audio quality is insufficient for an assessment."
        )
        return result

    result.band, _ = similarity_band(
        result.score,
        thresholds.comparison_high,
        thresholds.comparison_intermediate,
    )
    return result


@dataclass
class GalleryMatch:
    """One subject's score in a gallery search."""

    subject_id: str
    display_name: str
    score: float
    band: str


@dataclass
class GalleryResult:
    """Ranked gallery search plus the open-set decision over it."""

    questioned_label: str = "Questioned speaker"
    matches: List[GalleryMatch] = field(default_factory=list)
    decision: Optional[MatchDecision] = None
    searched: int = 0
    skipped: List[str] = field(default_factory=list)
    thresholds: Thresholds = field(default_factory=active)
    # Set when the questioned audio cannot support any assessment; the ranking
    # is still shown, but no lead may be drawn from it.
    inadequate_reason: str = ""
    questioned_source_filename: Optional[str] = None
    questioned_source_sha256: Optional[str] = None
    questioned_selection: str = ""

    @property
    def accepted_name(self) -> Optional[str]:
        """The subject the search would call a lead - never on inadequate audio."""
        if self.inadequate_reason:
            return None
        if self.decision is not None and self.decision.accepted:
            return self.decision.best_name
        return None

    def summary_lines(self) -> List[str]:
        lines: List[str] = []
        if self.questioned_source_filename:
            lines += [
                f"Questioned recording: {self.questioned_source_filename}",
                f"Source SHA-256: {self.questioned_source_sha256 or 'not recorded'}",
            ]
            if self.questioned_selection:
                lines.append(f"Selection: {self.questioned_selection}")
            lines.append("")
        lines.append(f"Searched {self.searched} known profile(s).")
        for index, match in enumerate(self.matches[:10], 1):
            lines.append(
                f"{index}. {match.display_name}   {match.score:.2f}   {match.band}"
            )
        if not self.matches:
            lines.append("No comparable profiles.")
        lines.append("")
        if self.inadequate_reason:
            # Ranking a gallery on audio too poor to assess would turn an
            # arithmetic ordering into an accusation.
            lines.append(f"{BAND_INSUFFICIENT}: {self.inadequate_reason}")
            lines.append(
                "The ranking above is shown for context only and supports no "
                "conclusion about any subject."
            )
        elif self.accepted_name:
            lines.append(
                f"Lead: the questioned speaker produced high similarity to the "
                f"{self.accepted_name} reference profile. Further review is warranted."
            )
        else:
            lines.append("No known profile produced a sufficiently strong match.")
        if self.decision is not None and self.decision.reason:
            lines.append(self.decision.reason)
        for reason in self.skipped:
            lines.append(f"Skipped: {reason}")
        return lines

    def to_dict(self) -> Dict[str, Any]:
        return {
            "questioned_label": self.questioned_label,
            "searched": self.searched,
            "matches": [
                {
                    "subject_id": m.subject_id,
                    "display_name": m.display_name,
                    "score": round(m.score, 4),
                    "assessment": m.band,
                }
                for m in self.matches
            ],
            "decision": self.decision.to_dict() if self.decision else None,
            "accepted_name": self.accepted_name,
            "inadequate_reason": self.inadequate_reason,
            "questioned_source_filename": self.questioned_source_filename,
            "questioned_source_sha256": self.questioned_source_sha256,
            "questioned_selection": self.questioned_selection,
            "skipped": list(self.skipped),
            "thresholds": self.thresholds.to_dict(),
            "disclaimer": DISCLAIMER,
        }


def search_gallery(
    embedding: Sequence[float],
    profiles: Sequence[SpeakerProfile],
    *,
    questioned_seconds: float,
    questioned_quality: str = INSUFFICIENT,
    questioned_label: str = "Questioned speaker",
    questioned_model: Optional[Any] = None,
    thresholds: Optional[Thresholds] = None,
) -> GalleryResult:
    """Rank a questioned speaker against every known subject.

    Ranking is not identification: the same acceptance-and-margin rule as
    per-turn recognition decides whether the top hit is even a lead, so a gallery
    whose best score is 0.63 against a 0.62 runner-up yields nothing. The
    questioned audio must also be adequate - the same test the 1:1 comparison
    applies, so the two paths cannot disagree about the same clip.
    """
    thresholds = thresholds or active()
    result = GalleryResult(questioned_label=questioned_label, thresholds=thresholds)
    result.inadequate_reason = (
        questioned_adequacy(
            speech_seconds=questioned_seconds,
            quality=questioned_quality,
            thresholds=thresholds,
        )
        or ""
    )
    # Two subjects may share a display name; keying on it would let one stand in
    # for the other in the ranking and the decision.
    labels = display_labels(profiles)
    candidates: List[Tuple[str, List[float]]] = []
    by_label: Dict[str, SpeakerProfile] = {}
    for profile in profiles:
        label = labels.get(profile.subject_id, profile.display_name)
        verdict = check_compatibility(profile.embedding_model, questioned_model)
        if not verdict.ok:
            result.skipped.append(f"{label}: {verdict.reason}")
            continue
        centroid_vector = profile.centroid()
        if not centroid_vector:
            result.skipped.append(f"{label}: no trusted reference samples.")
            continue
        candidates.append((label, centroid_vector))
        by_label[label] = profile
    result.searched = len(candidates)

    if not result.inadequate_reason:
        # On audio too poor to assess, the acceptance test is not run at all.
        # Running it and then withholding the lead would still print its
        # rationale ("Score 0.95 >= 0.62 with margin ...") underneath the
        # refusal, which reads as the tool arguing with itself.
        result.decision = decide_identity(
            embedding,
            candidates,
            acceptance=thresholds.recognition_acceptance,
            margin=thresholds.recognition_margin,
            speech_seconds=questioned_seconds,
            min_speech_seconds=thresholds.min_questioned_seconds,
        )
    for name, score in _ranked(embedding, candidates):
        profile = by_label[name]
        band, _ = similarity_band(
            score, thresholds.comparison_high, thresholds.comparison_intermediate
        )
        result.matches.append(
            GalleryMatch(
                subject_id=profile.subject_id,
                display_name=name,
                score=score,
                band=band,
            )
        )
    return result


def _ranked(
    embedding: Sequence[float], candidates: Sequence[Tuple[str, Sequence[float]]]
) -> List[Tuple[str, float]]:
    from .voiceprints import rank_candidates

    return rank_candidates(embedding, candidates)


def compare_questioned_to_profile(
    questioned: "QuestionedSpeaker",
    profile: SpeakerProfile,
    *,
    questioned_model: Optional[Any] = None,
    thresholds: Optional[Thresholds] = None,
    allow_unverified_model: bool = False,
) -> ComparisonResult:
    """Compare a measured questioned speaker, carrying its provenance across.

    Preferred over :func:`compare_embedding_to_profile` wherever a
    :class:`~whispr.questioned.QuestionedSpeaker` is in hand: the duration,
    quality and warnings come from the audio that produced the embedding, and
    the result names the recording it measured, so a report cannot end up
    describing a different file.
    """
    result = compare_embedding_to_profile(
        questioned.embedding,
        profile,
        questioned_seconds=questioned.speech_seconds,
        questioned_quality=questioned.quality,
        questioned_warnings=questioned.warnings,
        questioned_label=questioned.label,
        questioned_model=questioned_model,
        thresholds=thresholds,
        allow_unverified_model=allow_unverified_model,
    )
    result.questioned_source_filename = questioned.source_filename or None
    result.questioned_source_sha256 = questioned.source_sha256
    result.questioned_selection = questioned.selection_mode
    # The number of *embeddings*, not of source intervals: joined turns make one
    # window map back to several intervals, so counting intervals would report
    # seven windows for two embeddings.
    result.questioned_window_count = questioned.window_count
    return result


def search_gallery_for_questioned(
    questioned: "QuestionedSpeaker",
    profiles: Sequence[SpeakerProfile],
    *,
    questioned_model: Optional[Any] = None,
    thresholds: Optional[Thresholds] = None,
) -> GalleryResult:
    """Gallery search for a measured questioned speaker, provenance included."""
    result = search_gallery(
        questioned.embedding,
        profiles,
        questioned_seconds=questioned.speech_seconds,
        questioned_quality=questioned.quality,
        questioned_label=questioned.label,
        questioned_model=questioned_model,
        thresholds=thresholds,
    )
    result.questioned_source_filename = questioned.source_filename or None
    result.questioned_source_sha256 = questioned.source_sha256
    result.questioned_selection = questioned.selection_mode
    return result
