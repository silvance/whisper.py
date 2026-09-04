"""Operational thresholds for diarization, speaker recognition and comparison.

These are deliberately *separate* numbers for separate systems. An earlier
version reused a single ``0.5`` for clustering, recognition and 1:1 comparison,
which conflated three unrelated decisions.

None of these values are calibrated against operational recordings yet. They are
conservative starting points; update them from the validation harness
(:mod:`whispr.validation`) run over a representative corpus, and record the
active values in exported reports so a result can be interpreted later.

Operators can override them in the settings file (see :func:`active`), and the
GUI shows which values are in force and where they came from - but they are not
exposed as casual GUI controls, because changing one silently changes how every
subsequent result should be read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# --- Diarization ----------------------------------------------------------
# sherpa clustering cut-off: how eagerly turns merge into one speaker. Purely a
# clustering knob - unrelated to whether a voice matches a known person.
DIARIZATION_CLUSTERING_THRESHOLD = 0.5

# --- Known-speaker recognition (open set) ---------------------------------
# A turn is attributed to a known subject only when its similarity clears this
# bar AND beats the runner-up by MARGIN. Otherwise it stays UNKNOWN: with an
# open set (the speaker may be nobody we have enrolled) guessing is worse than
# abstaining.
RECOGNITION_ACCEPTANCE_THRESHOLD = 0.62
RECOGNITION_MARGIN_THRESHOLD = 0.08

# --- 1:1 speaker comparison ----------------------------------------------
# Band edges for the comparison assessment. These describe *similarity*, not a
# probability of identity, and never assert that two recordings are the same
# person.
COMPARISON_HIGH_BAND = 0.62
COMPARISON_INTERMEDIATE_BAND = 0.45

# Below these durations a score is not meaningful and the assessment is reported
# as "Insufficient data" regardless of the number.
MIN_QUESTIONED_SPEECH_SECONDS = 3.0
MIN_REFERENCE_SPEECH_SECONDS = 10.0

# A learned (auto-enrolled) sample this far below the trusted centroid is treated
# as a probable mistaken correction and flagged for review instead of used.
LEARNED_OUTLIER_THRESHOLD = 0.45

# Assessment labels. Deliberately similarity language - never identity language.
BAND_HIGH = "High similarity"
BAND_INTERMEDIATE = "Intermediate similarity"
BAND_LOW = "Low similarity"
BAND_INSUFFICIENT = "Insufficient data"

DISCLAIMER = (
    "Speaker similarity results produced by Whispers are investigative "
    "indicators intended to support lead development and analyst review. They "
    "are not forensic speaker identification, are not a biometric probability "
    "of identity, and should not be treated as proof that two recordings "
    "contain the same person."
)


@dataclass(frozen=True)
class Thresholds:
    """The active threshold set, recorded alongside every result."""

    recognition_acceptance: float = RECOGNITION_ACCEPTANCE_THRESHOLD
    recognition_margin: float = RECOGNITION_MARGIN_THRESHOLD
    comparison_high: float = COMPARISON_HIGH_BAND
    comparison_intermediate: float = COMPARISON_INTERMEDIATE_BAND
    min_questioned_seconds: float = MIN_QUESTIONED_SPEECH_SECONDS
    min_reference_seconds: float = MIN_REFERENCE_SPEECH_SECONDS
    diarization_clustering: float = DIARIZATION_CLUSTERING_THRESHOLD

    def to_dict(self) -> Dict[str, float]:
        return {
            "recognition_acceptance": self.recognition_acceptance,
            "recognition_margin": self.recognition_margin,
            "comparison_high": self.comparison_high,
            "comparison_intermediate": self.comparison_intermediate,
            "min_questioned_seconds": self.min_questioned_seconds,
            "min_reference_seconds": self.min_reference_seconds,
            "diarization_clustering": self.diarization_clustering,
        }


DEFAULTS = Thresholds()

# Settings key under which overrides live (advanced; not a casual GUI control).
SETTINGS_KEY = "thresholds"

_FIELDS: Tuple[str, ...] = tuple(DEFAULTS.to_dict())


def from_settings(settings: Dict[str, Any]) -> Thresholds:
    """Build a :class:`Thresholds` from a settings dict, ignoring bad values.

    An operator can hand-edit ``settings.json`` to retune after validation; a
    malformed or out-of-range entry falls back to the default rather than
    silently skewing every subsequent assessment.
    """
    raw = settings.get(SETTINGS_KEY)
    if not isinstance(raw, dict):
        return DEFAULTS
    values: Dict[str, float] = {}
    for field_name in _FIELDS:
        candidate = raw.get(field_name)
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            values[field_name] = float(candidate)
    if not values:
        return DEFAULTS
    return Thresholds(**{**DEFAULTS.to_dict(), **values})  # type: ignore[arg-type]


_active: Optional[Thresholds] = None


def active(refresh: bool = False) -> Thresholds:
    """The threshold set actually in force for this run.

    Reads the settings file once and caches it, so every decision in a session
    uses one consistent set and a mid-run edit cannot change the meaning of
    results already produced. Pass ``refresh=True`` to re-read (the GUI does this
    when it displays them).
    """
    global _active
    if _active is None or refresh:
        from .settings import load_settings

        _active = from_settings(load_settings())
    return _active


def overrides(
    thresholds: Optional[Thresholds] = None,
) -> Dict[str, Tuple[float, float]]:
    """``{field: (configured, shipped_default)}`` for every overridden value."""
    current = (active() if thresholds is None else thresholds).to_dict()
    shipped = DEFAULTS.to_dict()
    return {
        name: (value, shipped[name])
        for name, value in current.items()
        if value != shipped[name]
    }


def describe(thresholds: Thresholds = DEFAULTS) -> List[str]:
    """Human-readable lines describing the active thresholds (for reports)."""
    return [
        f"Recognition acceptance: {thresholds.recognition_acceptance:.2f}",
        f"Recognition margin: {thresholds.recognition_margin:.2f}",
        f"Comparison 'High similarity' at/above: {thresholds.comparison_high:.2f}",
        (
            "Comparison 'Intermediate similarity' at/above: "
            f"{thresholds.comparison_intermediate:.2f}"
        ),
        (f"Minimum questioned speech: {thresholds.min_questioned_seconds:.1f}s"),
        f"Minimum reference speech: {thresholds.min_reference_seconds:.1f}s",
    ]


def describe_active() -> List[str]:
    """Lines for the GUI/self-test: the values in force and where they came from.

    Deliberately read-only. These numbers are not calibrated for any particular
    operational setting, and retuning them without validation changes how every
    result should be read, so they are shown rather than offered as a control.
    """
    from .settings import settings_path

    current = active(refresh=True)
    changed = overrides(current)
    lines = ["Active thresholds", "-----------------"]
    lines += describe(current)
    lines.append("")
    if changed:
        lines.append("Overridden from the shipped defaults:")
        for name, (value, default) in sorted(changed.items()):
            lines.append(f"  {name}: {value:g} (default {default:g})")
    else:
        lines.append("All values are the shipped defaults.")
    lines.append(f"Configured in: {settings_path()} (key '{SETTINGS_KEY}')")
    lines.append("")
    lines += _wrap(
        "These thresholds are conservative starting points, not values "
        "calibrated against operational recordings. Change them only from a "
        "validation run over a representative corpus (python -m whispr.validation), "
        "and note that a change alters how every subsequent result should be "
        "read. The values in force are recorded in exported reports."
    )
    return lines


def _wrap(text: str, width: int = 76) -> List[str]:
    import textwrap

    return textwrap.wrap(text, width=width)
