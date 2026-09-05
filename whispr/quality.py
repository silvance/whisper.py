"""Transparent audio-quality metrics for speaker enrolment and comparison.

Speaker similarity degrades badly on short, clipped, very quiet or mostly-silent
audio, and a score computed on such audio should never be presented as a
confident assessment. These metrics are deliberately simple and each maps to a
stated rule, so an operator can see *why* a span was judged poor. Nothing here
is a calibrated or perceptual score.

The measurements run on a 16 kHz mono PCM WAV (what ``convert_to_wav`` and the
diarizer produce) and need only numpy.
"""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

PathLike = Union[str, Path]

# Frame size for the energy analysis: 30 ms is long enough to be stable and
# short enough to resolve pauses between words.
_FRAME_SECONDS = 0.03
# A frame counts as speech when its RMS is this many times the recording's own
# noise floor (an adaptive, level-independent rule).
_VOICED_FACTOR = 3.0
# Absolute floor so digital silence is never counted as speech.
_ABSOLUTE_FLOOR = 1e-4
# Below this spread between the quiet and loud parts of a span, the span holds no
# real silence to estimate a noise floor from - see _voiced_threshold.
_FLAT_MARGIN_DB = 12.0
# The fallback threshold for such a span, relative to its own speech level: a
# frame 20 dB below the loud parts is a quiet syllable, not the room.
_FLAT_RELATIVE = 0.1
# |sample| at or above this is treated as clipped (16-bit full scale is 1.0).
_CLIP_LEVEL = 0.99

# Assessment bands.
GOOD = "Good"
FAIR = "Fair"
POOR = "Poor"
INSUFFICIENT = "Insufficient"

# Rule cut-offs, stated here so the report can explain the verdict.
MIN_USABLE_SECONDS = 3.0
# Totals, applied to a whole profile by :func:`combine`. They are NOT applied to
# a single span: enrolment measures fixed 8-second windows, so an absolute
# 10-second bar would put "Good" out of reach of every window there is, and a
# 5-second bar would call any window with normal pauses in it "Poor". A span is
# judged on the proportion of it that is speech instead.
GOOD_MIN_SECONDS = 10.0
FAIR_MIN_SECONDS = 5.0
LOW_VOICED_RATIO = 0.35
GOOD_VOICED_RATIO = 0.6
HIGH_CLIPPING_PCT = 1.0
QUIET_RMS = 0.01


@dataclass
class QualityReport:
    """What was measured on a span, the verdict, and why."""

    duration_seconds: float = 0.0
    voiced_seconds: float = 0.0
    voiced_ratio: float = 0.0
    rms: float = 0.0
    peak: float = 0.0
    clipping_percent: float = 0.0
    # Crude level spread: loud frames vs the noise floor, in dB. This is an
    # indicator of how far speech sits above the background - NOT a calibrated
    # signal-to-noise ratio.
    level_margin_db: float = 0.0
    # True when the span has no quiet part to measure a noise floor from, so the
    # speech/silence split had to be estimated from the speech level itself.
    level_spread_low: bool = False
    assessment: str = INSUFFICIENT
    warnings: List[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.assessment != INSUFFICIENT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration_seconds": round(self.duration_seconds, 2),
            "voiced_seconds": round(self.voiced_seconds, 2),
            "voiced_ratio": round(self.voiced_ratio, 3),
            "rms": round(self.rms, 5),
            "peak": round(self.peak, 4),
            "clipping_percent": round(self.clipping_percent, 3),
            "level_margin_db": round(self.level_margin_db, 1),
            "level_spread_low": self.level_spread_low,
            "assessment": self.assessment,
            "warnings": list(self.warnings),
        }


def _frames(samples: Any, sample_rate: int) -> Any:
    """Reshape a waveform into non-overlapping analysis frames."""
    import numpy as np

    size = max(1, int(_FRAME_SECONDS * sample_rate))
    usable = (len(samples) // size) * size
    if usable <= 0:
        return np.zeros((0, size), dtype="float32")
    return samples[:usable].reshape(-1, size)


def analyse_samples(samples: Any, sample_rate: int) -> QualityReport:
    """Measure a mono float waveform and classify its usability."""
    import numpy as np

    report = QualityReport()
    if sample_rate <= 0 or samples is None or len(samples) == 0:
        report.warnings.append("No audio in the selected span.")
        return report

    report.duration_seconds = len(samples) / float(sample_rate)
    absolute = np.abs(samples)
    report.peak = float(absolute.max())
    report.rms = float(np.sqrt(np.mean(np.square(samples))))
    report.clipping_percent = float((absolute >= _CLIP_LEVEL).mean() * 100.0)

    frames = _frames(samples, sample_rate)
    if len(frames) == 0:
        report.warnings.append("Span is shorter than one analysis frame.")
        return report
    frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))
    # Noise floor: the 10th percentile frame, i.e. the quiet parts of this very
    # recording, so the rule adapts to level rather than assuming one.
    noise_floor = float(np.percentile(frame_rms, 10))
    speech_level = float(np.percentile(frame_rms, 90))
    if noise_floor > 0 and speech_level > 0:
        report.level_margin_db = 20.0 * math.log10(
            max(speech_level, 1e-9) / max(noise_floor, 1e-9)
        )

    threshold, report.level_spread_low = _voiced_threshold(
        noise_floor, speech_level, report.level_margin_db
    )
    voiced = frame_rms >= threshold
    frame_seconds = frames.shape[1] / float(sample_rate)
    report.voiced_seconds = float(voiced.sum()) * frame_seconds
    report.voiced_ratio = float(voiced.mean())

    _classify(report)
    return report


def _voiced_threshold(
    noise_floor: float, speech_level: float, margin_db: float
) -> "Tuple[float, bool]":
    """The frame level above which audio counts as speech, and whether it is a guess.

    The adaptive rule - three times the quietest tenth of the span - assumes that
    quietest tenth is *silence*. In a span of continuous speech it is not: it is
    the quietest syllables, and demanding three times their level rules most of
    the speech out. A person talking steadily into a microphone, the cleanest
    audio there is, would be measured as almost no speech at all and thrown away.

    So when the span shows too little spread between its quiet and loud parts to
    hold any silence, the level is taken from the speech instead - anything
    within 20 dB of the loud parts counts - and the report says the split was
    estimated, because this cannot tell continuous speech from continuous noise.
    """
    if noise_floor > 0 and margin_db < _FLAT_MARGIN_DB:
        return max(speech_level * _FLAT_RELATIVE, _ABSOLUTE_FLOOR), True
    return max(noise_floor * _VOICED_FACTOR, _ABSOLUTE_FLOOR), False


def _classify(report: QualityReport) -> None:
    """Apply the stated rules to set the assessment and warnings."""
    if report.clipping_percent >= HIGH_CLIPPING_PCT:
        report.warnings.append(
            f"{report.clipping_percent:.1f}% of samples are clipped; distortion "
            "reduces speaker-comparison reliability."
        )
    if report.rms < QUIET_RMS:
        report.warnings.append(
            "Audio is very quiet, which makes the voice measurement less reliable."
        )
    if report.voiced_ratio < LOW_VOICED_RATIO:
        report.warnings.append(
            f"Only {report.voiced_ratio * 100:.0f}% of the span contains speech; "
            "the rest is silence or noise."
        )
    if report.level_spread_low:
        report.warnings.append(
            "The span has no quiet passage to measure a background level from, "
            "so how much of it is speech is an estimate."
        )

    if report.voiced_seconds < MIN_USABLE_SECONDS:
        report.assessment = INSUFFICIENT
        report.warnings.append(
            f"Only {report.voiced_seconds:.1f}s of usable speech "
            f"(at least {MIN_USABLE_SECONDS:.0f}s is needed for a meaningful "
            "comparison)."
        )
        return

    poor = (
        report.clipping_percent >= HIGH_CLIPPING_PCT
        or report.rms < QUIET_RMS
        or report.voiced_ratio < LOW_VOICED_RATIO
    )
    if poor:
        report.assessment = POOR
    elif report.voiced_ratio >= GOOD_VOICED_RATIO and not report.level_spread_low:
        # Judged on the share of the span that is speech, not on an absolute
        # duration: the span is whatever length the caller chose to measure, and
        # a fixed bar would grade the window rather than the audio in it. A span
        # whose speech/silence split had to be estimated is held at Fair.
        report.assessment = GOOD
    else:
        report.assessment = FAIR


def read_wav_span(
    wav_path: PathLike, start: Optional[float] = None, end: Optional[float] = None
) -> "Tuple[Any, int]":
    """Read ``[start, end]`` of a 16-bit PCM WAV as mono float32 plus its rate.

    ``start``/``end`` default to the whole file. Shared by the quality metrics
    and enrolment so both see exactly the same audio.
    """
    import numpy as np

    with wave.open(str(wav_path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        width = wav.getsampwidth()
        total = wav.getnframes()
        if width != 2:
            raise RuntimeError("expected a 16-bit PCM WAV")
        first = 0 if start is None else max(0, int(start * sample_rate))
        last = total if end is None else min(total, int(math.ceil(end * sample_rate)))
        if last <= first:
            return np.zeros(0, dtype=np.float32), sample_rate
        wav.setpos(first)
        raw = wav.readframes(last - first)

    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, sample_rate


def analyse_span(
    wav_path: PathLike, start: Optional[float] = None, end: Optional[float] = None
) -> QualityReport:
    """Measure a span of a WAV file on disk."""
    samples, sample_rate = read_wav_span(wav_path, start, end)
    return analyse_samples(samples, sample_rate)


def combine(reports: List[QualityReport]) -> QualityReport:
    """Aggregate per-sample reports into one profile-level view.

    Durations add up and the band follows the *total* usable speech, because a
    reference profile assembled from several short clips can legitimately be
    strong. It is downgraded to ``Poor`` when most of that speech came from
    poor-quality material, so a few good clips cannot hide a mostly-bad profile.
    Every contributing warning is carried through either way.
    """
    combined = QualityReport()
    if not reports:
        combined.warnings.append("No audio has been enrolled yet.")
        return combined
    combined.duration_seconds = sum(r.duration_seconds for r in reports)
    combined.voiced_seconds = sum(r.voiced_seconds for r in reports)
    combined.voiced_ratio = (
        combined.voiced_seconds / combined.duration_seconds
        if combined.duration_seconds
        else 0.0
    )
    combined.rms = max(r.rms for r in reports)
    combined.peak = max(r.peak for r in reports)
    combined.clipping_percent = max(r.clipping_percent for r in reports)
    combined.level_margin_db = min(r.level_margin_db for r in reports)
    seen: List[str] = []
    for report in reports:
        for warning in report.warnings:
            if warning not in seen:
                seen.append(warning)
    combined.warnings = seen

    poor_seconds = sum(
        r.voiced_seconds for r in reports if r.assessment in (POOR, INSUFFICIENT)
    )
    if combined.voiced_seconds < MIN_USABLE_SECONDS:
        combined.assessment = INSUFFICIENT
        combined.warnings.append(
            f"Only {combined.voiced_seconds:.1f}s of usable speech in total "
            f"(at least {MIN_USABLE_SECONDS:.0f}s is needed)."
        )
    elif poor_seconds > combined.voiced_seconds / 2:
        combined.assessment = POOR
        combined.warnings.append(
            "Most of the measured speech came from poor-quality audio."
        )
    elif combined.voiced_seconds >= GOOD_MIN_SECONDS:
        combined.assessment = GOOD
    else:
        combined.assessment = FAIR
    return combined
