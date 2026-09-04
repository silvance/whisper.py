import wave

import numpy as np

from whispr import quality
from whispr.quality import (
    FAIR,
    GOOD,
    INSUFFICIENT,
    POOR,
    QualityReport,
    analyse_samples,
    analyse_span,
    combine,
    read_wav_span,
)

RATE = 16000
rng = np.random.default_rng(1234)


def _speechlike(seconds, level=0.2, voiced_fraction=0.7):
    """Loud bursts separated by quiet gaps - stands in for speech vs pauses."""
    total = int(seconds * RATE)
    signal = rng.normal(0, 0.0005, total).astype(np.float32)  # quiet noise floor
    burst = int(0.3 * RATE)
    gap = int(burst * (1 - voiced_fraction) / max(voiced_fraction, 0.01))
    pos = 0
    while pos + burst < total:
        signal[pos : pos + burst] += rng.normal(0, level, burst).astype(np.float32)
        pos += burst + max(gap, 1)
    return np.clip(signal, -1.0, 1.0)


def _write_wav(path, samples, rate=RATE):
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm.tobytes())


def test_silence_is_insufficient():
    report = analyse_samples(np.zeros(RATE * 20, dtype=np.float32), RATE)
    assert report.assessment == INSUFFICIENT
    assert report.usable is False
    assert any("usable speech" in w for w in report.warnings)


def test_empty_span_is_insufficient():
    report = analyse_samples(np.zeros(0, dtype=np.float32), RATE)
    assert report.assessment == INSUFFICIENT
    assert report.duration_seconds == 0.0


def test_short_speech_is_insufficient():
    report = analyse_samples(_speechlike(2.0), RATE)
    assert report.assessment == INSUFFICIENT
    assert report.voiced_seconds < quality.MIN_USABLE_SECONDS


def test_long_clean_speech_is_good():
    report = analyse_samples(_speechlike(30.0), RATE)
    assert report.assessment == GOOD
    assert report.voiced_seconds >= quality.GOOD_MIN_SECONDS
    assert report.warnings == []


def test_medium_speech_is_fair():
    report = analyse_samples(_speechlike(9.0), RATE)
    assert report.assessment in (FAIR, GOOD)
    assert report.voiced_seconds >= quality.MIN_USABLE_SECONDS


def test_clipping_is_detected_and_downgrades():
    samples = _speechlike(30.0, level=0.9)
    samples[: int(RATE * 5)] = 1.0  # hard clip a stretch
    report = analyse_samples(samples, RATE)
    assert report.clipping_percent > quality.HIGH_CLIPPING_PCT
    assert report.assessment == POOR
    assert any("clipped" in w for w in report.warnings)


def test_very_quiet_audio_is_flagged():
    report = analyse_samples(_speechlike(30.0, level=0.002), RATE)
    assert report.rms < quality.QUIET_RMS
    assert any("very quiet" in w for w in report.warnings)
    assert report.assessment == POOR


def test_mostly_silence_is_flagged():
    report = analyse_samples(_speechlike(40.0, voiced_fraction=0.15), RATE)
    assert report.voiced_ratio < quality.LOW_VOICED_RATIO
    assert any("contains speech" in w for w in report.warnings)


def test_metrics_round_trip_to_dict():
    report = analyse_samples(_speechlike(30.0), RATE)
    data = report.to_dict()
    assert data["assessment"] == GOOD
    assert data["voiced_seconds"] > 0
    assert "level_margin_db" in data


def test_read_and_analyse_span_from_file(tmp_path):
    path = tmp_path / "clip.wav"
    _write_wav(path, _speechlike(30.0))
    samples, rate = read_wav_span(path)
    assert rate == RATE and len(samples) == RATE * 30

    # A 5-second slice reads back as 5 seconds.
    partial, _ = read_wav_span(path, 1.0, 6.0)
    assert abs(len(partial) - RATE * 5) <= 1
    assert analyse_span(path).assessment == GOOD


def test_combine_sums_duration_and_carries_warnings():
    good = analyse_samples(_speechlike(30.0), RATE)
    quiet = analyse_samples(_speechlike(4.0, level=0.002), RATE)
    merged = combine([good, quiet])
    assert merged.voiced_seconds > good.voiced_seconds
    # A minority weak clip does not condemn an otherwise strong profile...
    assert merged.assessment == GOOD
    # ...but its problem is still reported.
    assert any("very quiet" in w for w in merged.warnings)


def test_combine_downgrades_when_most_speech_is_poor():
    good = analyse_samples(_speechlike(6.0), RATE)
    quiet = analyse_samples(_speechlike(40.0, level=0.002), RATE)
    merged = combine([good, quiet])
    assert merged.assessment == POOR
    assert any("poor-quality" in w for w in merged.warnings)


def test_combine_of_nothing_is_insufficient():
    merged = combine([])
    assert merged.assessment == INSUFFICIENT
    assert isinstance(merged, QualityReport)
