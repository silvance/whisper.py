"""The decision thresholds: separation, configuration and how they are shown."""

import pytest

from whispr import thresholds
from whispr.thresholds import DEFAULTS, SETTINGS_KEY, Thresholds, from_settings


@pytest.fixture(autouse=True)
def _clear_cache():
    """The active set is cached for the session; don't leak it between tests."""
    thresholds._active = None
    yield
    thresholds._active = None


def test_clustering_recognition_and_comparison_are_separate_numbers():
    # An earlier version reused one 0.5 for all three; conflating them again
    # would make a clustering tweak silently change who gets named.
    assert (
        thresholds.DIARIZATION_CLUSTERING_THRESHOLD
        != thresholds.RECOGNITION_ACCEPTANCE_THRESHOLD
    )
    assert thresholds.RECOGNITION_MARGIN_THRESHOLD > 0
    assert thresholds.COMPARISON_HIGH_BAND > thresholds.COMPARISON_INTERMEDIATE_BAND


def test_settings_override_a_single_value_and_keep_the_rest(monkeypatch):
    monkeypatch.setattr(
        thresholds,
        "from_settings",
        from_settings,  # keep the real implementation
    )
    configured = from_settings({SETTINGS_KEY: {"recognition_acceptance": 0.71}})
    assert configured.recognition_acceptance == 0.71
    assert configured.recognition_margin == DEFAULTS.recognition_margin


def test_malformed_overrides_fall_back_rather_than_skewing_results():
    for bad in (
        {},
        {SETTINGS_KEY: "0.7"},
        {SETTINGS_KEY: {"recognition_acceptance": "x"}},
    ):
        assert from_settings(bad) == DEFAULTS
    # A bool is not a threshold, even though it is an int in Python.
    assert from_settings({SETTINGS_KEY: {"recognition_margin": True}}) == DEFAULTS


def test_unknown_keys_are_ignored():
    assert from_settings({SETTINGS_KEY: {"not_a_threshold": 0.9}}) == DEFAULTS


def _configure(monkeypatch, payload):
    monkeypatch.setattr(thresholds, "_active", None)
    import whispr.settings as settings

    monkeypatch.setattr(settings, "load_settings", lambda: payload)


def test_active_reads_the_settings_file_and_then_caches_it(monkeypatch):
    calls = []

    import whispr.settings as settings

    def _load():
        calls.append(1)
        return {SETTINGS_KEY: {"recognition_acceptance": 0.8}}

    monkeypatch.setattr(settings, "load_settings", _load)
    assert thresholds.active().recognition_acceptance == 0.8
    thresholds.active()
    # One read per session: a mid-run edit must not change the meaning of
    # results already produced.
    assert len(calls) == 1
    assert thresholds.active(refresh=True).recognition_acceptance == 0.8
    assert len(calls) == 2


def test_overrides_report_configured_against_shipped(monkeypatch):
    _configure(monkeypatch, {SETTINGS_KEY: {"comparison_high": 0.7}})
    changed = thresholds.overrides(thresholds.active(refresh=True))
    assert changed == {"comparison_high": (0.7, DEFAULTS.comparison_high)}
    assert thresholds.overrides(DEFAULTS) == {}


def test_describe_active_states_the_values_the_source_and_the_caveat(monkeypatch):
    _configure(monkeypatch, {SETTINGS_KEY: {"recognition_acceptance": 0.77}})
    text = "\n".join(thresholds.describe_active())
    assert "0.77" in text
    assert "Overridden from the shipped defaults" in text
    assert "recognition_acceptance: 0.77 (default 0.62)" in text
    # It has to say where the numbers live and that they need validating.
    assert "settings.json" in text
    assert "not values calibrated" in text


def test_describe_active_says_so_when_nothing_is_overridden(monkeypatch):
    _configure(monkeypatch, {})
    text = "\n".join(thresholds.describe_active())
    assert "All values are the shipped defaults." in text


def test_describe_reports_similarity_bands_not_identity_claims():
    text = " ".join(thresholds.describe(Thresholds())).lower()
    for forbidden in ("probability", "identity", "same person", "confidence"):
        assert forbidden not in text


def test_disclaimer_denies_the_claims_it_names():
    text = thresholds.DISCLAIMER
    assert "not forensic speaker identification" in text
    assert "not a biometric probability" in text
    assert "should not be treated as proof" in text
