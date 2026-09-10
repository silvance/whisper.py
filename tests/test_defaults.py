"""What a first run does before anyone touches a setting.

The operators this is built for open the application, choose a recording and
press the button. Whatever these defaults are is, in practice, what the tool
does - so they are pinned here rather than left to drift.
"""

from whispr import diarization, resources
from whispr.transcription import DEFAULT_MODEL, MODEL_PREFERENCE, MODEL_SIZES


def test_the_default_model_is_the_thorough_one():
    """Covert recordings are muffled and distant; the default has to cope."""
    assert DEFAULT_MODEL == "medium"
    assert MODEL_PREFERENCE[0] == DEFAULT_MODEL


def test_the_default_model_is_multilingual():
    """The .en models cannot transcribe the languages this build translates."""
    assert not DEFAULT_MODEL.endswith(".en")


def test_every_fallback_is_a_real_model_the_bundler_can_fetch():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "packaging" / "fetch_assets.py"
    spec = importlib.util.spec_from_file_location("fetch_assets", path)
    fetch_assets = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fetch_assets)
    for name in MODEL_PREFERENCE:
        assert name in MODEL_SIZES, f"{name} is not offered in the application"
        assert name in fetch_assets.MODEL_REPOS, f"{name} cannot be bundled"


def test_the_fallback_order_runs_from_most_to_least_capable():
    assert list(MODEL_PREFERENCE) == ["medium", "small", "base.en", "base"]


# -- speaker separation ----------------------------------------------------


def test_speaker_separation_is_on_when_the_build_can_do_it(monkeypatch):
    """Almost every recording has more than one voice in it, and who said what
    is half the answer - an operator should not have to ask for it."""
    monkeypatch.setattr(diarization, "_pyannote_available", lambda: True)
    monkeypatch.setattr(diarization, "_module_available", lambda name: True)
    monkeypatch.setattr(resources, "pyannote_cache_dir", lambda: "/somewhere")
    assert diarization.is_available() is True


def test_speaker_separation_is_off_when_no_engine_is_bundled(monkeypatch):
    """A transcribe-only build must not start out asking for something it was
    never given: the first run would fail on a box the operator did not tick."""
    monkeypatch.setattr(diarization, "_pyannote_available", lambda: False)
    monkeypatch.setattr(diarization, "_module_available", lambda name: False)
    assert diarization.is_available() is False


def test_a_library_without_its_models_is_not_a_diarizer(monkeypatch):
    monkeypatch.setattr(diarization, "_pyannote_available", lambda: True)
    monkeypatch.setattr(diarization, "_module_available", lambda name: True)
    monkeypatch.setattr(resources, "pyannote_cache_dir", lambda: None)
    monkeypatch.setattr(resources, "bundled_diarization_models", lambda: None)
    assert diarization.available_backends() == []


def test_pyannote_is_preferred_over_sherpa_when_both_are_ready(monkeypatch):
    """pyannote is the one that holds up on hard audio, which is the material."""
    monkeypatch.setattr(diarization, "_pyannote_available", lambda: True)
    monkeypatch.setattr(diarization, "_module_available", lambda name: True)
    monkeypatch.setattr(resources, "pyannote_cache_dir", lambda: "/a")
    monkeypatch.setattr(resources, "bundled_diarization_models", lambda: ("/a", "/b"))
    assert diarization.available_backends() == ["pyannote", "sherpa"]


# -- the release bundle ----------------------------------------------------


def test_the_default_bundle_carries_the_default_model():
    """A build whose default model is not in it starts on a fallback instead."""
    import re
    from pathlib import Path

    workflow = (
        Path(__file__).resolve().parent.parent
        / ".github"
        / "workflows"
        / "release-bundles.yml"
    ).read_text(encoding="utf-8")
    match = re.search(r'default: "(?P<models>[^"]*base\.en[^"]*)"', workflow)
    assert match, "could not find the models input default"
    bundled = [name.strip() for name in match.group("models").split(",")]
    assert DEFAULT_MODEL in bundled
