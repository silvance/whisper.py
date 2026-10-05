"""Noise reduction, and the things it must not be allowed to claim.

An operator asked whether a conversation in a bar can be isolated. The answer
is in whispr/enhance.py and in the numbers behind it: the bundled denoiser is
good against steady noise and useless-to-harmful against a room full of other
voices, which is the case people want it for. What these hold shut is not the
model's behaviour - that is the model's - but the application's claims about
it, and the paths cleaned audio must never reach.
"""

from pathlib import Path

import pytest

from whispr import enhance


def test_a_cleaned_copy_is_named_so_it_can_be_recognised(tmp_path):
    out = enhance.cleaned_path(tmp_path / "carpark.m4a")
    assert out.name == "carpark.m4a" + enhance.SUFFIX
    assert out.parent == tmp_path


def test_a_cleaned_copy_can_be_written_somewhere_else(tmp_path):
    out = enhance.cleaned_path(tmp_path / "a" / "carpark.m4a", tmp_path / "out")
    assert out.parent == tmp_path / "out"


def test_the_guard_recognises_what_this_application_produced():
    assert enhance.looks_cleaned("carpark.m4a" + enhance.SUFFIX)
    assert enhance.looks_cleaned(Path("/case/interview.wav" + enhance.SUFFIX))


def test_the_guard_leaves_ordinary_recordings_alone():
    for name in ("carpark.m4a", "interview.wav", "cleaned.wav", "a.cleaned.mp3"):
        assert not enhance.looks_cleaned(name)


# -- What it is allowed to say ---------------------------------------------
#
# The first metric tried here was this project's own level-margin figure,
# before and after. On mixtures with known answers it reported +35 dB on hiss
# and +4 dB on babble - where the ground truth says cleaning made the babble
# recording 1.7 dB *worse*. It was measuring how hard the denoiser gated the
# quiet parts, which it always does, not whether speech got clearer. A number
# that says "this helped" about a recording it has just damaged is worse than
# no number, so the report carries facts and no verdict.


def test_the_report_does_not_claim_the_recording_got_better():
    report = enhance.EnhancementReport(removed_fraction=0.55)
    fields = report.to_dict()
    for forbidden in ("improved", "better", "snr", "quality", "margin", "helped"):
        assert not any(forbidden in key for key in fields), forbidden


def test_the_summary_states_what_was_done_and_warns_about_crowds():
    line = enhance.EnhancementReport(removed_fraction=0.55).summary()
    assert "55%" in line
    assert "other people talking" in line
    for forbidden in ("improved", "clearer", "better"):
        assert forbidden not in line.lower()


def test_how_much_was_removed_is_a_share_of_what_went_in():
    np = pytest.importorskip("numpy")
    before = np.array([1.0, -1.0, 1.0, -1.0], dtype="float32")
    assert enhance._removed_fraction(before, before) == 0.0
    assert enhance._removed_fraction(before, np.zeros(4, dtype="float32")) == 1.0
    half = enhance._removed_fraction(before, before * 0.5)
    assert 0.2 < half < 0.3


def test_silence_in_is_not_a_division_by_zero():
    np = pytest.importorskip("numpy")
    zeros = np.zeros(8, dtype="float32")
    assert enhance._removed_fraction(zeros, zeros) == 0.0
    assert enhance._removed_fraction(np.array([], dtype="float32"), zeros) == 0.0


# -- What the operator is told ---------------------------------------------


def test_the_guidance_names_the_case_it_does_not_solve():
    """The question that prompted this feature must be answered where it is offered."""
    text = f"{enhance.HELPS_WITH} {enhance.DOES_NOT}".lower()
    assert "bar" in text or "crowd" in text
    assert "does not separate" in text
    assert "worse" in text


def test_the_guidance_keeps_cleaned_audio_away_from_comparison():
    assert "comparison" in enhance.NOT_FOR_COMPARISON.lower()
    assert "profile" in enhance.NOT_FOR_COMPARISON.lower()


# -- A build without the model ---------------------------------------------


def test_a_build_without_the_model_simply_cannot_offer_it(monkeypatch):
    monkeypatch.setattr(enhance.resources, "bundled_denoiser_model", lambda: None)
    assert not enhance.available()


def test_asking_anyway_fails_with_something_an_operator_can_act_on(monkeypatch):
    monkeypatch.setattr(enhance.resources, "bundled_denoiser_model", lambda: None)
    with pytest.raises(enhance.EnhancementError) as caught:
        enhance._denoiser()
    message = str(caught.value)
    assert "rebuilt" in message
    # Never a suggestion that the machine could fetch it.
    assert "download" not in message.lower() and "internet" not in message.lower()


def test_a_recording_that_is_not_there_says_so(tmp_path):
    with pytest.raises(enhance.EnhancementError):
        enhance.denoise(tmp_path / "missing.wav")
