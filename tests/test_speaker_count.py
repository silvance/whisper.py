"""The speaker-count question: what answers it, and what each answer means."""

import pytest

from whispr import speaker_count


def test_the_first_choice_is_not_an_answer():
    """The dropdown opens on a prompt, and a prompt does not start a run."""
    assert speaker_count.choices()[0] == speaker_count.UNSET
    assert not speaker_count.is_answered(speaker_count.UNSET)


def test_not_sure_is_a_real_answer():
    """Declining to guess is a decision; it just isn't a count."""
    assert speaker_count.is_answered(speaker_count.UNKNOWN)
    assert speaker_count.to_setting(speaker_count.UNKNOWN) == ""


def test_every_count_from_one_to_the_cap_is_offered():
    labels = speaker_count.choices()
    assert len(labels) == speaker_count.MAX_SPEAKERS + 2
    for count in range(1, speaker_count.MAX_SPEAKERS + 1):
        assert speaker_count.to_setting(labels[count + 1]) == str(count)


def test_one_person_reads_as_one_person():
    """A monologue is a legitimate answer, and the wording has to survive it."""
    assert "1 person" in speaker_count.choices()
    assert "2 people" in speaker_count.choices()


def test_an_unrecognised_label_is_not_an_answer():
    """A stale value from another build asks the question again."""
    assert not speaker_count.is_answered("four")
    assert not speaker_count.is_answered("")
    assert speaker_count.to_setting("four") == ""


@pytest.mark.parametrize("stored", ["2", 2, " 2 "])
def test_a_stored_count_comes_back_as_its_label(stored):
    assert speaker_count.from_setting(stored) == "2 people"
    assert speaker_count.is_answered(speaker_count.from_setting(stored))


@pytest.mark.parametrize("stored", ["", None, "0", "-1", "abc", True, False])
def test_no_usable_count_comes_back_unanswered(stored):
    """Including a blank from a build that never asked - the case this is for."""
    assert speaker_count.from_setting(stored) == speaker_count.UNSET
    assert speaker_count.parse(stored) is None


def test_a_count_beyond_the_cap_is_asked_again():
    """The dropdown cannot express it, so leaving it set would hide it."""
    assert speaker_count.parse(speaker_count.MAX_SPEAKERS + 1) is None
    assert speaker_count.from_setting(99) == speaker_count.UNSET


def test_label_and_setting_round_trip():
    for label in speaker_count.choices():
        assert speaker_count.from_setting(speaker_count.to_setting(label)) in (
            label,
            speaker_count.UNSET,
        )


def test_a_chosen_count_reaches_the_engine():
    """The number the operator picked is the number diarization is given."""
    assert speaker_count.parse(speaker_count.to_setting("3 people")) == 3
