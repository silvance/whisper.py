"""Which names travel across a re-split, and which must not.

The failure these hold shut: an operator types "Smith" into Speaker 1, the
transcript is split again with a different count, and "Smith" lands on whoever
the new split happens to number first. The tool would look like it had kept
their work when it had in fact guessed - and in this application the guess is
a person's name on somebody else's words.
"""

from whispr.speaker_names import preset_names


def test_typed_names_are_matched_in_label_order_on_a_first_run():
    names = preset_names(["SPEAKER_01", "SPEAKER_00"], {}, ["Smith", "Jones"])
    assert names == {"SPEAKER_00": "Smith", "SPEAKER_01": "Jones"}


def test_a_typed_name_never_crosses_a_re_split():
    """The regression this module exists for."""
    typed = ["Smith", "Jones"]
    first = preset_names(["SPEAKER_00", "SPEAKER_01"], {}, typed)
    assert first == {"SPEAKER_00": "Smith", "SPEAKER_01": "Jones"}

    # The same fields are still on screen; the split is redone into 3.
    redone = preset_names(
        ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"], {}, typed, use_typed=False
    )
    assert redone == {}
    assert "Smith" not in redone.values()


def test_a_recognised_voice_does_cross_a_re_split():
    """It was measured against the new turns, so it is earned again."""
    recognized = {"voice::Hostile 1": "Hostile 1"}
    redone = preset_names(
        ["voice::Hostile 1", "SPEAKER_00"], recognized, ["Smith"], use_typed=False
    )
    assert redone == {"voice::Hostile 1": "Hostile 1"}


def test_a_recognised_voice_outranks_a_typed_field():
    """A name the audio supports beats one typed into a positional box."""
    names = preset_names(
        ["voice::Hostile 1", "SPEAKER_00"],
        {"voice::Hostile 1": "Hostile 1"},
        ["Smith"],
    )
    assert names["voice::Hostile 1"] == "Hostile 1"
    # The typed field falls to the first speaker that has no recognised name.
    assert names["SPEAKER_00"] == "Smith"


def test_blank_fields_name_nobody():
    names = preset_names(["SPEAKER_00", "SPEAKER_01"], {}, ["", "   "])
    assert names == {}


def test_a_blank_field_does_not_shift_the_names_after_it():
    """Positional means positional: a gap is a gap, not a shuffle."""
    names = preset_names(
        ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"], {}, ["Smith", "", "Jones"]
    )
    assert names == {"SPEAKER_00": "Smith", "SPEAKER_02": "Jones"}


def test_more_fields_than_speakers_is_harmless():
    names = preset_names(["SPEAKER_00"], {}, ["Smith", "Jones", "Brown"])
    assert names == {"SPEAKER_00": "Smith"}


def test_more_speakers_than_fields_leaves_the_rest_unnamed():
    names = preset_names(["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"], {}, ["Smith"])
    assert names == {"SPEAKER_00": "Smith"}


def test_segments_with_no_speaker_are_not_speakers():
    names = preset_names(["SPEAKER_00", "", "", "SPEAKER_01"], {}, ["Smith"])
    assert names == {"SPEAKER_00": "Smith"}


def test_names_are_trimmed():
    names = preset_names(["SPEAKER_00"], {}, ["  Smith  "])
    assert names == {"SPEAKER_00": "Smith"}
