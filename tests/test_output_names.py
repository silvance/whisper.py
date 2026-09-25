"""Two recordings called the same thing must not become one transcript.

The failure this holds shut arrived with folder batches. A case folder
organised by day holds 2026-09-01/interview.wav and 2026-09-02/interview.wav;
both were written to interview.wav.txt in the output folder, the second over
the first, and the run still reported success. An hour of work replaced by
another hour of work, with nothing on screen to say it had happened.
"""

from pathlib import Path

from whispr.output_names import (
    converted_audio_name,
    plan,
    subtitle_name,
    transcript_name,
)


def test_a_lone_recording_keeps_its_own_name():
    names = plan([Path("/case/carpark.m4a")])
    assert names == {Path("/case/carpark.m4a"): "carpark.m4a"}


def test_recordings_with_different_names_are_left_alone():
    sources = [Path("/case/a.wav"), Path("/case/day2/b.wav")]
    assert list(plan(sources).values()) == ["a.wav", "b.wav"]


def test_the_collision_that_lost_a_transcript():
    """The regression, in one test."""
    first = Path("/case/2026-09-01/interview.wav")
    second = Path("/case/2026-09-02/interview.wav")
    names = plan([first, second])
    assert names[first] != names[second]
    assert names[first] == "2026-09-01 - interview.wav"
    assert names[second] == "2026-09-02 - interview.wav"


def test_only_the_colliding_recordings_are_renamed():
    """A batch of 50 should not have 50 awkward names because two clashed."""
    sources = [
        Path("/case/day1/interview.wav"),
        Path("/case/day2/interview.wav"),
        Path("/case/carpark.m4a"),
    ]
    names = plan(sources)
    assert names[Path("/case/carpark.m4a")] == "carpark.m4a"
    assert len(set(names.values())) == 3


def test_a_deeper_folder_is_used_when_the_nearest_one_matches():
    sources = [
        Path("/case/alpha/audio/interview.wav"),
        Path("/case/bravo/audio/interview.wav"),
    ]
    names = plan(sources)
    assert set(names.values()) == {
        "alpha - audio - interview.wav",
        "bravo - audio - interview.wav",
    }
    assert len(set(names.values())) == 2


def test_three_recordings_of_the_same_name_all_differ():
    sources = [Path(f"/case/day{n}/call.wav") for n in (1, 2, 3)]
    names = plan(sources)
    assert len(set(names.values())) == 3


def test_every_name_in_a_batch_is_unique():
    """The property that matters, whatever the paths look like."""
    sources = [
        Path("/case/a/x.wav"),
        Path("/case/b/x.wav"),
        Path("/case/x.wav"),
        Path("/other/x.wav"),
        Path("/case/a/y.wav"),
    ]
    names = plan(sources)
    assert len(set(names.values())) == len(sources)


def test_a_name_that_would_clash_with_another_group_is_still_unique():
    """Extending one pair must not collide with a file already called that."""
    sources = [
        Path("/case/day1/interview.wav"),
        Path("/case/day2/interview.wav"),
        Path("/elsewhere/day1 - interview.wav"),
    ]
    names = plan(sources)
    assert len(set(names.values())) == 3


def test_characters_windows_will_not_accept_are_replaced():
    """The output folder is on Windows; a colon in a name is a failed write."""
    names = plan([Path("/case/C:weird/a:b.wav")])
    name = next(iter(names.values()))
    assert ":" not in name and "/" not in name


def test_the_order_of_the_batch_does_not_change_the_names():
    """The same folder run twice must write the same files, not a second set."""
    sources = [
        Path("/case/day1/interview.wav"),
        Path("/case/day2/interview.wav"),
        Path("/case/carpark.m4a"),
    ]
    assert plan(sources) == plan(list(reversed(sources)))


def test_the_output_suffixes_are_the_ones_already_shipped():
    assert transcript_name("interview.wav") == "interview.wav.txt"
    assert subtitle_name("interview.wav") == "interview.wav.srt"


def test_a_converted_video_lands_beside_its_own_transcript():
    """Two interview.mp4 files must not fight over one interview.wav either."""
    first = Path("/case/day1/interview.mp4")
    second = Path("/case/day2/interview.mp4")
    names = plan([first, second])
    assert converted_audio_name(names[first]) == "day1 - interview.wav"
    assert converted_audio_name(names[second]) == "day2 - interview.wav"
