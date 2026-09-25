"""What a finished run says, and about which recordings.

Two failures these hold shut, both found by review of live code rather than by
a test that was missing:

* a batch reported only the last file, so a gutted first file finished silently;
* the warning was announced a line before the success banner, which replaced it,
  so the warning reached nobody at all.
"""

import pytest

from whispr.run_summary import (
    LOW_KEPT_FRACTION,
    FailedRun,
    SkippedRun,
    completion,
    much_was_skipped,
    per_file_note,
)


def test_an_unfiltered_run_is_never_flagged():
    """No filter, no loss to report - not a loss of 100%."""
    assert not much_was_skipped(None)


@pytest.mark.parametrize("kept", [0.0, 0.19, 0.34])
def test_losing_most_of_a_recording_is_flagged(kept):
    assert much_was_skipped(kept)


@pytest.mark.parametrize("kept", [LOW_KEPT_FRACTION, 0.5, 0.9, 1.0])
def test_a_merely_quiet_recording_is_not(kept):
    """Long dead air is what silence skipping is for; this must stay quiet."""
    assert not much_was_skipped(kept)


def test_a_clean_run_says_only_that_it_finished():
    kind, message = completion(1, 1, [])
    assert kind == "success"
    assert message == "Transcription complete."


def test_a_clean_batch_counts_the_files():
    kind, message = completion(4, 4, [])
    assert kind == "success"
    assert "4 recording(s)" in message


def test_a_lossy_run_still_says_it_finished():
    """The run did finish. Dropping that would read as a failure, which it isn't."""
    kind, message = completion(1, 1, [SkippedRun("carpark.m4a", 9.7, 0.19)])
    assert kind == "warning"
    assert message.startswith("Transcription complete")


def test_a_lossy_run_names_the_recording_and_what_to_do():
    _, message = completion(1, 1, [SkippedRun("carpark.m4a", 9.7, 0.19)])
    assert "carpark.m4a" in message
    assert "9.7 min" in message
    assert "19%" in message
    assert "Skip silence" in message


def test_a_batch_names_every_recording_that_lost_audio():
    """The bug this exists for: only the last file used to be checked."""
    runs = [
        SkippedRun("first.m4a", 40.0, 0.11),
        SkippedRun("third.m4a", 12.5, 0.30),
    ]
    kind, message = completion(3, 3, runs)
    assert kind == "warning"
    assert "first.m4a" in message
    assert "third.m4a" in message
    assert "2 recordings" in message


def test_a_batch_where_only_the_first_file_suffered_still_warns():
    """The displayed result is the last one; the warning must not follow it."""
    kind, message = completion(3, 3, [SkippedRun("first.m4a", 40.0, 0.11)])
    assert kind == "warning"
    assert "first.m4a" in message


def test_there_is_exactly_one_banner_to_show():
    """Completion and the caveat are one message; a second would replace it."""
    for skipped in ([], [SkippedRun("a.m4a", 9.0, 0.2)]):
        kind, message = completion(1, 1, skipped)
        assert kind in {"success", "warning"}
        assert message


def test_the_status_line_names_the_file_it_is_about():
    """In a batch an unnamed line could be about any of them."""
    line = per_file_note(SkippedRun("carpark.m4a", 9.7, 0.19))
    assert line.startswith("carpark.m4a:")
    assert "9.7 min" in line and "19%" in line


# -- Recordings that were not there ---------------------------------------
#
# A file that was never opened must not be able to finish behind a green
# "complete". The single-file case was the worst of it: one missing recording
# and the banner read "Transcription complete." with nothing transcribed.


def test_one_missing_file_is_not_a_complete_transcription():
    kind, message = completion(0, 1, [], ["carpark.m4a"])
    assert kind == "warning"
    assert "Nothing was transcribed" in message
    assert "carpark.m4a" in message
    assert "complete" not in message.lower()


def test_a_whole_batch_of_missing_files_says_nothing_was_done():
    kind, message = completion(0, 3, [], ["a.m4a", "b.m4a", "c.m4a"])
    assert kind == "warning"
    assert "Nothing was transcribed" in message
    for name in ("a.m4a", "b.m4a", "c.m4a"):
        assert name in message


def test_a_partly_missing_batch_says_how_many_of_how_many():
    kind, message = completion(4, 5, [], ["fifth.m4a"])
    assert kind == "warning"
    assert "4 of 5" in message
    assert "fifth.m4a" in message


def test_a_missing_file_outranks_a_silence_caveat():
    """Both are true; not finding the recording is the one to lead with."""
    kind, message = completion(
        1, 2, [SkippedRun("first.m4a", 9.0, 0.2)], ["second.m4a"]
    )
    assert kind == "warning"
    assert "second.m4a" in message


def test_a_run_with_nothing_missing_is_unaffected():
    assert completion(2, 2, [], []) == (
        "success",
        "Transcription complete — 2 recording(s).",
    )


# -- Recordings that could not be transcribed ------------------------------
#
# A folder handed over at five o'clock runs unattended. One recording that
# cannot be read is that recording's problem: the run carries on, and the
# banner accounts for it afterwards rather than the operator finding an empty
# output folder in the morning.


def test_one_bad_recording_in_a_batch_does_not_read_as_success():
    kind, message = completion(
        11, 12, [], [], [FailedRun("carpark.m4a", "The file is not readable.")]
    )
    assert kind == "warning"
    assert "11 of 12" in message
    assert "carpark.m4a" in message
    assert "The file is not readable." in message


def test_a_partly_failed_batch_says_the_rest_was_saved():
    """Otherwise the operator cannot tell whether any of it is worth keeping."""
    _, message = completion(11, 12, [], [], [FailedRun("a.m4a", "Unreadable.")])
    assert "the rest were transcribed and saved" in message


def test_a_whole_folder_failing_for_one_reason_says_it_once():
    """A model missing fails all fifty the same way. That is one problem."""
    reason = "The model 'medium' isn't in this build."
    failed = [FailedRun(f"{n}.m4a", reason) for n in range(50)]
    kind, message = completion(0, 50, [], [], failed)
    assert kind == "warning"
    assert "Nothing was transcribed" in message
    assert message.count(reason) == 1
    assert "50 could not be transcribed" in message
    assert "0.m4a" not in message


def test_different_reasons_name_the_recordings():
    failed = [FailedRun("a.m4a", "Unreadable."), FailedRun("b.m4a", "No audio track.")]
    _, message = completion(1, 3, [], [], failed)
    assert "a.m4a" in message and "b.m4a" in message


def test_missing_and_failed_are_both_accounted_for():
    kind, message = completion(
        8, 10, [], ["gone.m4a"], [FailedRun("bad.m4a", "Unreadable.")]
    )
    assert kind == "warning"
    assert "gone.m4a" in message and "bad.m4a" in message


def test_a_failure_outranks_a_silence_caveat():
    """Both true; not transcribing a recording is the one to lead with."""
    _, message = completion(
        1,
        2,
        [SkippedRun("first.m4a", 9.0, 0.2)],
        [],
        [FailedRun("b.m4a", "Unreadable.")],
    )
    assert "b.m4a" in message
    assert "Skip silence" not in message


def test_a_clean_batch_is_unaffected_by_the_new_argument():
    assert completion(3, 3, [], [], []) == (
        "success",
        "Transcription complete — 3 recording(s).",
    )


# -- Output that could not be written --------------------------------------
#
# Transcribed and saved are different claims. A batch whose destination went
# away - an unplugged drive, a share that dropped - used to log a line and
# finish green, so an operator came back to a folder with nothing in it and a
# banner saying the run had completed.


def test_a_recording_that_could_not_be_written_is_not_a_clean_finish():
    kind, message = completion(
        4, 4, [], [], [], [FailedRun("carpark.m4a", "The drive is not there.")], 3
    )
    assert kind == "warning"
    assert "saved 3" in message
    assert "carpark.m4a" in message
    assert "The drive is not there." in message


def test_transcribing_everything_and_saving_nothing_says_so():
    """The whole point: a green banner over an empty output folder."""
    unsaved = [
        FailedRun(f"{n}.wav", "The output folder is not there.") for n in range(6)
    ]
    kind, message = completion(6, 6, [], [], [], unsaved, 0)
    assert kind == "warning"
    assert "Nothing was saved" in message
    assert "complete" not in message.lower()


def test_one_reason_for_every_failed_save_is_stated_once():
    unsaved = [
        FailedRun(f"{n}.wav", "The output folder is not there.") for n in range(40)
    ]
    _, message = completion(40, 40, [], [], [], unsaved, 0)
    assert message.count("The output folder is not there.") == 1
    assert "40" in message


def test_not_transcribing_anything_outranks_not_saving_it():
    """Both are true when nothing ran; the first explains the second."""
    _, message = completion(
        0, 2, [], ["a.wav"], [FailedRun("b.wav", "Unreadable.")], [], 0
    )
    assert "Nothing was transcribed" in message
    assert "Nothing was saved" not in message


def test_a_save_failure_and_a_transcription_failure_are_told_apart():
    kind, message = completion(
        3,
        4,
        [],
        [],
        [FailedRun("bad.wav", "Unreadable.")],
        [FailedRun("ok.wav", "Disk full.")],
        2,
    )
    assert "could not be transcribed" in message
    assert "could not be written" in message


def test_a_run_that_saved_everything_is_unaffected():
    assert completion(3, 3, [], [], [], [], 3) == (
        "success",
        "Transcription complete — 3 recording(s).",
    )
