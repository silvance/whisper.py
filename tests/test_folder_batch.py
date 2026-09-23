"""Handing over a folder instead of picking its contents one at a time.

What a case folder actually contains is recordings, notes, photographs, last
run's transcripts, and subfolders by day or by source. So the walk has to
decide what is a recording - and then say what it decided against, because an
operator who drops a folder of 50 and sees 12 queued must not be left
wondering whether the tool lost the rest.
"""

from pathlib import Path

from whispr.folder_batch import Found, describe, expand, is_media, plural


def make(root: Path, *names: str) -> Path:
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x00")
    return root


def names(found: Found):
    return [p.name for p in found.recordings]


def test_a_folder_becomes_the_recordings_inside_it(tmp_path):
    make(tmp_path, "one.m4a", "two.mp3", "notes.txt")
    found = expand([tmp_path])
    assert names(found) == ["one.m4a", "two.mp3"]
    assert found.ignored == 1


def test_subfolders_are_walked_too(tmp_path):
    """ "Every file" means every file; a case folder is organised by day."""
    make(tmp_path, "day1/a.wav", "day2/b.wav", "day2/later/c.wav")
    assert names(expand([tmp_path])) == ["a.wav", "b.wav", "c.wav"]


def test_the_order_is_the_one_an_operator_would_read(tmp_path):
    make(tmp_path, "b.wav", "a.wav", "sub/c.wav")
    found = expand([tmp_path])
    assert names(found) == ["a.wav", "b.wav", "c.wav"]


def test_a_folder_of_nothing_useful_queues_nothing(tmp_path):
    make(tmp_path, "notes.txt", "photo.jpg")
    found = expand([tmp_path])
    assert found.recordings == []
    assert found.ignored == 2


def test_a_file_the_operator_named_is_taken_at_its_word(tmp_path):
    """An extension is a guess; someone pointing at a file knows more than it."""
    make(tmp_path, "interview.rec")
    found = expand([tmp_path / "interview.rec"])
    assert names(found) == ["interview.rec"]


def test_only_folder_contents_are_filtered(tmp_path):
    make(tmp_path, "interview.rec", "clip.wav")
    assert names(expand([tmp_path])) == ["clip.wav"]


def test_files_and_folders_can_be_dropped_together(tmp_path):
    make(tmp_path, "folder/a.wav", "loose.mp3")
    found = expand([tmp_path / "loose.mp3", tmp_path / "folder"])
    assert set(names(found)) == {"loose.mp3", "a.wav"}


def test_the_same_recording_is_never_queued_twice(tmp_path):
    """Dropping a folder and a file inside it is one recording, not two."""
    make(tmp_path, "sub/a.wav")
    found = expand([tmp_path, tmp_path / "sub" / "a.wav"])
    assert names(found) == ["a.wav"]


def test_hidden_files_and_operating_system_litter_are_left_alone(tmp_path):
    make(tmp_path, "real.wav", ".hidden.wav", ".DS_Store", "Thumbs.db")
    found = expand([tmp_path])
    assert names(found) == ["real.wav"]
    # Litter the operator did not put there is not worth reporting as ignored.
    assert found.ignored == 0


def test_a_folder_that_cannot_be_read_is_reported_not_raised(tmp_path, monkeypatch):
    """One unreadable subfolder must not cost the other forty."""
    import os

    real = os.walk

    def walk(top, onerror=None, **kwargs):
        if onerror is not None:
            onerror(PermissionError(13, "Permission denied", str(tmp_path / "locked")))
        return real(top, **kwargs)

    make(tmp_path, "a.wav")
    monkeypatch.setattr(os, "walk", walk)
    found = expand([tmp_path])
    assert names(found) == ["a.wav"]
    assert found.unreadable and "locked" in found.unreadable[0]


def test_not_recursing_stops_at_the_top_level(tmp_path):
    make(tmp_path, "a.wav", "sub/b.wav")
    assert names(expand([tmp_path], recursive=False)) == ["a.wav"]


# -- The WAV a previous run left behind -----------------------------------
#
# A batch converts video to WAV beside the source. Run the same folder twice
# without this and every video is transcribed again as its own WAV: the hours
# doubled, the output duplicated, and nothing on screen saying why.


def test_a_wav_beside_the_video_it_came_from_is_left_out(tmp_path):
    make(tmp_path, "interview.mp4", "interview.wav")
    found = expand([tmp_path])
    assert names(found) == ["interview.mp4"]
    assert found.converted == ["interview.wav"]


def test_a_wav_with_no_video_of_that_name_is_a_recording(tmp_path):
    make(tmp_path, "interview.mp4", "carpark.wav")
    assert set(names(expand([tmp_path]))) == {"interview.mp4", "carpark.wav"}


def test_a_wav_in_a_different_folder_is_not_assumed_to_be_a_conversion(tmp_path):
    make(tmp_path, "video/interview.mp4", "audio/interview.wav")
    assert len(expand([tmp_path]).recordings) == 2


def test_a_named_wav_is_still_taken_at_its_word(tmp_path):
    """The operator who says "this one" overrides the assumption about it."""
    make(tmp_path, "interview.mp4", "interview.wav")
    found = expand([tmp_path / "interview.wav"])
    assert names(found) == ["interview.wav"]


# -- What the queue says --------------------------------------------------


def test_the_queue_counts_what_it_took(tmp_path):
    make(tmp_path, "a.wav", "b.wav")
    assert describe(expand([tmp_path])) == "2 recordings queued — a.wav, b.wav."


def test_the_queue_accounts_for_what_it_passed_over(tmp_path):
    make(tmp_path, "a.wav", "notes.txt", "sheet.xlsx")
    assert "2 other files ignored" in describe(expand([tmp_path]))


def test_the_queue_names_the_folders_it_walked(tmp_path):
    make(tmp_path, "day1/a.wav", "day2/b.wav")
    assert "from 3 folders" in describe(expand([tmp_path]))


def test_an_empty_folder_says_so_rather_than_nothing(tmp_path):
    """Silence here reads as a broken drop, which is the wrong thing to learn."""
    make(tmp_path, "notes.txt")
    message = describe(expand([tmp_path]))
    assert "No recordings in that folder." in message
    assert "1 file in there is not audio or video" in message


def test_the_queue_mentions_a_left_out_conversion(tmp_path):
    make(tmp_path, "interview.mp4", "interview.wav")
    assert "already-converted" in describe(expand([tmp_path]))


def test_the_queue_mentions_a_folder_it_could_not_read():
    found = Found(recordings=[Path("a.wav")], folders=2, unreadable=["/locked"])
    assert "1 folder could not be read" in describe(found)


def test_media_is_decided_by_extension_case_insensitively():
    assert is_media(Path("A.WAV")) and is_media(Path("b.Mp4"))
    assert not is_media(Path("notes.txt"))


def test_the_queue_counts_things_the_way_people_write_them():
    """ "1 recording(s)" is the sound of software talking to itself."""
    assert plural(1, "recording") == "1 recording"
    assert plural(2, "recording") == "2 recordings"
    assert plural(1, "already-converted copy", "already-converted copies") == (
        "1 already-converted copy"
    )
    assert plural(3, "already-converted copy", "already-converted copies") == (
        "3 already-converted copies"
    )


def test_a_long_queue_names_the_first_few_and_counts_the_rest(tmp_path):
    make(tmp_path, *[f"{n:02d}.wav" for n in range(9)])
    message = describe(expand([tmp_path]))
    assert "9 recordings queued" in message
    assert "00.wav, 01.wav, 02.wav, 03.wav (+5 more)" in message


def test_nothing_dropped_at_all_says_nothing(tmp_path):
    """An empty queue on a fresh page has no story to tell."""
    assert describe(Found()) == ""
