import copy

import pytest

from whispr import diarization
from whispr.diarization import SpeakerSegment, assign_speakers, diarize
from whispr.transcription import Segment, Word


def test_assign_speakers_by_overlap():
    segments = [
        Segment(start=0.0, end=2.0, text="hello"),
        Segment(start=2.0, end=4.0, text="world"),
        Segment(start=4.0, end=6.0, text="again"),
    ]
    speakers = [
        SpeakerSegment(start=0.0, end=2.1, speaker="SPEAKER_00"),
        SpeakerSegment(start=2.1, end=6.0, speaker="SPEAKER_01"),
    ]
    assign_speakers(segments, speakers)
    assert [s.speaker for s in segments] == [
        "SPEAKER_00",
        "SPEAKER_01",
        "SPEAKER_01",
    ]


def test_assign_speakers_no_overlap_leaves_none():
    segments = [Segment(start=10.0, end=12.0, text="orphan")]
    speakers = [SpeakerSegment(start=0.0, end=5.0, speaker="SPEAKER_00")]
    assign_speakers(segments, speakers)
    assert segments[0].speaker is None


def test_assign_speakers_picks_majority_overlap():
    # Segment 1-5: overlaps SPEAKER_00 for 1s (1-2) and SPEAKER_01 for 3s (2-5).
    segments = [Segment(start=1.0, end=5.0, text="mostly one")]
    speakers = [
        SpeakerSegment(start=0.0, end=2.0, speaker="SPEAKER_00"),
        SpeakerSegment(start=2.0, end=8.0, speaker="SPEAKER_01"),
    ]
    assign_speakers(segments, speakers)
    assert segments[0].speaker == "SPEAKER_01"


def test_assign_speakers_splits_segment_by_word():
    # One Whisper segment spanning a speaker change mid-way.
    segment = Segment(
        start=0.0,
        end=4.0,
        text="hello there how are you",
        words=[
            Word(start=0.0, end=1.0, word=" hello"),
            Word(start=1.0, end=2.0, word=" there"),
            Word(start=2.0, end=3.0, word=" how"),
            Word(start=3.0, end=4.0, word=" you"),
        ],
    )
    speakers = [
        SpeakerSegment(start=0.0, end=2.0, speaker="SPEAKER_00"),
        SpeakerSegment(start=2.0, end=4.0, speaker="SPEAKER_01"),
    ]
    out = assign_speakers([segment], speakers)
    assert [s.speaker for s in out] == ["SPEAKER_00", "SPEAKER_01"]
    assert out[0].text == "hello there"
    assert out[1].text == "how you"


def test_assign_speakers_small_gap_uses_nearest():
    # Word ~1s after the turn ends -> within the gap tolerance, snaps to nearest.
    segment = Segment(
        start=6.0,
        end=6.5,
        text="orphan",
        words=[Word(start=6.0, end=6.5, word="orphan")],
    )
    speakers = [SpeakerSegment(start=0.0, end=5.0, speaker="SPEAKER_00")]
    out = assign_speakers([segment], speakers)
    assert out[0].speaker == "SPEAKER_00"  # nearest turn, not None


def test_assign_speakers_large_gap_is_unassigned():
    # Word many seconds from any turn -> unattributable, left as None.
    segment = Segment(
        start=10.0,
        end=11.0,
        text="orphan",
        words=[Word(start=10.0, end=11.0, word="orphan")],
    )
    speakers = [SpeakerSegment(start=0.0, end=5.0, speaker="SPEAKER_00")]
    out = assign_speakers([segment], speakers)
    assert out[0].speaker is None


def test_assign_speakers_smooths_spurious_short_run():
    # A spurious 1-word island (from overlap/jitter) wedged between two longer
    # runs of the same speaker should be absorbed back into them.
    segment = Segment(
        start=0.0,
        end=4.0,
        text="alpha beta gamma delta epsilon",
        words=[
            Word(start=0.0, end=0.8, word=" alpha"),
            Word(start=0.8, end=1.6, word=" beta"),
            Word(start=1.6, end=1.9, word=" gamma"),  # spurious single-word blip
            Word(start=1.9, end=2.8, word=" delta"),
            Word(start=2.8, end=4.0, word=" epsilon"),
        ],
    )
    speakers = [
        SpeakerSegment(start=0.0, end=1.6, speaker="SPEAKER_00"),
        SpeakerSegment(start=1.6, end=1.9, speaker="SPEAKER_01"),  # blip turn
        SpeakerSegment(start=1.9, end=4.0, speaker="SPEAKER_00"),
    ]
    out = assign_speakers([segment], speakers)
    # Without smoothing this would split into three runs; the blip is absorbed.
    assert [s.speaker for s in out] == ["SPEAKER_00"]
    assert out[0].text == "alpha beta gamma delta epsilon"


def test_assign_speakers_keeps_real_short_turn_at_min_length():
    # A clearly-spoken short turn longer than the smoothing window survives.
    segment = Segment(
        start=0.0,
        end=4.0,
        text="long opening yes long closing",
        words=[
            Word(start=0.0, end=1.2, word=" long"),
            Word(start=1.2, end=1.6, word=" opening"),
            Word(start=1.6, end=2.6, word=" yes"),  # 1.0s > 0.8s window
            Word(start=2.6, end=3.2, word=" long"),
            Word(start=3.2, end=4.0, word=" closing"),
        ],
    )
    speakers = [
        SpeakerSegment(start=0.0, end=1.6, speaker="SPEAKER_00"),
        SpeakerSegment(start=1.6, end=2.6, speaker="SPEAKER_01"),
        SpeakerSegment(start=2.6, end=4.0, speaker="SPEAKER_00"),
    ]
    out = assign_speakers([segment], speakers)
    assert [s.speaker for s in out] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]


def test_diarize_without_backend(tmp_path, monkeypatch):
    # With no pyannote, auto falls back to sherpa; sherpa-onnx isn't installed in
    # this environment -> clear RuntimeError.
    import whispr.diarization as d

    monkeypatch.setattr(d, "_pyannote_available", lambda: False)
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"\x00")
    with pytest.raises(RuntimeError, match="sherpa-onnx is not installed"):
        diarize(wav)


def test_diarize_unknown_backend(tmp_path):
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"\x00")
    with pytest.raises(ValueError, match="unknown diarization backend"):
        diarize(wav, backend="bogus")


def test_diarize_pyannote_backend_requires_pyannote(tmp_path, monkeypatch):
    import whispr.diarization as d

    monkeypatch.setattr(d, "_pyannote_available", lambda: False)
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"\x00")
    with pytest.raises(RuntimeError, match="pyannote.audio is not installed"):
        diarize(wav, backend="pyannote")


def test_diarize_sherpa_backend_forced(tmp_path, monkeypatch):
    import whispr.diarization as d

    # Even when pyannote is available, backend="sherpa" must route to sherpa.
    monkeypatch.setattr(d, "_pyannote_available", lambda: True)
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"\x00")
    with pytest.raises(RuntimeError, match="sherpa-onnx is not installed"):
        diarize(wav, backend="sherpa")


# -- progress through a pyannote pass ---------------------------------------
# pyannote runs as one call. Without a hook the whole pass is an opaque wait,
# which on a long recording is minutes of an interface that cannot say more than
# "working". These cover the hook that reports it.


def _hook(progress=None, on_progress=None, cancelled=None):
    return diarization._pyannote_hook(progress, on_progress, cancelled)


def test_progress_advances_through_a_step():
    seen = []
    hook = _hook(on_progress=seen.append)
    hook("segmentation", total=10, completed=0)
    hook("segmentation", total=10, completed=5)
    hook("segmentation", total=10, completed=10)
    assert seen == [0.0, pytest.approx(0.225), pytest.approx(0.45)]


def test_progress_carries_on_into_the_next_step():
    seen = []
    hook = _hook(on_progress=seen.append)
    hook("segmentation", total=4, completed=4)
    hook("embeddings", total=4, completed=2)
    assert seen[-1] == pytest.approx(0.7)


def test_progress_never_goes_backwards():
    """Steps overlap and report unevenly; a bar that retreats reads as a fault."""
    seen = []
    hook = _hook(on_progress=seen.append)
    hook("embeddings", total=10, completed=10)
    hook("segmentation", total=10, completed=1)
    assert seen == [pytest.approx(0.95), pytest.approx(0.95)]


def test_an_unknown_step_holds_the_bar_and_says_so():
    """Better to admit the step is not understood than to invent a number."""
    seen = []
    said = []
    hook = _hook(progress=said.append, on_progress=seen.append)
    hook("segmentation", total=2, completed=2)
    hook("some_future_step", total=100, completed=50)
    assert seen[-1] == pytest.approx(0.45)
    assert any("some_future_step" in message for message in said)


def test_each_step_is_named_once_not_on_every_call():
    said = []
    hook = _hook(progress=said.append)
    for done in range(4):
        hook("segmentation", total=4, completed=done)
    assert len(said) == 1


def test_a_step_with_no_total_still_marks_where_it_started():
    seen = []
    hook = _hook(on_progress=seen.append)
    hook("embeddings")
    assert seen == [pytest.approx(0.45)]


def test_cancelling_stops_the_pass_rather_than_waiting_for_it_to_finish():
    hook = _hook(cancelled=lambda: True)
    with pytest.raises(diarization.CancelledError):
        hook("segmentation", total=10, completed=1)


def test_a_pipeline_without_hook_support_is_detected_not_attempted():
    """Calling with an unsupported keyword raises TypeError, which cannot be
    told apart from a TypeError raised by a real fault inside the pipeline."""

    class WithHook:
        def apply(self, file, num_speakers=None, hook=None):
            return None

    class WithoutHook:
        def apply(self, file, num_speakers=None):
            return None

    assert diarization._accepts_hook(WithHook()) is True
    assert diarization._accepts_hook(WithoutHook()) is False
    assert diarization._accepts_hook(object()) is False


# -- Redoing the split from the words as transcribed ----------------------
#
# When the speaker count comes back wrong, the remedy is to split again rather
# than transcribe again. That only holds if a second split works from the words
# as the model produced them: splitting the output of the first split would
# build the second attempt on the first one's mistakes.


def _spoken_words(segments):
    """Every word in order, ignoring how the segments are cut up."""
    return [w.word for seg in segments for w in seg.words]


def _transcript():
    def words(text, t0, step=0.5):
        return [
            Word(start=t0 + i * step, end=t0 + (i + 1) * step, word=" " + tok)
            for i, tok in enumerate(text.split())
        ]

    lines = [
        ("we are on the corner now", 0.0),
        ("copy that hold there", 3.0),
        ("the second car just arrived", 6.0),
        ("i see it stand by", 9.0),
    ]
    out = []
    for text, start in lines:
        ws = words(text, start)
        out.append(Segment(start=ws[0].start, end=ws[-1].end, text=text, words=ws))
    return out


def _even_split(count, total=12.0):
    span = total / count
    return [
        SpeakerSegment(start=i * span, end=(i + 1) * span, speaker=f"SPEAKER_{i:02d}")
        for i in range(count)
    ]


def test_a_second_split_keeps_every_word_the_model_produced():
    pristine = _transcript()
    spoken = _spoken_words(pristine)

    five = assign_speakers(copy.deepcopy(pristine), _even_split(5))
    assert len({s.speaker for s in five}) == 5

    two = assign_speakers(copy.deepcopy(pristine), _even_split(2))
    assert len({s.speaker for s in two}) == 2
    # The point of the whole exercise: the words are untouched by either split.
    assert _spoken_words(two) == spoken
    assert _spoken_words(five) == spoken


def test_the_words_as_transcribed_survive_the_first_split():
    """The kept copy has to stay pristine, or a redo inherits the first split."""
    pristine = _transcript()
    before = [(s.start, s.end, s.text, s.speaker) for s in pristine]
    assign_speakers(copy.deepcopy(pristine), _even_split(5))
    assert [(s.start, s.end, s.text, s.speaker) for s in pristine] == before
    assert all(s.speaker is None for s in pristine)


def test_a_hand_corrected_tag_cannot_leak_into_the_redo():
    """Why the pristine copy is kept rather than re-splitting what is on screen.

    Corrections were made against the old split and are meaningless against a
    new one - two speakers merged into one person under a name the operator
    typed would silently carry that name onto whoever the new split puts there.
    Working from the words as transcribed makes that impossible rather than
    unlikely.
    """
    pristine = _transcript()
    on_screen = assign_speakers(copy.deepcopy(pristine), _even_split(5))
    on_screen[0].speaker = "voice::Hostile 1"  # the operator named this one

    redone = assign_speakers(copy.deepcopy(pristine), _even_split(2))
    assert all(s.speaker != "voice::Hostile 1" for s in redone)
    assert {s.speaker for s in redone} == {"SPEAKER_00", "SPEAKER_01"}


def test_redoing_twice_with_the_same_count_gives_the_same_split():
    """A redo is the run's speaker pass again, not a nudge to the last one."""
    pristine = _transcript()
    first = assign_speakers(copy.deepcopy(pristine), _even_split(3))
    second = assign_speakers(copy.deepcopy(pristine), _even_split(3))
    assert [(s.start, s.end, s.text, s.speaker) for s in first] == [
        (s.start, s.end, s.text, s.speaker) for s in second
    ]
