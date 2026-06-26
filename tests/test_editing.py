from whispr.editing import (
    coalesce_segments,
    split_segment_on_span,
    split_segment_on_word,
)
from whispr.transcription import Segment, Word


def _seg(words, speaker):
    return Segment(
        start=words[0].start,
        end=words[-1].end,
        text="".join(w.word for w in words).strip(),
        speaker=speaker,
        words=list(words),
    )


def test_split_single_word_into_three_parts():
    words = [
        Word(start=0.0, end=1.0, word=" a"),
        Word(start=1.0, end=2.0, word=" b"),
        Word(start=2.0, end=3.0, word=" c"),
    ]
    segment = _seg(words, "SPEAKER_00")
    parts = split_segment_on_word(segment, 1, "SPEAKER_01", to_end=False)
    assert [p.speaker for p in parts] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
    assert [p.text for p in parts] == ["a", "b", "c"]


def test_split_to_end_moves_tail():
    words = [
        Word(start=0.0, end=1.0, word=" a"),
        Word(start=1.0, end=2.0, word=" b"),
        Word(start=2.0, end=3.0, word=" c"),
    ]
    segment = _seg(words, "SPEAKER_00")
    parts = split_segment_on_word(segment, 1, "SPEAKER_01", to_end=True)
    assert [p.speaker for p in parts] == ["SPEAKER_00", "SPEAKER_01"]
    assert [p.text for p in parts] == ["a", "b c"]


def test_split_first_word_has_no_before_part():
    words = [Word(start=0.0, end=1.0, word=" a"), Word(start=1.0, end=2.0, word=" b")]
    segment = _seg(words, "SPEAKER_00")
    parts = split_segment_on_word(segment, 0, "SPEAKER_01", to_end=False)
    assert [p.speaker for p in parts] == ["SPEAKER_01", "SPEAKER_00"]


def test_split_without_words_returns_unchanged():
    segment = Segment(start=0.0, end=1.0, text="hi", speaker="SPEAKER_00")
    assert split_segment_on_word(segment, 0, "SPEAKER_01", to_end=False) == [segment]


def test_split_span_moves_middle_words():
    words = [
        Word(start=0.0, end=1.0, word=" a"),
        Word(start=1.0, end=2.0, word=" b"),
        Word(start=2.0, end=3.0, word=" c"),
        Word(start=3.0, end=4.0, word=" d"),
    ]
    segment = _seg(words, "SPEAKER_00")
    parts = split_segment_on_span(segment, 1, 2, "SPEAKER_01")
    assert [p.speaker for p in parts] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
    assert [p.text for p in parts] == ["a", "b c", "d"]


def test_split_span_whole_line():
    words = [Word(start=0.0, end=1.0, word=" a"), Word(start=1.0, end=2.0, word=" b")]
    segment = _seg(words, "SPEAKER_00")
    parts = split_segment_on_span(segment, 0, 1, "SPEAKER_01")
    assert [p.speaker for p in parts] == ["SPEAKER_01"]
    assert parts[0].text == "a b"


def test_split_span_invalid_range_returns_unchanged():
    words = [Word(start=0.0, end=1.0, word=" a")]
    segment = _seg(words, "SPEAKER_00")
    assert split_segment_on_span(segment, 2, 1, "SPEAKER_01") == [segment]


def test_coalesce_merges_adjacent_same_speaker():
    a = _seg([Word(0.0, 1.0, " a")], "SPEAKER_00")
    b = _seg([Word(1.0, 2.0, " b")], "SPEAKER_00")
    c = _seg([Word(2.0, 3.0, " c")], "SPEAKER_01")
    out = coalesce_segments([a, b, c])
    assert [s.speaker for s in out] == ["SPEAKER_00", "SPEAKER_01"]
    assert out[0].text == "a b"
    assert out[0].end == 2.0


def test_split_then_coalesce_round_trips_a_noop_move():
    # Moving the middle word to a new speaker, then back, coalesces to one run.
    words = [
        Word(start=0.0, end=1.0, word=" a"),
        Word(start=1.0, end=2.0, word=" b"),
        Word(start=2.0, end=3.0, word=" c"),
    ]
    segment = _seg(words, "SPEAKER_00")
    parts = split_segment_on_word(segment, 1, "SPEAKER_00", to_end=False)
    merged = coalesce_segments(parts)
    assert len(merged) == 1
    assert merged[0].text == "a b c"
