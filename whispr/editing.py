"""Pure transcript-editing operations used by the GUI's speaker corrections.

These were previously methods on the GUI class; pulling them out makes the
speaker/word reassignment logic unit-testable without a running Tk window.
They operate only on :class:`~whispr.transcription.Segment` /
:class:`~whispr.transcription.Word` values and return new segment lists.
"""

from __future__ import annotations

from typing import List, Optional

from .transcription import Segment, Word


def _segment_from_words(words: List[Word], speaker: Optional[str]) -> Optional[Segment]:
    """Build a segment spanning ``words`` (or ``None`` when the list is empty)."""
    if not words:
        return None
    return Segment(
        start=words[0].start,
        end=words[-1].end,
        text="".join(w.word for w in words).strip(),
        speaker=speaker,
        words=list(words),
    )


def split_segment_on_word(
    segment: Segment,
    word_index: int,
    speaker_id: str,
    *,
    to_end: bool,
) -> List[Segment]:
    """Reassign a word (or from it to the line's end) to another speaker.

    Splits ``segment`` around the target word into up to three parts - the words
    before it (original speaker), the moved span (``speaker_id``), and the words
    after it (original speaker). ``to_end`` moves from ``word_index`` to the end of
    the line; otherwise only the single word moves. Empty parts are dropped. When
    the segment has no word timestamps, the segment is returned unchanged.
    """
    words = segment.words
    if not words or word_index >= len(words):
        return [segment]
    end = len(words) if to_end else word_index + 1
    parts = [
        seg
        for seg in (
            _segment_from_words(words[:word_index], segment.speaker),
            _segment_from_words(words[word_index:end], speaker_id),
            _segment_from_words(words[end:], segment.speaker),
        )
        if seg is not None
    ]
    return parts


def coalesce_segments(segments: List[Segment]) -> List[Segment]:
    """Merge adjacent segments that share a speaker (rebuilding text/words).

    Only segments carrying word timestamps are merged, so plain (non-diarized)
    segments are left untouched.
    """
    out: List[Segment] = []
    for seg in segments:
        if out and out[-1].speaker == seg.speaker and out[-1].words and seg.words:
            prev = out[-1]
            prev.words = prev.words + seg.words
            prev.end = seg.end
            prev.text = "".join(w.word for w in prev.words).strip()
        else:
            out.append(seg)
    return out
