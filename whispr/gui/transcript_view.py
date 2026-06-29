"""The transcript pane: renders a result and supports speaker/word corrections.

Diarization on hard or overlapping audio is never perfect, so the transcript is
interactive: click a ``[speaker]`` tag to rename that speaker everywhere or move
the whole line to another speaker, or click a single word to move just that word
(or from it onward) to another speaker. The underlying split/merge logic lives in
:mod:`whispr.editing`; this module is the Tk view around it.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import simpledialog
from tkinter.scrolledtext import ScrolledText
from typing import Callable, Dict, List, Optional

from ..editing import (
    coalesce_segments,
    split_segment_on_span,
    split_segment_on_word,
)
from ..transcription import (
    TranscriptionResult,
    is_low_confidence_segment,
    is_low_confidence_word,
)

# Foreground colour for low-confidence text when highlighting is enabled.
_LOW_CONFIDENCE_COLOR = "#ff7a7a"


class TranscriptView:
    """Owns the transcript text widget plus the current result and speaker names."""

    def __init__(
        self,
        parent: tk.Misc,
        root: tk.Misc,
        blank_lines_var: tk.BooleanVar,
        on_change: Callable[[], None],
        highlight_var: Optional[tk.BooleanVar] = None,
        on_play: Optional[Callable[[float, float], None]] = None,
    ) -> None:
        self.root = root
        self.blank_lines_var = blank_lines_var
        # When set/enabled, low-confidence words (or segments) are colored.
        self.highlight_var = highlight_var
        # Called after an edit mutates the result, so the owner can re-save outputs.
        self._on_change = on_change
        # Play a [start, end] span of the source audio; None disables playback.
        self._on_play = on_play
        self._result: Optional[TranscriptionResult] = None
        self._speaker_names: Dict[str, str] = {}
        self.widget = ScrolledText(
            parent, wrap="word", state="disabled", height=14, font="TkFixedFont"
        )
        # Right-click: move a highlighted span of words to another speaker (for the
        # case where a sentence mid-line belongs to someone else). Button-3 on
        # Windows/Linux, Button-2 on macOS.
        self.widget.bind("<Button-3>", self._selection_menu)
        self.widget.bind("<Button-2>", self._selection_menu)

    def set_result(
        self, result: Optional[TranscriptionResult], speaker_names: Dict[str, str]
    ) -> None:
        """Show ``result`` (sharing the ``speaker_names`` dict with the owner)."""
        self._result = result
        self._speaker_names = speaker_names
        self.render()

    def get_text(self) -> str:
        return self.widget.get("1.0", "end-1c")

    def render(self) -> None:
        """Render the current result, with clickable speaker tags / words."""

        highlight = self.highlight_var is not None and self.highlight_var.get()

        def _lowconf_seg(segment) -> tuple:
            return (
                ("lowconf",) if highlight and is_low_confidence_segment(segment) else ()
            )

        def _do() -> None:
            result = self._result
            self.widget.configure(state="normal")
            self.widget.delete("1.0", "end")
            self.widget.tag_config("lowconf", foreground=_LOW_CONFIDENCE_COLOR)
            if result is None:
                self.widget.configure(state="disabled")
                return
            # Blank line between segments/turns when enabled (easier to read/paste).
            line_end = "\n\n" if self.blank_lines_var.get() else "\n"
            if not result.has_speakers:
                if result.segments:
                    for index, segment in enumerate(result.segments):
                        if index:
                            self.widget.insert("end", line_end)
                        line_tag = f"line::{index}"
                        tags = (line_tag, *_lowconf_seg(segment))
                        # Ctrl-click (or the menu) plays this line - works the same
                        # whether or not the transcript is diarized.
                        if self._on_play is not None:
                            # ButtonRelease (not press) so a click-drag can select
                            # words instead of immediately popping the menu.
                            self.widget.tag_bind(
                                line_tag,
                                "<ButtonRelease-1>",
                                self._line_menu_handler(index),
                            )
                            self.widget.tag_bind(
                                line_tag,
                                "<Control-Button-1>",
                                self._play_line_handler(index),
                            )
                            self.widget.tag_bind(
                                line_tag, "<Enter>", self._cursor_handler("hand2")
                            )
                            self.widget.tag_bind(
                                line_tag, "<Leave>", self._cursor_handler("")
                            )
                        self.widget.insert("end", segment.text, tags)
                    self.widget.insert("end", "\n")
                else:
                    self.widget.insert("end", result.text + "\n")
            else:
                bound: set[str] = set()
                for index, segment in enumerate(result.segments):
                    sid = segment.speaker or "UNKNOWN"
                    name = self._speaker_names.get(sid, sid)
                    spk_tag = f"spk::{sid}"
                    line_tag = f"line::{index}"
                    if sid not in bound:
                        bound.add(sid)
                        self.widget.tag_config(spk_tag, underline=True)
                    # Bind on a per-line tag so a click knows which segment it hit:
                    # the menu can both fix this one line and rename globally. On
                    # ButtonRelease so a click-drag selects words instead.
                    self.widget.tag_bind(
                        line_tag, "<ButtonRelease-1>", self._speaker_menu_handler(index)
                    )
                    self.widget.tag_bind(
                        line_tag, "<Enter>", self._cursor_handler("hand2")
                    )
                    self.widget.tag_bind(line_tag, "<Leave>", self._cursor_handler(""))
                    if self._on_play is not None:
                        self.widget.tag_bind(
                            line_tag,
                            "<Control-Button-1>",
                            self._play_line_handler(index),
                        )
                    self.widget.insert("end", f"[{name}]", (spk_tag, line_tag))
                    if segment.words:
                        # Render words individually so a single misattributed word
                        # can be clicked and moved to another speaker.
                        for w_index, word in enumerate(segment.words):
                            text = word.word
                            if w_index == 0 and not text[:1].isspace():
                                text = " " + text
                            wtag = f"word::{index}::{w_index}"
                            self.widget.tag_bind(
                                wtag,
                                "<ButtonRelease-1>",
                                self._word_menu_handler(index, w_index),
                            )
                            self.widget.tag_bind(
                                wtag, "<Enter>", self._cursor_handler("hand2")
                            )
                            self.widget.tag_bind(
                                wtag, "<Leave>", self._cursor_handler("")
                            )
                            if self._on_play is not None:
                                self.widget.tag_bind(
                                    wtag,
                                    "<Control-Button-1>",
                                    self._play_word_handler(index, w_index),
                                )
                            wtags: tuple[str, ...] = (wtag,)
                            if highlight and is_low_confidence_word(word):
                                wtags = (wtag, "lowconf")
                            self.widget.insert("end", text, wtags)
                        self.widget.insert("end", line_end)
                    else:
                        self.widget.insert(
                            "end", f" {segment.text}", _lowconf_seg(segment)
                        )
                        self.widget.insert("end", line_end)
            self.widget.configure(state="disabled")

        self.root.after(0, _do)

    def _changed(self) -> None:
        """Re-render after an edit and notify the owner to persist the change."""
        self.render()
        self._on_change()

    def _cursor_handler(self, cursor: str) -> Callable[[object], None]:
        def handler(_event: object) -> None:
            self.widget.config(cursor=cursor)

        return handler

    def _ordered_speaker_ids(self) -> List[str]:
        """Distinct speaker ids in first-appearance order across the result."""
        ids: List[str] = []
        seen: set[str] = set()
        if self._result is not None:
            for segment in self._result.segments:
                sid = segment.speaker or "UNKNOWN"
                if sid not in seen:
                    seen.add(sid)
                    ids.append(sid)
        return ids

    def _speaker_menu_handler(self, index: int) -> Callable[[object], None]:
        def handler(event: object) -> None:
            if self._has_selection():
                return  # a drag-select; leave it for right-click "Move selection"
            result = self._result
            if result is None or index >= len(result.segments):
                return
            current = result.segments[index].speaker or "UNKNOWN"
            menu = tk.Menu(self.root, tearoff=0)
            if self._on_play is not None:
                menu.add_command(
                    label="▶ Play this line",
                    command=lambda: self._play_line(index),
                )
                menu.add_separator()
            # Reassign just this line to the correct speaker - the fix for the
            # boundary/overlap errors diarization can't get right on its own.
            for sid in self._ordered_speaker_ids():
                name = self._speaker_names.get(sid, sid)
                mark = "  ✓" if sid == current else ""
                menu.add_command(
                    label=f"This line is {name}{mark}",
                    command=self._reassign_command(index, sid),
                )
            menu.add_separator()
            cur_name = self._speaker_names.get(current, current)
            menu.add_command(
                label=f"Rename '{cur_name}' everywhere…",
                command=lambda: self._rename_speaker(current),
            )
            try:
                menu.tk_popup(event.x_root, event.y_root)  # type: ignore[attr-defined]
            finally:
                menu.grab_release()

        return handler

    def _reassign_command(self, index: int, speaker_id: str) -> Callable[[], None]:
        def command() -> None:
            self._reassign_segment(index, speaker_id)

        return command

    def _reassign_segment(self, index: int, speaker_id: str) -> None:
        result = self._result
        if result is None or index >= len(result.segments):
            return
        result.segments[index].speaker = speaker_id
        self._changed()

    # -- Playback ----------------------------------------------------------

    def _line_menu_handler(self, index: int) -> Callable[[object], None]:
        """Single-click menu for a non-diarized line (just Play)."""

        def handler(event: object) -> None:
            if self._on_play is None or self._has_selection():
                return
            menu = tk.Menu(self.root, tearoff=0)
            menu.add_command(
                label="▶ Play this line", command=lambda: self._play_line(index)
            )
            try:
                menu.tk_popup(event.x_root, event.y_root)  # type: ignore[attr-defined]
            finally:
                menu.grab_release()

        return handler

    def _play_line_handler(self, index: int) -> Callable[[object], str]:
        def handler(_event: object) -> str:
            self._play_line(index)
            return "break"  # don't also fire the speaker-menu single-click

        return handler

    def _play_word_handler(
        self, seg_index: int, word_index: int
    ) -> Callable[[object], str]:
        def handler(_event: object) -> str:
            self._play_from_word(seg_index, word_index)
            return "break"

        return handler

    def _play_line(self, index: int) -> None:
        result = self._result
        if self._on_play is None or result is None or index >= len(result.segments):
            return
        segment = result.segments[index]
        self._on_play(segment.start, segment.end)

    def _play_from_word(self, seg_index: int, word_index: int) -> None:
        result = self._result
        if self._on_play is None or result is None or seg_index >= len(result.segments):
            return
        segment = result.segments[seg_index]
        if word_index >= len(segment.words):
            return
        self._on_play(segment.words[word_index].start, segment.end)

    def _word_menu_handler(
        self, seg_index: int, word_index: int
    ) -> Callable[[object], None]:
        def handler(event: object) -> None:
            if self._has_selection():
                return  # a drag-select; leave it for right-click "Move selection"
            result = self._result
            if result is None or seg_index >= len(result.segments):
                return
            segment = result.segments[seg_index]
            if word_index >= len(segment.words):
                return
            current = segment.speaker or "UNKNOWN"
            word_text = segment.words[word_index].word.strip()
            menu = tk.Menu(self.root, tearoff=0)
            if self._on_play is not None:
                menu.add_command(
                    label="▶ Play from here",
                    command=lambda: self._play_from_word(seg_index, word_index),
                )
                menu.add_separator()
            # Move just this word (splits the segment around it).
            for sid in self._ordered_speaker_ids():
                if sid == current:
                    continue
                name = self._speaker_names.get(sid, sid)
                menu.add_command(
                    label=f"Move “{word_text}” → {name}",
                    command=self._word_command(seg_index, word_index, sid, False),
                )
            menu.add_separator()
            # Reassign from this word to the end of the line (the common case: a
            # new speaker's turn actually starts mid-line).
            for sid in self._ordered_speaker_ids():
                if sid == current:
                    continue
                name = self._speaker_names.get(sid, sid)
                menu.add_command(
                    label=f"From “{word_text}” onward → {name}",
                    command=self._word_command(seg_index, word_index, sid, True),
                )
            try:
                menu.tk_popup(event.x_root, event.y_root)  # type: ignore[attr-defined]
            finally:
                menu.grab_release()

        return handler

    def _word_command(
        self, seg_index: int, word_index: int, speaker_id: str, to_end: bool
    ) -> Callable[[], None]:
        def command() -> None:
            self._reassign_word_span(seg_index, word_index, speaker_id, to_end)

        return command

    def _reassign_word_span(
        self, seg_index: int, word_index: int, speaker_id: str, to_end: bool
    ) -> None:
        result = self._result
        if result is None or seg_index >= len(result.segments):
            return
        segment = result.segments[seg_index]
        if word_index >= len(segment.words):
            return
        parts = split_segment_on_word(segment, word_index, speaker_id, to_end=to_end)
        result.segments[seg_index : seg_index + 1] = parts
        result.segments = coalesce_segments(result.segments)
        self._changed()

    # -- Move a highlighted span of words ----------------------------------

    def _has_selection(self) -> bool:
        """True if a non-empty text selection exists (i.e. the user drag-selected).

        A plain click clears the selection on press, so by ButtonRelease this is
        only true after a drag - letting us suppress the per-word/line menu and
        leave the selection for a right-click "Move selection".
        """
        try:
            return self.widget.index("sel.first") != self.widget.index("sel.last")
        except tk.TclError:
            return False

    def _selected_word_span(self) -> "Optional[tuple[int, int, int]]":
        """Map the current text selection to ``(seg_index, first_word, last_word)``.

        Returns ``None`` unless the selection covers a contiguous run of words
        within a single segment (the supported case: a sentence mid-line).
        """
        result = self._result
        if result is None:
            return None
        try:
            sel_first = self.widget.index("sel.first")
            sel_last = self.widget.index("sel.last")
        except tk.TclError:
            return None  # nothing selected
        hits: List[tuple[int, int]] = []
        for s_idx, segment in enumerate(result.segments):
            for w_idx in range(len(segment.words)):
                ranges = self.widget.tag_ranges(f"word::{s_idx}::{w_idx}")
                if not ranges:
                    continue
                w_start, w_end = str(ranges[0]), str(ranges[1])
                # Overlap test between the word's range and the selection.
                if self.widget.compare(w_start, "<", sel_last) and self.widget.compare(
                    w_end, ">", sel_first
                ):
                    hits.append((s_idx, w_idx))
        if not hits:
            return None
        if len({s for s, _ in hits}) != 1:
            return None  # selection spans multiple segments - not supported
        s_idx = hits[0][0]
        w_idxs = sorted(w for _, w in hits)
        if w_idxs != list(range(w_idxs[0], w_idxs[-1] + 1)):
            return None  # non-contiguous (shouldn't happen for a normal drag)
        return (s_idx, w_idxs[0], w_idxs[-1])

    def _selection_menu(self, event: object) -> None:
        result = self._result
        span = self._selected_word_span()
        if result is None or span is None:
            return
        seg_index, first_word, last_word = span
        current = result.segments[seg_index].speaker or "UNKNOWN"
        targets = [sid for sid in self._ordered_speaker_ids() if sid != current]
        if not targets:
            return
        menu = tk.Menu(self.root, tearoff=0)
        for sid in targets:
            name = self._speaker_names.get(sid, sid)
            menu.add_command(
                label=f"Move selection → {name}",
                command=self._span_command(seg_index, first_word, last_word, sid),
            )
        try:
            menu.tk_popup(event.x_root, event.y_root)  # type: ignore[attr-defined]
        finally:
            menu.grab_release()

    def _span_command(
        self, seg_index: int, first_word: int, last_word: int, speaker_id: str
    ) -> Callable[[], None]:
        def command() -> None:
            self._reassign_span(seg_index, first_word, last_word, speaker_id)

        return command

    def _reassign_span(
        self, seg_index: int, first_word: int, last_word: int, speaker_id: str
    ) -> None:
        result = self._result
        if result is None or seg_index >= len(result.segments):
            return
        segment = result.segments[seg_index]
        parts = split_segment_on_span(segment, first_word, last_word, speaker_id)
        result.segments[seg_index : seg_index + 1] = parts
        result.segments = coalesce_segments(result.segments)
        self._changed()

    def _rename_speaker(self, speaker_id: str) -> None:
        current = self._speaker_names.get(speaker_id, speaker_id)
        new_name = simpledialog.askstring(
            "Rename speaker",
            f"New name for {current}:",
            initialvalue=current,
            parent=self.root,
        )
        if not new_name or not new_name.strip():
            return
        self._speaker_names[speaker_id] = new_name.strip()
        self._changed()
