"""What a finished transcription run leaves on screen.

One banner, not two. A run that lost most of a recording to the silence filter
still finished, so "complete" is true - but true on its own is the wrong thing
to leave in front of an operator, and a warning announced a line before a
success banner is a warning nobody sees, because the second replaces the
first. So completion and the caveat are one message, or there is no caveat.

Kept out of the GUI so the wording can be tested without a display.
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional, Sequence, Tuple


class SkippedRun(NamedTuple):
    """One recording the silence filter removed most of, and by how much."""

    name: str
    minutes: float
    kept: float


class FailedRun(NamedTuple):
    """One recording that could not be transcribed, and the plain reason why."""

    name: str
    reason: str


# How little of a recording has to reach the model before the operator is told.
# Long dead air is exactly what silence skipping is for, so this is deliberately
# not sensitive: it exists to catch the filter eating speech, not to comment on
# a quiet recording.
LOW_KEPT_FRACTION = 0.35


def much_was_skipped(kept_fraction: "Optional[float]") -> bool:
    """True when so little reached the model that somebody should be told."""
    return kept_fraction is not None and kept_fraction < LOW_KEPT_FRACTION


def per_file_note(run: SkippedRun) -> str:
    """The Status line for one recording, named - a batch needs the name."""
    return (
        f"{run.name}: skip silence passed over {run.minutes:.1f} min — "
        f"only {run.kept * 100:.0f}% of it was transcribed."
    )


def completion(
    done: int,
    total: int,
    skipped: "Sequence[SkippedRun]",
    missing: "Sequence[str]" = (),
    failed: "Sequence[FailedRun]" = (),
    unsaved: "Sequence[FailedRun]" = (),
    saved: "Optional[int]" = None,
) -> "Tuple[str, str]":
    """The banner a finished run leaves behind: its kind, and its words.

    A run that could not find its recordings did not succeed, whatever the
    count says. "Transcription complete." over a file that was never opened is
    the worst thing this banner could say, so a missing or failed file makes it
    amber and names what was not done.

    Order matters when several are true at once. Not transcribing a recording
    outranks transcribing one badly: a caveat about the silence filter can wait
    for the run where every recording actually ran.
    """
    if failed or missing or unsaved:
        return _shortfall(done, total, missing, failed, unsaved, saved)
    if not skipped:
        message = (
            f"Transcription complete — {done} recording(s)."
            if total > 1
            else "Transcription complete."
        )
        return "success", message
    if len(skipped) == 1:
        run = skipped[0]
        return (
            "warning",
            f"Transcription complete — but skip silence passed over "
            f"{run.minutes:.1f} min of {run.name}; only {run.kept * 100:.0f}% "
            "of it was transcribed. If speech is missing, turn off “Skip "
            "silence” in Advanced options and run it again.",
        )
    names = ", ".join(run.name for run in skipped)
    return (
        "warning",
        f"Transcription complete — but skip silence passed over most of "
        f"{len(skipped)} recordings ({names}). If speech is missing from them, "
        "turn off “Skip silence” in Advanced options and run them again.",
    )


def _shortfall(
    done: int,
    total: int,
    missing: "Sequence[str]",
    failed: "Sequence[FailedRun]",
    unsaved: "Sequence[FailedRun]" = (),
    saved: "Optional[int]" = None,
) -> "Tuple[str, str]":
    """The banner for a run that did not get through everything it was given."""
    reasons: List[str] = []
    if missing:
        names = ", ".join(missing)
        reasons.append(
            f"{names} could not be found"
            if len(missing) == 1
            else f"{len(missing)} could not be found ({names})"
        )
    if failed:
        reasons.append(_why_failed(failed))
    if unsaved:
        reasons.append(_why_unsaved(unsaved))
    detail = "; ".join(reasons)
    written = done if saved is None else saved
    if done == 0:
        # Not transcribing anything is the stronger fact, and the one to lead
        # with; nothing was saved either, but that is a consequence of it.
        return "warning", f"Nothing was transcribed — {detail}."
    if written == 0:
        # Transcribing and writing none of it is not a partial success: there
        # is nothing to go and look at afterwards.
        return "warning", f"Nothing was saved — {detail}."
    tally = f"Transcribed {done} of {total} recordings"
    if unsaved:
        tally = f"Transcribed {done} of {total} recordings and saved {written}"
    # Only claim the remainder is safely on disk when it is. With failed saves
    # in the same run that sentence would be contradicting the one before it.
    closing = (
        "The Status tab lists them."
        if unsaved
        else "The Status tab lists them; the rest were transcribed and saved."
    )
    return "warning", f"{tally} — {detail}. {closing}"


def _why_unsaved(unsaved: "Sequence[FailedRun]") -> str:
    """One phrase for output that could not be written.

    Kept apart from a transcription failure because the remedy is different:
    the recording was fine and the destination was not, so the work is not
    lost, it is just not where it was asked to go.
    """
    names = ", ".join(run.name for run in unsaved)
    reasons = {run.reason for run in unsaved}
    if len(reasons) == 1:
        only = next(iter(reasons))
        if len(unsaved) == 1:
            return f"the output for {names} could not be written ({only})"
        return (
            f"the output for {len(unsaved)} could not be written — "
            f"all for the same reason: {only}"
        )
    return f"the output for {len(unsaved)} could not be written ({names})"


def _why_failed(failed: "Sequence[FailedRun]") -> str:
    """One phrase for the failures - grouped, because one cause is one problem.

    A model that is not in the build fails every recording in the folder for
    the same reason. Fifty lines saying so is not fifty problems, and an
    operator reading a wall of them learns less than one sentence would.
    """
    names = ", ".join(run.name for run in failed)
    reasons = {run.reason for run in failed}
    if len(reasons) == 1:
        only = next(iter(reasons))
        if len(failed) == 1:
            return f"{names} could not be transcribed ({only})"
        return (
            f"{len(failed)} could not be transcribed — all for the same reason: {only}"
        )
    return f"{len(failed)} could not be transcribed ({names})"


__all__ = [
    "LOW_KEPT_FRACTION",
    "FailedRun",
    "SkippedRun",
    "completion",
    "much_was_skipped",
    "per_file_note",
]
