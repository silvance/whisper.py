"""Offline playback of a transcript segment's audio.

Lets an analyst click a line to re-listen to exactly that span - essential for
verifying low-quality/covert audio. Fully offline: the segment is extracted from
the source media with the bundled ffmpeg, then played with the OS's built-in
mechanism (``winsound`` on Windows; ``ffplay``/``aplay``/``afplay``/``paplay`` on
other platforms when present). No new runtime dependency.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from shutil import which
from typing import List, Optional, Union

from .resources import find_ffmpeg

PathLike = Union[str, Path]

# Non-Windows CLI players we try, in order. Windows uses the built-in winsound.
_PLAYER_CANDIDATES = ("ffplay", "aplay", "afplay", "paplay")


class PlaybackError(RuntimeError):
    """Raised when a segment can't be extracted or played."""


def _player_command() -> Optional[str]:
    """Path to a CLI audio player on PATH (non-Windows), or ``None``."""
    for name in _PLAYER_CANDIDATES:
        found = which(name)
        if found:
            return found
    return None


def _player_args(command: str, wav: Path) -> List[str]:
    """Build the argv to play ``wav`` with ``command`` (ffplay needs flags)."""
    if "ffplay" in os.path.basename(command).lower():
        return [command, "-autoexit", "-nodisp", "-loglevel", "quiet", str(wav)]
    return [command, str(wav)]


def playback_available() -> bool:
    """True if a segment can be extracted (ffmpeg) and played on this system."""
    if find_ffmpeg() is None:
        return False
    return os.name == "nt" or _player_command() is not None


class SegmentPlayer:
    """Plays one segment at a time, replacing any currently-playing audio."""

    def __init__(self) -> None:
        self._temp: Optional[Path] = None
        self._proc: Optional[subprocess.Popen] = None

    def play_segment(self, source: PathLike, start: float, end: float) -> None:
        """Extract ``[start, end]`` from ``source`` and play it (async)."""
        ffmpeg = find_ffmpeg()
        if ffmpeg is None:
            raise PlaybackError("ffmpeg was not found, so audio can't be played.")
        self.stop()
        duration = max(0.05, end - start)
        handle, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(handle)
        out = Path(tmp)
        # -ss before -i is fast (keyframe) seek - plenty accurate for re-listening.
        result = subprocess.run(
            [
                str(ffmpeg),
                "-y",
                "-ss",
                f"{max(0.0, start):.3f}",
                "-i",
                str(source),
                "-t",
                f"{duration:.3f}",
                "-ac",
                "1",
                "-ar",
                "22050",
                "-acodec",
                "pcm_s16le",
                str(out),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            try:
                out.unlink()
            except OSError:
                pass
            raise PlaybackError(
                result.stderr.strip() or "ffmpeg failed to extract the audio segment."
            )
        self._temp = out
        self._start_playback(out)

    def _start_playback(self, wav: Path) -> None:
        if os.name == "nt":
            import winsound  # type: ignore[import-not-found,unused-ignore]

            winsound.PlaySound(  # type: ignore[attr-defined,unused-ignore]
                str(wav),
                winsound.SND_FILENAME | winsound.SND_ASYNC,  # type: ignore[attr-defined,unused-ignore]
            )
            return
        command = _player_command()
        if command is None:
            raise PlaybackError("No audio player is available on this system.")
        self._proc = subprocess.Popen(
            _player_args(command, wav),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def stop(self) -> None:
        """Stop any playback and clean up the temp file."""
        if os.name == "nt":
            try:
                import winsound  # type: ignore[import-not-found,unused-ignore]

                winsound.PlaySound(  # type: ignore[attr-defined,unused-ignore]
                    None,
                    winsound.SND_PURGE,  # type: ignore[attr-defined,unused-ignore]
                )
            except Exception:  # noqa: BLE001 - best-effort stop
                pass
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:  # noqa: BLE001 - best-effort stop
                pass
            self._proc = None
        if self._temp is not None:
            try:
                self._temp.unlink()
            except OSError:
                pass
            self._temp = None
