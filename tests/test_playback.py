from pathlib import Path

import pytest

from whispr import playback
from whispr.playback import (
    PlaybackError,
    SegmentPlayer,
    _player_args,
    playback_available,
)


def test_playback_available_false_without_ffmpeg(monkeypatch):
    monkeypatch.setattr(playback, "find_ffmpeg", lambda: None)
    assert playback_available() is False


def test_playback_available_with_ffmpeg_and_player(monkeypatch):
    monkeypatch.setattr(playback, "find_ffmpeg", lambda: Path("/usr/bin/ffmpeg"))
    monkeypatch.setattr(playback, "_player_command", lambda: "/usr/bin/aplay")
    assert playback_available() is True


def test_play_segment_without_ffmpeg_raises(monkeypatch):
    monkeypatch.setattr(playback, "find_ffmpeg", lambda: None)
    with pytest.raises(PlaybackError, match="ffmpeg"):
        SegmentPlayer().play_segment("clip.wav", 0.0, 1.0)


def test_player_args_ffplay_uses_headless_flags():
    args = _player_args("/usr/bin/ffplay", Path("/tmp/x.wav"))
    assert "-autoexit" in args and "-nodisp" in args
    assert args[-1] == "/tmp/x.wav"


def test_player_args_simple_player_is_command_and_file():
    assert _player_args("/usr/bin/aplay", Path("/tmp/x.wav")) == [
        "/usr/bin/aplay",
        "/tmp/x.wav",
    ]


def test_stop_is_safe_when_nothing_playing():
    # No exception when there's no temp file or process.
    SegmentPlayer().stop()
