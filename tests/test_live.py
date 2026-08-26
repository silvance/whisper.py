import io
from pathlib import Path

import pytest

from whispr import live
from whispr.live import LiveTranscriber, LiveTranscriptionError


class _FakePopen:
    """Stands in for a finished ffmpeg process so start()/stop() run without one."""

    def __init__(self, args, **_kwargs):
        self.args = args
        self.stdin = io.BytesIO()
        self.stderr = io.BytesIO(b"")

    def poll(self):
        return 0  # already exited -> the watch thread drains and stops promptly

    def wait(self, timeout=None):
        return 0

    def terminate(self):
        pass

    def kill(self):
        pass


def test_start_without_ffmpeg_raises(monkeypatch):
    monkeypatch.setattr(live, "find_ffmpeg", lambda: None)
    lt = LiveTranscriber("rtmp://localhost/live/stream")
    with pytest.raises(LiveTranscriptionError, match="ffmpeg"):
        lt.start(on_text=lambda _t: None)


def test_segment_seconds_is_clamped():
    lt = LiveTranscriber("rtmp://x", segment_seconds=0)
    assert lt._segment_seconds >= 2


def test_ffmpeg_args_pull_mode(monkeypatch):
    captured = {}

    def _fake_popen(args, **kwargs):
        captured["args"] = args
        return _FakePopen(args, **kwargs)

    monkeypatch.setattr(live, "find_ffmpeg", lambda: Path("ffmpeg"))
    monkeypatch.setattr(live.subprocess, "Popen", _fake_popen)

    lt = LiveTranscriber("rtmp://localhost/live/stream", segment_seconds=8)
    lt.start(on_text=lambda _t: None)
    lt.stop()

    args = captured["args"]
    assert "rtmp://localhost/live/stream" in args
    assert "-vn" in args  # audio only
    assert "-listen" not in args  # pull mode
    # 16 kHz mono segmented output at the requested length
    assert args[args.index("-segment_time") + 1] == "8"
    assert args[args.index("-ar") + 1] == str(live.LIVE_SAMPLE_RATE)
    # temp dir is cleaned up on stop
    assert lt._tmpdir is None


def test_ffmpeg_args_listen_mode(monkeypatch):
    captured = {}

    def _fake_popen(args, **kwargs):
        captured["args"] = args
        return _FakePopen(args, **kwargs)

    monkeypatch.setattr(live, "find_ffmpeg", lambda: Path("ffmpeg"))
    monkeypatch.setattr(live.subprocess, "Popen", _fake_popen)

    lt = LiveTranscriber("rtmp://0.0.0.0/live", listen=True)
    lt.start(on_text=lambda _t: None)
    lt.stop()

    args = captured["args"]
    assert args.index("-listen") < args.index("-i")  # -listen precedes the input
