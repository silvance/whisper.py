"""Near-live transcription of a stream (RTMP/UDP/HTTP/device/file) via ffmpeg.

faster-whisper is not a streaming recogniser, so this does *chunked* near-live
transcription: ffmpeg reads the incoming stream and splits its audio into short
fixed-length WAV segments; each finished segment is transcribed with
:func:`whispr.transcription.transcribe_audio` and appended. End-to-end latency is
roughly one segment length plus the model's inference time, so a small model
(``base.en`` / ``small.en``) on CPU stays a few seconds behind live.

The engine runs on background threads and reports results through callbacks, so a
GUI can start/stop it and update a transcript as text arrives. It has no GUI or
optional-dependency imports itself.
"""

from __future__ import annotations

import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Callable, List, Optional, Union

from .resources import find_ffmpeg
from .transcription import transcribe_audio

PathLike = Union[str, Path]

# Segments are emitted as 16 kHz mono WAV (what the models expect).
LIVE_SAMPLE_RATE = 16000
# Default chunk length: short enough to feel live, long enough for coherent
# recognition and to amortise per-segment model overhead.
DEFAULT_SEGMENT_SECONDS = 8
# How much prior text to feed back as the decoder prompt for continuity across
# segment boundaries (bounded so the prompt stays small).
_CONTEXT_CHARS = 200


class LiveTranscriptionError(RuntimeError):
    """Raised when the stream can't be opened or ffmpeg fails to run."""


def test_connection(
    source: str,
    *,
    listen: bool = False,
    probe_seconds: int = 3,
    timeout: float = 12.0,
) -> "tuple[bool, str]":
    """Quick reachability check for a stream, before committing to a session.

    Runs ffmpeg for a few seconds against ``source`` (discarding output) and
    reports ``(ok, message)``. Success means audio actually came through; failure
    returns a plain-language reason (bad URL, nothing connected in time, etc.).
    Blocks up to ``timeout`` seconds, so call it off the UI thread.
    """
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        return False, "ffmpeg was not found, so the stream can't be tested."
    args: List[str] = [str(ffmpeg), "-hide_banner", "-loglevel", "error"]
    if listen:
        args += ["-listen", "1"]
    args += ["-i", source, "-t", str(max(1, probe_seconds)), "-f", "null", "-"]
    try:
        result = subprocess.run(args, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, (
            "No data within the timeout. If this PC hosts the stream, tick "
            "'Wait for an incoming connection' and make sure the source is "
            "pushing; otherwise check the URL."
        )
    except OSError as exc:
        return False, f"Couldn't run ffmpeg: {exc}"
    if result.returncode == 0:
        return True, "Stream reached — audio is coming through."
    lines = result.stderr.decode("utf-8", "replace").strip().splitlines()
    detail = lines[-1] if lines else f"ffmpeg exited with code {result.returncode}."
    return False, f"Couldn't read the stream: {detail}"


class LiveTranscriber:
    """Transcribes a live stream in near-real-time, one short segment at a time.

    Parameters
    ----------
    source
        Anything ffmpeg can open as input: an ``rtmp://`` / ``udp://`` /
        ``http://`` URL, a device, or a file. For a GoPro pushing RTMP to this PC
        via a local RTMP server, pass that server's URL (e.g.
        ``rtmp://localhost/live/stream``).
    model_size
        A bundled model name, a size, or a path to a CTranslate2 model directory
        (same as :func:`whispr.transcription.transcribe_audio`). Prefer a small
        model (``base.en`` / ``small.en``) so CPU inference keeps up.
    listen
        When True, ffmpeg *waits for an incoming connection* (``-listen 1``)
        instead of connecting out - use it when the source pushes directly to a
        URL this machine hosts.
    """

    def __init__(
        self,
        source: str,
        *,
        model_size: str = "base.en",
        language: Optional[str] = None,
        segment_seconds: int = DEFAULT_SEGMENT_SECONDS,
        vad_filter: bool = True,
        listen: bool = False,
    ) -> None:
        self._source = source
        self._model_size = model_size
        self._language = language
        self._segment_seconds = max(2, int(segment_seconds))
        self._vad_filter = vad_filter
        self._listen = listen

        self._stop = threading.Event()
        self._proc: Optional[subprocess.Popen] = None
        self._tmpdir: Optional[Path] = None
        self._threads: List[threading.Thread] = []
        self._stderr: List[str] = []

    # -- Lifecycle ---------------------------------------------------------

    def start(
        self,
        *,
        on_text: Callable[[str], None],
        on_status: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> None:
        """Begin capturing and transcribing. Returns immediately (runs in threads).

        ``on_text`` is called with each finished segment's transcript (non-empty),
        ``on_status`` with progress notes, and ``on_error`` if the stream fails.
        """
        ffmpeg = find_ffmpeg()
        if ffmpeg is None:
            raise LiveTranscriptionError(
                "ffmpeg was not found, so a live stream can't be read."
            )
        self._stop.clear()
        self._tmpdir = Path(tempfile.mkdtemp(prefix="whispr-live-"))
        pattern = str(self._tmpdir / "seg_%05d.wav")

        args: List[str] = [str(ffmpeg), "-hide_banner", "-loglevel", "error"]
        if self._listen:
            args += ["-listen", "1"]
        args += [
            "-i",
            self._source,
            "-vn",  # ignore video (GoPro sends both); we only need audio
            "-ac",
            "1",
            "-ar",
            str(LIVE_SAMPLE_RATE),
            "-f",
            "segment",
            "-segment_time",
            str(self._segment_seconds),
            "-reset_timestamps",
            "1",
            pattern,
        ]
        if on_status is not None:
            on_status("Connecting to the stream…")
        try:
            self._proc = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise LiveTranscriptionError(f"Couldn't start ffmpeg: {exc}") from exc

        self._threads = [
            threading.Thread(target=self._drain_stderr, daemon=True),
            threading.Thread(
                target=self._run_watch,
                args=(on_text, on_status, on_error),
                daemon=True,
            ),
        ]
        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        """Stop capture and transcription, and clean up (safe to call twice)."""
        self._stop.set()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            # 'q' asks ffmpeg to finish and finalise the current segment cleanly
            # (works on Windows too, where terminate() would truncate it).
            try:
                if proc.stdin:
                    proc.stdin.write(b"q")
                    proc.stdin.flush()
            except OSError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(timeout=10)
        self._cleanup_tmpdir()

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # -- Workers -----------------------------------------------------------

    def _drain_stderr(self) -> None:
        """Collect ffmpeg's stderr so a failure reason is available (and unblocked)."""
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for raw in iter(proc.stderr.readline, b""):
            try:
                self._stderr.append(raw.decode("utf-8", "replace"))
            except Exception:  # noqa: BLE001 - best-effort diagnostics
                pass

    def _run_watch(
        self,
        on_text: Callable[[str], None],
        on_status: Optional[Callable[[str], None]],
        on_error: Optional[Callable[[Exception], None]],
    ) -> None:
        """Watch for finished segments and transcribe them in order."""
        assert self._tmpdir is not None and self._proc is not None
        processed: set = set()
        context = ""
        announced = False
        error_reported = False
        try:
            while True:
                running = self._proc.poll() is None
                segments = sorted(self._tmpdir.glob("seg_*.wav"))
                # While ffmpeg runs, the newest file is still being written - hold
                # it back; once ffmpeg has exited, every remaining file is final.
                ready = segments if not running else segments[:-1]
                for segment in ready:
                    if segment in processed:
                        continue
                    processed.add(segment)
                    if on_status is not None and not announced:
                        announced = True
                        on_status("Transcribing…")
                    try:
                        text = self._transcribe_segment(segment, context)
                    except Exception as exc:  # noqa: BLE001 - surfaced, non-fatal
                        if on_error is not None and not error_reported:
                            error_reported = True
                            on_error(exc)
                        text = ""
                    if text:
                        context = (context + " " + text)[-_CONTEXT_CHARS:]
                        on_text(text)
                    try:
                        segment.unlink()
                    except OSError:
                        pass
                if not running:
                    break
                self._stop.wait(0.3)
        finally:
            rc = self._proc.poll()
            # ffmpeg failed before producing anything - surface why (bad URL, no
            # connection, etc.) rather than finishing silently.
            if (
                rc not in (0, None)
                and not processed
                and on_error is not None
                and not error_reported
            ):
                detail = "".join(self._stderr).strip()
                on_error(
                    LiveTranscriptionError(detail or f"ffmpeg exited with code {rc}.")
                )
            if on_status is not None:
                on_status("Stopped.")

    def _transcribe_segment(self, segment: Path, context: str) -> str:
        """Transcribe one finished segment, primed with the recent transcript."""
        result = transcribe_audio(
            segment,
            model_size=self._model_size,
            language=self._language,
            vad_filter=self._vad_filter,
            word_timestamps=False,
            initial_prompt=context or None,
            cancelled=self._stop.is_set,
        )
        return result.text.strip()

    def _cleanup_tmpdir(self) -> None:
        if self._tmpdir is None:
            return
        for leftover in self._tmpdir.glob("seg_*.wav"):
            try:
                leftover.unlink()
            except OSError:
                pass
        try:
            self._tmpdir.rmdir()
        except OSError:
            pass
        self._tmpdir = None
