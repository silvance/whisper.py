"""The Live tab: near-real-time transcription of a stream (RTMP/UDP/HTTP…).

Point it at a stream ffmpeg can read - e.g. a GoPro pushing RTMP to a local RTMP
server - and it transcribes short segments as they arrive (see
:mod:`whispr.live`). Kept deliberately simple: a URL, a model, Start/Stop, and a
live-updating transcript.
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import filedialog, ttk
from tkinter.scrolledtext import ScrolledText
from typing import IO, Callable, Optional

from ..live import DEFAULT_SEGMENT_SECONDS, LiveTranscriber, test_connection
from ..resources import bundled_models
from ..transcription import MODEL_SIZES
from .errors import friendly_error
from .widgets import append_line, bind_wheel, scrollable_body

_LANGUAGES = ["Auto", "en", "es", "fr", "de", "it", "pt", "ru", "ar", "zh", "ja", "ko"]


class LiveTab:
    """Builds and drives the Live (stream) tab inside ``parent``."""

    def __init__(
        self,
        parent: ttk.Frame,
        root: tk.Misc,
        cancel_event: threading.Event,
        on_cancel: Callable[[], None],
    ) -> None:
        self.parent = parent
        self.root = root
        self._on_cancel = on_cancel

        self._bundled_models = bundled_models()
        default_model = "base.en"
        if self._bundled_models:
            for preferred in ("base.en", "small.en", "small"):
                if preferred in self._bundled_models:
                    default_model = preferred
                    break
            else:
                default_model = next(iter(self._bundled_models))

        self.source_var = tk.StringVar(value="rtmp://localhost/live/stream")
        self.listen_var = tk.BooleanVar(value=False)
        self.model_var = tk.StringVar(value=default_model)
        self.language_var = tk.StringVar(value="Auto")
        self.vad_var = tk.BooleanVar(value=True)
        self.segment_var = tk.StringVar(value=str(DEFAULT_SEGMENT_SECONDS))
        self.save_var = tk.BooleanVar(value=False)
        self.save_path_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Idle")

        self._live: Optional[LiveTranscriber] = None
        self._save_fh: Optional[IO[str]] = None

        self._build()

    # -- UI ----------------------------------------------------------------

    def _build(self) -> None:
        canvas, container = scrollable_body(self.parent)

        settings = ttk.LabelFrame(container, text="Stream", padding=10)
        settings.pack(fill="x")
        settings.columnconfigure(1, weight=1)

        ttk.Label(settings, text="Stream URL").grid(
            row=0, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Entry(settings, textvariable=self.source_var).grid(
            row=0, column=1, columnspan=2, sticky="ew", pady=4
        )
        ttk.Checkbutton(
            settings,
            text="Wait for an incoming connection (this PC hosts the stream)",
            variable=self.listen_var,
        ).grid(row=1, column=1, columnspan=2, sticky="w")
        ttk.Label(
            settings,
            text=(
                "Point this at a stream ffmpeg can read — e.g. a GoPro pushing "
                "RTMP to a local server (rtmp://localhost/live/stream). Tick the "
                "box above only if this PC should listen for the incoming stream "
                "directly."
            ),
            wraplength=460,
            justify="left",
            font=("", 8),
        ).grid(row=2, column=1, columnspan=2, sticky="w", pady=(2, 0))

        model_values = (
            list(self._bundled_models) if self._bundled_models else list(MODEL_SIZES)
        )
        ttk.Label(settings, text="Model").grid(
            row=3, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Combobox(
            settings, textvariable=self.model_var, values=model_values, width=18
        ).grid(row=3, column=1, sticky="w", pady=4)
        ttk.Label(
            settings,
            text="Use a small model (base.en / small.en) so it keeps up with live.",
            font=("", 8),
        ).grid(row=4, column=1, columnspan=2, sticky="w")

        ttk.Label(settings, text="Language").grid(
            row=5, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Combobox(
            settings,
            textvariable=self.language_var,
            values=_LANGUAGES,
            width=10,
        ).grid(row=5, column=1, sticky="w", pady=4)

        ttk.Label(settings, text="Chunk length (seconds)").grid(
            row=6, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Entry(settings, textvariable=self.segment_var, width=8).grid(
            row=6, column=1, sticky="w", pady=4
        )
        ttk.Label(
            settings,
            text="Shorter = more live but choppier; longer = smoother but more lag.",
            font=("", 8),
        ).grid(row=7, column=1, columnspan=2, sticky="w")

        ttk.Checkbutton(
            settings,
            text="Skip silence (voice activity detection)",
            variable=self.vad_var,
        ).grid(row=8, column=1, columnspan=2, sticky="w", pady=(4, 0))

        # Optional: append the live transcript to a file as it comes in.
        save_row = ttk.Frame(settings)
        save_row.grid(row=9, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        save_row.columnconfigure(1, weight=1)
        ttk.Checkbutton(
            save_row,
            text="Save to file",
            variable=self.save_var,
            command=self._update_save_state,
        ).grid(row=0, column=0, sticky="w")
        self.save_entry = ttk.Entry(save_row, textvariable=self.save_path_var)
        self.save_entry.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        self.save_button = ttk.Button(
            save_row, text="Choose…", command=self._choose_save_path
        )
        self.save_button.grid(row=0, column=2, padx=(8, 0))

        # --- Controls ------------------------------------------------------
        controls = ttk.Frame(container)
        controls.pack(fill="x", pady=(12, 0))
        self.test_button = ttk.Button(
            controls, text="Test connection", command=self._test_connection
        )
        self.test_button.pack(side="left")
        self.start_button = ttk.Button(controls, text="▶ Start", command=self._start)
        self.start_button.pack(side="left", padx=(8, 0))
        self.stop_button = ttk.Button(
            controls, text="⏹ Stop", command=self._stop, state="disabled"
        )
        self.stop_button.pack(side="left", padx=(8, 0))
        ttk.Label(controls, textvariable=self.status_var).pack(
            side="left", padx=(12, 0)
        )

        # --- Live transcript ----------------------------------------------
        ttk.Label(container, text="Live transcript").pack(anchor="w", pady=(12, 2))
        self.transcript = ScrolledText(
            container, wrap="word", state="disabled", height=16, font="TkFixedFont"
        )
        self.transcript.pack(fill="both", expand=True)

        actions = ttk.Frame(container)
        actions.pack(fill="x", pady=(6, 0))
        ttk.Button(actions, text="Copy", command=self._copy).pack(side="left")
        ttk.Button(actions, text="Clear", command=self._clear).pack(
            side="left", padx=(8, 0)
        )

        self._update_save_state()
        bind_wheel(canvas, container)

    # -- Save-to-file state ------------------------------------------------

    def _update_save_state(self) -> None:
        state = "normal" if self.save_var.get() else "disabled"
        self.save_entry.configure(state=state)
        self.save_button.configure(state=state)

    def _choose_save_path(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save live transcript",
            defaultextension=".txt",
            initialfile="live-transcript.txt",
            filetypes=[("Text file", "*.txt"), ("All files", "*.*")],
        )
        if path:
            self.save_path_var.set(path)

    # -- Run/stop ----------------------------------------------------------

    def _parse_segment_seconds(self) -> int:
        try:
            return max(2, int(self.segment_var.get().strip()))
        except ValueError:
            return DEFAULT_SEGMENT_SECONDS

    def _test_connection(self) -> None:
        """Probe the stream for a few seconds and report if it's reachable."""
        source = self.source_var.get().strip()
        if not source:
            self.status_var.set("Enter a stream URL first.")
            return
        listen = self.listen_var.get()
        self.test_button.configure(state="disabled")
        self.status_var.set("Testing connection…")

        def _worker() -> None:
            ok, message = test_connection(source, listen=listen)

            def _apply() -> None:
                self.status_var.set(("✓ " if ok else "✗ ") + message)
                # Re-enable only if no session started in the meantime.
                if self._live is None:
                    self.test_button.configure(state="normal")

            self.root.after(0, _apply)

        threading.Thread(target=_worker, daemon=True).start()

    def _start(self) -> None:
        source = self.source_var.get().strip()
        if not source:
            self.status_var.set("Enter a stream URL first.")
            return
        language = self.language_var.get().strip()
        language_arg = None if language in ("", "Auto") else language
        model_sel = self.model_var.get().strip()
        model = str(self._bundled_models.get(model_sel, model_sel))

        # Open the save file (append) if requested.
        self._save_fh = None
        if self.save_var.get() and self.save_path_var.get().strip():
            try:
                self._save_fh = open(
                    self.save_path_var.get().strip(), "a", encoding="utf-8"
                )
            except OSError as exc:
                self.status_var.set(friendly_error(exc))
                return

        self._live = LiveTranscriber(
            source,
            model_size=model,
            language=language_arg,
            segment_seconds=self._parse_segment_seconds(),
            vad_filter=self.vad_var.get(),
            listen=self.listen_var.get(),
        )
        try:
            self._live.start(
                on_text=self._on_text,
                on_status=self._on_status,
                on_error=self._on_error,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            self.status_var.set(friendly_error(exc))
            self._close_save_fh()
            self._live = None
            return
        self.start_button.configure(state="disabled")
        self.test_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_var.set("Listening…")

    def _stop(self) -> None:
        # stop() blocks (it joins the ffmpeg process + threads), so run it off the
        # UI thread and re-enable the controls when it returns.
        live = self._live
        if live is None:
            return
        self.stop_button.configure(state="disabled")
        self.status_var.set("Stopping…")

        def _worker() -> None:
            try:
                live.stop()
            finally:
                self.root.after(0, self._on_stopped)

        threading.Thread(target=_worker, daemon=True).start()

    def _on_stopped(self) -> None:
        self._live = None
        self._close_save_fh()
        self.start_button.configure(state="normal")
        self.test_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.status_var.set("Stopped.")

    # -- Callbacks (from background threads) -------------------------------

    def _on_text(self, text: str) -> None:
        append_line(self.transcript, text)
        fh = self._save_fh
        if fh is not None:
            try:
                fh.write(text + "\n")
                fh.flush()
            except OSError:
                pass

    def _on_status(self, message: str) -> None:
        self.root.after(0, lambda: self.status_var.set(message))

    def _on_error(self, exc: Exception) -> None:
        self.root.after(0, lambda: self.status_var.set(friendly_error(exc)))

    def _close_save_fh(self) -> None:
        if self._save_fh is not None:
            try:
                self._save_fh.close()
            except OSError:
                pass
            self._save_fh = None

    # -- Transcript actions ------------------------------------------------

    def _copy(self) -> None:
        text = self.transcript.get("1.0", "end-1c")
        if not text.strip():
            self.status_var.set("Nothing to copy yet.")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.status_var.set("Transcript copied to clipboard.")

    def _clear(self) -> None:
        self.transcript.configure(state="normal")
        self.transcript.delete("1.0", "end")
        self.transcript.configure(state="disabled")

    # -- Shared-cancel + shutdown hooks ------------------------------------

    def notify_cancelling(self) -> None:
        """The shared Cancel button stops a running live session."""
        if self._live is not None:
            self._stop()

    def close(self) -> None:
        """Release resources when the app closes."""
        live = self._live
        if live is not None:
            try:
                live.stop()
            except Exception:  # noqa: BLE001 - never block closing
                pass
        self._close_save_fh()

    # Kept for API parity with the other tabs (no persisted settings here).
    def get_settings(self) -> "dict[str, object]":
        return {}

    def apply_settings(self, data: "dict[str, object]") -> None:
        return None
