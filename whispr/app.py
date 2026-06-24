"""W.H.I.S.P.R. desktop GUI for audio/video transcription.

A small Tkinter application that wraps :mod:`whispr.transcription` (faster-whisper)
so an operator can pick a recording, choose a model/language, and get a transcript
without touching the command line. Transcription runs on a background thread so the
UI stays responsive, and output is streamed segment-by-segment as it is produced.

Run with ``python -m whispr`` or the ``whispr`` console script.
"""

from __future__ import annotations

import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, simpledialog, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Callable, Dict, Optional

from .diarization import assign_speakers, diarize
from .resources import bundled_models
from .transcription import (
    AUDIO_EXTENSIONS,
    MODEL_SIZES,
    TranscriptionResult,
    convert_to_wav,
    is_video,
    transcribe_audio,
)

# A handful of common languages for the dropdown; "Auto" lets Whisper detect.
COMMON_LANGUAGES = [
    "Auto",
    "en",
    "es",
    "fr",
    "de",
    "it",
    "pt",
    "nl",
    "ru",
    "ar",
    "zh",
    "ja",
    "ko",
    "hi",
]


class WhisprApp:
    """The main application window."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("W.H.I.S.P.R. - Audio Transcription")

        # Bundled (offline) models take priority so the app works air-gapped.
        self._bundled_models = bundled_models()
        default_model = "base"
        if self._bundled_models:
            default_model = (
                "small"
                if "small" in self._bundled_models
                else next(iter(self._bundled_models))
            )

        self.input_file_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()
        self.write_output_var = tk.BooleanVar(value=True)
        self.model_var = tk.StringVar(value=default_model)
        self.task_var = tk.StringVar(value="transcribe")
        self.language_var = tk.StringVar(value="Auto")
        self.vad_var = tk.BooleanVar(value=True)
        self.convert_video_var = tk.BooleanVar(value=True)
        self.diarize_var = tk.BooleanVar(value=False)
        self.num_speakers_var = tk.StringVar(value="")
        self.sensitivity_var = tk.StringVar(value="0.5")
        self.srt_var = tk.BooleanVar(value=False)
        self.progress_label_var = tk.StringVar(value="Idle")

        # State for the last result, so speakers can be renamed after a run.
        self._result: Optional[TranscriptionResult] = None
        self._result_source: Optional[Path] = None
        self._result_outdir: Optional[Path] = None
        self._speaker_names: Dict[str, str] = {}

        self._build_ui()

    # -- UI construction ---------------------------------------------------

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="Audio/Video File:").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.input_file_var, width=70).grid(
            row=0, column=1, sticky="ew"
        )
        ttk.Button(top, text="Browse", command=self.choose_file).grid(row=0, column=2)

        ttk.Checkbutton(
            top, text="Write output to folder", variable=self.write_output_var
        ).grid(row=1, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.output_dir_var, width=70).grid(
            row=1, column=1, sticky="ew"
        )
        ttk.Button(top, text="Select Output Dir", command=self.choose_output_dir).grid(
            row=1, column=2
        )

        # Model: a bundled model name, a size name, or a path to a local
        # CTranslate2 model. Bundled (offline) models are listed first.
        model_values = list(self._bundled_models) + [
            size for size in MODEL_SIZES if size not in self._bundled_models
        ]
        ttk.Label(top, text="Model (size or path):").grid(row=2, column=0, sticky="w")
        ttk.Combobox(
            top,
            textvariable=self.model_var,
            values=model_values,
            width=40,
        ).grid(row=2, column=1, sticky="ew")
        ttk.Button(top, text="Browse Model", command=self.choose_model_dir).grid(
            row=2, column=2
        )

        ttk.Label(top, text="Task:").grid(row=3, column=0, sticky="w")
        ttk.Combobox(
            top,
            textvariable=self.task_var,
            values=["transcribe", "translate"],
            width=20,
            state="readonly",
        ).grid(row=3, column=1, sticky="w")

        ttk.Label(top, text="Language:").grid(row=4, column=0, sticky="w")
        ttk.Combobox(
            top,
            textvariable=self.language_var,
            values=COMMON_LANGUAGES,
            width=20,
        ).grid(row=4, column=1, sticky="w")

        ttk.Checkbutton(
            top, text="Voice activity detection (skip silence)", variable=self.vad_var
        ).grid(row=5, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(
            top,
            text="Convert video to WAV first (ffmpeg)",
            variable=self.convert_video_var,
        ).grid(row=6, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(
            top, text="Also save .srt subtitles", variable=self.srt_var
        ).grid(row=7, column=0, columnspan=2, sticky="w")

        ttk.Checkbutton(
            top,
            text="Identify speakers (diarization)",
            variable=self.diarize_var,
        ).grid(row=8, column=0, columnspan=2, sticky="w")
        ttk.Label(top, text="Number of speakers (blank = auto):").grid(
            row=9, column=0, sticky="w"
        )
        ttk.Entry(top, textvariable=self.num_speakers_var, width=8).grid(
            row=9, column=1, sticky="w"
        )
        ttk.Label(top, text="Speaker sensitivity (higher = fewer, auto only):").grid(
            row=10, column=0, sticky="w"
        )
        ttk.Entry(top, textvariable=self.sensitivity_var, width=8).grid(
            row=10, column=1, sticky="w"
        )

        self.run_button = ttk.Button(top, text="Run", command=self.run_in_thread)
        self.run_button.grid(row=11, column=1, pady=8, sticky="w")

        ttk.Label(top, textvariable=self.progress_label_var).grid(
            row=12, column=0, sticky="w"
        )
        self.progress_bar = ttk.Progressbar(top, mode="indeterminate", length=420)
        self.progress_bar.grid(row=12, column=1, columnspan=2, sticky="ew", pady=(2, 8))

        tabs = ttk.Notebook(self.root)
        tabs.pack(fill="both", expand=True)
        self.output = ScrolledText(tabs, wrap="word", state="disabled")
        self.status = ScrolledText(tabs, wrap="word", state="disabled")
        tabs.add(self.output, text="Transcript")
        tabs.add(self.status, text="Status")

    # -- Thread-safe widget helpers ---------------------------------------

    def _append(self, widget: ScrolledText, text: str) -> None:
        def _do() -> None:
            widget.configure(state="normal")
            widget.insert("end", str(text) + "\n")
            widget.see("end")
            widget.configure(state="disabled")

        widget.after(0, _do)

    def _clear(self, widget: ScrolledText) -> None:
        def _do() -> None:
            widget.configure(state="normal")
            widget.delete("1.0", "end")
            widget.configure(state="disabled")

        widget.after(0, _do)

    def _set_busy(self, busy: bool, message: Optional[str] = None) -> None:
        def _do() -> None:
            if busy:
                self.run_button.configure(state="disabled")
                # Indeterminate while we don't yet have a measurable fraction
                # (setup, ffmpeg conversion, model loading).
                self.progress_bar.configure(mode="indeterminate")
                self.progress_bar.start(12)
                self.progress_label_var.set(message or "Processing...")
            else:
                self.progress_bar.stop()
                self.progress_bar.configure(mode="determinate")
                self.progress_bar["value"] = 0
                self.progress_label_var.set(message or "Idle")
                self.run_button.configure(state="normal")

        self.root.after(0, _do)

    def _set_progress(self, fraction: float, message: str) -> None:
        """Show real progress on a determinate bar (fraction is 0..1)."""
        pct = max(0.0, min(1.0, fraction)) * 100.0

        def _do() -> None:
            self.progress_bar.stop()
            self.progress_bar.configure(mode="determinate")
            self.progress_bar["value"] = pct
            self.progress_label_var.set(f"{message} {pct:.0f}%")

        self.root.after(0, _do)

    # -- Callbacks ---------------------------------------------------------

    def choose_file(self) -> None:
        patterns = " ".join(f"*{ext}" for ext in AUDIO_EXTENSIONS)
        path = filedialog.askopenfilename(
            filetypes=[("Audio/Video", patterns), ("All files", "*.*")]
        )
        if path:
            self.input_file_var.set(path)

    def choose_output_dir(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.output_dir_var.set(path)

    def choose_model_dir(self) -> None:
        path = filedialog.askdirectory(title="Select a CTranslate2 model directory")
        if path:
            self.model_var.set(path)

    def run_in_thread(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        task = self.task_var.get()
        self._set_busy(
            True, "Translating..." if task == "translate" else "Transcribing..."
        )
        temp_wav: Optional[Path] = None
        try:
            self._clear(self.output)
            path = self.input_file_var.get()
            if not path or not Path(path).exists():
                self._append(self.status, f"Input file does not exist: {path}")
                return

            language = self.language_var.get().strip()
            language_arg = None if language in ("", "Auto") else language
            outdir = self.output_dir_var.get() if self.write_output_var.get() else None

            self._append(self.status, f"Processing: {path}")

            # Optionally pre-convert video to WAV with ffmpeg before transcribing.
            media_path = Path(path)
            media_is_normalized = False  # True when media_path is our 16 kHz mono WAV
            if self.convert_video_var.get() and is_video(media_path):
                if outdir and Path(outdir).is_dir():
                    wav_dest: Optional[Path] = Path(outdir) / (media_path.stem + ".wav")
                else:
                    wav_dest = None  # convert to a temp file we clean up afterwards
                media_path = convert_to_wav(
                    media_path,
                    wav_dest,
                    progress=lambda msg: self._append(self.status, msg),
                )
                media_is_normalized = True
                if wav_dest is None:
                    temp_wav = media_path
                self._append(self.status, f"Converted to {media_path}")

            # Resolve a bundled model name to its local directory so we never
            # try to download on an air-gapped machine.
            model_sel = self.model_var.get()
            model = str(self._bundled_models.get(model_sel, model_sel))

            transcribe_label = "Translating" if task == "translate" else "Transcribing"
            result = transcribe_audio(
                media_path,
                model_size=model,
                task=task,
                language=language_arg,
                vad_filter=self.vad_var.get(),
                progress=lambda msg: self._append(self.status, msg),
                on_progress=lambda f: self._set_progress(f, transcribe_label),
            )

            self._append(
                self.status,
                f"Detected language: {result.language} "
                f"({result.language_probability:.0%}), "
                f"duration: {result.duration:.1f}s",
            )

            if self.diarize_var.get():
                self._diarize_into(result, Path(path), media_path, media_is_normalized)

            # Remember the result so speakers can be renamed afterwards.
            self._result = result
            self._result_source = Path(path)
            self._result_outdir = Path(outdir) if outdir else None
            self._speaker_names = {}
            self._render_transcript()

            if outdir:
                self._save_outputs(result, Path(path), Path(outdir))

            self._append(self.status, "Finished.")
        except Exception:
            self._append(self.status, "UNEXPECTED ERROR:")
            self._append(self.status, traceback.format_exc())
        finally:
            if temp_wav is not None:
                try:
                    temp_wav.unlink()
                except OSError:
                    pass
            self._set_busy(False, "Finished")

    def _parse_num_speakers(self) -> Optional[int]:
        raw = self.num_speakers_var.get().strip()
        if not raw:
            return None
        try:
            value = int(raw)
        except ValueError:
            self._append(self.status, f"Ignoring invalid speaker count: {raw!r}")
            return None
        return value if value > 0 else None

    def _parse_threshold(self) -> float:
        raw = self.sensitivity_var.get().strip()
        if not raw:
            return 0.5
        try:
            value = float(raw)
        except ValueError:
            self._append(self.status, f"Ignoring invalid sensitivity: {raw!r}")
            return 0.5
        return min(max(value, 0.05), 1.0)

    def _diarize_into(
        self,
        result: TranscriptionResult,
        source: Path,
        media_path: Path,
        media_is_normalized: bool,
    ) -> None:
        # Diarization needs a 16 kHz mono WAV; convert to a temp file unless the
        # media we already have is one we normalized.
        diar_wav = media_path
        diar_temp: Optional[Path] = None
        if not media_is_normalized:
            self._append(self.status, "Preparing audio for diarization...")
            diar_wav = convert_to_wav(
                source, progress=lambda msg: self._append(self.status, msg)
            )
            diar_temp = diar_wav
        try:
            speaker_segments = diarize(
                diar_wav,
                num_speakers=self._parse_num_speakers(),
                threshold=self._parse_threshold(),
                progress=lambda msg: self._append(self.status, msg),
                on_progress=lambda f: self._set_progress(f, "Identifying speakers"),
            )
            result.segments = assign_speakers(result.segments, speaker_segments)
            count = len({seg.speaker for seg in speaker_segments})
            self._append(self.status, f"Identified {count} speaker(s).")
        finally:
            if diar_temp is not None:
                try:
                    diar_temp.unlink()
                except OSError:
                    pass

    def _save_outputs(
        self, result: TranscriptionResult, source: Path, outdir: Path
    ) -> None:
        if not outdir.is_dir():
            self._append(self.status, f"Output folder does not exist: {outdir}")
            return
        names = self._speaker_names
        txt_path = outdir / (source.name + ".txt")
        txt_path.write_text(result.to_txt(names), encoding="utf-8")
        self._append(self.status, f"Wrote transcript to {txt_path}")
        if self.srt_var.get():
            srt_path = outdir / (source.name + ".srt")
            srt_path.write_text(result.to_srt(names), encoding="utf-8")
            self._append(self.status, f"Wrote subtitles to {srt_path}")

    # -- Transcript rendering + speaker renaming ---------------------------

    def _render_transcript(self) -> None:
        """Render the current result into the Transcript tab. When diarized,
        each speaker tag is clickable to rename that speaker."""

        def _do() -> None:
            result = self._result
            self.output.configure(state="normal")
            self.output.delete("1.0", "end")
            if result is None:
                self.output.configure(state="disabled")
                return
            if not result.has_speakers:
                self.output.insert("end", result.text + "\n")
            else:
                bound: set[str] = set()
                for segment in result.segments:
                    sid = segment.speaker or "UNKNOWN"
                    name = self._speaker_names.get(sid, sid)
                    tag = f"spk::{sid}"
                    if sid not in bound:
                        bound.add(sid)
                        self.output.tag_config(tag, underline=True)
                        self.output.tag_bind(
                            tag, "<Button-1>", self._rename_handler(sid)
                        )
                        self.output.tag_bind(
                            tag, "<Enter>", self._cursor_handler("hand2")
                        )
                        self.output.tag_bind(tag, "<Leave>", self._cursor_handler(""))
                    self.output.insert("end", f"[{name}]", (tag,))
                    self.output.insert("end", f" {segment.text}\n")
            self.output.configure(state="disabled")

        self.root.after(0, _do)

    def _rename_handler(self, speaker_id: str) -> Callable[[object], None]:
        def handler(_event: object) -> None:
            self._rename_speaker(speaker_id)

        return handler

    def _cursor_handler(self, cursor: str) -> Callable[[object], None]:
        def handler(_event: object) -> None:
            self.output.config(cursor=cursor)

        return handler

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
        self._render_transcript()
        # Keep saved files in sync if an output folder was used.
        if self._result is not None and self._result_source and self._result_outdir:
            self._save_outputs(self._result, self._result_source, self._result_outdir)


def main() -> None:
    """Launch the W.H.I.S.P.R. GUI."""
    try:
        # ttkbootstrap gives a modern theme; fall back to stock Tk if absent.
        import ttkbootstrap as tb

        root = tb.Window(themename="darkly")
    except ImportError:
        root = tk.Tk()

    WhisprApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
