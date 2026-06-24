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
from tkinter import filedialog, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Optional

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
        self.srt_var = tk.BooleanVar(value=False)
        self.progress_label_var = tk.StringVar(value="Idle")

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

        self.run_button = ttk.Button(top, text="Run", command=self.run_in_thread)
        self.run_button.grid(row=10, column=1, pady=8, sticky="w")

        ttk.Label(top, textvariable=self.progress_label_var).grid(
            row=11, column=0, sticky="w"
        )
        self.progress_bar = ttk.Progressbar(top, mode="indeterminate", length=420)
        self.progress_bar.grid(row=11, column=1, columnspan=2, sticky="ew", pady=(2, 8))

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
                self.progress_bar.start(12)
                self.progress_label_var.set(message or "Processing...")
            else:
                self.progress_bar.stop()
                self.progress_label_var.set(message or "Idle")
                self.run_button.configure(state="normal")

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

            result = transcribe_audio(
                media_path,
                model_size=model,
                task=task,
                language=language_arg,
                vad_filter=self.vad_var.get(),
                progress=lambda msg: self._append(self.status, msg),
            )

            self._append(
                self.status,
                f"Detected language: {result.language} "
                f"({result.language_probability:.0%}), "
                f"duration: {result.duration:.1f}s",
            )
            self._append(self.output, result.text)

            if self.diarize_var.get():
                self._diarize_into(result, Path(path), media_path, media_is_normalized)

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
                progress=lambda msg: self._append(self.status, msg),
            )
            assign_speakers(result.segments, speaker_segments)
            count = len({seg.speaker for seg in speaker_segments})
            self._append(self.status, f"Identified {count} speaker(s).")
            # Re-render the transcript with speaker labels.
            self._clear(self.output)
            self._append(self.output, result.to_txt())
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
        txt_path = outdir / (source.name + ".txt")
        txt_path.write_text(result.to_txt(), encoding="utf-8")
        self._append(self.status, f"Wrote transcript to {txt_path}")
        if self.srt_var.get():
            srt_path = outdir / (source.name + ".srt")
            srt_path.write_text(result.to_srt(), encoding="utf-8")
            self._append(self.status, f"Wrote subtitles to {srt_path}")


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
