"""The Translate tab: offline foreign->English text translation, with OCR.

Paste text or batch files (translated to ``<name>.en.<ext>`` beside each
original); images and PDFs are OCR'd first when the OCR engine is bundled. The
source language can be chosen explicitly or auto-detected.
"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Callable, List, Optional

from ..export import text_to_docx
from ..ocr import OCR_EXTENSIONS, extract_text, is_ocr_file, tesseract_lang
from ..transcription import CancelledError
from ..translation import detect_language
from .errors import friendly_error
from .widgets import bind_wheel, register_drop, scrollable_body

# Sentinel shown in the "From" dropdown for automatic language detection.
AUTO_DETECT_LABEL = "Auto-detect language"
AUTO_DETECT_CODE = "__auto__"
# Lets users OCR an English document by picking English explicitly. There is no
# English->English translation, so this is extract-only (the OCR'd text is the
# output). Mapped to the target code "en".
ENGLISH_LABEL = "English (OCR only, no translation)"
# OCR needs a script up front; when "From" is Auto-detect we default to Latin/
# English so the common case (English/Latin documents) works without a manual pick.
DEFAULT_OCR_LANG = "eng"


class TranslateTab:
    """Builds and drives the Translate tab inside ``parent``."""

    def __init__(
        self,
        parent: ttk.Frame,
        root: tk.Misc,
        cancel_event: threading.Event,
        on_cancel: Callable[[], None],
        *,
        ocr_available: bool,
        detect_available: bool,
        dnd_ok: bool,
    ) -> None:
        self.parent = parent
        self.root = root
        self._cancel_event = cancel_event
        self._on_cancel = on_cancel
        self._ocr_available = ocr_available
        self._detect_available = detect_available
        self._dnd_ok = dnd_ok
        self._translate_lang_codes: dict[str, str] = {}
        self._build()

    # -- UI construction ---------------------------------------------------

    def _build(self) -> None:
        translate_canvas, container = scrollable_body(self.parent)

        # --- Languages -----------------------------------------------------
        lang_frame = ttk.LabelFrame(container, text="Languages", padding=10)
        lang_frame.pack(fill="x")
        lang_frame.columnconfigure(1, weight=1)
        ttk.Label(lang_frame, text="From").grid(
            row=0, column=0, sticky="w", padx=(0, 8), pady=4
        )
        self.translate_from_var = tk.StringVar()
        self.translate_from_combo = ttk.Combobox(
            lang_frame, textvariable=self.translate_from_var, state="readonly", width=28
        )
        self.translate_from_combo.grid(row=0, column=1, sticky="w", pady=4)
        ttk.Button(lang_frame, text="Refresh", command=self._refresh_languages).grid(
            row=0, column=2, padx=(8, 0), pady=4
        )
        ttk.Label(lang_frame, text="To").grid(
            row=1, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Label(lang_frame, text="English").grid(row=1, column=1, sticky="w", pady=4)
        self.translate_hint_var = tk.StringVar()
        ttk.Label(
            lang_frame,
            textvariable=self.translate_hint_var,
            wraplength=460,
            justify="left",
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 0))

        # --- Paste box -----------------------------------------------------
        paste_frame = ttk.LabelFrame(container, text="Translate text", padding=10)
        paste_frame.pack(fill="both", expand=True, pady=(10, 0))
        ttk.Label(paste_frame, text="Paste text to translate:").pack(anchor="w")
        self.translate_input = ScrolledText(
            paste_frame, wrap="word", height=6, font="TkFixedFont"
        )
        self.translate_input.pack(fill="both", expand=True, pady=(2, 6))
        paste_buttons = ttk.Frame(paste_frame)
        paste_buttons.pack(fill="x")
        ttk.Button(
            paste_buttons, text="Translate", command=self._translate_paste_in_thread
        ).pack(side="left")
        if self._ocr_available:
            # OCR a single image/PDF into the box so its text can be reviewed and
            # corrected before translating (OCR is rarely perfect).
            ttk.Button(
                paste_buttons,
                text="Extract from image/PDF…",
                command=self._extract_to_paste,
            ).pack(side="left", padx=(8, 0))
        result_header = ttk.Frame(paste_frame)
        result_header.pack(fill="x", pady=(6, 0))
        ttk.Label(result_header, text="Result:").pack(side="left")
        ttk.Button(result_header, text="Copy", command=self._copy_translation).pack(
            side="right"
        )
        ttk.Button(
            result_header, text="Save as Word…", command=self._save_translation_docx
        ).pack(side="right", padx=(0, 8))
        self.translate_output = ScrolledText(
            paste_frame, wrap="word", height=6, state="disabled", font="TkFixedFont"
        )
        self.translate_output.pack(fill="both", expand=True, pady=(2, 0))
        # Drop an image/PDF (or text file) onto the paste box to extract its text.
        register_drop(
            self.root, self._dnd_ok, self.translate_input, self._on_drop_source
        )

        # --- Batch files ---------------------------------------------------
        batch_frame = ttk.LabelFrame(
            container, text="Translate files (batch)", padding=10
        )
        batch_frame.pack(fill="x", pady=(10, 0))
        self._translate_files: List[Path] = []
        self.translate_files_var = tk.StringVar(value="No files selected.")
        row = ttk.Frame(batch_frame)
        row.pack(fill="x")
        ttk.Button(row, text="Add files…", command=self._add_translate_files).pack(
            side="left"
        )
        ttk.Button(row, text="Clear", command=self._clear_translate_files).pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(
            row, text="Translate files", command=self._translate_files_in_thread
        ).pack(side="left", padx=(8, 0))
        ttk.Label(
            batch_frame,
            textvariable=self.translate_files_var,
            wraplength=460,
            justify="left",
        ).pack(anchor="w", pady=(6, 0))
        batch_help = "Each file is translated to <name>.en.<ext> next to the original."
        if self._ocr_available:
            batch_help += (
                " Images and PDFs are OCR'd first (the extracted text is also saved "
                "as <name>.ocr.txt)."
            )
        ttk.Label(
            batch_frame,
            text=batch_help,
            wraplength=460,
            justify="left",
        ).pack(anchor="w")
        # Drop files onto the batch box to add them to the queue.
        register_drop(self.root, self._dnd_ok, batch_frame, self._on_drop_files)

        # --- Run controls + progress --------------------------------------
        run_frame = ttk.Frame(container)
        run_frame.pack(fill="x", pady=(12, 0))
        run_frame.columnconfigure(1, weight=1)
        self.translate_cancel_button = ttk.Button(
            run_frame, text="Cancel", command=self._on_cancel, state="disabled"
        )
        self.translate_cancel_button.grid(row=0, column=0, sticky="w")
        self.translate_progress = ttk.Progressbar(run_frame, mode="determinate")
        self.translate_progress.grid(row=0, column=1, sticky="ew", padx=(10, 0))
        self.translate_status_var = tk.StringVar(value="Idle")
        ttk.Label(run_frame, textvariable=self.translate_status_var).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )

        self._refresh_languages()
        bind_wheel(translate_canvas, container)

    # -- Cancellation ------------------------------------------------------

    def notify_cancelling(self) -> None:
        self.translate_cancel_button.configure(state="disabled")
        self.translate_status_var.set("Cancelling…")

    # -- Languages ---------------------------------------------------------

    def _refresh_languages(self) -> None:
        self.translate_hint_var.set("Loading languages…")
        threading.Thread(target=self._load_languages, daemon=True).start()

    def _load_languages(self) -> None:
        try:
            from ..translation import available_source_languages

            langs = available_source_languages()
            err: Optional[Exception] = None
        except Exception as exc:  # noqa: BLE001 - surfaced to the user as a hint
            langs, err = [], exc

        def _apply() -> None:
            self._translate_lang_codes = {name: code for code, name in langs}
            names = list(self._translate_lang_codes)
            # Offer automatic detection (langdetect) as the first choice.
            if names and self._detect_available:
                self._translate_lang_codes[AUTO_DETECT_LABEL] = AUTO_DETECT_CODE
                names = [AUTO_DETECT_LABEL] + names
            # Offer English explicitly so English images/PDFs can be OCR'd
            # (extract-only; there's no English->English translation).
            if names and self._ocr_available:
                self._translate_lang_codes[ENGLISH_LABEL] = "en"
                names = names + [ENGLISH_LABEL]
            self.translate_from_combo.configure(values=names)
            if names:
                if self.translate_from_var.get() not in names:
                    self.translate_from_var.set(names[0])
                self.translate_hint_var.set("")
            elif err is not None:
                self.translate_hint_var.set(friendly_error(err))
            else:
                self.translate_hint_var.set(
                    "No language packs found. Use a build with bundled packs, or "
                    "install Argos packs."
                )

        self.root.after(0, _apply)

    def _selected_from_code(self) -> Optional[str]:
        return self._translate_lang_codes.get(self.translate_from_var.get())

    def _installed_from_codes(self) -> set[str]:
        """Set of source language codes that have an installed pack to English."""
        # Exclude the auto sentinel and the target ("en"): there's no en->en pack.
        return {
            code
            for code in self._translate_lang_codes.values()
            if code not in (AUTO_DETECT_CODE, "en")
        }

    def _resolve_from_code(self, text: str) -> Optional[str]:
        """Resolve the translation source language for ``text``.

        Returns a concrete foreign language code with an installed pack, or
        ``None`` when there's nothing to translate (no selection, English source,
        auto-detect failed, or the detected language has no bundled pack) - the
        caller reports a friendly message or treats it as extract-only.
        """
        selected = self._selected_from_code()
        if selected == AUTO_DETECT_CODE:
            detected = detect_language(text)
            if detected and detected in self._installed_from_codes():
                return detected
            return None
        # A concrete foreign language; English ("en") is extract-only (no en->en).
        if selected and selected != "en":
            return selected
        return None

    def _ocr_lang_code(self) -> str:
        """Tesseract language for OCR, from the selected 'From' language.

        When a specific 'From' language is chosen we use it; otherwise (Auto-detect)
        we default to Latin/English so the common case - English/Latin-script
        documents - just works. For a non-Latin scan, pick that language first.
        """
        selected = self._selected_from_code()
        if not selected or selected == AUTO_DETECT_CODE:
            return DEFAULT_OCR_LANG
        return tesseract_lang(selected)

    # -- Batch file list ---------------------------------------------------

    def _add_translate_files(self) -> None:
        if self._ocr_available:
            ocr_patterns = " ".join(f"*{ext}" for ext in OCR_EXTENSIONS)
            filetypes = [
                ("Text, images & PDFs", f"*.txt {ocr_patterns}"),
                ("All files", "*.*"),
            ]
        else:
            filetypes = [("Text files", "*.txt"), ("All files", "*.*")]
        paths = filedialog.askopenfilenames(filetypes=filetypes)
        self._add_translate_paths([Path(raw) for raw in paths if raw])

    def _add_translate_paths(self, paths: List[Path]) -> None:
        for path in paths:
            if path not in self._translate_files:
                self._translate_files.append(path)
        self._update_translate_files_label()

    def _clear_translate_files(self) -> None:
        self._translate_files = []
        self._update_translate_files_label()

    def _update_translate_files_label(self) -> None:
        count = len(self._translate_files)
        if not count:
            self.translate_files_var.set("No files selected.")
            return
        names = ", ".join(p.name for p in self._translate_files[:5])
        more = "" if count <= 5 else f" (+{count - 5} more)"
        self.translate_files_var.set(f"{count} file(s): {names}{more}")

    # -- Thread-safe widget updates ---------------------------------------

    def _set_translate_busy(self, busy: bool, message: Optional[str] = None) -> None:
        def _do() -> None:
            self.translate_cancel_button.configure(
                state="normal" if busy else "disabled"
            )
            if not busy:
                self.translate_progress["value"] = 0
            self.translate_status_var.set(message or ("Working…" if busy else "Idle"))

        self.root.after(0, _do)

    def _set_translate_progress(self, fraction: float) -> None:
        pct = max(0.0, min(1.0, fraction)) * 100.0
        self.root.after(0, lambda: self.translate_progress.configure(value=pct))

    def _set_translate_status(self, message: str) -> None:
        self.root.after(0, lambda: self.translate_status_var.set(message))

    def _set_translate_output(self, text: str) -> None:
        def _do() -> None:
            self.translate_output.configure(state="normal")
            self.translate_output.delete("1.0", "end")
            self.translate_output.insert("end", text)
            self.translate_output.configure(state="disabled")

        self.root.after(0, _do)

    def _set_translate_input(self, text: str) -> None:
        def _do() -> None:
            self.translate_input.delete("1.0", "end")
            self.translate_input.insert("end", text)

        self.root.after(0, _do)

    # -- Translate (paste) -------------------------------------------------

    def _translate_paste_in_thread(self) -> None:
        if not self._selected_from_code():
            self.translate_status_var.set("Pick a 'From' language first.")
            return
        text = self.translate_input.get("1.0", "end-1c")
        if not text.strip():
            self.translate_status_var.set("Nothing to translate.")
            return
        self._cancel_event.clear()
        threading.Thread(
            target=self._translate_paste, args=(text,), daemon=True
        ).start()

    def _translate_paste(self, text: str) -> None:
        from ..translation import translate_text

        self._set_translate_busy(True, "Translating…")
        final = "Done"
        try:
            from_code = self._resolve_from_code(text)
            if not from_code:
                if self._selected_from_code() == "en":
                    self._set_translate_output(
                        "That's English already — there's nothing to translate. To "
                        "pull text out of an image or PDF, use 'Extract from "
                        "image/PDF…'."
                    )
                else:
                    self._set_translate_output(
                        "Couldn't determine the source language. Pick a specific "
                        "'From' language (auto-detect found no bundled pack for this "
                        "text)."
                    )
                return
            result = translate_text(
                text,
                from_code=from_code,
                to_code="en",
                on_progress=self._set_translate_progress,
                cancelled=self._cancel_event.is_set,
            )
            self._set_translate_output(result)
        except CancelledError:
            final = "Cancelled"
        except Exception as exc:  # noqa: BLE001
            self._set_translate_output(friendly_error(exc))
            final = "Error"
        finally:
            self._set_translate_busy(False, final)

    # -- Translate (batch files) ------------------------------------------

    def _translate_files_in_thread(self) -> None:
        if not self._selected_from_code():
            self.translate_status_var.set("Pick a 'From' language first.")
            return
        if not self._translate_files:
            self.translate_status_var.set("Add files first.")
            return
        self._cancel_event.clear()
        files = list(self._translate_files)
        threading.Thread(
            target=self._translate_files_worker, args=(files,), daemon=True
        ).start()

    def _translate_files_worker(self, files: List[Path]) -> None:
        from ..translation import translate_text

        self._set_translate_busy(True, "Translating files…")
        final = "Done"
        translated_count = 0
        extracted_count = 0
        skipped = 0
        try:
            total = max(1, len(files))
            for index, src in enumerate(files):
                if self._cancel_event.is_set():
                    raise CancelledError("Translation cancelled.")
                # Get the foreign text: read text files directly, OCR images/PDFs.
                is_ocr = is_ocr_file(src)
                if is_ocr:
                    self._set_translate_status(f"Reading {src.name} (OCR)…")
                    text = extract_text(
                        src,
                        lang=self._ocr_lang_code(),
                        progress=self._set_translate_status,
                        cancelled=self._cancel_event.is_set,
                    )
                    # Always keep the extracted text, even if we can't translate it.
                    ocr_dest = src.with_name(f"{src.stem}.ocr.txt")
                    ocr_dest.write_text(text, encoding="utf-8")
                    extracted_count += 1
                    self._set_translate_status(f"Extracted {ocr_dest.name}")
                    dest = src.with_name(f"{src.stem}.en.txt")
                else:
                    text = src.read_text(encoding="utf-8", errors="replace")
                    dest = src.with_name(f"{src.stem}.en{src.suffix}")

                from_code = self._resolve_from_code(text)
                if not from_code:
                    # OCR files already produced .ocr.txt; text files yield nothing.
                    if is_ocr:
                        self._set_translate_status(
                            f"{src.name}: extracted text only (no foreign language "
                            "to translate)."
                        )
                    else:
                        self._set_translate_status(
                            f"Skipped {src.name}: couldn't determine its language."
                        )
                        skipped += 1
                    self._set_translate_progress((index + 1) / total)
                    continue
                self._set_translate_status(f"Translating {src.name}…")
                translated = translate_text(
                    text,
                    from_code=from_code,
                    to_code="en",
                    cancelled=self._cancel_event.is_set,
                )
                dest.write_text(translated, encoding="utf-8")
                translated_count += 1
                self._set_translate_status(f"Wrote {dest.name}")
                self._set_translate_progress((index + 1) / total)
            parts = []
            if translated_count:
                parts.append(f"translated {translated_count}")
            if extracted_count:
                parts.append(f"extracted {extracted_count}")
            if skipped:
                parts.append(f"skipped {skipped}")
            final = "Done — " + (", ".join(parts) if parts else "nothing to do")
        except CancelledError:
            final = "Cancelled"
        except Exception as exc:  # noqa: BLE001
            # Show the cause in the Result box; the status line is too short and
            # gets overwritten by the final "busy off" message below.
            self._set_translate_output(friendly_error(exc))
            final = "Error — see Result box"
        finally:
            self._set_translate_busy(False, final)

    # -- OCR extract to the paste box --------------------------------------

    def _extract_to_paste(self) -> None:
        """OCR a single image/PDF into the paste box for review before translating."""
        lang = self._ocr_lang_code()
        patterns = " ".join(f"*{ext}" for ext in OCR_EXTENSIONS)
        path = filedialog.askopenfilename(
            title="Choose an image or PDF",
            filetypes=[("Images & PDFs", patterns), ("All files", "*.*")],
        )
        if not path:
            return
        self._cancel_event.clear()
        threading.Thread(
            target=self._extract_to_paste_worker, args=(Path(path), lang), daemon=True
        ).start()

    def _extract_to_paste_worker(self, path: Path, lang: str) -> None:
        self._set_translate_busy(True, f"Reading {path.name} (OCR)…")
        final = "Extracted — review, then Translate"
        try:
            text = extract_text(
                path,
                lang=lang,
                progress=self._set_translate_status,
                cancelled=self._cancel_event.is_set,
            )
            self._set_translate_input(text)
            if not text.strip():
                final = "No text found in that file."
        except CancelledError:
            final = "Cancelled"
        except Exception as exc:  # noqa: BLE001
            self._set_translate_output(friendly_error(exc))
            final = "Error — see Result box"
        finally:
            self._set_translate_busy(False, final)

    # -- Export / clipboard ------------------------------------------------

    def _copy_translation(self) -> None:
        text = self.translate_output.get("1.0", "end-1c")
        if not text.strip():
            self.translate_status_var.set("Nothing to copy yet.")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.translate_status_var.set("Translation copied to clipboard.")

    def _save_translation_docx(self) -> None:
        text = self.translate_output.get("1.0", "end-1c")
        if not text.strip():
            self.translate_status_var.set("Nothing to save yet.")
            return
        path = filedialog.asksaveasfilename(
            title="Save translation as Word",
            defaultextension=".docx",
            initialfile="translation.docx",
            filetypes=[("Word document", "*.docx")],
        )
        if not path:
            return
        try:
            text_to_docx(text, path)
            self.translate_status_var.set(f"Saved {Path(path).name}")
        except Exception as exc:  # noqa: BLE001
            self.translate_status_var.set(friendly_error(exc))

    # -- Drag-and-drop -----------------------------------------------------

    def _on_drop_source(self, paths: List[Path]) -> None:
        """A file dropped on the paste box: OCR an image/PDF, or load a text file."""
        path = paths[0]
        if is_ocr_file(path):
            lang = self._ocr_lang_code()
            self._cancel_event.clear()
            threading.Thread(
                target=self._extract_to_paste_worker, args=(path, lang), daemon=True
            ).start()
        else:
            try:
                self._set_translate_input(
                    path.read_text(encoding="utf-8", errors="replace")
                )
            except OSError as exc:
                self.translate_status_var.set(friendly_error(exc))

    def _on_drop_files(self, paths: List[Path]) -> None:
        """Files dropped on the batch box: add them to the queue."""
        self._add_translate_paths(paths)
        self.translate_status_var.set(f"Added {len(paths)} file(s) to the batch.")
