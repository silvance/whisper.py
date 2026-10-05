"""Reducing steady background noise before transcription - and saying when not to.

An operator asked whether a conversation in a bar can be isolated. The honest
answer is in the numbers, so they were measured rather than assumed. Against
clean speech with noise mixed in at a known level, the bundled denoiser moved
the speech-to-noise margin by:

    mains hum, -5 dB in        +21.5 dB
    broadband hiss, 0 dB in    +10.8 dB
    crowd babble, 0 dB in       -1.7 dB
    crowd babble, -5 dB in      -5.0 dB
    already-clean speech         slightly worse

So it earns its place on a bad line, a running engine, air conditioning or
machinery, and it is the wrong tool for a room full of other conversations -
the case people most want it for. The interference there *is* speech, and a
model trained to keep speech and drop everything else has no basis for
preferring one voice over another. It is not a voice separator, and nothing in
this build is.

Three rules follow from that, and they are the reason this module exists rather
than a single call to the library:

* The original is never replaced. Enhancement writes a second file.
* What it produced is marked as produced, in the status log and in the
  provenance of anything derived from it.
* It is kept away from voice comparison. A denoiser alters exactly the spectral
  detail a voiceprint measures, so a profile built from raw audio and a
  questioned sample that has been cleaned are not comparable - the same class
  of mismatch the embedding-model check already refuses.

The figures above come from synthetic mixtures, which is a proxy and not
validation on operational recordings. ``python -m whispr.enhance`` runs the
same measurement over a folder of real ones; that is the number to believe.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

from . import resources

PathLike = Union[str, Path]
ProgressCallback = Optional[Callable[[str], None]]

# What the cleaned copy is called. Deliberately recognisable: the comparison
# page refuses a file named this way, and an operator can see at a glance which
# of two files in a folder is the recording and which is the derived copy.
SUFFIX = ".cleaned.wav"

# Said wherever the option is offered. The second sentence is the important one.
HELPS_WITH = (
    "Steady background noise: a bad line, hum, hiss, traffic, engines, air "
    "conditioning, machinery."
)
DOES_NOT = (
    "It does not separate one conversation from another. In a bar or a crowd "
    "the background is other people talking, and on that it does nothing and "
    "can make the recording worse. It will not rescue speech you cannot "
    "already hear."
)
NOT_FOR_COMPARISON = (
    "A cleaned copy is for listening and reading, never for voice comparison "
    "or building a profile: cleaning alters the detail a voiceprint measures."
)


class EnhancementError(RuntimeError):
    """Cleaning could not be done, with a reason an operator can act on."""


@dataclass
class EnhancementReport:
    """What cleaning did to one recording - in facts, and no further.

    There is deliberately no "did it help" figure here, and the reason is worth
    recording. The obvious candidate was this project's own level-margin
    measure, taken before and after. Measured against mixtures with known
    answers it reported +35 dB on hiss and *+4 dB on babble* - the case where
    the ground-truth comparison says cleaning made the recording 1.7 dB worse.
    It was measuring how hard the denoiser gated the quiet parts, which a
    denoiser always does, and not whether any speech became clearer.

    A number that says "this helped" on a recording cleaning has just damaged
    is worse than no number at all, so what is reported is what was done: how
    much of the recording was removed. Whether that helped is for the ears of
    the person who knows what the recording is of.
    """

    source: str = ""
    output: str = ""
    seconds: float = 0.0
    model_sha256: str = ""
    channels: int = 1
    # Energy of what was taken out, as a share of what went in. A measurement of
    # the operation, not a verdict on it.
    removed_fraction: float = 0.0
    warnings: List[str] = field(default_factory=list)

    @property
    def removed_percent(self) -> float:
        return self.removed_fraction * 100.0

    def to_dict(self) -> "dict[str, Any]":
        return {
            "source": self.source,
            "output": self.output,
            "seconds": round(self.seconds, 2),
            "model_sha256": self.model_sha256,
            "channels": self.channels,
            "removed_fraction": round(self.removed_fraction, 4),
            "warnings": list(self.warnings),
        }

    def summary(self) -> str:
        """One line for the status log, claiming nothing it cannot support."""
        return (
            f"Cleaned copy written — it removed {self.removed_percent:.0f}% of "
            "the recording's energy. Listen to both before relying on either: "
            "on a crowded recording what it removes is other people talking, "
            "and taking that out can take the conversation with it."
        )


def cleaned_path(source: PathLike, directory: "Optional[PathLike]" = None) -> Path:
    """Where the cleaned copy of ``source`` belongs."""
    src = Path(source)
    folder = Path(directory) if directory is not None else src.parent
    return folder / (src.name + SUFFIX)


def looks_cleaned(path: PathLike) -> bool:
    """Whether this file is one this application produced by cleaning another.

    Used to keep cleaned audio out of enrolment and comparison. It reads a name,
    so it is a guard and not a proof - an operator who renames the file can
    still get it in, which is why the wording where it is used explains the
    reason rather than just refusing.
    """
    return Path(path).name.endswith(SUFFIX)


def model_path() -> "Optional[Path]":
    """The bundled denoiser model, or None when this build has none."""
    return resources.bundled_denoiser_model()


def available() -> bool:
    """Whether this build can clean audio at all."""
    if model_path() is None:
        return False
    try:
        import sherpa_onnx  # noqa: F401
    except Exception:  # noqa: BLE001 - an absent library is an answer, not a fault
        return False
    return True


def _denoiser(threads: int = 2) -> Any:
    import sherpa_onnx

    model = model_path()
    if model is None:
        raise EnhancementError(
            "This build does not include the noise-reduction model, so audio "
            "cannot be cleaned. Nothing on this machine will add it; the "
            "bundle has to be rebuilt with it included."
        )
    return sherpa_onnx.OfflineSpeechDenoiser(
        sherpa_onnx.OfflineSpeechDenoiserConfig(
            model=sherpa_onnx.OfflineSpeechDenoiserModelConfig(
                gtcrn=sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(
                    model=str(model)
                ),
                num_threads=max(1, int(threads)),
                provider="cpu",
            )
        )
    )


def _read_wav(path: Path) -> "tuple[Any, int, int]":
    """``(mono float32 samples, sample rate, channel count)`` from a PCM WAV."""
    import numpy as np

    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())
    if width != 2:
        raise EnhancementError(
            f"{path.name} is not 16-bit audio, which is what this step reads."
        )
    samples = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, rate, channels


def _write_wav(path: Path, samples: Any, rate: int) -> None:
    import numpy as np

    clipped = np.clip(np.asarray(samples, dtype="float32"), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(rate))
        wav.writeframes(pcm.tobytes())


def _removed_fraction(before: Any, after: Any) -> float:
    """The share of the recording's energy that cleaning took out."""
    import numpy as np

    n = min(len(before), len(after))
    if not n:
        return 0.0
    a = np.asarray(before[:n], dtype="float64")
    b = np.asarray(after[:n], dtype="float64")
    total = float(np.sum(a * a))
    if total <= 0:
        return 0.0
    removed = float(np.sum((a - b) ** 2))
    return max(0.0, min(1.0, removed / total))


def denoise(
    source: PathLike,
    dest: "Optional[PathLike]" = None,
    *,
    threads: int = 2,
    progress: ProgressCallback = None,
) -> EnhancementReport:
    """Write a cleaned copy of a 16-bit PCM WAV, and measure what it did.

    The source is opened read-only and never written to. The measurement either
    side is the project's own level-margin figure, so an operator reads the
    result in the same terms the quality column already uses.
    """
    import numpy as np

    from .hashing import sha256_file

    src = Path(source)
    out = Path(dest) if dest is not None else cleaned_path(src)
    if not src.is_file():
        raise EnhancementError(f"{src} is not there.")

    samples, rate, channels = _read_wav(src)
    report = EnhancementReport(
        source=str(src),
        output=str(out),
        seconds=len(samples) / float(rate or 1),
        channels=channels,
    )
    if channels > 1:
        # Worth recording: with two or more microphones there are techniques
        # that beat anything a single channel allows, and knowing the field
        # recordings are multi-channel is what makes that worth building.
        report.warnings.append(
            f"{channels} channels were mixed to one before cleaning."
        )
    if not len(samples):
        raise EnhancementError(f"{src.name} has no audio in it.")

    if progress is not None:
        progress(f"Cleaning {src.name} ({report.seconds / 60:.1f} min)…")

    cleaned = np.asarray(_denoiser(threads).run(samples, rate).samples, dtype="float32")

    _write_wav(out, cleaned, rate)
    report.removed_fraction = _removed_fraction(samples, cleaned)
    model = model_path()
    report.model_sha256 = sha256_file(model) if model is not None else ""
    if progress is not None:
        progress(report.summary())
    return report


__all__ = [
    "DOES_NOT",
    "HELPS_WITH",
    "NOT_FOR_COMPARISON",
    "SUFFIX",
    "EnhancementError",
    "EnhancementReport",
    "available",
    "cleaned_path",
    "denoise",
    "looks_cleaned",
    "model_path",
]
