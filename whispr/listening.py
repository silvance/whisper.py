"""Make a recording easier to listen to, without a model and without guessing.

This is the plain-signal-processing counterpart to :mod:`whispr.enhance`. Where
that module runs a neural denoiser, everything here is a fixed filter and a
gain curve: arithmetic on the samples, decided by numbers an operator can read
off this file. That has four consequences that matter for investigative work.

It cannot invent speech. The output is a linear filter of the input followed by
a per-sample gain, so every sample out is a weighted sum of samples in. Silence
in gives silence out. No model is choosing what a muffled word probably was,
which is the standing objection to the generative tools that do this better.

It is explainable. "85 Hz high-pass, 3 dB down at 350 Hz, 5 dB up at 2.6 kHz,
50 Hz hum and 3 harmonics notched 40 dB, 3:1 toward -20 dBFS above 6 dB over
the noise floor" is a complete and checkable account of what was done to an
exhibit. :class:`ListeningReport`
carries it so the processing can be stated rather than alluded to.

It costs nothing to ship. NumPy is already a dependency, so this works in every
build - including ones with no denoiser model in them - and adds no megabytes.

And it leaves the timing alone. The spectral stage is a symmetric linear-phase
kernel whose group delay is removed afterwards, so the output is sample-aligned
with the input and transcript timestamps still point at the right audio. There
is a test for this; it is easy to break and silently wrong when broken.

What it does, measured against mixtures built with known answers (the figures
are reproduced by tests/test_listening.py, which fails if they stop holding):

    Mains hum           Found and notched without being asked, and told apart
                        from a voice: on a hum-free recording of a 120 Hz
                        talker an earlier version of the detector reported
                        60 Hz hum at 42 dB and would have notched the speech
                        out of the file. The hum band ends up 33 dB down
                        relative to the speech band, at a cost of under 0.5 dB
                        to the speech itself.
    Quiet talker        +16 dB on a passage recorded at -49 dBFS, with the gap
                        to a loud bang in the same file falling from 27.6 dB
                        to 9.7 dB and the peak from 0.0 dBFS (clipping) to
                        exactly the -1 dBFS ceiling.
    The two together    Removing the hum is what lets the leveller work: the
                        same recording gets +12.5 dB with the hum notched and
                        +1.9 dB with it left in, because until it goes the hum
                        is most of what the leveller can see.
    Speed and size      About 150x real time on one CPU core, so an hour of
                        audio takes well under a minute, in roughly 800 MB of
                        working memory. Both of those were worse: the first
                        version measured its levels by band-passing the whole
                        recording in float64 and needed 1.9 GB, on a machine
                        that is also running the transcriber.

And what it does not do, also measured: on a target talker against five
interfering talkers the ratio between them moved by 0.02 dB. That is not a
shortcoming of the tuning. A linear filter multiplies the target and the
interference by the same response, so it cannot change the ratio between them;
no choice of numbers in this file would. Separating one conversation from
another needs a different kind of tool, and this is not one.
"""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Union

PathLike = Union[str, Path]
ProgressCallback = Optional[Callable[[str], None]]

# What the listening copy is called. Recognisable for the same reason the
# cleaned copy is: the comparison page refuses a file named this way, the batch
# walker skips it, and an operator can tell at a glance which file in a folder
# is the recording and which is derived from it.
SUFFIX = ".listening.wav"

# Said wherever the option is offered.
HELPS_WITH = (
    "Hum from mains power, low rumble, handling thumps, and recordings where "
    "the person you want is much quieter than everything else. It also stops "
    "sudden bangs being painful to listen to for an hour."
)
DOES_NOT = (
    "It does not separate one conversation from another. In a bar or a crowd "
    "the problem is other people talking, and this cannot touch that - it "
    "turns the other voices up exactly as much as the one you want. It will "
    "not rescue speech you cannot already hear."
)
NOT_FOR_COMPARISON = (
    "A listening copy is for listening and reading, never for voice comparison "
    "or building a profile: it deliberately changes the balance of the voice, "
    "which is part of what a voiceprint measures."
)

# Processing strength. The names are what the operator sees.
GENTLE = "Gentle"
STANDARD = "Standard"
STRONG = "Strong"
STRENGTHS = (GENTLE, STANDARD, STRONG)
_STRENGTH_SCALE = {GENTLE: 0.5, STANDARD: 1.0, STRONG: 1.5}

# Mains frequencies worth looking for. Which one a recording has depends on
# where it was made, so the default is to find out rather than ask.
MAINS_CANDIDATES = (50.0, 60.0)
AUTO_HUM = "Auto"
NO_HUM = "Off"

# --- Fixed filter geometry. Changing these changes what the tool does, so they
# --- are named and in one place rather than buried in the arithmetic.
_HIGHPASS_HZ = 85.0        # below a low male voice; rumble lives under it
_HIGHPASS_ORDER = 4        # 24 dB/octave
_MUD_HZ = 350.0            # room boom, which masks the consonant band
_MUD_DB = -3.0
_MUD_Q = 1.1
_PRESENCE_HZ = 2600.0      # where consonants are, and intelligibility with them
_PRESENCE_DB = 5.0
_PRESENCE_Q = 1.3
_HISS_HZ = 7000.0          # speech is essentially complete below this
_HISS_DB_PER_OCTAVE = -9.0

# Hum notches. Depth is limited on purpose: a notch to zero rings audibly
# around transients, and -40 dB is already inaudible under speech.
_NOTCH_WIDTH_HZ = 3.0
_NOTCH_DEPTH_DB = -40.0
_MAX_HARMONICS = 8
# A harmonic has to stand this far above the spectrum around it to be treated
# as hum. Without this the notches gouge holes wherever they happen to land.
_HUM_PROMINENCE_DB = 6.0
_HUM_MIN_HARMONICS = 2
# And it has to be steady. Prominence alone cannot tell hum from a voice: a
# talker pitched at 120 Hz puts strong energy at 120, 240, 360 and 480 Hz,
# which are the even harmonics of 60, and a talker at 100 Hz does the same to
# 50. Measured on a hum-free recording of a 120 Hz voice, the prominence test
# on its own reported 60 Hz hum with four harmonics at up to 42 dB - it would
# have notched the speech out of the file. What separates them is time: mains
# hum runs at a constant level from end to end, while a voice harmonic appears
# and disappears with every syllable. This is the spread, in dB, that a line's
# level may vary across the recording and still be called hum.
_HUM_STEADY_DB = 6.0

# Leveller. Lifts quiet speech toward a comfortable level and holds down what
# is already loud, with the gain moving slowly enough not to pump.
_TARGET_DBFS = -20.0
# Which frames the leveller will lift. This has to be measured against the
# recording's own noise floor rather than a fixed level. It was -45 dBFS, and
# on a covert recording whose target talker sits at -49 dBFS that gate did
# nothing at all - it refused to help in exactly the case the tool is for.
_GATE_ABOVE_FLOOR_DB = 6.0
_FLOOR_PERCENTILE = 10.0   # frames this quiet are taken to be the noise floor
_SILENCE_DBFS = -70.0      # and below this there is nothing to lift
_RATIO = 3.0
_MAX_LIFT_DB = 24.0
_MAX_CUT_DB = -12.0
_ATTACK_S = 0.02
_RELEASE_S = 0.30
_GAIN_FRAME_S = 0.010
_CEILING_DBFS = -1.0
_LIMIT_FRAME_S = 0.005


class ListeningError(RuntimeError):
    """A listening copy could not be made, with a reason an operator can act on."""


@dataclass
class ListeningSettings:
    """What to do to the recording. Defaults are the tuning measured above."""

    strength: str = STANDARD
    # AUTO_HUM to find the mains frequency, NO_HUM to leave hum alone, or a
    # frequency in Hz to notch that one without looking.
    hum: Union[str, float] = AUTO_HUM
    level: bool = True

    @property
    def scale(self) -> float:
        return _STRENGTH_SCALE.get(self.strength, 1.0)


@dataclass
class ListeningReport:
    """What was done to one recording, in terms that can be put in a report.

    As with :class:`whispr.enhance.EnhancementReport` there is no "did it help"
    figure, and for the same reason: the obvious candidates measure how hard
    the processing worked, which is not the same question and answers it wrongly
    on exactly the recordings where it matters. What is recorded here is what
    was applied and what the levels did. Whether the speech got easier to
    follow is for the ears of the person who knows what the recording is of,
    which is why the option to hear the original is next to the option to make
    this copy.
    """

    source: str = ""
    output: str = ""
    seconds: float = 0.0
    rate: int = 0
    channels: int = 1
    strength: str = STANDARD
    # Mains frequency notched, and how many harmonics with it. 0.0 means none
    # was found or none was asked for.
    hum_hz: float = 0.0
    hum_harmonics: int = 0
    levelled: bool = False
    # Levels, in dBFS, before and after. Facts about the two files.
    quiet_dbfs_before: float = 0.0
    quiet_dbfs_after: float = 0.0
    peak_dbfs_before: float = 0.0
    peak_dbfs_after: float = 0.0
    warnings: List[str] = field(default_factory=list)

    @property
    def quiet_lift_db(self) -> float:
        """How much the quietest speech in the recording came up."""
        return self.quiet_dbfs_after - self.quiet_dbfs_before

    def filters(self) -> str:
        """The processing as one checkable line, for a report or a log."""
        parts = [
            f"{_HIGHPASS_HZ:.0f} Hz high-pass",
            f"{_MUD_DB * self.scale:+.1f} dB at {_MUD_HZ:.0f} Hz",
            f"{_PRESENCE_DB * self.scale:+.1f} dB at {_PRESENCE_HZ / 1000:.1f} kHz",
            f"{_HISS_DB_PER_OCTAVE * self.scale:+.0f} dB/octave above "
            f"{_HISS_HZ / 1000:.0f} kHz",
        ]
        if self.hum_harmonics:
            parts.append(
                f"{self.hum_hz:.0f} Hz hum and {self.hum_harmonics - 1} harmonics "
                f"notched {_NOTCH_DEPTH_DB:.0f} dB"
            )
        if self.levelled:
            parts.append(
                f"{_RATIO:.0f}:1 toward {_TARGET_DBFS:.0f} dBFS above "
                f"{_GATE_ABOVE_FLOOR_DB:.0f} dB over the noise floor"
            )
        return "; ".join(parts)

    @property
    def scale(self) -> float:
        return _STRENGTH_SCALE.get(self.strength, 1.0)

    def to_dict(self) -> "dict[str, Any]":
        return {
            "source": self.source,
            "output": self.output,
            "seconds": round(self.seconds, 2),
            "rate": self.rate,
            "channels": self.channels,
            "strength": self.strength,
            "hum_hz": round(self.hum_hz, 2),
            "hum_harmonics": self.hum_harmonics,
            "levelled": self.levelled,
            "quiet_dbfs_before": round(self.quiet_dbfs_before, 2),
            "quiet_dbfs_after": round(self.quiet_dbfs_after, 2),
            "peak_dbfs_before": round(self.peak_dbfs_before, 2),
            "peak_dbfs_after": round(self.peak_dbfs_after, 2),
            "filters": self.filters(),
            "warnings": list(self.warnings),
        }

    def summary(self) -> str:
        """One line for the status log, claiming nothing it cannot support."""
        said: List[str] = []
        if self.hum_harmonics:
            said.append(f"took out {self.hum_hz:.0f} Hz hum")
        if self.levelled and self.quiet_lift_db >= 1.0:
            said.append(f"brought the quietest speech up {self.quiet_lift_db:.0f} dB")
        if self.peak_dbfs_before > _CEILING_DBFS >= self.peak_dbfs_after:
            said.append("stopped the loud parts clipping")
        did = ", ".join(said) if said else "applied the speech filters"
        return (
            f"Listening copy written — {did}. Compare it with the original "
            "before relying on either: if the background is other people "
            "talking, this will not have helped."
        )


def listening_path(source: PathLike, directory: "Optional[PathLike]" = None) -> Path:
    """Where the listening copy of ``source`` belongs."""
    src = Path(source)
    folder = Path(directory) if directory is not None else src.parent
    return folder / (src.name + SUFFIX)


def looks_polished(path: PathLike) -> bool:
    """Whether this file is one this application produced for listening.

    It reads a name, so it is a guard and not a proof - an operator who renames
    the file can still get it into a comparison, which is why the wording where
    this is used explains the reason rather than just refusing.
    """
    return Path(path).name.endswith(SUFFIX)


def derived_kind(path: PathLike) -> "Optional[str]":
    """What processing produced ``path``, or ``None`` if it looks like an original.

    One question for the places that have to keep derived audio out of
    enrolment and comparison, so a third kind of processed copy cannot be added
    later without those places being told about it.
    """
    from . import enhance

    if looks_polished(path):
        return "listening copy"
    if enhance.looks_cleaned(path):
        return "cleaned copy"
    return None


def available() -> bool:
    """Whether this build can make a listening copy.

    Always true: this needs NumPy, which the application already requires, and
    no model file. It exists so callers can ask the same question they ask of
    :func:`whispr.enhance.available`.
    """
    try:
        import numpy  # noqa: F401
    except Exception:  # noqa: BLE001 - an absent library is an answer, not a fault
        return False
    return True


def _read_wav(path: Path) -> "tuple[Any, int, int]":
    """``(mono float32 samples, sample rate, channel count)`` from a PCM WAV."""
    import numpy as np

    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())
    if width != 2:
        raise ListeningError(
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


def _dbfs(value: float) -> float:
    return 20.0 * math.log10(max(float(value), 1e-12))


def response_db(freqs: Any, scale: float = 1.0,
                notches: Sequence[float] = ()) -> Any:
    """The speech-focus magnitude response, in dB, on the given frequencies.

    Public because it is the honest description of this tool: anyone wondering
    what the filter does can ask it, and the GUI draws it.
    """
    import numpy as np

    f = np.maximum(np.asarray(freqs, dtype="float64"), 1e-9)
    db = np.zeros_like(f)

    # High-pass: rumble, HVAC, handling, wind. Not scaled by strength - there
    # is nothing below 85 Hz that helps anyone understand speech.
    ratio = f / _HIGHPASS_HZ
    power = ratio ** _HIGHPASS_ORDER
    db += 20.0 * np.log10(power / np.sqrt(1.0 + power ** 2))

    def bell(centre: float, gain_db: float, q: float) -> Any:
        return gain_db * np.exp(-((np.log2(f / centre) * q) ** 2) * 2.0)

    db += bell(_MUD_HZ, _MUD_DB * scale, _MUD_Q)
    db += bell(_PRESENCE_HZ, _PRESENCE_DB * scale, _PRESENCE_Q)
    db += _HISS_DB_PER_OCTAVE * scale * np.log2(np.maximum(f, _HISS_HZ) / _HISS_HZ)

    for centre in notches:
        inside = np.abs(f - centre) <= _NOTCH_WIDTH_HZ
        db = np.where(inside, np.minimum(db, _NOTCH_DEPTH_DB), db)
    return db


def _kernel(rate: int, scale: float, notches: Sequence[float]) -> Any:
    """A symmetric linear-phase FIR for the response, long enough for the notches.

    Length is set by the narrowest feature: a 3 Hz notch needs a frequency
    resolution finer than that, which means a kernel of roughly half the sample
    rate. At 16 kHz that is 8001 taps - a quarter-second kernel, convolved by
    FFT, which is cheap. Its delay is removed in :func:`_convolve`.
    """
    import numpy as np

    taps = int(rate // 2) | 1
    taps = max(1025, min(taps, 32769))
    size = 1 << (taps * 2 - 1).bit_length()
    freqs = np.fft.rfftfreq(size, 1.0 / rate)
    mag = 10.0 ** (response_db(freqs, scale, notches) / 20.0)
    mag[0] = 0.0  # no DC offset in the output
    impulse = np.roll(np.fft.irfft(mag, size), taps // 2)[:taps]
    # float32 throughout: at float64 an hour of audio needed about 1.9 GB of
    # working memory, on a machine that is also running the transcriber.
    return (impulse * np.hanning(taps)).astype("float32")


def _convolve(samples: Any, kernel: Any) -> Any:
    """Overlap-add FFT convolution, trimmed so the output stays sample-aligned.

    The trim is the point. A symmetric kernel delays everything by half its
    length; leaving that in would shift the audio under every transcript
    timestamp by a quarter of a second.
    """
    import numpy as np

    taps = len(kernel)
    lag = taps // 2
    block = 1 << (max(4096, taps * 4) - 1).bit_length()
    step = block - taps + 1
    out = np.zeros(len(samples) + taps - 1, dtype="float32")
    spectrum = np.fft.rfft(kernel, block)
    for start in range(0, len(samples), step):
        chunk = samples[start:start + step]
        segment = np.fft.irfft(np.fft.rfft(chunk, block) * spectrum, block)
        room = len(out) - start
        out[start:start + len(segment)] += segment[:room]
    return out[lag:lag + len(samples)]


def _window_spectra(samples: Any, rate: int, windows: int = 48) -> "tuple[Any, Any]":
    """``(frequencies, power per window)`` from windows spread through the recording.

    Per window rather than averaged, because telling hum from a voice needs to
    know whether a spectral line is steady over time, and an average has
    thrown that away.
    """
    import numpy as np

    size = 1 << max(12, int(math.log2(max(rate, 2))) + 1)
    if len(samples) < size:
        size = 1 << max(8, (len(samples)).bit_length() - 1)
    if size < 2 or len(samples) < size:
        return np.zeros(0), np.zeros((0, 0))
    starts = np.unique(
        np.linspace(0, len(samples) - size, num=max(1, windows)).astype(int)
    )
    window = np.hanning(size)
    stack = np.empty((len(starts), size // 2 + 1), dtype="float64")
    for index, start in enumerate(starts):
        block = samples[start:start + size].astype("float64") * window
        stack[index] = np.abs(np.fft.rfft(block)) ** 2
    return np.fft.rfftfreq(size, 1.0 / rate), stack


def detect_hum(samples: Any, rate: int) -> "tuple[float, int]":
    """``(mains frequency, harmonics found)`` for the hum in this recording.

    Returns ``(0.0, 0)`` when nothing steady stands out, which is the answer
    for most recordings and has to be, or the notches gouge holes in speech.

    A line counts as hum only if it stands above the spectrum beside it *and*
    holds a near-constant level from one end of the recording to the other,
    and only if the mains fundamental itself is one of them. Each of those
    three conditions is there because dropping it makes the detector notch a
    voice; see ``_HUM_STEADY_DB``.
    """
    import numpy as np

    freqs, stack = _window_spectra(samples, rate)
    if not len(freqs) or not len(stack):
        return 0.0, 0
    mean_db = 10.0 * np.log10(np.maximum(stack.mean(axis=0), 1e-20))
    per_window_db = 10.0 * np.log10(np.maximum(stack, 1e-20))
    best = (0.0, 0, 0.0)
    for mains in MAINS_CANDIDATES:
        found, strength, has_fundamental = 0, 0.0, False
        for harmonic in range(1, _MAX_HARMONICS + 1):
            centre = mains * harmonic
            if centre >= rate / 2.0:
                break
            offset = np.abs(freqs - centre)
            peak = offset <= _NOTCH_WIDTH_HZ
            # The spectrum beside the line, skipping a guard band so the line
            # itself does not raise its own reference.
            around = (offset > _NOTCH_WIDTH_HZ * 3) & (offset <= mains * 0.45)
            if not peak.any() or not around.any():
                continue
            prominence = float(mean_db[peak].max() - np.median(mean_db[around]))
            if prominence < _HUM_PROMINENCE_DB:
                continue
            # Steady? Spread of this line's own level across the recording.
            bin_index = int(np.argmax(np.where(peak, mean_db, -np.inf)))
            levels = per_window_db[:, bin_index]
            spread = float(np.percentile(levels, 75) - np.percentile(levels, 25))
            if spread > _HUM_STEADY_DB:
                continue
            found += 1
            has_fundamental = has_fundamental or harmonic == 1
            # Scored by how much hum is there, not how many lines were found.
            # Counting lines let the wrong candidate win: a 50 Hz hum loud
            # enough leaks a steady line into the 60 Hz bin, and on a count
            # both candidates then look equally good.
            strength += prominence
        if (has_fundamental and found >= _HUM_MIN_HARMONICS
                and strength > best[2]):
            best = (mains, found, strength)
    return best[0], best[1]


def _harmonics(mains: float, rate: int, count: int) -> List[float]:
    """The harmonic frequencies to notch for a mains frequency."""
    out: List[float] = []
    for harmonic in range(1, max(count, 1) + 1):
        centre = mains * harmonic
        if centre >= rate / 2.0:
            break
        out.append(centre)
    return out


def _gain_curve(want_db: Any, frame_seconds: float) -> Any:
    """Smooth a wanted-gain curve with attack and release, frame by frame.

    Sequential by nature - a gain that reacts instantly to every frame is
    audible as pumping - but it runs over frames, not samples, so an hour of
    audio is a loop of a few hundred thousand cheap steps.
    """
    import numpy as np

    rising = math.exp(-frame_seconds / _RELEASE_S)
    falling = math.exp(-frame_seconds / _ATTACK_S)
    smooth = np.empty_like(want_db)
    gain = 0.0
    for index in range(len(want_db)):
        wanted = float(want_db[index])
        coefficient = falling if wanted < gain else rising
        gain = coefficient * gain + (1.0 - coefficient) * wanted
        smooth[index] = gain
    return smooth


def _frames(samples: Any, rate: int, seconds: float) -> "tuple[Any, int]":
    """``(frames, hop)`` as a *view* where possible, not a copy.

    Padding to a whole number of frames copied the entire recording, twice per
    run. Truncating instead drops at most one frame of under 50 ms from the end
    of the analysis, and the gain curve covers the tail anyway because it is
    interpolated across the full length from the frame centres.
    """
    hop = max(1, int(rate * seconds))
    usable = len(samples) - len(samples) % hop
    if usable < hop:
        return samples[:0].reshape(0, hop), hop
    return samples[:usable].reshape(-1, hop), hop


def _spread(values: Any, hop: int, length: int) -> Any:
    """Interpolate a per-frame curve up to one value per sample."""
    import numpy as np

    centres = np.arange(len(values)) * hop + hop / 2.0
    # One value per sample, so float32: np.interp returns float64 and at an
    # hour of audio that array alone is 460 MB.
    return np.interp(np.arange(length), centres, values).astype("float32")


def _gate_dbfs(frame_db: Any) -> float:
    """The level below which this recording is noise rather than speech."""
    import numpy as np

    if not len(frame_db):
        return _SILENCE_DBFS
    floor = float(np.percentile(frame_db, _FLOOR_PERCENTILE))
    return max(floor + _GATE_ABOVE_FLOOR_DB, _SILENCE_DBFS)


def _level(samples: Any, rate: int, scale: float) -> Any:
    """Lift quiet speech toward a comfortable level and hold down the loud."""
    import numpy as np

    frames, hop = _frames(samples, rate, _GAIN_FRAME_S)
    rms = np.sqrt((frames.astype("float64") ** 2).mean(axis=1) + 1e-12)
    db = 20.0 * np.log10(np.maximum(rms, 1e-9))
    gate = _gate_dbfs(db)
    reduction = 1.0 - 1.0 / max(_RATIO, 1.0001)
    want = np.clip((_TARGET_DBFS - db) * reduction * scale,
                   _MAX_CUT_DB, _MAX_LIFT_DB)
    # Below the gate the gain holds where it was instead of dropping to zero.
    # Driving it to zero in the gaps looks right and is badly wrong: speech is
    # modulated at a syllable rate, so the gain collapsed several times a
    # second with a fast attack and crawled back with a slow release, and the
    # lift never arrived. Measured on a talker at -49 dBFS it gave 1.7 dB
    # instead of the 19 dB the compression curve asks for. Holding also stops
    # the background swelling up in every pause, which is the other thing a
    # leveller must not do.
    quiet = db < gate
    if quiet.all():
        want = np.zeros_like(want)
    elif quiet.any():
        index = np.where(quiet, 0, np.arange(len(want)))
        want = want[np.maximum.accumulate(index)]
    gain = _spread(_gain_curve(want, _GAIN_FRAME_S), hop, len(samples))
    gain /= 20.0
    np.power(10.0, gain, out=gain)
    samples *= gain
    return samples


def _limit(samples: Any, rate: int) -> Any:
    """Hold peaks under the ceiling so an hour of listening is not painful."""
    import numpy as np

    frames, hop = _frames(samples, rate, _LIMIT_FRAME_S)
    # max(|x|) per frame without materialising abs() over the whole recording.
    peak = np.maximum(frames.max(axis=1), -frames.min(axis=1))
    ceiling = 10.0 ** (_CEILING_DBFS / 20.0)
    needed = np.minimum(1.0, ceiling / np.maximum(peak, 1e-9))
    # Look-ahead and look-behind by a frame, so the gain is already down when
    # the transient arrives instead of clamping after it.
    needed = np.minimum.reduce([needed, np.roll(needed, 1), np.roll(needed, -1)])
    samples *= _spread(needed, hop, len(samples))
    return np.clip(samples, -1.0, 1.0, out=samples)


# How the level measurement samples the recording. Measuring "how loud is the
# quietest speech" needs a percentile, not every frame, and band-passing a
# whole hour to get one cost about 900 MB of working memory on a 30-minute file
# - on a machine that is also running the transcriber. Spans spread through the
# recording answer the same question in a few megabytes.
_MEASURE_SPANS = 200
_MEASURE_SECONDS = 1.0
_MEASURE_FRAME_S = 0.05
_BAND_TAPS = 257


def _measure_spans(length: int, rate: int) -> "tuple[List[int], int]":
    """``(start offsets, span length)`` for the spans the measurement uses.

    The same spans are used before and after, so the two figures describe the
    same moments of the recording rather than two different populations.
    """
    import numpy as np

    span = max(int(rate * _MEASURE_SECONDS), _BAND_TAPS * 4)
    if length <= span:
        return [0], length
    # Never more spans than the recording has room for without them sitting on
    # top of each other. Replaying one line is a few seconds of audio, and 200
    # overlapping spans would measure sixty times the clip to level it.
    count = max(1, min(_MEASURE_SPANS, length // span))
    starts = np.unique(
        np.linspace(0, length - span, num=count).astype(int)
    )
    return [int(start) for start in starts], span


def _band_kernel(rate: int) -> Any:
    """A band-pass for 300-3400 Hz, used only to measure, never written out."""
    import numpy as np

    size = 1 << (_BAND_TAPS * 2 - 1).bit_length()
    freqs = np.fft.rfftfreq(size, 1.0 / rate)
    keep = ((freqs >= 300.0) & (freqs <= min(3400.0, rate * 0.45))).astype("float64")
    impulse = np.roll(np.fft.irfft(keep, size), _BAND_TAPS // 2)[:_BAND_TAPS]
    return (impulse * np.hanning(_BAND_TAPS)).astype("float32")


def _speech_frame_db(samples: Any, rate: int, starts: Sequence[int],
                     span: int) -> Any:
    """Frame levels in the speech band, in dB, over the sampled spans.

    In the speech band because the question is how loud the *speech* is.
    Measured full-band on a recording with heavy mains hum, the lift came out
    at -7.8 dB: arithmetically right and a lie in substance, because what had
    got quieter was the hum being notched out and not the talker.
    """
    import numpy as np

    kernel = _band_kernel(rate)
    hop = max(1, int(rate * _MEASURE_FRAME_S))
    skirt = _BAND_TAPS
    levels: List[Any] = []
    for start in starts:
        chunk = np.asarray(samples[start:start + span], dtype="float32")
        if len(chunk) < skirt * 2 + hop:
            continue
        inner = _convolve(chunk, kernel)[skirt:len(chunk) - skirt]
        usable = len(inner) - len(inner) % hop
        if usable < hop:
            continue
        frames = inner[:usable].reshape(-1, hop)
        levels.append(np.sqrt((frames.astype("float64") ** 2).mean(axis=1) + 1e-12))
    if not levels:
        return np.zeros(0)
    return 20.0 * np.log10(np.maximum(np.concatenate(levels), 1e-9))


def _voiced_mask(frame_db: Any) -> Any:
    """Which measured frames carry something, as a boolean mask.

    Chosen once, from the original, and then reused to measure the processed
    copy. Choosing again afterwards would compare two different populations of
    frames and report a difference that is partly just the change of
    population - the same mistake that made the denoiser's old level-margin
    figure say cleaning had helped when it had not.
    """
    import numpy as np

    if not len(frame_db):
        return np.zeros(0, dtype=bool)
    mask = frame_db > float(np.median(frame_db)) - 6.0
    return mask if mask.any() else np.ones(len(frame_db), dtype=bool)


def _quiet_dbfs(frame_db: Any, voiced: Any) -> float:
    """Level of the quietest speech-like part of the measured frames.

    The 20th percentile, not the minimum - which is silence in any recording -
    and not the mean, which one loud bang dominates. What this answers is "did
    the person I can barely hear get louder".
    """
    import numpy as np

    if not len(frame_db) or not len(voiced):
        return -120.0
    count = min(len(frame_db), len(voiced))
    usable = frame_db[:count][voiced[:count]]
    if not len(usable):
        return -120.0
    return float(np.percentile(usable, 20))


def polish(
    source: PathLike,
    dest: "Optional[PathLike]" = None,
    *,
    settings: "Optional[ListeningSettings]" = None,
    progress: ProgressCallback = None,
) -> ListeningReport:
    """Write a copy of a 16-bit PCM WAV that is easier to listen to.

    The source is opened read-only and never written to. What was applied is
    recorded in the report so it can be stated in writing later.
    """
    import numpy as np

    config = settings or ListeningSettings()
    src = Path(source)
    out = Path(dest) if dest is not None else listening_path(src)
    if not src.is_file():
        raise ListeningError(f"{src} is not there.")

    samples, rate, channels = _read_wav(src)
    if not len(samples):
        raise ListeningError(f"{src.name} has no audio in it.")
    if rate <= 0:
        raise ListeningError(f"{src.name} does not say what sample rate it is.")

    report = ListeningReport(
        source=str(src),
        output=str(out),
        seconds=len(samples) / float(rate),
        rate=rate,
        channels=channels,
        strength=config.strength,
        levelled=bool(config.level),
    )
    if channels > 1:
        # Worth recording: with two or more microphones there are techniques
        # that beat anything a single channel allows, and knowing the field
        # recordings are multi-channel is what makes that worth building.
        report.warnings.append(
            f"{channels} channels were mixed to one before processing."
        )
    if progress is not None:
        progress(f"Preparing a listening copy of {src.name} "
                 f"({report.seconds / 60:.1f} min)…")

    notches: List[float] = []
    if config.hum == AUTO_HUM:
        mains, found = detect_hum(samples, rate)
        if mains:
            notches = _harmonics(mains, rate, max(found, _HUM_MIN_HARMONICS))
            report.hum_hz, report.hum_harmonics = mains, len(notches)
    elif config.hum != NO_HUM:
        try:
            mains = float(config.hum)
        except (TypeError, ValueError):
            raise ListeningError(
                f"{config.hum!r} is not a mains frequency. Use "
                f"{AUTO_HUM!r}, {NO_HUM!r}, 50 or 60."
            ) from None
        if mains > 0:
            notches = _harmonics(mains, rate, _MAX_HARMONICS)
            report.hum_hz, report.hum_harmonics = mains, len(notches)

    # Measured in the speech band, over spans chosen once from the original.
    starts, span = _measure_spans(len(samples), rate)
    before_db = _speech_frame_db(samples, rate, starts, span)
    voiced = _voiced_mask(before_db)
    report.quiet_dbfs_before = _quiet_dbfs(before_db, voiced)
    report.peak_dbfs_before = _dbfs(float(np.abs(samples).max()))

    processed = _convolve(samples, _kernel(rate, config.scale, notches))
    if config.level:
        processed = _level(processed, rate, config.scale)
    processed = _limit(processed, rate)

    report.quiet_dbfs_after = _quiet_dbfs(
        _speech_frame_db(processed, rate, starts, span), voiced)
    report.peak_dbfs_after = _dbfs(float(np.abs(processed).max()))

    _write_wav(out, processed, rate)
    if progress is not None:
        progress(report.summary())
    return report


__all__ = [
    "AUTO_HUM",
    "DOES_NOT",
    "GENTLE",
    "HELPS_WITH",
    "MAINS_CANDIDATES",
    "NOT_FOR_COMPARISON",
    "NO_HUM",
    "STANDARD",
    "STRENGTHS",
    "STRONG",
    "SUFFIX",
    "ListeningError",
    "ListeningReport",
    "ListeningSettings",
    "available",
    "derived_kind",
    "detect_hum",
    "listening_path",
    "looks_polished",
    "polish",
    "response_db",
]
