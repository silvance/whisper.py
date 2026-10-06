"""The listening copy: what plain filtering does, and what it must never claim.

An operator asked for something built in house that cleans a recording up
slightly so a listener can focus on the speech. whispr/listening.py is that,
and it is deliberately arithmetic rather than a model, so its behaviour is
measurable rather than a matter of opinion. These tests measure it.

Three of them are load-bearing and will go silently wrong if the module is
refactored carelessly:

  * the output must stay sample-aligned with the input, or every transcript
    timestamp points at the wrong audio;
  * the processing must not be able to invent signal;
  * it must not change the ratio between a target talker and other talkers,
    because it cannot, and the module says so in writing.
"""

import math
import wave
from pathlib import Path

import numpy as np
import pytest

from whispr import listening

RATE = 16000


# --- Signals with known answers ---------------------------------------------


def speechlike(seconds, f0=120.0, amp=1.0, rate=RATE, seed=7):
    """A voiced-speech-shaped signal: harmonics under a formant envelope."""
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    t = np.arange(n) / rate
    sig = np.zeros(n)
    for k in range(1, 40):
        f = f0 * k
        if f > rate * 0.47:
            break
        env = sum(
            math.exp(-(((f - c) / bw) ** 2))
            for c, bw in ((500, 180), (1500, 250), (2500, 300))
        )
        sig += env * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)) / k**0.3
    sig *= 0.5 + 0.5 * np.maximum(0, np.sin(2 * np.pi * 4.0 * t))
    return amp * sig / (np.abs(sig).max() + 1e-9)


def hum(seconds, mains=50.0, amp=0.08, rate=RATE, harmonics=3):
    t = np.arange(int(seconds * rate)) / rate
    return amp * sum(
        np.sin(2 * np.pi * mains * k * t) / k for k in range(1, harmonics + 1)
    )


def write_wav(path, samples, rate=RATE):
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm.tobytes())
    return Path(path)


def read_wav(path):
    with wave.open(str(path), "rb") as wav:
        rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype("float64") / 32768.0, rate


def band_db(x, lo, hi, rate=RATE):
    f = np.fft.rfftfreq(len(x), 1.0 / rate)
    power = np.abs(np.fft.rfft(x)) ** 2
    mask = (f >= lo) & (f < hi)
    return 10.0 * math.log10(max(float(power[mask].sum()), 1e-20))


# --- The response is the one the module documents ----------------------------


def test_the_filter_cuts_rumble_and_lifts_the_consonant_band():
    freqs = np.fft.rfftfreq(8192, 1.0 / RATE)
    db = listening.response_db(freqs)

    def at(hz):
        return float(db[int(np.argmin(np.abs(freqs - hz)))])

    assert at(40) < -18.0, "rumble must be well down"
    assert -4.0 < at(85) < -2.0, "the high-pass corner is where it says it is"
    assert at(150) > -1.0, "a low voice's fundamental must survive"
    assert at(2600) > 4.0, "the consonant band must be lifted"
    assert at(350) < -2.0, "the masking band must be dipped"
    assert at(7800) < at(5000), "hiss must roll off above speech"


def test_strength_scales_the_shaping_but_never_the_high_pass():
    freqs = np.array([40.0, 2600.0])
    gentle = listening.response_db(
        freqs, listening.ListeningSettings(strength=listening.GENTLE).scale
    )
    strong = listening.response_db(
        freqs, listening.ListeningSettings(strength=listening.STRONG).scale
    )
    assert strong[1] > gentle[1], "Strong lifts presence further"
    assert gentle[0] == pytest.approx(strong[0], abs=0.01), (
        "nothing below 85 Hz helps anyone understand speech, at any strength"
    )


# --- Load-bearing property: the audio must not move -------------------------


def test_the_output_is_sample_aligned_with_the_input():
    """A shifted output silently breaks every transcript timestamp."""
    kernel = listening._kernel(RATE, 1.0, [])
    impulse = np.zeros(RATE)
    impulse[RATE // 2] = 1.0
    out = listening._convolve(impulse, kernel)
    assert len(out) == len(impulse)
    assert int(np.argmax(np.abs(out))) == RATE // 2


def test_a_whole_file_keeps_its_length_and_its_timing(tmp_path):
    """One distinct event in the file must come out at the same moment.

    A marker, not continuous speech: speech modulated at a syllable rate has
    many near-equal peaks and which one is largest tells you nothing about
    timing.
    """
    samples = speechlike(4.0, amp=0.05)
    marker = RATE * 2
    samples[marker : marker + 400] += 0.6 * np.exp(-np.arange(400) / 80.0)
    src = write_wav(tmp_path / "a.wav", samples)
    report = listening.polish(
        src, tmp_path / "out.wav", settings=listening.ListeningSettings(level=False)
    )
    out, rate = read_wav(report.output)
    assert rate == RATE
    assert len(out) == len(samples)
    moved = abs(int(np.argmax(np.abs(out))) - marker)
    assert moved < RATE // 100, f"the marker moved {moved / RATE * 1000:.0f} ms"


# --- Load-bearing property: it cannot invent signal -------------------------


def test_silence_in_gives_silence_out(tmp_path):
    src = write_wav(tmp_path / "quiet.wav", np.zeros(RATE * 2))
    report = listening.polish(src, tmp_path / "out.wav")
    out, _ = read_wav(report.output)
    assert float(np.abs(out).max()) == 0.0


def test_processing_does_not_reach_beyond_where_there_was_sound(tmp_path):
    """A burst in the middle must not grow speech into the silence around it."""
    samples = np.zeros(RATE * 3)
    samples[RATE : RATE * 2] = speechlike(1.0, amp=0.5)
    src = write_wav(tmp_path / "burst.wav", samples)
    out, _ = read_wav(listening.polish(src, tmp_path / "o.wav").output)
    # Allow the filter kernel's quarter-second skirt, and nothing more.
    skirt = RATE // 2
    assert float(np.abs(out[: RATE - skirt]).max()) < 0.02
    assert float(np.abs(out[RATE * 2 + skirt :]).max()) < 0.02


# --- Load-bearing property: babble is untouched, and provably --------------


def test_it_cannot_change_the_ratio_between_one_talker_and_others():
    """The claim in the module docstring, as a test.

    A linear filter multiplies the target and the interference by the same
    response. If someone later makes this stage non-linear to chase a bar
    recording, this fails - which is the point.
    """
    target = speechlike(5.0, f0=115, amp=0.06, seed=1)
    others = sum(
        speechlike(5.0, f0=f, amp=0.05, seed=i)
        for i, f in enumerate((98.0, 142.0, 171.0, 205.0, 231.0))
    )
    kernel = listening._kernel(RATE, 1.0, [])
    before = band_db(target, 300, 3400) - band_db(others, 300, 3400)
    after = band_db(listening._convolve(target, kernel), 300, 3400) - band_db(
        listening._convolve(others, kernel), 300, 3400
    )
    assert abs(after - before) < 0.2, (
        "filtering cannot separate one conversation from another"
    )


def test_what_it_does_not_do_is_said_where_the_option_is_offered():
    assert "does not separate" in listening.DOES_NOT
    assert "crowd" in listening.DOES_NOT
    assert "never for voice comparison" in listening.NOT_FOR_COMPARISON


def test_the_report_cannot_grow_a_claim_that_it_helped():
    """The lesson of the denoiser's level-margin figure, held shut here too."""
    report = listening.ListeningReport()
    fields = set(report.to_dict())
    for word in (
        "improve",
        "improved",
        "improvement",
        "clarity",
        "clearer",
        "intelligibility",
        "quality",
        "score",
        "better",
        "snr",
    ):
        assert not any(word in name for name in fields), (
            f"{word!r} appeared in the report: it would be claiming an outcome "
            "this processing cannot measure"
        )


# --- Hum removal -----------------------------------------------------------


def test_mains_hum_is_found_and_told_apart_from_the_other_candidate():
    speech = speechlike(6.0, amp=0.05)
    mains, harmonics = listening.detect_hum(speech + hum(6.0, 50.0), RATE)
    assert mains == 50.0
    assert harmonics >= listening._HUM_MIN_HARMONICS
    mains, _ = listening.detect_hum(speech + hum(6.0, 60.0), RATE)
    assert mains == 60.0


def test_no_hum_is_reported_when_there_is_none():
    """Notching a recording with no hum in it gouges holes in speech."""
    rng = np.random.default_rng(3)
    speech = speechlike(6.0, amp=0.2) + 0.01 * rng.standard_normal(RATE * 6)
    assert listening.detect_hum(speech, RATE) == (0.0, 0)


def test_removing_hum_costs_the_speech_almost_nothing():
    speech = speechlike(6.0, amp=0.2)
    kernel = listening._kernel(RATE, 1.0, [50.0, 100.0, 150.0, 200.0])
    plain = listening._kernel(RATE, 1.0, [])
    notched = listening._convolve(speech, kernel)
    unnotched = listening._convolve(speech, plain)
    cost = band_db(notched, 300, 3400) - band_db(unnotched, 300, 3400)
    assert abs(cost) < 0.5, f"notches cost the speech band {cost:+.2f} dB"


def test_hum_is_measurably_removed(tmp_path):
    speech = speechlike(6.0, amp=0.03)
    src = write_wav(tmp_path / "hum.wav", speech + hum(6.0, 50.0))
    report = listening.polish(src, tmp_path / "o.wav")
    assert report.hum_hz == 50.0
    assert report.hum_harmonics >= 2
    before, _ = read_wav(src)
    after, _ = read_wav(report.output)

    # Relative to the speech band, because the leveller deliberately raises
    # the whole recording and absolute band levels would hide that.
    def relative(x):
        return band_db(x, 40, 160) - band_db(x, 300, 3400)

    assert relative(after) < relative(before) - 20.0


def test_hum_removal_can_be_turned_off_and_forced(tmp_path):
    src = write_wav(tmp_path / "h.wav", speechlike(6.0, amp=0.05) + hum(6.0, 50.0))
    off = listening.polish(
        src,
        tmp_path / "off.wav",
        settings=listening.ListeningSettings(hum=listening.NO_HUM),
    )
    assert off.hum_harmonics == 0
    forced = listening.polish(
        src, tmp_path / "f.wav", settings=listening.ListeningSettings(hum=60.0)
    )
    assert forced.hum_hz == 60.0


def test_a_nonsense_mains_frequency_is_refused_in_plain_words(tmp_path):
    src = write_wav(tmp_path / "h.wav", speechlike(1.0, amp=0.1))
    with pytest.raises(listening.ListeningError) as caught:
        listening.polish(
            src,
            tmp_path / "o.wav",
            settings=listening.ListeningSettings(hum="sometimes"),
        )
    assert "mains frequency" in str(caught.value)


def test_the_lift_is_measured_on_the_speech_and_not_on_the_hum(tmp_path):
    """Measured full-band, notching hum reports the speech getting quieter.

    Arithmetically true and substantively a lie: what got quieter was the hum.
    The figure is measured in the speech band so that it answers the question
    its name asks.
    """
    rng = np.random.default_rng(7)
    speech = speechlike(6.0, amp=0.03)
    noisy = speech + hum(6.0, 50.0) + 0.004 * rng.standard_normal(RATE * 6)
    src = write_wav(tmp_path / "h.wav", noisy)
    report = listening.polish(src, tmp_path / "a.wav")
    assert report.hum_harmonics >= 2, "the fixture has hum in it"
    assert report.quiet_lift_db > 5.0, (
        "removing hum must not be reported as the speech getting quieter"
    )


def test_taking_the_hum_out_is_what_lets_the_leveller_work(tmp_path):
    """The knock-on effect, as a number: the hum was using up the headroom."""
    rng = np.random.default_rng(7)
    noisy = (
        speechlike(6.0, amp=0.03)
        + hum(6.0, 50.0)
        + 0.004 * rng.standard_normal(RATE * 6)
    )
    src = write_wav(tmp_path / "h.wav", noisy)
    notched = listening.polish(src, tmp_path / "a.wav")
    left_in = listening.polish(
        src,
        tmp_path / "b.wav",
        settings=listening.ListeningSettings(hum=listening.NO_HUM),
    )
    assert notched.quiet_lift_db > left_in.quiet_lift_db + 5.0


# --- The leveller ----------------------------------------------------------


def test_a_quiet_talker_comes_up_and_a_bang_comes_down(tmp_path):
    quiet = speechlike(4.0, amp=0.02)
    bang = np.zeros(RATE * 2)
    bang[:1200] = 0.9 * np.exp(-np.arange(1200) / 300.0)
    loud = speechlike(2.0, amp=0.4, seed=5)
    src = write_wav(tmp_path / "wide.wav", np.concatenate([quiet, bang + loud]))
    report = listening.polish(src, tmp_path / "o.wav")

    assert report.quiet_lift_db > 3.0, "the quiet talker must become audible"
    assert report.peak_dbfs_before > -1.0, "the fixture was meant to clip"
    assert report.peak_dbfs_after <= -0.9, "the copy must not clip"
    out, _ = read_wav(report.output)
    assert float(np.abs(out).max()) < 1.0


def test_the_leveller_can_be_turned_off(tmp_path):
    src = write_wav(tmp_path / "a.wav", speechlike(3.0, amp=0.02))
    report = listening.polish(
        src, tmp_path / "o.wav", settings=listening.ListeningSettings(level=False)
    )
    assert report.levelled is False
    assert abs(report.quiet_lift_db) < 3.0


def test_the_gain_curve_moves_slowly_enough_not_to_pump():
    """A gain that jumps frame to frame is audible. Attack is fast, release slow."""
    want = np.zeros(200)
    want[100:] = 12.0
    smoothed = listening._gain_curve(want, listening._GAIN_FRAME_S)
    assert smoothed[100] < 1.0, "the rise must not be instant"
    assert smoothed[-1] > 11.0, "but it must get there"
    steps = np.abs(np.diff(smoothed))
    assert steps.max() < 1.0, "no step in the gain may be a jump"


def test_framing_does_not_copy_the_whole_recording():
    """A field recording is an hour long and the transcriber wants the memory.

    Padding each analysis to a whole number of frames copied the entire
    recording twice per run; at float64 the whole chain needed 1.9 GB for an
    hour of audio.
    """
    samples = np.arange(RATE * 3 + 7, dtype="float32")
    frames, hop = listening._frames(samples, RATE, 0.05)
    assert frames.base is not None, "framing must be a view, not a copy"
    assert np.shares_memory(frames, samples)
    assert frames.shape[1] == hop


def test_the_chain_works_in_single_precision():
    """float64 doubles the memory for no audible benefit."""
    kernel = listening._kernel(RATE, 1.0, [])
    assert kernel.dtype == np.float32
    out = listening._convolve(np.zeros(RATE, dtype="float32"), kernel)
    assert out.dtype == np.float32


def test_a_long_recording_is_processed_without_falling_over(tmp_path):
    speech = np.tile(speechlike(5.0, amp=0.1), 24)  # two minutes
    src = write_wav(tmp_path / "long.wav", speech)
    report = listening.polish(src, tmp_path / "o.wav")
    out, _ = read_wav(report.output)
    assert len(out) == len(speech)
    assert report.seconds == pytest.approx(len(speech) / RATE, abs=0.1)


def test_a_short_clip_is_not_measured_sixty_times_over():
    """Replaying one line must feel instant, so measurement cannot dominate."""
    rate = 22050
    starts, span = listening._measure_spans(rate * 3, rate)
    assert len(starts) <= 3, "spans must not pile up on a three-second clip"
    # A long recording still gets a proper sample of it.
    long_starts, _ = listening._measure_spans(rate * 3600, rate)
    assert len(long_starts) == listening._MEASURE_SPANS


def test_a_playback_length_segment_is_processed_promptly(tmp_path):
    """The shape SegmentPlayer hands it: a few seconds of 22 kHz mono."""
    rate = 22050
    signal = speechlike(3.0, amp=0.04, rate=rate) + hum(3.0, 50.0, rate=rate)
    src = write_wav(tmp_path / "span.wav", signal, rate=rate)
    report = listening.polish(src, tmp_path / "o.wav")
    assert report.hum_hz == 50.0, "hum must still be found in a short span"
    assert report.quiet_lift_db > 3.0
    out, out_rate = read_wav(report.output)
    assert out_rate == rate
    assert len(out) == len(signal)


# --- Naming, guards and the report ----------------------------------------


def test_a_listening_copy_is_named_so_it_can_be_recognised(tmp_path):
    out = listening.listening_path(tmp_path / "carpark.m4a")
    assert out.name == "carpark.m4a" + listening.SUFFIX
    assert out.parent == tmp_path


def test_a_listening_copy_can_be_written_somewhere_else(tmp_path):
    out = listening.listening_path(tmp_path / "a" / "x.wav", tmp_path / "out")
    assert out.parent == tmp_path / "out"


def test_the_guard_recognises_both_kinds_of_derived_copy():
    from whispr import enhance

    assert listening.derived_kind("a.wav" + listening.SUFFIX) == "listening copy"
    assert listening.derived_kind("a.wav" + enhance.SUFFIX) == "cleaned copy"
    assert listening.derived_kind("a.wav") is None


def test_the_source_recording_is_never_written_to(tmp_path):
    speech = speechlike(2.0, amp=0.2)
    src = write_wav(tmp_path / "exhibit.wav", speech)
    before = src.read_bytes()
    listening.polish(src, tmp_path / "o.wav")
    assert src.read_bytes() == before


def test_the_report_states_the_processing_in_checkable_terms(tmp_path):
    src = write_wav(tmp_path / "h.wav", speechlike(6.0, amp=0.05) + hum(6.0, 50.0))
    report = listening.polish(src, tmp_path / "o.wav")
    line = report.filters()
    assert "85 Hz high-pass" in line
    assert "2.6 kHz" in line
    assert "50 Hz hum" in line
    assert "3:1" in line
    assert "noise floor" in line
    assert report.to_dict()["filters"] == line


def test_the_summary_warns_about_the_case_it_cannot_help(tmp_path):
    src = write_wav(tmp_path / "a.wav", speechlike(3.0, amp=0.05))
    report = listening.polish(src, tmp_path / "o.wav")
    assert "other people" in report.summary()


def test_a_missing_file_is_refused_in_plain_words(tmp_path):
    with pytest.raises(listening.ListeningError) as caught:
        listening.polish(tmp_path / "nope.wav")
    assert "is not there" in str(caught.value)


def test_an_empty_recording_is_refused_in_plain_words(tmp_path):
    src = write_wav(tmp_path / "empty.wav", np.zeros(0))
    with pytest.raises(listening.ListeningError) as caught:
        listening.polish(src, tmp_path / "o.wav")
    assert "no audio" in str(caught.value)


def test_stereo_is_noted_because_two_microphones_allow_more(tmp_path):
    speech = speechlike(2.0, amp=0.2)
    stereo = np.stack([speech, speech * 0.8], axis=1).reshape(-1)
    pcm = (stereo * 32767.0).astype("<i2")
    path = tmp_path / "pair.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes(pcm.tobytes())
    report = listening.polish(path, tmp_path / "o.wav")
    assert report.channels == 2
    assert any("channels" in w for w in report.warnings)


def test_it_works_in_a_build_with_no_model_files():
    """The point of doing this with arithmetic: nothing has to be bundled."""
    assert listening.available() is True


def test_an_unusual_sample_rate_is_handled(tmp_path):
    speech = speechlike(2.0, amp=0.2, rate=44100)
    src = write_wav(tmp_path / "hi.wav", speech, rate=44100)
    report = listening.polish(src, tmp_path / "o.wav")
    out, rate = read_wav(report.output)
    assert rate == 44100, "the listening copy keeps the recording's sample rate"
    assert len(out) == len(speech)
