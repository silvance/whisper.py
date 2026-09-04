import wave
from pathlib import Path

import numpy as np
import pytest

from whispr import enrollment
from whispr.enrollment import (
    EnrollmentResult,
    enroll_from_wav,
    spans_for_speaker,
    speaker_totals,
    windows,
)
from whispr.speaker_profiles import (
    SAMPLE_LEARNED,
    SAMPLE_REFERENCE,
    EmbeddingModelIdentity,
    ProfileError,
    SpeakerProfile,
)

RATE = 16000
rng = np.random.default_rng(7)


def _speechlike(seconds, level=0.2):
    total = int(seconds * RATE)
    signal = rng.normal(0, 0.0005, total).astype(np.float32)
    burst = int(0.3 * RATE)
    pos = 0
    while pos + burst < total:
        signal[pos : pos + burst] += rng.normal(0, level, burst).astype(np.float32)
        pos += burst + int(0.1 * RATE)
    return np.clip(signal, -1.0, 1.0)


@pytest.fixture
def wav(tmp_path):
    path = tmp_path / "subject.wav"
    pcm = (_speechlike(40.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(pcm.tobytes())
    return path


class _FakeEmbedder:
    """Returns a fixed-dimension vector; enough to exercise enrolment wiring."""

    def __init__(self, dim=192):
        self.dim = dim
        self.calls = []

    def embed_span(self, _wav, start, end):
        self.calls.append((start, end))
        return [1.0] + [0.0] * (self.dim - 1)


@pytest.fixture(autouse=True)
def _model(monkeypatch):
    monkeypatch.setattr(
        enrollment,
        "bundled_model_identity",
        lambda: EmbeddingModelIdentity(name="titanet-large", sha256="c" * 64),
    )


class _Seg:
    def __init__(self, start, end, speaker):
        self.start, self.end, self.speaker = start, end, speaker


# -- windowing -------------------------------------------------------------


def test_windows_drops_too_short_span():
    assert windows(0.0, 1.0) == []


def test_windows_keeps_short_but_usable_span_whole():
    assert windows(0.0, 5.0) == [(0.0, 5.0)]


def test_windows_splits_long_span_into_several():
    pieces = windows(0.0, 30.0, window=8.0, minimum=3.0)
    assert len(pieces) == 4  # 8+8+8+6
    assert pieces[0] == (0.0, 8.0)
    assert pieces[-1][1] == 30.0


def test_windows_drops_tiny_remainder():
    pieces = windows(0.0, 17.0, window=8.0, minimum=3.0)
    assert pieces == [(0.0, 8.0), (8.0, 16.0)]  # trailing 1s dropped


# -- enrolment -------------------------------------------------------------


def test_enrolls_multiple_embeddings_not_one(wav):
    profile = SpeakerProfile(display_name="Subject A")
    embedder = _FakeEmbedder()
    result = enroll_from_wav(
        profile, wav, [(0.0, 32.0)], embedder, source_filename="subject.wav"
    )
    assert isinstance(result, EnrollmentResult)
    # One recording must yield several independent samples.
    assert result.added_count >= 4
    assert len(profile.reference_samples()) == result.added_count
    assert len(embedder.calls) == result.added_count


def test_enrolled_samples_are_trusted_reference_with_provenance(wav):
    profile = SpeakerProfile(display_name="Subject B")
    result = enroll_from_wav(
        profile,
        wav,
        [(0.0, 16.0)],
        _FakeEmbedder(),
        source_filename="subject.wav",
        source_sha256="d" * 64,
    )
    sample = result.added[0]
    assert sample.sample_type == SAMPLE_REFERENCE
    assert sample.approved is True
    assert sample.source_filename == "subject.wav"
    assert sample.source_sha256 == "d" * 64
    assert sample.source_start == 0.0
    assert sample.speech_duration > 0
    assert sample.quality["assessment"] in ("Good", "Fair")
    assert profile.total_reference_seconds > 0


def test_embedding_model_identity_is_stamped(wav):
    profile = SpeakerProfile(display_name="Subject C")
    enroll_from_wav(profile, wav, [(0.0, 16.0)], _FakeEmbedder(dim=192))
    assert profile.embedding_model is not None
    assert profile.embedding_model.sha256 == "c" * 64
    assert profile.embedding_model.vector_dimension == 192
    assert profile.is_legacy is False


def test_refuses_to_mix_embedding_models(wav):
    profile = SpeakerProfile(
        display_name="Subject D",
        embedding_model=EmbeddingModelIdentity(
            name="other", sha256="e" * 64, vector_dimension=192
        ),
    )
    with pytest.raises(ProfileError, match="different speaker-embedding model"):
        enroll_from_wav(profile, wav, [(0.0, 16.0)], _FakeEmbedder())


def test_short_span_is_skipped_with_a_reason(wav):
    profile = SpeakerProfile(display_name="Subject E")
    result = enroll_from_wav(profile, wav, [(0.0, 1.0)], _FakeEmbedder())
    assert result.added_count == 0
    assert result.skipped and "shorter than" in result.skipped[0]


def test_silent_span_is_skipped_not_enrolled(tmp_path):
    silent = tmp_path / "silence.wav"
    with wave.open(str(silent), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(np.zeros(RATE * 30, dtype=np.int16).tobytes())
    profile = SpeakerProfile(display_name="Subject F")
    result = enroll_from_wav(profile, silent, [(0.0, 24.0)], _FakeEmbedder())
    assert result.added_count == 0
    assert profile.samples == []
    assert any("insufficient" in reason for reason in result.skipped)


def test_duplicate_source_span_is_not_enrolled_twice(wav):
    profile = SpeakerProfile(display_name="Subject G")
    common = dict(source_filename="subject.wav", source_sha256="f" * 64)
    first = enroll_from_wav(profile, wav, [(0.0, 16.0)], _FakeEmbedder(), **common)
    second = enroll_from_wav(profile, wav, [(0.0, 16.0)], _FakeEmbedder(), **common)
    assert first.added_count > 0
    assert second.added_count == 0
    assert any("already enrolled" in reason for reason in second.skipped)


def test_duplicate_allowed_when_requested(wav):
    profile = SpeakerProfile(display_name="Subject H")
    common = dict(source_filename="s.wav", source_sha256="a" * 64)
    enroll_from_wav(profile, wav, [(0.0, 16.0)], _FakeEmbedder(), **common)
    again = enroll_from_wav(
        profile, wav, [(0.0, 16.0)], _FakeEmbedder(), allow_duplicates=True, **common
    )
    assert again.added_count > 0


def test_learned_sample_type_is_not_auto_trusted(wav):
    profile = SpeakerProfile(display_name="Subject I")
    result = enroll_from_wav(
        profile, wav, [(0.0, 16.0)], _FakeEmbedder(), sample_type=SAMPLE_LEARNED
    )
    assert result.added_count > 0
    assert all(s.sample_type == SAMPLE_LEARNED for s in result.added)
    assert all(not s.approved for s in result.added)
    assert profile.trusted_samples() == []


def test_a_correction_cannot_move_a_trusted_reference_centroid(wav):
    """The transcript side proposes; it never enrols into the trusted model."""
    profile = SpeakerProfile(display_name="Subject I2")
    enroll_from_wav(profile, wav, [(0.0, 16.0)], _FakeEmbedder())
    before = list(profile.centroid())
    trusted_before = len(profile.trusted_samples())

    # A correction that was in fact wrong arrives as a learned proposal.
    enroll_from_wav(
        profile,
        wav,
        [(16.0, 32.0)],
        _FakeEmbedder(),
        sample_type=SAMPLE_LEARNED,
        notes="Proposed automatically from a transcript correction.",
    )
    assert profile.centroid() == before
    assert len(profile.trusted_samples()) == trusted_before
    assert profile.pending_samples()  # visible for review, not silently applied


def test_run_quality_is_reported(wav):
    profile = SpeakerProfile(display_name="Subject J")
    result = enroll_from_wav(profile, wav, [(0.0, 32.0)], _FakeEmbedder())
    assert result.quality is not None
    assert result.quality.assessment in ("Good", "Fair")
    assert result.added_seconds > 0


# -- diarized cluster selection -------------------------------------------


def test_spans_and_totals_for_diarized_clusters():
    segments = [
        _Seg(0.0, 10.0, "SPEAKER_00"),
        _Seg(10.0, 14.0, "SPEAKER_01"),
        _Seg(14.0, 30.0, "SPEAKER_00"),
    ]
    assert spans_for_speaker(segments, "SPEAKER_00") == [(0.0, 10.0), (14.0, 30.0)]
    assert speaker_totals(segments) == [("SPEAKER_00", 26.0), ("SPEAKER_01", 4.0)]


# -- operator-typed time ranges -------------------------------------------


def test_parse_time_ranges_mm_ss():
    from whispr.enrollment import parse_time_ranges

    assert parse_time_ranges("0:10-0:45, 1:20-2:00") == [(10.0, 45.0), (80.0, 120.0)]


def test_parse_time_ranges_plain_seconds_and_hours():
    from whispr.enrollment import parse_time_ranges

    assert parse_time_ranges("5-12") == [(5.0, 12.0)]
    assert parse_time_ranges("1:00:00-1:00:30") == [(3600.0, 3630.0)]


def test_parse_time_ranges_separators():
    from whispr.enrollment import parse_time_ranges

    assert parse_time_ranges("1-2; 3-4\n5-6") == [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]


def test_parse_time_ranges_rejects_bad_input():
    from whispr.enrollment import parse_time_ranges

    for bad in ("", "10", "abc-def", "10-5", "1:2:3:4-9"):
        with pytest.raises(ValueError):
            parse_time_ranges(bad)


# -- Preparing a source recording -------------------------------------------


def test_a_ready_wav_is_used_as_is_without_ffmpeg(wav, monkeypatch):
    """A 16 kHz mono 16-bit WAV needs no conversion - and no ffmpeg."""
    import whispr.transcription as transcription

    def _no_ffmpeg(*args, **kwargs):
        raise AssertionError("convert_to_wav should not be called for a ready WAV")

    monkeypatch.setattr(transcription, "convert_to_wav", _no_ffmpeg)
    prepared, digest, temporary = enrollment.prepare_source(wav)
    assert prepared == Path(wav)
    assert not temporary  # not a temp copy, so the caller must not delete it
    assert digest and len(digest) == 64


def test_the_hash_is_of_the_source_the_operator_chose(wav):
    from whispr.hashing import sha256_file

    _prepared, digest, _temporary = enrollment.prepare_source(wav)
    assert digest == sha256_file(wav)


def test_only_the_exact_analysis_format_skips_conversion(tmp_path, wav):
    import wave as wave_module

    assert enrollment._is_analysis_wav(Path(wav))
    # Stereo, a different rate, or 8-bit all still need converting.
    stereo = tmp_path / "stereo.wav"
    with wave_module.open(str(stereo), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00" * 400)
    assert not enrollment._is_analysis_wav(stereo)

    resampled = tmp_path / "44k.wav"
    with wave_module.open(str(resampled), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(44100)
        handle.writeframes(b"\x00" * 400)
    assert not enrollment._is_analysis_wav(resampled)

    assert not enrollment._is_analysis_wav(tmp_path / "missing.wav")
    not_a_wav = tmp_path / "clip.mp3"
    not_a_wav.write_bytes(b"not audio")
    assert not enrollment._is_analysis_wav(not_a_wav)


def test_a_wav_extension_on_a_non_wav_file_is_not_trusted(tmp_path):
    impostor = tmp_path / "actually-mp3.wav"
    impostor.write_bytes(b"ID3\x00\x00")
    assert not enrollment._is_analysis_wav(impostor)


def test_enroll_from_media_covers_the_whole_recording_by_default(wav):
    """The scripted entry point: convert if needed, then enrol every span."""
    profile = SpeakerProfile(display_name="Subject M")
    result = enrollment.enroll_from_media(profile, wav, _FakeEmbedder())
    assert result.added_count > 0
    assert profile.trusted_samples()
    # Provenance is recorded from the source the caller passed, not the copy.
    assert all(s.source_filename == Path(wav).name for s in result.added)
    assert all(s.source_sha256 for s in result.added)


def test_enroll_from_media_honours_explicit_spans(wav):
    profile = SpeakerProfile(display_name="Subject N")
    result = enrollment.enroll_from_media(
        profile, wav, _FakeEmbedder(), spans=[(0.0, 16.0)]
    )
    assert result.added_count > 0
    assert all(s.source_end <= 16.0 for s in result.added)
