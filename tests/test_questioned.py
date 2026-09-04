"""Measuring a questioned speaker: what was embedded, and from which recording.

The duration a comparison reports has to be the duration behind the embedding.
An earlier version embedded one span and reported the speech of all of them,
which let a 2-second measurement pass a 3-second minimum by claiming 24.
"""

import wave

import numpy as np
import pytest

from whispr import questioned
from whispr.questioned import (
    SELECTION_DIARIZED,
    SELECTION_RANGES,
    QuestionedSpeaker,
    measure,
    measure_from_wav,
)

RATE = 16000


def _speechlike(seconds, rate=RATE, f0=120.0):
    """A voiced-looking signal: harmonics under a syllable-rate envelope."""
    t = np.arange(int(seconds * rate)) / rate
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 3.0 * t))
    tone = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in (1, 2, 3))
    return (0.3 * envelope * tone).astype(np.float32)


def _silence(seconds, rate=RATE):
    return np.zeros(int(seconds * rate), dtype=np.float32)


def _write(path, chunks):
    pcm = (np.concatenate(chunks) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(pcm.tobytes())
    return path


@pytest.fixture
def long_wav(tmp_path):
    """40 seconds of continuous usable speech."""
    return _write(tmp_path / "questioned.wav", [_speechlike(40.0)])


@pytest.fixture
def scattered_wav(tmp_path):
    """Short bursts of speech separated by silence - a speaker in a conversation."""
    chunks = []
    for _ in range(8):
        chunks.append(_speechlike(2.0))
        chunks.append(_silence(2.0))
    return _write(tmp_path / "scattered.wav", chunks)


class _CountingEmbedder:
    """Records every span it was asked to embed."""

    def __init__(self, dim=4):
        self.dim = dim
        self.calls = []

    def embed_span(self, _wav, start, end):
        self.calls.append((round(start, 2), round(end, 2)))
        return [1.0] + [0.0] * (self.dim - 1)


class _PerSpanEmbedder:
    """A different unit vector per span, so a centroid is visibly an average."""

    def __init__(self):
        self.calls = []

    def embed_span(self, _wav, start, end):
        self.calls.append((start, end))
        return [1.0, 0.0] if len(self.calls) % 2 else [0.0, 1.0]


# -- The duration reported is the duration measured --------------------------


def test_all_usable_windows_are_embedded_not_just_the_longest(long_wav):
    embedder = _CountingEmbedder()
    result = measure_from_wav(long_wav, [(0.0, 40.0)], embedder)
    assert len(embedder.calls) > 1
    assert len(result.embedded_spans) == len(embedder.calls)
    assert result.usable


def test_short_conversational_turns_are_joined_into_measurable_speech(
    scattered_wav,
):
    """A target speaking in bursts is measurable; each burst alone is not."""
    spans = [(i * 4.0, i * 4.0 + 2.0) for i in range(8)]
    result = measure_from_wav(
        scattered_wav, spans, _CountingEmbedder(), selection_mode=SELECTION_DIARIZED
    )
    # Sixteen seconds of speech in eight two-second turns: none of them reaches
    # the three-second window minimum on its own, so measuring turn by turn
    # would throw all of it away.
    assert result.usable
    assert result.aggregated
    assert result.speech_seconds > 10.0
    assert result.window_count >= 1


def test_a_single_turn_too_short_to_window_is_still_not_measured(tmp_path):
    """Joining turns is not a way around the minimum - 2 seconds is 2 seconds."""
    wav = _write(tmp_path / "brief.wav", [_speechlike(2.0)])
    result = measure_from_wav(wav, [(0.0, 2.0)], _CountingEmbedder())
    assert not result.usable
    assert not result.aggregated
    assert result.speech_seconds == 0.0
    assert result.skipped and "needed to measure a voice" in result.skipped[0]


def test_reported_speech_never_exceeds_the_selection_it_came_from(scattered_wav):
    spans = [(i * 4.0, i * 4.0 + 2.0) for i in range(8)]
    result = measure_from_wav(scattered_wav, spans, _CountingEmbedder())
    assert result.speech_seconds <= result.selected_seconds + 0.01


def test_joined_windows_are_reported_in_the_source_timeline(scattered_wav):
    """A window over the joined audio still says where in the recording it was."""
    spans = [(i * 4.0, i * 4.0 + 2.0) for i in range(8)]
    result = measure_from_wav(scattered_wav, spans, _CountingEmbedder())
    assert result.embedded_spans
    # Every reported interval lies inside one of the selected turns - never in
    # the silence or another speaker's audio between them.
    for start, end in result.embedded_spans:
        assert any(a - 0.05 <= start and end <= b + 0.05 for a, b in spans), (
            start,
            end,
        )


def test_only_the_selected_audio_is_joined(tmp_path):
    """Nothing between the selected turns may enter the measurement."""
    wav = _write(
        tmp_path / "two-speakers.wav",
        [_speechlike(6.0, f0=120), _speechlike(6.0, f0=260), _speechlike(6.0, f0=120)],
    )
    target = [(0.0, 6.0), (12.0, 18.0)]
    result = measure_from_wav(wav, target, _CountingEmbedder())
    assert result.usable
    for start, end in result.embedded_spans:
        assert any(a - 0.05 <= start and end <= b + 0.05 for a, b in target)
    # The 6-18s middle speaker contributes nothing.
    assert all(not (6.5 < start < 11.5) for start, _end in result.embedded_spans)


# -- Overlapping and repeated ranges -----------------------------------------


def test_a_repeated_range_is_measured_once(long_wav):
    once = measure_from_wav(long_wav, [(0.0, 20.0)], _CountingEmbedder())
    twice = measure_from_wav(long_wav, [(0.0, 20.0), (0.0, 20.0)], _CountingEmbedder())
    # The same ten seconds entered twice is one stretch of speech, not two.
    assert twice.speech_seconds == pytest.approx(once.speech_seconds, abs=0.5)
    assert twice.window_count == once.window_count
    assert any("measured once" in w for w in twice.warnings)


def test_overlapping_ranges_do_not_double_count_their_overlap(long_wav):
    overlapping = measure_from_wav(
        long_wav, [(0.0, 20.0), (10.0, 30.0)], _CountingEmbedder()
    )
    union = measure_from_wav(long_wav, [(0.0, 30.0)], _CountingEmbedder())
    assert overlapping.speech_seconds == pytest.approx(union.speech_seconds, abs=0.5)


def test_the_operators_original_ranges_are_still_recorded(long_wav):
    result = measure_from_wav(
        long_wav, [(0.0, 20.0), (10.0, 30.0)], _CountingEmbedder()
    )
    # What they typed is kept for the record even though it is not what was read.
    assert result.selected_spans == [(0.0, 20.0), (10.0, 30.0)]
    assert result.to_dict()["selected_spans"] == [[0.0, 20.0], [10.0, 30.0]]


def test_an_empty_selection_measures_nothing(long_wav):
    result = measure_from_wav(long_wav, [], _CountingEmbedder())
    assert not result.usable
    assert any("empty" in w for w in result.warnings)


# -- Span arithmetic ---------------------------------------------------------


def test_source_intervals_map_a_joined_window_back():
    from whispr.questioned import source_intervals

    pieces = [(0.0, 5.0, 100.0, 105.0), (5.0, 9.0, 200.0, 204.0)]
    # A window spanning the join maps to both original stretches.
    assert source_intervals(pieces, 3.0, 7.0) == [(103.0, 105.0), (200.0, 202.0)]
    # One wholly inside a piece maps to just that piece.
    assert source_intervals(pieces, 1.0, 2.0) == [(101.0, 102.0)]
    assert source_intervals(pieces, 20.0, 25.0) == []


def test_speech_seconds_never_exceeds_the_speech_actually_embedded(long_wav):
    result = measure_from_wav(long_wav, [(0.0, 40.0)], _CountingEmbedder())
    # Every embedded window is 8s or less, so the total cannot exceed the sum
    # of the windows that were embedded.
    window_span = sum(end - start for start, end in result.embedded_spans)
    assert result.speech_seconds <= window_span + 0.01
    assert result.speech_seconds > 0


def test_unusable_windows_are_skipped_with_a_reason_and_excluded(tmp_path):
    wav = _write(tmp_path / "half.wav", [_speechlike(16.0), _silence(16.0)])
    result = measure_from_wav(wav, [(0.0, 32.0)], _CountingEmbedder())
    assert result.skipped
    assert result.usable
    # The silent half contributes no speech to the reported duration.
    assert result.speech_seconds < 20.0


def test_a_selection_with_nothing_usable_measures_nothing(tmp_path):
    wav = _write(tmp_path / "quiet.wav", [_silence(30.0)])
    result = measure_from_wav(wav, [(0.0, 30.0)], _CountingEmbedder())
    assert not result.usable
    assert result.embedding == []
    assert result.speech_seconds == 0.0
    assert any("usable speech" in w for w in result.warnings)


def test_the_embedding_is_the_average_of_its_windows(long_wav):
    result = measure_from_wav(long_wav, [(0.0, 40.0)], _PerSpanEmbedder())
    # Alternating unit vectors: the result is a mixture of both, not whichever
    # single window happened to be longest, and it is unit length.
    assert len(result.embedded_spans) > 1
    assert result.embedding[0] > 0.1 and result.embedding[1] > 0.1
    assert sum(v * v for v in result.embedding) == pytest.approx(1.0, abs=1e-6)


def test_a_partly_usable_selection_warns_that_it_measured_less(tmp_path):
    wav = _write(
        tmp_path / "mixed.wav", [_speechlike(16.0), _silence(8.0), _speechlike(16.0)]
    )
    result = measure_from_wav(wav, [(0.0, 40.0)], _CountingEmbedder())
    assert result.usable
    assert any("was not usable" in w for w in result.warnings)


# -- Provenance --------------------------------------------------------------


def test_the_measurement_identifies_the_recording_it_came_from(long_wav):
    from whispr.hashing import sha256_file

    result = measure(long_wav, [(0.0, 40.0)], _CountingEmbedder())
    assert result.source_filename == long_wav.name
    assert result.source_sha256 == sha256_file(long_wav)
    assert result.source_size == long_wav.stat().st_size
    described = "\n".join(result.describe())
    assert long_wav.name in described
    assert result.source_sha256 in described


def test_the_hash_is_of_the_operators_file_not_a_converted_copy(long_wav):
    from whispr.hashing import sha256_file

    before = sha256_file(long_wav)
    result = measure(long_wav, None, _CountingEmbedder())
    assert result.source_sha256 == before
    # The source is never modified.
    assert sha256_file(long_wav) == before


def test_no_spans_means_the_whole_recording(long_wav):
    result = measure(long_wav, None, _CountingEmbedder())
    assert result.selection_mode == questioned.SELECTION_WHOLE
    assert result.selected_spans and result.selected_spans[0][0] == 0.0
    assert result.selected_spans[0][1] == pytest.approx(40.0, abs=0.1)


def test_the_selection_mode_is_recorded_for_the_report(long_wav):
    result = measure(
        long_wav, [(0.0, 20.0)], _CountingEmbedder(), selection_mode=SELECTION_RANGES
    )
    assert result.selection_mode == SELECTION_RANGES
    assert "time ranges" in "\n".join(result.describe())
    assert result.to_dict()["selection_mode"] == SELECTION_RANGES


def test_to_dict_records_both_what_was_selected_and_what_was_measured(long_wav):
    result = measure(long_wav, [(0.0, 40.0)], _CountingEmbedder())
    data = result.to_dict()
    assert data["selected_spans"] == [[0.0, 40.0]]
    assert len(data["embedded_spans"]) == len(result.embedded_spans)
    assert data["speech_seconds"] == result.speech_seconds
    assert data["source_sha256"] == result.source_sha256


def test_an_unmeasured_speaker_describes_itself_honestly():
    empty = QuestionedSpeaker()
    assert not empty.usable
    assert empty.label == "Questioned speaker"
    described = "\n".join(empty.describe())
    assert "not recorded" in described


def test_a_missing_file_is_reported_not_swallowed(tmp_path):
    with pytest.raises((FileNotFoundError, RuntimeError)):
        measure(tmp_path / "nope.wav", None, _CountingEmbedder())


def test_short_digest_abbreviates_for_status_lines():
    assert questioned.short_digest("a" * 64) == "a" * 12
    assert questioned.short_digest(None) == "unknown"


# -- The comparison built from a measurement ---------------------------------


def test_a_comparison_inherits_the_measurement_and_its_provenance(long_wav):
    from whispr.matching import compare_questioned_to_profile
    from whispr.quality import GOOD
    from whispr.speaker_profiles import (
        EmbeddingModelIdentity,
        EnrollmentSample,
        SpeakerProfile,
    )

    model = EmbeddingModelIdentity(name="m", sha256="a" * 64, vector_dimension=4)
    profile = SpeakerProfile(display_name="Subject A", embedding_model=model)
    profile.add_reference_sample(
        EnrollmentSample(
            embedding=[1.0, 0.0, 0.0, 0.0],
            speech_duration=30.0,
            quality={"assessment": GOOD},
        )
    )
    measured = measure(long_wav, [(0.0, 40.0)], _CountingEmbedder())
    comparison = compare_questioned_to_profile(
        measured, profile, questioned_model=model
    )
    # The comparison's duration is the measurement's, not the selection's.
    assert comparison.questioned_seconds == measured.speech_seconds
    assert comparison.questioned_quality == measured.quality
    assert comparison.questioned_source_sha256 == measured.source_sha256
    assert comparison.questioned_source_filename == long_wav.name
    assert comparison.questioned_window_count == len(measured.embedded_spans)


def test_a_comparison_counts_embeddings_not_source_intervals(scattered_wav):
    """Joined turns make one window map to several intervals - report windows."""
    from whispr.matching import compare_questioned_to_profile
    from whispr.quality import GOOD
    from whispr.speaker_profiles import (
        EmbeddingModelIdentity,
        EnrollmentSample,
        SpeakerProfile,
    )

    spans = [(i * 4.0, i * 4.0 + 2.0) for i in range(8)]
    measured = measure_from_wav(scattered_wav, spans, _CountingEmbedder())
    assert measured.aggregated
    # Several source intervals behind a smaller number of embeddings.
    assert len(measured.embedded_spans) > measured.window_count

    model = EmbeddingModelIdentity(name="m", sha256="a" * 64, vector_dimension=4)
    profile = SpeakerProfile(display_name="Subject A", embedding_model=model)
    profile.add_reference_sample(
        EnrollmentSample(
            embedding=[1.0, 0.0, 0.0, 0.0],
            speech_duration=30.0,
            quality={"assessment": GOOD},
        )
    )
    comparison = compare_questioned_to_profile(
        measured, profile, questioned_model=model
    )
    assert comparison.questioned_window_count == measured.window_count
    assert comparison.questioned_window_count < len(measured.embedded_spans)
    text = "\n".join(comparison.format_lines())
    assert f"measured across {measured.window_count} window(s)" in text
