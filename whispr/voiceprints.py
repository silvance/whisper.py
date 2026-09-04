"""Speaker voiceprints: recognise enrolled voices and learn them from corrections.

A *voiceprint* is a set of speaker embeddings - fixed-length vectors computed
from short spans of one person's speech with the same ONNX embedding model the
diarizer already uses. Two things build on it:

* **Recognition** - after diarization has split the audio into turns, each turn
  is embedded and compared against the enrolled voices. Attribution is
  **open set** and decided per turn: a turn is relabelled only when its own audio
  clears the acceptance threshold *and* beats the runner-up by a margin.
  Otherwise it keeps its ``SPEAKER_xx`` cluster label. The speaker in a given
  recording may simply be someone nobody enrolled, so abstaining is often the
  correct answer, and an identity is never propagated across a diarization
  cluster: sharing a cluster with a matched turn is evidence about clustering,
  not evidence that this audio is that person.
* **Enrolment** - when an operator corrects who-said-what, the corrected span's
  audio is embedded and folded into that speaker's voiceprint. For *known
  subjects* prefer the deliberate workflow in :mod:`whispr.enrollment`, which
  produces trusted reference samples with full provenance.

The heavy lifting (the ONNX embedding extractor) uses sherpa-onnx and numpy and
is loaded lazily, so importing this module carries no optional-dependency cost.
The vector maths (centroid, cosine similarity, ranking, the accept/margin
decision) is dependency-free and unit-tested; only the audio embedding needs
sherpa-onnx. Note that unit tests over synthetic vectors validate *software
behaviour* only - they say nothing about speaker-recognition accuracy, which
needs the corpus harness in :mod:`whispr.validation`.
"""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .diarization import SpeakerSegment
from .resources import bundled_embedding_model
from .thresholds import (
    BAND_HIGH,
    BAND_INTERMEDIATE,
    BAND_LOW,
    COMPARISON_HIGH_BAND,
    COMPARISON_INTERMEDIATE_BAND,
    RECOGNITION_ACCEPTANCE_THRESHOLD,
    RECOGNITION_MARGIN_THRESHOLD,
)

PathLike = Union[str, Path]

# Embedding models emit unit-less vectors; cosine similarity in [-1, 1]. Voices
# from the same person on these embeddings typically score well above 0.5, and
# different people well below, so 0.5 is a sane default acceptance bar.
DEFAULT_THRESHOLD = 0.5

# A turn shorter than this yields an unreliable embedding (too little speech to
# characterise the voice), so we neither enrol from nor recognise on it.
MIN_TURN_SECONDS = 1.0

# Cosine similarity at/above which two voiceprints are, cautiously, "could be the
# same speaker" in the 1:1 comparison tool. Conservative on purpose - this is an
# investigative aid, not a forensic identification, and quality shifts the score.
SAME_SPEAKER_THRESHOLD = 0.5

# Cap the enrolment samples kept per speaker: recent corrections matter most and
# an unbounded list would bloat the profile file. The centroid stays stable.
_MAX_VECTORS = 32

# Voiceprint audio must be 16 kHz mono (what convert_to_wav / the diarizer emit).
VOICEPRINT_SAMPLE_RATE = 16000


# -- Pure vector maths (no numpy; unit-tested) -----------------------------


def _normalize(vector: Sequence[float]) -> List[float]:
    """Return ``vector`` scaled to unit length (a zero vector is left as zeros)."""
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        return [float(x) for x in vector]
    return [x / norm for x in vector]


def centroid(vectors: Sequence[Sequence[float]]) -> List[float]:
    """Unit-length mean of ``vectors`` (empty in -> empty out)."""
    if not vectors:
        return []
    dim = len(vectors[0])
    sums = [0.0] * dim
    for vec in vectors:
        for i, value in enumerate(vec):
            sums[i] += value
    mean = [s / len(vectors) for s in sums]
    return _normalize(mean)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two equal-length vectors (0.0 for empty/degenerate)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@dataclass
class Voiceprint:
    """One enrolled speaker: a display name plus the embeddings seen for them."""

    name: str
    vectors: List[List[float]] = field(default_factory=list)

    def add(self, vector: Sequence[float]) -> None:
        """Fold a new (normalized) enrolment embedding in, capped to the most recent."""
        self.vectors.append(_normalize(vector))
        if len(self.vectors) > _MAX_VECTORS:
            self.vectors = self.vectors[-_MAX_VECTORS:]

    @property
    def centroid(self) -> List[float]:
        return centroid(self.vectors)

    def to_dict(self) -> Dict[str, object]:
        return {"name": self.name, "vectors": self.vectors}

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "Voiceprint":
        name = str(data.get("name", ""))
        raw = data.get("vectors") or []
        vectors: List[List[float]] = []
        if isinstance(raw, list):
            for vec in raw:
                if isinstance(vec, list):
                    vectors.append([float(x) for x in vec])
        return cls(name=name, vectors=vectors)


def best_match(
    embedding: Sequence[float],
    voiceprints: Sequence[Voiceprint],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> Tuple[Optional[str], float]:
    """The enrolled name whose centroid best matches ``embedding`` (>= threshold).

    Returns ``(name, score)`` for the best match at or above ``threshold``, or
    ``(None, best_score)`` when nothing clears the bar.
    """
    best_name: Optional[str] = None
    best_score = 0.0
    for vp in voiceprints:
        score = cosine_similarity(embedding, vp.centroid)
        if score > best_score:
            best_score = score
            if score >= threshold:
                best_name = vp.name
    return (best_name if best_score >= threshold else None), best_score


# -- 1:1 speaker comparison (verification) ---------------------------------


def compare_voiceprints(a: Voiceprint, b: Voiceprint) -> float:
    """Similarity of two voiceprints: cosine of their centroids (~0..1).

    Higher means the two sets of enrolled speech are more alike. This is an
    investigative *indicator*, not a calibrated probability and not forensic
    identification - see :func:`similarity_band`.
    """
    return cosine_similarity(a.centroid, b.centroid)


def similarity_band(
    score: float,
    high: float = COMPARISON_HIGH_BAND,
    intermediate: float = COMPARISON_INTERMEDIATE_BAND,
) -> Tuple[str, str]:
    """Map a comparison ``score`` to a qualitative ``(band, explanation)``.

    Deliberately *similarity* language. A cosine similarity is not a probability
    that two recordings contain the same person, so nothing here claims identity;
    the caller adds the investigative-only disclaimer.

    The band edges default to the shipped values; callers that hold a configured
    :class:`~whispr.thresholds.Thresholds` pass theirs, so a retuned build bands
    by the numbers it actually recorded in its reports.
    """
    if score >= high:
        return (
            BAND_HIGH,
            "the questioned speech is highly similar to the reference voice",
        )
    if score >= intermediate:
        return (
            BAND_INTERMEDIATE,
            "the questioned speech is moderately similar to the reference voice",
        )
    return (
        BAND_LOW,
        "the questioned speech is not notably similar to the reference voice",
    )


@dataclass
class MatchDecision:
    """The outcome of an open-set match, with the numbers behind it.

    Retained on every attribution so a result can be explained later: which
    candidate won, by how much over the runner-up, against which thresholds, and
    on how much speech.
    """

    best_name: Optional[str] = None
    best_score: float = 0.0
    second_name: Optional[str] = None
    second_score: float = 0.0
    acceptance_threshold: float = RECOGNITION_ACCEPTANCE_THRESHOLD
    margin_threshold: float = RECOGNITION_MARGIN_THRESHOLD
    speech_seconds: float = 0.0
    accepted: bool = False
    reason: str = ""

    @property
    def margin(self) -> float:
        """How far the winner beat the runner-up (its own score when alone)."""
        return round(self.best_score - self.second_score, 4)

    def to_dict(self) -> Dict[str, object]:
        return {
            "best_name": self.best_name,
            "best_score": round(self.best_score, 4),
            "second_name": self.second_name,
            "second_score": round(self.second_score, 4),
            "margin": self.margin,
            "acceptance_threshold": self.acceptance_threshold,
            "margin_threshold": self.margin_threshold,
            "speech_seconds": round(self.speech_seconds, 2),
            "accepted": self.accepted,
            "reason": self.reason,
        }


def rank_candidates(
    embedding: Sequence[float], candidates: Sequence[Tuple[str, Sequence[float]]]
) -> List[Tuple[str, float]]:
    """Score ``embedding`` against every ``(name, centroid)``, best first."""
    scored = [
        (name, cosine_similarity(embedding, centroid_vector))
        for name, centroid_vector in candidates
        if centroid_vector
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def decide_identity(
    embedding: Sequence[float],
    candidates: Sequence[Tuple[str, Sequence[float]]],
    *,
    acceptance: float = RECOGNITION_ACCEPTANCE_THRESHOLD,
    margin: float = RECOGNITION_MARGIN_THRESHOLD,
    speech_seconds: float = 0.0,
    min_speech_seconds: float = 0.0,
) -> MatchDecision:
    """Decide whether ``embedding`` belongs to a known candidate - or nobody.

    This is an **open-set** decision: the speaker may well be someone who was
    never enrolled, so abstaining is a valid and frequently correct answer. An
    identity is accepted only when the top score clears ``acceptance`` *and*
    beats the runner-up by ``margin``. Two candidates scoring 0.68 and 0.67 are
    ambiguous, not a match.
    """
    decision = MatchDecision(
        acceptance_threshold=acceptance,
        margin_threshold=margin,
        speech_seconds=speech_seconds,
    )
    ranked = rank_candidates(embedding, candidates)
    if not ranked:
        decision.reason = "No enrolled voices to compare against."
        return decision
    decision.best_name, decision.best_score = ranked[0]
    if len(ranked) > 1:
        decision.second_name, decision.second_score = ranked[1]

    if min_speech_seconds and speech_seconds < min_speech_seconds:
        decision.reason = (
            f"Only {speech_seconds:.1f}s of speech; at least "
            f"{min_speech_seconds:.1f}s is required to attribute a known speaker."
        )
        return decision
    if decision.best_score < acceptance:
        decision.reason = (
            f"Best score {decision.best_score:.2f} is below the acceptance "
            f"threshold {acceptance:.2f}."
        )
        return decision
    if decision.second_name is not None and decision.margin < margin:
        decision.reason = (
            f"Ambiguous: {decision.best_name} {decision.best_score:.2f} vs "
            f"{decision.second_name} {decision.second_score:.2f} "
            f"(margin {decision.margin:.2f} < {margin:.2f})."
        )
        return decision
    decision.accepted = True
    decision.reason = (
        f"Score {decision.best_score:.2f} >= {acceptance:.2f} with margin "
        f"{decision.margin:.2f} >= {margin:.2f}."
    )
    return decision


# -- Audio embedding (sherpa-onnx; lazy) -----------------------------------


def _read_wav_span(wav_path: PathLike, start: float, end: float) -> "Tuple[Any, int]":
    """Read ``[start, end]`` seconds of a 16-bit PCM WAV as mono float32 + rate."""
    import numpy as np

    with wave.open(str(wav_path), "rb") as wav:
        n_channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        n_frames = wav.getnframes()
        if sample_width != 2:
            raise RuntimeError("voiceprints expect a 16-bit PCM WAV")
        first = max(0, int(start * sample_rate))
        last = min(n_frames, int(math.ceil(end * sample_rate)))
        if last <= first:
            return np.zeros(0, dtype=np.float32), sample_rate
        wav.setpos(first)
        raw = wav.readframes(last - first)

    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return data, sample_rate


class SpeakerEmbedder:
    """Wraps sherpa-onnx's speaker-embedding extractor over the bundled model."""

    def __init__(self, model_path: Optional[PathLike] = None) -> None:
        import os

        try:
            import sherpa_onnx
        except ImportError as exc:
            # Plain language, and no attempt to fetch anything: on an air-gapped
            # machine a missing component is a fact to report, not to fix.
            raise RuntimeError(
                "Speaker features need sherpa-onnx, which is not installed in "
                "this build. Use a bundle that includes it (see the Self-test), "
                "or install it with:  pip install 'silvance-whisper[gui]'"
            ) from exc

        if model_path is None:
            model_path = bundled_embedding_model()
            if model_path is None:
                raise RuntimeError(
                    "No speaker-embedding model found. Bundle it under "
                    "whispr_assets/diarization/embedding.onnx (fetch_assets.py "
                    "embedding) or pass an explicit model path."
                )
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(model_path), num_threads=max(1, os.cpu_count() or 1)
        )
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)

    @property
    def dim(self) -> int:
        return int(self._extractor.dim)

    def embed_samples(self, samples: object, sample_rate: int) -> List[float]:
        """Embed a mono float32 waveform into a voiceprint vector."""
        stream = self._extractor.create_stream()
        stream.accept_waveform(sample_rate=sample_rate, waveform=samples)
        stream.input_finished()
        return list(self._extractor.compute(stream))

    def embed_span(
        self, wav_path: PathLike, start: float, end: float
    ) -> Optional[List[float]]:
        """Embed ``[start, end]`` of ``wav_path``; ``None`` if the span is empty."""
        samples, sample_rate = _read_wav_span(wav_path, start, end)
        if len(samples) == 0:
            return None
        return self.embed_samples(samples, sample_rate)


def enroll_spans(
    voiceprint: Voiceprint,
    wav_path: PathLike,
    spans: Sequence[Tuple[float, float]],
    embedder: SpeakerEmbedder,
    *,
    min_seconds: float = MIN_TURN_SECONDS,
) -> int:
    """Embed each usable ``[start, end]`` span and add it to ``voiceprint``.

    Spans shorter than ``min_seconds`` are skipped (too little speech to be a
    reliable sample). Returns the number of samples actually added.
    """
    added = 0
    for start, end in spans:
        if end - start < min_seconds:
            continue
        vector = embedder.embed_span(wav_path, start, end)
        if vector:
            voiceprint.add(vector)
            added += 1
    return added


def recognize(
    wav_path: PathLike,
    speaker_segments: Sequence[SpeakerSegment],
    voiceprints: Sequence[Voiceprint],
    embedder: SpeakerEmbedder,
    *,
    threshold: float = RECOGNITION_ACCEPTANCE_THRESHOLD,
    margin: float = RECOGNITION_MARGIN_THRESHOLD,
    min_seconds: float = MIN_TURN_SECONDS,
) -> "Tuple[List[SpeakerSegment], Dict[str, str]]":
    """Attribute diarizer turns to enrolled voices, conservatively and per turn.

    Each turn is judged **on its own audio**: it is relabelled only when its own
    embedding clears the acceptance threshold and beats the runner-up by the
    required margin. Turns that are too short to embed, that match nothing, or
    that are ambiguous keep their original ``SPEAKER_xx`` cluster label.

    An identity is deliberately *not* propagated across a diarization cluster.
    Sharing a cluster with a matched turn is evidence about clustering, not
    evidence that this audio is the known person - and when the label names a
    specific individual, guessing is the wrong trade.

    Returns the relabelled turns plus a ``{speaker_id: display_name}`` map for
    the ids that were recognised (recognised ids look like ``voice::Name``).
    """
    if not voiceprints:
        return list(speaker_segments), {}

    candidates = [(vp.name, vp.centroid) for vp in voiceprints]
    out: List[SpeakerSegment] = []
    name_map: Dict[str, str] = {}
    for seg in speaker_segments:
        duration = seg.end - seg.start
        speaker_id = seg.speaker
        if duration >= min_seconds:
            vector = embedder.embed_span(wav_path, seg.start, seg.end)
            if vector:
                decision = decide_identity(
                    vector,
                    candidates,
                    acceptance=threshold,
                    margin=margin,
                    speech_seconds=duration,
                )
                if decision.accepted and decision.best_name:
                    speaker_id = f"voice::{decision.best_name}"
                    name_map[speaker_id] = decision.best_name
        out.append(SpeakerSegment(start=seg.start, end=seg.end, speaker=speaker_id))
    return out, name_map
