"""Speaker voiceprints: recognise enrolled voices and learn them from corrections.

A *voiceprint* is a set of speaker embeddings - fixed-length vectors computed
from short spans of one person's speech with the same ONNX embedding model the
diarizer already uses. Two things build on it:

* **Recognition** - after diarization has split the audio into turns, each turn
  is embedded and compared against the voiceprints saved in the active profile.
  A turn that confidently matches an enrolled voice is attributed to that person,
  *overriding the diarizer's clustering when it disagrees*. This is what fixes the
  common failure where a quieter speaker's turns get lumped in with a louder one:
  the quiet turns still match the quiet voiceprint.
* **Enrolment** - when an operator corrects who-said-what, the corrected span's
  audio is embedded and folded into that speaker's voiceprint, so the next
  recording for the same operation recognises them automatically.

The heavy lifting (the ONNX embedding extractor) uses sherpa-onnx and numpy and
is loaded lazily, so importing this module carries no optional-dependency cost.
The vector maths (centroid, cosine similarity, matching) is dependency-free and
unit-tested; only the actual audio embedding needs sherpa-onnx.
"""

from __future__ import annotations

import math
import wave
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .diarization import SpeakerSegment
from .resources import bundled_diarization_models

PathLike = Union[str, Path]

# Embedding models emit unit-less vectors; cosine similarity in [-1, 1]. Voices
# from the same person on these embeddings typically score well above 0.5, and
# different people well below, so 0.5 is a sane default acceptance bar.
DEFAULT_THRESHOLD = 0.5

# A turn shorter than this yields an unreliable embedding (too little speech to
# characterise the voice), so we neither enrol from nor recognise on it.
MIN_TURN_SECONDS = 1.0

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


# -- Audio embedding (sherpa-onnx; lazy) -----------------------------------


def _read_wav_span(
    wav_path: PathLike, start: float, end: float
) -> "Tuple[Any, int]":
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

        import sherpa_onnx

        if model_path is None:
            bundled = bundled_diarization_models()
            if bundled is None:
                raise RuntimeError(
                    "No speaker-embedding model found. Bundle the diarization "
                    "models under whispr_assets/diarization/ (segmentation.onnx + "
                    "embedding.onnx) or pass an explicit model path."
                )
            model_path = bundled[1]
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
    threshold: float = DEFAULT_THRESHOLD,
    min_seconds: float = MIN_TURN_SECONDS,
) -> "Tuple[List[SpeakerSegment], Dict[str, str]]":
    """Re-attribute diarizer turns to enrolled voices where they match.

    For every turn long enough to embed, we compare it to the enrolled
    voiceprints. Each original diarizer cluster is then given a "dominant" name -
    the enrolled voice most of its (matched) speech points to - and every turn is
    resolved as:

    * its own confident match, when it disagrees with the cluster's dominant name
      (this is the re-attribution that rescues a quiet speaker wrongly clustered
      with a louder one); else
    * the cluster's dominant name, when it has one (unifies and auto-names); else
    * its original cluster label, untouched.

    Returns a new ``SpeakerSegment`` list plus a ``{speaker_id: display_name}``
    map for the ids that were recognised (recognised ids look like ``voice::Name``).
    """
    if not voiceprints:
        return list(speaker_segments), {}

    # Embed each turn once (skip the too-short ones), caching by index.
    embeddings: Dict[int, List[float]] = {}
    turn_match: Dict[int, str] = {}
    for i, seg in enumerate(speaker_segments):
        if seg.end - seg.start < min_seconds:
            continue
        vector = embedder.embed_span(wav_path, seg.start, seg.end)
        if not vector:
            continue
        embeddings[i] = vector
        name, _score = best_match(vector, voiceprints, threshold=threshold)
        if name is not None:
            turn_match[i] = name

    # Dominant enrolled name per original cluster, weighted by matched duration.
    cluster_weight: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for i, seg in enumerate(speaker_segments):
        name = turn_match.get(i)
        if name is not None:
            cluster_weight[seg.speaker][name] += seg.end - seg.start
    cluster_name: Dict[str, str] = {}
    for cluster, weights in cluster_weight.items():
        cluster_name[cluster] = max(weights.items(), key=lambda kv: kv[1])[0]

    out: List[SpeakerSegment] = []
    name_map: Dict[str, str] = {}
    for i, seg in enumerate(speaker_segments):
        dominant = cluster_name.get(seg.speaker)
        turn = turn_match.get(i)
        if turn is not None and turn != dominant:
            chosen: Optional[str] = turn  # per-turn override
        else:
            chosen = dominant
        if chosen is not None:
            speaker_id = f"voice::{chosen}"
            name_map[speaker_id] = chosen
        else:
            speaker_id = seg.speaker
        out.append(SpeakerSegment(start=seg.start, end=seg.end, speaker=speaker_id))
    return out, name_map
