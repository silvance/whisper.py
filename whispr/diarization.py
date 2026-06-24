"""Speaker diarization backend using sherpa-onnx (ONNX / CPU, no PyTorch).

Diarization answers "who spoke when": it splits the audio into speaker turns and
labels each with a speaker id. We use `sherpa-onnx
<https://github.com/k2-fsa/sherpa-onnx>`_, which runs on onnxruntime (already a
dependency for VAD), needs no PyTorch, and uses freely downloadable models that
bundle cleanly for air-gapped use.

``sherpa_onnx`` and ``numpy`` are imported lazily so that importing this module -
and the rest of the package - does not require the optional dependencies.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

from .resources import bundled_diarization_models
from .transcription import ProgressCallback, Segment

PathLike = Union[str, Path]

# Diarization models operate on 16 kHz mono audio (the format convert_to_wav emits).
DIARIZATION_SAMPLE_RATE = 16000


@dataclass
class SpeakerSegment:
    """A span of audio attributed to a single speaker."""

    start: float
    end: float
    speaker: str


def _resolve_models(
    segmentation_model: Optional[PathLike],
    embedding_model: Optional[PathLike],
) -> Tuple[str, str]:
    if segmentation_model and embedding_model:
        return str(segmentation_model), str(embedding_model)
    bundled = bundled_diarization_models()
    if bundled is None:
        raise RuntimeError(
            "No diarization models found. Bundle them under "
            "whispr_assets/diarization/ (segmentation.onnx + embedding.onnx), set "
            "WHISPR_ASSETS, or pass explicit model paths."
        )
    seg, emb = bundled
    return str(seg), str(emb)


def _read_wav_mono16k(path: PathLike):
    """Read a 16-bit PCM WAV as a mono float32 numpy array, with its sample rate."""
    import numpy as np

    with wave.open(str(path), "rb") as wav:
        n_channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        raw = wav.readframes(wav.getnframes())

    if sample_width != 2:
        raise RuntimeError("diarization expects a 16-bit PCM WAV")

    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return data, sample_rate


def diarize(
    wav_path: PathLike,
    *,
    segmentation_model: Optional[PathLike] = None,
    embedding_model: Optional[PathLike] = None,
    num_speakers: Optional[int] = None,
    threshold: float = 0.5,
    progress: Optional[ProgressCallback] = None,
    on_progress: Optional[Callable[[float], None]] = None,
) -> List[SpeakerSegment]:
    """Diarize a 16 kHz mono WAV into speaker-labelled segments.

    Parameters
    ----------
    wav_path
        Path to a 16 kHz mono PCM WAV (see ``transcription.convert_to_wav``).
    segmentation_model, embedding_model
        Paths to the sherpa-onnx segmentation and speaker-embedding ONNX models.
        If omitted, bundled models under ``whispr_assets/diarization/`` are used.
    num_speakers
        Expected number of speakers. ``None`` (default) auto-detects via
        clustering with ``threshold``.
    threshold
        Clustering threshold used when ``num_speakers`` is not given.
    progress
        Optional status callback.
    """
    try:
        import sherpa_onnx
    except ImportError as exc:
        raise RuntimeError(
            "sherpa-onnx is not installed. Install the GUI/transcription extras "
            "with:  pip install 'silvance-whisper[gui]'"
        ) from exc

    seg_model, emb_model = _resolve_models(segmentation_model, embedding_model)

    if progress is not None:
        progress("Loading diarization models...")

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=seg_model
            ),
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=emb_model),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers if num_speakers else -1,
            threshold=threshold,
        ),
    )
    if not config.validate():
        raise RuntimeError(
            "Invalid diarization configuration; check the model paths "
            f"(segmentation={seg_model}, embedding={emb_model})."
        )

    sd = sherpa_onnx.OfflineSpeakerDiarization(config)

    samples, sample_rate = _read_wav_mono16k(wav_path)
    if sample_rate != sd.sample_rate:
        raise RuntimeError(
            f"diarization expects {sd.sample_rate} Hz audio, got {sample_rate} Hz"
        )

    if progress is not None:
        progress("Identifying speakers...")

    def _on_chunk(num_processed: int, num_total: int, *_extra) -> int:
        if on_progress is not None and num_total:
            on_progress(min(num_processed / num_total, 1.0))
        return 0

    if on_progress is not None:
        result = sd.process(samples, callback=_on_chunk).sort_by_start_time()
        on_progress(1.0)
    else:
        result = sd.process(samples).sort_by_start_time()
    return [
        SpeakerSegment(start=s.start, end=s.end, speaker=f"SPEAKER_{s.speaker:02d}")
        for s in result
    ]


def assign_speakers(
    segments: List[Segment],
    speaker_segments: Sequence[SpeakerSegment],
) -> List[Segment]:
    """Label each transcript segment with the speaker it overlaps most.

    Mutates and returns ``segments`` (sets ``Segment.speaker``). A transcript
    segment that overlaps no speaker turn is left unlabelled (``None``).
    """
    for segment in segments:
        best_speaker: Optional[str] = None
        best_overlap = 0.0
        for sp in speaker_segments:
            overlap = min(segment.end, sp.end) - max(segment.start, sp.start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = sp.speaker
        segment.speaker = best_speaker
    return segments
