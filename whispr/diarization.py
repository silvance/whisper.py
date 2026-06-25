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

import os
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

from .resources import bundled_diarization_models, pyannote_cache_dir
from .transcription import ProgressCallback, Segment, Word

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
    """Diarize a WAV into speaker-labelled segments.

    Uses pyannote.audio when it is installed (much better on hard/low-quality
    audio); otherwise falls back to the sherpa-onnx ONNX pipeline. Both return
    the same ``SpeakerSegment`` list, so the rest of the app is unaffected.
    """
    if _pyannote_available():
        return _diarize_pyannote(
            wav_path,
            num_speakers=num_speakers,
            progress=progress,
            on_progress=on_progress,
        )
    return _diarize_sherpa(
        wav_path,
        segmentation_model=segmentation_model,
        embedding_model=embedding_model,
        num_speakers=num_speakers,
        threshold=threshold,
        progress=progress,
        on_progress=on_progress,
    )


def _pyannote_available() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("pyannote.audio") is not None
    except ModuleNotFoundError:
        # find_spec raises (rather than returning None) when the parent package
        # itself is absent.
        return False


def _diarize_pyannote(
    wav_path: PathLike,
    *,
    num_speakers: Optional[int] = None,
    progress: Optional[ProgressCallback] = None,
    on_progress: Optional[Callable[[float], None]] = None,
) -> List[SpeakerSegment]:
    """Diarize with pyannote.audio (speaker-diarization-3.1) on CPU.

    Loads from a bundled offline HF cache (``whispr_assets/pyannote``) when
    present; otherwise from the normal HF cache using ``HF_TOKEN``.
    """
    import torch
    from pyannote.audio import Pipeline

    cache = pyannote_cache_dir()
    if cache is not None:
        os.environ.setdefault("HF_HOME", str(cache))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    if progress is not None:
        progress("Loading pyannote pipeline...")
    model_id = "pyannote/speaker-diarization-3.1"
    try:
        # pyannote 3.x / current huggingface_hub use `token`.
        pipeline = Pipeline.from_pretrained(model_id, token=token)
    except TypeError:
        # Older pyannote used `use_auth_token`.
        pipeline = Pipeline.from_pretrained(model_id, use_auth_token=token)
    if pipeline is None:
        raise RuntimeError(
            "Could not load the pyannote pipeline. Accept the licenses for "
            "pyannote/speaker-diarization-3.1 and pyannote/segmentation-3.0 on "
            "Hugging Face and set HF_TOKEN, or bundle the models under "
            "whispr_assets/pyannote."
        )
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    pipeline.to(torch.device("cpu"))

    if progress is not None:
        progress("Identifying speakers (pyannote)...")
    params = {"num_speakers": num_speakers} if num_speakers else {}
    output = pipeline(str(wav_path), **params)
    annotation = _as_annotation(output)

    segments = [
        SpeakerSegment(start=turn.start, end=turn.end, speaker=speaker)
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    segments.sort(key=lambda seg: seg.start)
    if on_progress is not None:
        on_progress(1.0)
    return segments


def _as_annotation(output):
    """Return the pyannote ``Annotation`` from a pipeline result.

    Older pyannote returns the ``Annotation`` directly; newer versions wrap it in
    a ``DiarizeOutput`` exposing it as ``.speaker_diarization`` (or ``.diarization``).
    """
    if hasattr(output, "itertracks"):
        return output
    for attr in ("speaker_diarization", "diarization"):
        inner = getattr(output, attr, None)
        if inner is not None and hasattr(inner, "itertracks"):
            return inner
    raise RuntimeError(f"Unexpected pyannote output type: {type(output).__name__}")


def _diarize_sherpa(
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

    # sherpa-onnx defaults each model to a single thread, which makes CPU
    # diarization extremely slow; use all available cores.
    num_threads = max(1, os.cpu_count() or 1)

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=seg_model
            ),
            num_threads=num_threads,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=emb_model, num_threads=num_threads
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers if num_speakers else -1,
            threshold=threshold,
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
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


def _best_overlap_speaker(
    start: float, end: float, speaker_segments: Sequence[SpeakerSegment]
) -> Optional[str]:
    best_speaker: Optional[str] = None
    best_overlap = 0.0
    for sp in speaker_segments:
        overlap = min(end, sp.end) - max(start, sp.start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = sp.speaker
    return best_speaker


def _speaker_at(t: float, speaker_segments: Sequence[SpeakerSegment]) -> Optional[str]:
    """Speaker active at time ``t``; falls back to the nearest turn if in a gap."""
    nearest: Optional[str] = None
    nearest_distance = float("inf")
    for sp in speaker_segments:
        if sp.start <= t <= sp.end:
            return sp.speaker
        distance = sp.start - t if t < sp.start else t - sp.end
        if distance < nearest_distance:
            nearest_distance = distance
            nearest = sp.speaker
    return nearest


@dataclass
class _Run:
    """A run of consecutive words attributed to one speaker (pre-segment)."""

    speaker: Optional[str]
    words: List[Word]

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end

    @property
    def duration(self) -> float:
        return self.end - self.start


def _coalesce(runs: List[_Run]) -> List[_Run]:
    """Merge adjacent runs that share a speaker."""
    out: List[_Run] = []
    for run in runs:
        if out and out[-1].speaker == run.speaker:
            out[-1].words.extend(run.words)
        else:
            out.append(_Run(speaker=run.speaker, words=list(run.words)))
    return out


def _merge_short_runs(
    runs: List[_Run],
    *,
    min_seconds: float = 0.8,
    max_words: int = 2,
) -> List[_Run]:
    """Absorb stray sub-second / 1-2 word runs into an adjacent speaker.

    pyannote's turn boundaries rarely line up exactly with Whisper's word
    boundaries, so the first word or two of a new turn can be attributed to the
    previous speaker (and vice versa). These show up as tiny runs wedged between
    longer ones. We relabel any such fragment to a neighbouring speaker - the
    only neighbour at the edges, or the longer (by word count) neighbour in the
    middle - then coalesce. This trades the occasional genuinely short
    interjection for far fewer boundary leaks.
    """
    runs = [_Run(speaker=r.speaker, words=list(r.words)) for r in runs]
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for i, run in enumerate(runs):
            if run.duration >= min_seconds or len(run.words) > max_words:
                continue
            prev = runs[i - 1] if i > 0 else None
            nxt = runs[i + 1] if i + 1 < len(runs) else None
            if prev is None and nxt is None:
                continue
            if prev is None:
                target = nxt
            elif nxt is None:
                target = prev
            else:
                target = prev if len(prev.words) >= len(nxt.words) else nxt
            if target is None or target.speaker == run.speaker:
                continue
            run.speaker = target.speaker
            changed = True
            break
        if changed:
            runs = _coalesce(runs)
    return runs


def assign_speakers(
    segments: List[Segment],
    speaker_segments: Sequence[SpeakerSegment],
) -> List[Segment]:
    """Attach speaker labels to a transcript, splitting on speaker changes.

    When per-word timestamps are available, each word is assigned to the speaker
    active at its midpoint and consecutive same-speaker words are grouped into
    runs - so a single Whisper segment that spans a speaker change is split
    correctly. A smoothing pass (:func:`_merge_short_runs`) then absorbs tiny
    stray fragments at turn boundaries, where pyannote's turn edges don't line
    up with Whisper's word edges. Each surviving run becomes one segment.
    Segments without word timestamps fall back to whole-segment overlap.
    """
    out: List[Segment] = []
    pending: List[_Run] = []

    def _emit_pending() -> None:
        for run in _merge_short_runs(pending):
            text = "".join(w.word for w in run.words).strip()
            if not text:
                continue
            out.append(
                Segment(
                    start=run.words[0].start,
                    end=run.words[-1].end,
                    text=text,
                    speaker=run.speaker,
                    words=list(run.words),
                )
            )
        pending.clear()

    for segment in segments:
        if not segment.words:
            # Mixed/no-word segment: flush what we have, then attribute whole.
            _emit_pending()
            segment.speaker = _best_overlap_speaker(
                segment.start, segment.end, speaker_segments
            )
            out.append(segment)
            continue

        for word in segment.words:
            midpoint = (word.start + word.end) / 2.0
            speaker = _speaker_at(midpoint, speaker_segments)
            if pending and pending[-1].speaker == speaker:
                pending[-1].words.append(word)
            else:
                pending.append(_Run(speaker=speaker, words=[word]))

    _emit_pending()
    return out
