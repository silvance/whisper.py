"""Save and reload a transcript as a small JSON "project" sidecar.

Transcription/diarization is expensive and the manual speaker corrections are
hand work, so this lets an operator close the app and resume editing later: the
full result (segments, words, confidences), the speaker name mapping and the
source media path are written to a ``.whispr.json`` file and read back.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from .transcription import Segment, TranscriptionResult, Word

PathLike = Union[str, Path]

# Suffix for saved projects, and the on-disk format version (so a future change
# can migrate older files instead of failing on them).
PROJECT_SUFFIX = ".whispr.json"
_VERSION = 1


def save_project(
    path: PathLike,
    result: TranscriptionResult,
    speaker_names: Optional[Dict[str, str]] = None,
    source: Optional[PathLike] = None,
) -> Path:
    """Write ``result`` (+ speaker names and source path) to ``path`` as JSON."""
    data = {
        "version": _VERSION,
        "source": str(source) if source else None,
        "speaker_names": dict(speaker_names or {}),
        "result": {
            "text": result.text,
            "language": result.language,
            "language_probability": result.language_probability,
            "duration": result.duration,
            "segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "speaker": seg.speaker,
                    "avg_logprob": seg.avg_logprob,
                    "words": [
                        {
                            "start": w.start,
                            "end": w.end,
                            "word": w.word,
                            "probability": w.probability,
                        }
                        for w in seg.words
                    ],
                }
                for seg in result.segments
            ],
        },
    }
    out = Path(path)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load_project(
    path: PathLike,
) -> Tuple[TranscriptionResult, Dict[str, str], Optional[str]]:
    """Read a project file, returning ``(result, speaker_names, source)``.

    Missing optional fields default sensibly so files written by older/newer
    versions still load.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = data.get("result", {})
    segments = [
        Segment(
            start=seg.get("start", 0.0),
            end=seg.get("end", 0.0),
            text=seg.get("text", ""),
            speaker=seg.get("speaker"),
            words=[
                Word(
                    start=w.get("start", 0.0),
                    end=w.get("end", 0.0),
                    word=w.get("word", ""),
                    probability=w.get("probability"),
                )
                for w in seg.get("words", [])
            ],
            avg_logprob=seg.get("avg_logprob"),
        )
        for seg in raw.get("segments", [])
    ]
    result = TranscriptionResult(
        text=raw.get("text", ""),
        language=raw.get("language", ""),
        language_probability=raw.get("language_probability", 0.0),
        duration=raw.get("duration", 0.0),
        segments=segments,
    )
    speaker_names = dict(data.get("speaker_names") or {})
    source = data.get("source")
    return result, speaker_names, source
