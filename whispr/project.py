"""Save and reload a transcript as a small JSON "project" sidecar.

Transcription/diarization is expensive and the manual speaker corrections are
hand work, so this lets an operator close the app and resume editing later: the
full result (segments, words, confidences), the speaker name mapping and the
source media path are written to a ``.whispr.json`` file and read back.

Schema 2 adds a **provenance** block (see :mod:`whispr.provenance`): the source
recording's SHA-256, the models and settings that produced the analysis, and the
application build. Schema 1 files still load - they simply carry no provenance,
which is reported as unknown rather than invented. The source media itself is
never copied into the project.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from .provenance import AnalysisProvenance
from .speaker_profiles import write_json_atomic
from .transcription import Segment, TranscriptionResult, Word

PathLike = Union[str, Path]

# Suffix for saved projects, and the on-disk format version (so a future change
# can migrate older files instead of failing on them).
PROJECT_SUFFIX = ".whispr.json"
# 1: transcript + speaker names + source path.
# 2: adds the provenance block.
SCHEMA_VERSION = 2
_VERSION = SCHEMA_VERSION


def save_project(
    path: PathLike,
    result: TranscriptionResult,
    speaker_names: Optional[Dict[str, str]] = None,
    source: Optional[PathLike] = None,
    provenance: Optional[AnalysisProvenance] = None,
) -> Path:
    """Write ``result`` (+ speaker names, source path and provenance) as JSON.

    Written atomically: a crash mid-save leaves the previous project intact
    rather than a truncated file.
    """
    data = {
        "schema_version": SCHEMA_VERSION,
        "version": _VERSION,  # legacy key, for readers that predate schema_version
        "source": str(source) if source else None,
        "provenance": provenance.to_dict() if provenance else None,
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
    return write_json_atomic(path, data)


@dataclass
class ProjectRecord:
    """A loaded project: the transcript plus everything recorded around it."""

    result: TranscriptionResult
    speaker_names: Dict[str, str] = field(default_factory=dict)
    source: Optional[str] = None
    provenance: Optional[AnalysisProvenance] = None
    schema_version: int = 1

    @property
    def has_provenance(self) -> bool:
        return self.provenance is not None


def load_project_record(path: PathLike) -> ProjectRecord:
    """Read a project file into a :class:`ProjectRecord`.

    Handles both schema versions: a schema-1 file simply has no provenance.
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
    raw_provenance = data.get("provenance")
    version = data.get("schema_version") or data.get("version") or 1
    return ProjectRecord(
        result=result,
        speaker_names=dict(data.get("speaker_names") or {}),
        source=data.get("source"),
        provenance=(
            AnalysisProvenance.from_dict(raw_provenance)
            if isinstance(raw_provenance, dict)
            else None
        ),
        schema_version=int(version),
    )


def load_project(
    path: PathLike,
) -> Tuple[TranscriptionResult, Dict[str, str], Optional[str]]:
    """Read a project file, returning ``(result, speaker_names, source)``.

    Kept for existing callers; use :func:`load_project_record` when the
    provenance block is needed.
    """
    record = load_project_record(path)
    return record.result, record.speaker_names, record.source
