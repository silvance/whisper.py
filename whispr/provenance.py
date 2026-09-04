"""Provenance for an analysis: what produced this result, from what, with what.

A transcript or a similarity score is only defensible if you can say which
recording it came from, which models and settings processed it, and which build
of the application ran. This module assembles that record so it can be stored in
a project file and printed into a report.

Everything here is derived locally: file hashes, the bundled models already on
disk, the settings the operator chose, and the build manifest. The source media
is hashed, never modified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .buildinfo import UNKNOWN, BuildInfo, build_info, runtime_summary
from .hashing import sha256_file_or_none
from .thresholds import DEFAULTS, Thresholds

PathLike = Union[str, Path]

# Bump when this record's shape changes; readers tolerate older versions.
PROVENANCE_SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class SourceRecord:
    """The recording an analysis was performed on."""

    filename: str = ""
    original_path: str = ""
    file_size: Optional[int] = None
    sha256: Optional[str] = None

    @classmethod
    def from_path(cls, path: Optional[PathLike]) -> "SourceRecord":
        """Describe and hash ``path`` (read-only; missing files degrade cleanly)."""
        if path is None:
            return cls()
        source = Path(path)
        size: Optional[int] = None
        try:
            size = source.stat().st_size
        except OSError:
            size = None
        return cls(
            filename=source.name,
            original_path=str(source),
            file_size=size,
            sha256=sha256_file_or_none(source),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "original_path": self.original_path,
            "file_size": self.file_size,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceRecord":
        size = data.get("file_size")
        return cls(
            filename=str(data.get("filename") or ""),
            original_path=str(data.get("original_path") or ""),
            file_size=int(size) if isinstance(size, int) else None,
            sha256=(str(data["sha256"]) if data.get("sha256") else None),
        )


@dataclass
class TranscriptionProvenance:
    """How the words were produced."""

    model_name: str = UNKNOWN
    model_sha256: Optional[str] = None
    device: str = "cpu"
    compute_type: str = "int8"
    language_setting: str = "auto"
    detected_language: str = ""
    vad: bool = True
    beam_size: int = 5
    initial_prompt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_sha256": self.model_sha256,
            "device": self.device,
            "compute_type": self.compute_type,
            "language_setting": self.language_setting,
            "detected_language": self.detected_language,
            "vad": self.vad,
            "beam_size": self.beam_size,
            "initial_prompt": self.initial_prompt,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranscriptionProvenance":
        return cls(
            model_name=str(data.get("model_name") or UNKNOWN),
            model_sha256=(
                str(data["model_sha256"]) if data.get("model_sha256") else None
            ),
            device=str(data.get("device") or "cpu"),
            compute_type=str(data.get("compute_type") or "int8"),
            language_setting=str(data.get("language_setting") or "auto"),
            detected_language=str(data.get("detected_language") or ""),
            vad=bool(data.get("vad", True)),
            beam_size=int(data.get("beam_size") or 5),
            initial_prompt=str(data.get("initial_prompt") or ""),
        )


@dataclass
class DiarizationProvenance:
    """How speakers were separated (absent when diarization was not run)."""

    engine: str = ""
    engine_version: str = ""
    segmentation_model: str = ""
    segmentation_model_sha256: Optional[str] = None
    embedding_model: str = ""
    embedding_model_sha256: Optional[str] = None
    expected_speaker_count: Optional[int] = None
    clustering_threshold: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine": self.engine,
            "engine_version": self.engine_version,
            "segmentation_model": self.segmentation_model,
            "segmentation_model_sha256": self.segmentation_model_sha256,
            "embedding_model": self.embedding_model,
            "embedding_model_sha256": self.embedding_model_sha256,
            "expected_speaker_count": self.expected_speaker_count,
            "clustering_threshold": self.clustering_threshold,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiarizationProvenance":
        count = data.get("expected_speaker_count")
        threshold = data.get("clustering_threshold")
        return cls(
            engine=str(data.get("engine") or ""),
            engine_version=str(data.get("engine_version") or ""),
            segmentation_model=str(data.get("segmentation_model") or ""),
            segmentation_model_sha256=(
                str(data["segmentation_model_sha256"])
                if data.get("segmentation_model_sha256")
                else None
            ),
            embedding_model=str(data.get("embedding_model") or ""),
            embedding_model_sha256=(
                str(data["embedding_model_sha256"])
                if data.get("embedding_model_sha256")
                else None
            ),
            expected_speaker_count=int(count) if isinstance(count, int) else None,
            clustering_threshold=(
                float(threshold) if isinstance(threshold, (int, float)) else None
            ),
        )

    @classmethod
    def from_bundle(
        cls,
        *,
        engine: str,
        expected_speaker_count: Optional[int] = None,
        clustering_threshold: Optional[float] = None,
    ) -> "DiarizationProvenance":
        """Fill the model fields from the bundled diarization assets."""
        from .resources import (
            bundled_diarization_models,
            bundled_embedding_model,
            bundled_embedding_model_name,
        )

        record = cls(
            engine=engine,
            expected_speaker_count=expected_speaker_count,
            clustering_threshold=clustering_threshold,
        )
        bundled = bundled_diarization_models()
        if bundled is not None:
            segmentation, _embedding = bundled
            record.segmentation_model = segmentation.name
            record.segmentation_model_sha256 = sha256_file_or_none(segmentation)
        embedding_path = bundled_embedding_model()
        if embedding_path is not None:
            record.embedding_model = bundled_embedding_model_name() or (
                embedding_path.name
            )
            record.embedding_model_sha256 = sha256_file_or_none(embedding_path)
        return record


@dataclass
class AnalysisProvenance:
    """The full record: source, build, models and settings for one analysis."""

    source: SourceRecord = field(default_factory=SourceRecord)
    created_utc: str = field(default_factory=_utc_now)
    build: BuildInfo = field(default_factory=build_info)
    runtime: Dict[str, str] = field(default_factory=runtime_summary)
    transcription: Optional[TranscriptionProvenance] = None
    diarization: Optional[DiarizationProvenance] = None
    thresholds: Thresholds = field(default_factory=lambda: DEFAULTS)
    schema_version: int = PROVENANCE_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_utc": self.created_utc,
            "source": self.source.to_dict(),
            "application": self.build.to_dict(),
            "runtime": dict(self.runtime),
            "transcription": (
                self.transcription.to_dict() if self.transcription else None
            ),
            "diarization": (self.diarization.to_dict() if self.diarization else None),
            "speaker_matching": {
                "acceptance_threshold": self.thresholds.recognition_acceptance,
                "margin_threshold": self.thresholds.recognition_margin,
                "comparison_high": self.thresholds.comparison_high,
                "comparison_intermediate": self.thresholds.comparison_intermediate,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisProvenance":
        application = data.get("application")
        matching = data.get("speaker_matching")
        thresholds = DEFAULTS
        if isinstance(matching, dict):
            thresholds = Thresholds(
                recognition_acceptance=float(
                    matching.get(
                        "acceptance_threshold", DEFAULTS.recognition_acceptance
                    )
                ),
                recognition_margin=float(
                    matching.get("margin_threshold", DEFAULTS.recognition_margin)
                ),
                comparison_high=float(
                    matching.get("comparison_high", DEFAULTS.comparison_high)
                ),
                comparison_intermediate=float(
                    matching.get(
                        "comparison_intermediate", DEFAULTS.comparison_intermediate
                    )
                ),
            )
        transcription = data.get("transcription")
        diarization = data.get("diarization")
        runtime = data.get("runtime")
        return cls(
            source=SourceRecord.from_dict(data.get("source") or {}),
            created_utc=str(data.get("created_utc") or _utc_now()),
            build=(
                BuildInfo(**_build_kwargs(application))
                if isinstance(application, dict)
                else BuildInfo()
            ),
            runtime=(
                {str(k): str(v) for k, v in runtime.items()}
                if isinstance(runtime, dict)
                else {}
            ),
            transcription=(
                TranscriptionProvenance.from_dict(transcription)
                if isinstance(transcription, dict)
                else None
            ),
            diarization=(
                DiarizationProvenance.from_dict(diarization)
                if isinstance(diarization, dict)
                else None
            ),
            thresholds=thresholds,
            schema_version=int(data.get("schema_version") or PROVENANCE_SCHEMA_VERSION),
        )

    def describe(self) -> List[str]:
        """Report-ready lines covering source, build and models."""
        lines = [
            f"Source file: {self.source.filename or 'unknown'}",
            f"Source SHA-256: {self.source.sha256 or 'not recorded'}",
            f"Analysed (UTC): {self.created_utc}",
        ]
        lines += self.build.describe()
        if self.transcription:
            lines += [
                f"Transcription model: {self.transcription.model_name}",
                f"Transcription model SHA-256: "
                f"{self.transcription.model_sha256 or 'not recorded'}",
                f"Device / compute: {self.transcription.device} / "
                f"{self.transcription.compute_type}",
                f"Language: {self.transcription.language_setting} "
                f"(detected: {self.transcription.detected_language or 'n/a'})",
            ]
        if self.diarization:
            lines += [
                f"Diarization engine: {self.diarization.engine}",
                f"Speaker embedding model: {self.diarization.embedding_model or 'n/a'}",
                f"Speaker embedding SHA-256: "
                f"{self.diarization.embedding_model_sha256 or 'not recorded'}",
            ]
        lines += [
            f"Speaker acceptance threshold: "
            f"{self.thresholds.recognition_acceptance:.2f}",
            f"Speaker margin threshold: {self.thresholds.recognition_margin:.2f}",
        ]
        return lines


def _build_kwargs(data: Dict[str, Any]) -> Dict[str, Any]:
    """Filter a stored application record down to BuildInfo's fields."""
    allowed = {
        "build_id",
        "git_commit",
        "build_timestamp",
        "application_version",
        "platform",
        "python_version",
        "dependencies",
        "models",
    }
    return {key: value for key, value in data.items() if key in allowed}


def transcription_model_sha256(model: PathLike) -> Optional[str]:
    """SHA-256 of a CTranslate2 model's weights, when the path resolves to one.

    Bundled models are directories containing ``model.bin``; a bare size name
    (``small.en``) has no local file to hash and returns ``None`` rather than a
    fabricated value.
    """
    candidate = Path(model)
    if candidate.is_dir():
        weights = candidate / "model.bin"
        if weights.is_file():
            return sha256_file_or_none(weights)
    if candidate.is_file():
        return sha256_file_or_none(candidate)
    return None
