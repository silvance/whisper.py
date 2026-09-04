"""Versioned speaker (subject) profiles: the reference voices of known people.

This is distinct from :mod:`whispr.profiles`, which stores an *operation* profile
(settings plus the voices met during that operation). A
:class:`SpeakerProfile` is one **subject** - a known person whose reference voice
was deliberately enrolled from historical recordings - and is the thing a
questioned recording gets compared against.

Two properties matter operationally and drive the schema:

* **Provenance.** Every sample records where it came from (file, SHA-256, time
  span, duration, when, how) and the profile records the exact embedding model
  that produced its vectors, including that model file's SHA-256 and vector
  dimension. Embeddings from different models are not comparable, so a profile
  that cannot prove which model produced it must not be silently compared.
* **Trust.** Samples an operator deliberately enrolled from known audio
  (``reference``) are trusted. Samples picked up automatically from transcript
  corrections (``learned``) are not, and stay out of the trusted model until an
  operator approves them - so one mistaken correction cannot quietly shift a
  known subject's reference centroid.

Files are written atomically and save failures are raised, not swallowed: these
are operational records, not conveniences.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .settings import settings_path
from .thresholds import LEARNED_OUTLIER_THRESHOLD
from .voiceprints import centroid, cosine_similarity

PathLike = Union[str, Path]

# Bump when the on-disk shape changes; readers migrate older versions.
SCHEMA_VERSION = 2

# A single exported subject profile (v2).
SPEAKER_PROFILE_SUFFIX = ".whispr-speaker.json"
# Legacy formats we still import (see whispr.profiles).
LEGACY_PROFILE_SUFFIX = ".whispr-profile.json"
LEGACY_VOICEPRINT_SUFFIX = ".whispr-voiceprint.json"

# Sample classes.
SAMPLE_REFERENCE = "reference"
SAMPLE_LEARNED = "learned"

# Learned samples are capped (most recent kept) so an unattended session can't
# grow a profile without bound. Reference samples are operator-curated and are
# never dropped automatically.
MAX_LEARNED_SAMPLES = 32


class ProfileError(RuntimeError):
    """Raised when a profile cannot be read, written or migrated."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# -- Embedding model identity ---------------------------------------------


@dataclass
class EmbeddingModelIdentity:
    """Which embedding model produced a profile's vectors.

    Cosine similarity between vectors from *different* models is meaningless, so
    this travels with the profile and gates comparison.
    """

    name: str
    sha256: Optional[str] = None
    vector_dimension: Optional[int] = None
    backend: str = "sherpa-onnx"

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "sha256": self.sha256,
            "vector_dimension": self.vector_dimension,
            "backend": self.backend,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingModelIdentity":
        dim = data.get("vector_dimension")
        return cls(
            name=str(data.get("name") or "unknown"),
            sha256=(str(data["sha256"]) if data.get("sha256") else None),
            vector_dimension=int(dim) if isinstance(dim, int) else None,
            backend=str(data.get("backend") or "sherpa-onnx"),
        )

    def describe(self) -> str:
        from .hashing import short

        parts = [self.name]
        if self.vector_dimension:
            parts.append(f"dim {self.vector_dimension}")
        parts.append(f"sha256 {short(self.sha256)}")
        return ", ".join(parts)


@dataclass
class Compatibility:
    """Whether two profiles' embedding models allow a meaningful comparison."""

    ok: bool
    reason: str
    # True when nothing is provably wrong but provenance is missing, so an
    # operator may explicitly choose to proceed.
    needs_confirmation: bool = False


def check_compatibility(
    a: Optional[EmbeddingModelIdentity], b: Optional[EmbeddingModelIdentity]
) -> Compatibility:
    """Decide whether vectors from ``a`` and ``b`` may be compared.

    A hash or dimension mismatch is a hard refusal. Missing identity (a legacy
    profile) is not provably wrong, but cannot be verified either, so it comes
    back as ``needs_confirmation`` for the operator to accept deliberately.
    """
    if a is None or b is None:
        return Compatibility(
            ok=False,
            reason=(
                "One of these profiles does not record which embedding model "
                "produced it (imported from an older format), so model "
                "compatibility cannot be proven."
            ),
            needs_confirmation=True,
        )
    if (
        a.vector_dimension
        and b.vector_dimension
        and (a.vector_dimension != b.vector_dimension)
    ):
        return Compatibility(
            ok=False,
            reason=(
                "Embedding dimensions differ "
                f"({a.vector_dimension} vs {b.vector_dimension}): these profiles "
                "were built with different speaker-embedding models and cannot "
                "be compared."
            ),
        )
    if a.sha256 and b.sha256:
        if a.sha256 != b.sha256:
            return Compatibility(
                ok=False,
                reason=(
                    "Different speaker-embedding models "
                    f"({a.describe()} vs {b.describe()}). Scores from different "
                    "models are not comparable; re-enrol one profile with the "
                    "other's model."
                ),
            )
        return Compatibility(ok=True, reason="Same embedding model.")
    return Compatibility(
        ok=False,
        reason=(
            "At least one profile has no embedding-model hash, so an identical "
            "model cannot be confirmed."
        ),
        needs_confirmation=True,
    )


def bundled_model_identity() -> Optional[EmbeddingModelIdentity]:
    """Identity of this build's speaker-embedding model, or ``None`` if absent.

    Hashes the bundled model file so a profile enrolled by this build can later
    be proven to match (or not match) another build's model.
    """
    from .hashing import sha256_file_or_none
    from .resources import bundled_embedding_model, bundled_embedding_model_name

    path = bundled_embedding_model()
    if path is None:
        return None
    return EmbeddingModelIdentity(
        name=bundled_embedding_model_name() or "titanet-large",
        sha256=sha256_file_or_none(path),
        vector_dimension=None,  # filled in on first embedding (see enrollment)
        backend="sherpa-onnx",
    )


# -- Samples ---------------------------------------------------------------


@dataclass
class EnrollmentSample:
    """One embedding plus everything needed to justify and audit it."""

    embedding: List[float]
    sample_type: str = SAMPLE_REFERENCE
    approved: bool = True
    sample_id: str = field(default_factory=lambda: _new_id("smp"))
    source_filename: Optional[str] = None
    source_sha256: Optional[str] = None
    source_start: Optional[float] = None
    source_end: Optional[float] = None
    speech_duration: float = 0.0
    created_utc: str = field(default_factory=_utc_now)
    quality: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @property
    def is_trusted(self) -> bool:
        """In the trusted model: enrolled reference, or an approved learned sample."""
        if self.sample_type == SAMPLE_REFERENCE:
            return self.approved
        return self.sample_type == SAMPLE_LEARNED and self.approved

    def to_dict(self) -> Dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "embedding": self.embedding,
            "sample_type": self.sample_type,
            "approved": self.approved,
            "source_filename": self.source_filename,
            "source_sha256": self.source_sha256,
            "source_start": self.source_start,
            "source_end": self.source_end,
            "speech_duration": self.speech_duration,
            "created_utc": self.created_utc,
            "quality": self.quality,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnrollmentSample":
        raw = data.get("embedding") or []
        embedding = [float(x) for x in raw] if isinstance(raw, list) else []
        sample_type = str(data.get("sample_type") or SAMPLE_REFERENCE)
        if sample_type not in (SAMPLE_REFERENCE, SAMPLE_LEARNED):
            sample_type = SAMPLE_LEARNED
        quality = data.get("quality")
        return cls(
            embedding=embedding,
            sample_type=sample_type,
            approved=bool(data.get("approved", True)),
            sample_id=str(data.get("sample_id") or _new_id("smp")),
            source_filename=(
                str(data["source_filename"]) if data.get("source_filename") else None
            ),
            source_sha256=(
                str(data["source_sha256"]) if data.get("source_sha256") else None
            ),
            source_start=_opt_float(data.get("source_start")),
            source_end=_opt_float(data.get("source_end")),
            speech_duration=float(data.get("speech_duration") or 0.0),
            created_utc=str(data.get("created_utc") or _utc_now()),
            quality=quality if isinstance(quality, dict) else {},
            notes=str(data.get("notes") or ""),
        )


def _opt_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


# -- Subject profile -------------------------------------------------------


@dataclass
class SpeakerProfile:
    """A known subject's reference voice: trusted samples plus their provenance."""

    display_name: str
    subject_id: str = field(default_factory=lambda: _new_id("subj"))
    embedding_model: Optional[EmbeddingModelIdentity] = None
    samples: List[EnrollmentSample] = field(default_factory=list)
    created_utc: str = field(default_factory=_utc_now)
    updated_utc: str = field(default_factory=_utc_now)
    notes: str = ""
    # Carried through from an imported v1 operation profile so its settings are
    # not lost on migration.
    settings: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    # -- Views over the samples -------------------------------------------

    @property
    def is_legacy(self) -> bool:
        """True when the profile cannot prove which embedding model made it."""
        return self.embedding_model is None or not self.embedding_model.sha256

    def trusted_samples(self) -> List[EnrollmentSample]:
        """Samples that form the reference model used for comparison."""
        return [s for s in self.samples if s.is_trusted and s.embedding]

    def pending_samples(self) -> List[EnrollmentSample]:
        """Learned samples awaiting operator approval (excluded from matching)."""
        return [
            s
            for s in self.samples
            if s.sample_type == SAMPLE_LEARNED and not s.approved
        ]

    def reference_samples(self) -> List[EnrollmentSample]:
        return [s for s in self.samples if s.sample_type == SAMPLE_REFERENCE]

    def learned_samples(self) -> List[EnrollmentSample]:
        return [s for s in self.samples if s.sample_type == SAMPLE_LEARNED]

    def centroid(self) -> List[float]:
        """Unit-length mean of the *trusted* embeddings (empty when none)."""
        return centroid([s.embedding for s in self.trusted_samples()])

    @property
    def total_reference_seconds(self) -> float:
        """Usable speech behind the trusted model."""
        return round(sum(s.speech_duration for s in self.trusted_samples()), 2)

    def source_files(self) -> List[str]:
        """Distinct source recordings this profile was enrolled from."""
        seen: List[str] = []
        for sample in self.samples:
            name = sample.source_filename
            if name and name not in seen:
                seen.append(name)
        return seen

    def summary(self) -> Dict[str, Any]:
        trusted = self.trusted_samples()
        return {
            "reference_sample_count": len(self.reference_samples()),
            "learned_sample_count": len(self.learned_samples()),
            "pending_sample_count": len(self.pending_samples()),
            "trusted_sample_count": len(trusted),
            "total_reference_seconds": self.total_reference_seconds,
            "source_file_count": len(self.source_files()),
        }

    # -- Mutation ----------------------------------------------------------

    def add_reference_sample(self, sample: EnrollmentSample) -> EnrollmentSample:
        """Add an operator-enrolled (trusted) sample from known audio."""
        sample.sample_type = SAMPLE_REFERENCE
        sample.approved = True
        self.samples.append(sample)
        self.updated_utc = _utc_now()
        return sample

    def propose_learned_sample(
        self,
        sample: EnrollmentSample,
        *,
        outlier_threshold: float = LEARNED_OUTLIER_THRESHOLD,
    ) -> Tuple[EnrollmentSample, bool]:
        """Record an automatically learned sample **without** trusting it.

        Learned samples never enter the trusted model on their own: they are
        stored unapproved for an operator to review, so a mistaken transcript
        correction cannot shift a known subject's reference centroid. When a
        trusted model already exists, a sample far away from it is additionally
        annotated as a probable mis-correction.

        Returns ``(sample, looks_consistent)``.
        """
        sample.sample_type = SAMPLE_LEARNED
        sample.approved = False
        consistent = True
        reference = self.centroid()
        if reference and sample.embedding:
            score = cosine_similarity(sample.embedding, reference)
            sample.quality = {
                **sample.quality,
                "similarity_to_reference": round(score, 4),
            }
            if score < outlier_threshold:
                consistent = False
                sample.notes = (sample.notes + " " if sample.notes else "") + (
                    f"Outlier: similarity {score:.2f} to the current reference is "
                    "below the review threshold; verify the correction before "
                    "approving."
                )
        self.samples.append(sample)
        self._trim_learned()
        self.updated_utc = _utc_now()
        return sample, consistent

    def approve_sample(self, sample_id: str) -> bool:
        """Promote a learned sample into the trusted model."""
        for sample in self.samples:
            if sample.sample_id == sample_id:
                sample.approved = True
                self.updated_utc = _utc_now()
                return True
        return False

    def remove_sample(self, sample_id: str) -> bool:
        """Drop a sample entirely (e.g. a rejected learned sample)."""
        before = len(self.samples)
        self.samples = [s for s in self.samples if s.sample_id != sample_id]
        if len(self.samples) != before:
            self.updated_utc = _utc_now()
            return True
        return False

    def _trim_learned(self) -> None:
        """Cap learned samples at the most recent, never touching reference ones."""
        learned = self.learned_samples()
        excess = len(learned) - MAX_LEARNED_SAMPLES
        if excess <= 0:
            return
        drop = {s.sample_id for s in learned[:excess]}
        self.samples = [s for s in self.samples if s.sample_id not in drop]

    # -- Serialisation -----------------------------------------------------

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "whispr-speaker-profile",
            "subject_id": self.subject_id,
            "display_name": self.display_name,
            "created_utc": self.created_utc,
            "updated_utc": self.updated_utc,
            "notes": self.notes,
            "settings": self.settings,
            "embedding_model": (
                self.embedding_model.to_dict() if self.embedding_model else None
            ),
            "samples": [s.to_dict() for s in self.samples],
            "summary": self.summary(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpeakerProfile":
        model = data.get("embedding_model")
        samples_raw = data.get("samples")
        samples = [
            EnrollmentSample.from_dict(item)
            for item in (samples_raw if isinstance(samples_raw, list) else [])
            if isinstance(item, dict)
        ]
        name = str(data.get("display_name") or data.get("name") or "Unnamed subject")
        settings = data.get("settings")
        return cls(
            display_name=name,
            subject_id=str(data.get("subject_id") or _new_id("subj")),
            embedding_model=(
                EmbeddingModelIdentity.from_dict(model)
                if isinstance(model, dict)
                else None
            ),
            samples=samples,
            created_utc=str(data.get("created_utc") or _utc_now()),
            updated_utc=str(data.get("updated_utc") or _utc_now()),
            notes=str(data.get("notes") or ""),
            settings=settings if isinstance(settings, dict) else {},
            schema_version=int(data.get("schema_version") or SCHEMA_VERSION),
        )


# -- Migration from the v1 formats ----------------------------------------

_MIGRATION_NOTE = (
    "Imported from a pre-versioned Whispers profile. The embedding model that "
    "produced these vectors was not recorded, so model compatibility cannot be "
    "proven."
)


def _samples_from_legacy_vectors(
    vectors: Sequence[Sequence[float]],
) -> List[EnrollmentSample]:
    """Turn v1 ``vectors`` into samples, preserving them as usable history.

    v1 vectors came from transcript corrections, so they are classed as
    ``learned``; they are kept **approved** because they were already acting as
    that voice's model - demoting them would silently change behaviour for
    existing users. The profile is still flagged legacy (no model identity).
    """
    samples: List[EnrollmentSample] = []
    for vector in vectors:
        if not vector:
            continue
        samples.append(
            EnrollmentSample(
                embedding=[float(x) for x in vector],
                sample_type=SAMPLE_LEARNED,
                approved=True,
                speech_duration=0.0,
                notes=_MIGRATION_NOTE,
            )
        )
    return samples


def migrate_legacy_voiceprint(data: Dict[str, Any]) -> SpeakerProfile:
    """Build a v2 subject profile from a v1 ``.whispr-voiceprint.json`` payload."""
    name = str(data.get("name") or "Unnamed subject")
    raw = data.get("vectors")
    vectors = raw if isinstance(raw, list) else []
    profile = SpeakerProfile(
        display_name=name,
        embedding_model=None,
        samples=_samples_from_legacy_vectors(vectors),
        notes=_MIGRATION_NOTE,
    )
    source_profile = data.get("source_profile")
    if source_profile:
        profile.notes = f"{profile.notes} Source operation profile: {source_profile}."
    return profile


def migrate_legacy_profile(data: Dict[str, Any]) -> List[SpeakerProfile]:
    """Build one v2 subject profile per voice in a v1 ``.whispr-profile.json``."""
    raw = data.get("voiceprints")
    entries = raw if isinstance(raw, list) else []
    settings = data.get("settings")
    operation = str(data.get("name") or "")
    out: List[SpeakerProfile] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        profile = migrate_legacy_voiceprint(entry)
        if isinstance(settings, dict):
            profile.settings = dict(settings)
        if operation:
            profile.notes = f"{profile.notes} Source operation profile: {operation}."
        out.append(profile)
    return out


def load_profile_file(path: PathLike) -> List[SpeakerProfile]:
    """Read any supported profile file, migrating older formats.

    Accepts a v2 subject profile, a v1 single voiceprint export, or a v1
    operation profile (which yields one subject per enrolled voice). Raises
    :class:`ProfileError` with a plain-language reason for anything else, so an
    old file is never silently discarded as "empty".
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProfileError(f"Couldn't read the profile file: {exc}") from exc
    if not isinstance(data, dict):
        raise ProfileError("This file is not a Whispers speaker profile.")

    version = data.get("schema_version")
    if data.get("kind") == "whispr-speaker-profile" or isinstance(version, int):
        if isinstance(version, int) and version > SCHEMA_VERSION:
            raise ProfileError(
                f"This profile uses schema version {version}, newer than this "
                f"build understands (version {SCHEMA_VERSION}). Update Whispers."
            )
        return [SpeakerProfile.from_dict(data)]
    if data.get("kind") == "whispr-voiceprint":
        return [migrate_legacy_voiceprint(data)]
    if isinstance(data.get("voiceprints"), list):
        migrated = migrate_legacy_profile(data)
        if not migrated:
            raise ProfileError("That profile contains no enrolled voices.")
        return migrated
    if data.get("vectors") is not None and data.get("name"):
        return [migrate_legacy_voiceprint(data)]
    raise ProfileError("This file is not a Whispers speaker profile.")


# -- Persistence -----------------------------------------------------------


def speakers_dir() -> Path:
    """Directory holding the per-user subject profiles."""
    return settings_path().parent / "speakers"


def _slug(name: str) -> str:
    cleaned = "".join(
        ch if (ch.isalnum() or ch in "-_") else "_" for ch in name.strip()
    ).strip("_")
    return cleaned.lower() or "subject"


def profile_path(profile: SpeakerProfile) -> Path:
    """Where ``profile`` is stored (stable per subject id)."""
    stem = f"{_slug(profile.display_name)}-{profile.subject_id}"
    return speakers_dir() / f"{stem}{SPEAKER_PROFILE_SUFFIX}"


def write_json_atomic(path: PathLike, payload: Mapping[str, object]) -> Path:
    """Write ``payload`` to ``path`` atomically (temp file + replace).

    A crash mid-write leaves the previous file intact rather than a truncated
    one. Errors propagate: losing an operational profile silently is worse than
    a visible failure.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(
        dir=str(out.parent), prefix=out.name, suffix=".tmp"
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, out)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return out


def save_speaker_profile(
    profile: SpeakerProfile, path: Optional[PathLike] = None
) -> Path:
    """Persist ``profile`` atomically. Raises :class:`ProfileError` on failure."""
    profile.updated_utc = _utc_now()
    target = Path(path) if path is not None else profile_path(profile)
    try:
        return write_json_atomic(target, profile.to_dict())
    except OSError as exc:
        raise ProfileError(
            f"Couldn't save the speaker profile to {target}: {exc}"
        ) from exc


def list_speaker_profiles() -> List[SpeakerProfile]:
    """Every stored subject profile, newest update first; unreadable ones skipped."""
    out: List[SpeakerProfile] = []
    try:
        entries = sorted(speakers_dir().glob(f"*{SPEAKER_PROFILE_SUFFIX}"))
    except OSError:
        return out
    for entry in entries:
        try:
            out.extend(load_profile_file(entry))
        except ProfileError:
            continue
    out.sort(key=lambda p: p.updated_utc, reverse=True)
    return out


def find_speaker_profile_by_name(display_name: str) -> Optional[SpeakerProfile]:
    """Fetch one stored subject profile by display name (case-insensitive).

    A transcript speaker label is a name, not a subject id, so this is how the
    transcript side finds the subject a correction might relate to. Ambiguity is
    resolved as "no match": with two subjects sharing a display name, guessing
    which one an operator meant is exactly the mistake this module exists to
    prevent.
    """
    wanted = (display_name or "").strip().casefold()
    if not wanted:
        return None
    matches = [
        p
        for p in list_speaker_profiles()
        if p.display_name.strip().casefold() == wanted
    ]
    return matches[0] if len(matches) == 1 else None


def delete_speaker_profile(profile: SpeakerProfile) -> None:
    """Remove a stored subject profile (best-effort)."""
    try:
        profile_path(profile).unlink()
    except OSError:
        pass
