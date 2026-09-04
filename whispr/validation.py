"""Offline speaker-verification validation over a corpus of real recordings.

The unit tests elsewhere in this package exercise *software behaviour* on
synthetic vectors. They say nothing about whether the system can tell two people
apart. That question can only be answered by running real recordings of known
speakers through the same embedding model the application uses, and measuring how
genuine comparisons (same person, different recordings) separate from impostor
comparisons (different people).

This module is the harness for that. It is a developer/analyst tool, not part of
the shipped GUI: point it at a corpus, and it produces the score distributions,
error rates by threshold, ROC data and equal error rate needed to choose an
operational threshold empirically instead of guessing.

    python -m whispr.validation /path/to/corpus --out results

Corpus layout - either a directory of per-speaker folders::

    corpus/SPEAKER_A/call1.wav
    corpus/SPEAKER_A/call2.wav
    corpus/SPEAKER_B/meeting.wav

or a JSON/CSV manifest with ``speaker_id``, ``path`` and optional ``condition``
(channel, microphone, environment) so results can be grouped by condition.

The rate maths is dependency-free and unit-tested; only the embedding step needs
the bundled model. Nothing here reaches the network.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

PathLike = Union[str, Path]

# Duration buckets used to report performance by how much speech was available -
# the single biggest driver of speaker-comparison reliability.
DURATION_BUCKETS: Tuple[Tuple[str, float, float], ...] = (
    ("under 5s", 0.0, 5.0),
    ("5-15s", 5.0, 15.0),
    ("15-30s", 15.0, 30.0),
    ("30s+", 30.0, float("inf")),
)


@dataclass
class CorpusItem:
    """One recording of one known speaker."""

    speaker_id: str
    path: Path
    condition: str = ""


@dataclass
class EmbeddedItem:
    """A corpus item reduced to an embedding plus how much speech backed it."""

    item: CorpusItem
    embedding: List[float]
    speech_seconds: float = 0.0
    quality: str = ""


@dataclass
class Trial:
    """One comparison between two recordings, labelled genuine or impostor."""

    score: float
    genuine: bool
    speaker_a: str
    speaker_b: str
    file_a: str
    file_b: str
    min_speech_seconds: float = 0.0
    condition_a: str = ""
    condition_b: str = ""

    @property
    def same_condition(self) -> bool:
        return self.condition_a == self.condition_b


@dataclass
class OperatingPoint:
    """Error rates at one candidate threshold."""

    threshold: float
    false_accept_rate: float
    false_reject_rate: float

    @property
    def genuine_accept_rate(self) -> float:
        """True-accept rate: the ROC's y-axis against ``false_accept_rate``."""
        return 1.0 - self.false_reject_rate

    def to_dict(self) -> Dict[str, float]:
        return {
            "threshold": round(self.threshold, 4),
            "false_accept_rate": round(self.false_accept_rate, 6),
            "false_reject_rate": round(self.false_reject_rate, 6),
            "genuine_accept_rate": round(self.genuine_accept_rate, 6),
        }


# -- Corpus loading --------------------------------------------------------

_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".mp4", ".mkv"}


def load_corpus(path: PathLike) -> List[CorpusItem]:
    """Load a corpus from a directory tree or a JSON/CSV manifest."""
    source = Path(path)
    if source.is_dir():
        return _load_corpus_dir(source)
    if source.suffix.lower() == ".json":
        return _load_corpus_json(source)
    if source.suffix.lower() == ".csv":
        return _load_corpus_csv(source)
    raise ValueError(
        f"{source} is neither a corpus directory nor a .json/.csv manifest."
    )


def _load_corpus_dir(root: Path) -> List[CorpusItem]:
    items: List[CorpusItem] = []
    for speaker_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for recording in sorted(speaker_dir.rglob("*")):
            if recording.is_file() and recording.suffix.lower() in _AUDIO_SUFFIXES:
                items.append(CorpusItem(speaker_id=speaker_dir.name, path=recording))
    return items


def _load_corpus_json(manifest: Path) -> List[CorpusItem]:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    rows = data.get("items") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError("JSON manifest must be a list (or {'items': [...]}).")
    return [_item_from_row(row, manifest.parent) for row in rows]


def _load_corpus_csv(manifest: Path) -> List[CorpusItem]:
    with open(manifest, newline="", encoding="utf-8") as handle:
        return [_item_from_row(row, manifest.parent) for row in csv.DictReader(handle)]


def _item_from_row(row: Any, base: Path) -> CorpusItem:
    if not isinstance(row, dict):
        raise ValueError(f"Corpus row is not an object: {row!r}")
    speaker = str(row.get("speaker_id") or row.get("speaker") or "").strip()
    raw_path = str(row.get("path") or row.get("file") or "").strip()
    if not speaker or not raw_path:
        raise ValueError(f"Corpus row needs speaker_id and path: {row!r}")
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = base / candidate
    return CorpusItem(
        speaker_id=speaker,
        path=candidate,
        condition=str(row.get("condition") or "").strip(),
    )


# -- Trials ----------------------------------------------------------------


def build_trials(embedded: Sequence[EmbeddedItem]) -> List[Trial]:
    """Every pairing of distinct recordings, labelled genuine or impostor.

    Genuine pairs come from *different recordings of the same speaker* - never a
    recording against itself, which would measure nothing.
    """
    from .voiceprints import cosine_similarity

    trials: List[Trial] = []
    for left, right in combinations(embedded, 2):
        if left.item.path == right.item.path:
            continue
        trials.append(
            Trial(
                score=cosine_similarity(left.embedding, right.embedding),
                genuine=left.item.speaker_id == right.item.speaker_id,
                speaker_a=left.item.speaker_id,
                speaker_b=right.item.speaker_id,
                file_a=left.item.path.name,
                file_b=right.item.path.name,
                min_speech_seconds=min(left.speech_seconds, right.speech_seconds),
                condition_a=left.item.condition,
                condition_b=right.item.condition,
            )
        )
    return trials


# -- Rates -----------------------------------------------------------------


def false_accept_rate(impostor: Sequence[float], threshold: float) -> float:
    """Fraction of impostor comparisons that would be accepted at ``threshold``."""
    if not impostor:
        return 0.0
    return sum(1 for score in impostor if score >= threshold) / len(impostor)


def false_reject_rate(genuine: Sequence[float], threshold: float) -> float:
    """Fraction of genuine comparisons that would be rejected at ``threshold``."""
    if not genuine:
        return 0.0
    return sum(1 for score in genuine if score < threshold) / len(genuine)


def operating_points(
    genuine: Sequence[float],
    impostor: Sequence[float],
    thresholds: Optional[Sequence[float]] = None,
) -> List[OperatingPoint]:
    """Error rates across candidate thresholds (defaults to every observed score)."""
    if thresholds is None:
        observed = sorted({round(score, 4) for score in list(genuine) + list(impostor)})
        thresholds = observed or [0.0]
    return [
        OperatingPoint(
            threshold=float(threshold),
            false_accept_rate=false_accept_rate(impostor, threshold),
            false_reject_rate=false_reject_rate(genuine, threshold),
        )
        for threshold in thresholds
    ]


def equal_error_rate(
    points: Sequence[OperatingPoint],
) -> Tuple[Optional[float], Optional[float]]:
    """The (rate, threshold) where false accepts and false rejects are closest.

    Reported as the conventional single-number summary. It is a *summary*, not an
    operating point to adopt blindly: an investigative tool usually wants a
    threshold biased away from false accepts.
    """
    if not points:
        return None, None
    best = min(points, key=lambda p: abs(p.false_accept_rate - p.false_reject_rate))
    rate = (best.false_accept_rate + best.false_reject_rate) / 2.0
    return rate, best.threshold


def summarise(scores: Sequence[float]) -> Dict[str, float]:
    """Count/min/mean/max and quartiles of a score distribution."""
    if not scores:
        return {"count": 0}
    ordered = sorted(scores)
    count = len(ordered)

    def _pct(fraction: float) -> float:
        index = min(count - 1, max(0, int(round(fraction * (count - 1)))))
        return ordered[index]

    return {
        "count": count,
        "min": ordered[0],
        "p25": _pct(0.25),
        "median": _pct(0.5),
        "mean": sum(ordered) / count,
        "p75": _pct(0.75),
        "p95": _pct(0.95),
        "max": ordered[-1],
    }


def bucket_for(seconds: float) -> str:
    for label, low, high in DURATION_BUCKETS:
        if low <= seconds < high:
            return label
    return DURATION_BUCKETS[-1][0]


@dataclass
class ValidationReport:
    """Everything measured over a corpus, ready to export."""

    trials: List[Trial] = field(default_factory=list)
    corpus_size: int = 0
    speaker_count: int = 0
    skipped: List[str] = field(default_factory=list)
    embedding_model: str = "unknown"

    @property
    def genuine_scores(self) -> List[float]:
        return [t.score for t in self.trials if t.genuine]

    @property
    def impostor_scores(self) -> List[float]:
        return [t.score for t in self.trials if not t.genuine]

    def points(self) -> List[OperatingPoint]:
        return operating_points(self.genuine_scores, self.impostor_scores)

    def by_duration(self) -> Dict[str, Dict[str, Any]]:
        """Genuine/impostor summaries grouped by the shorter side's speech."""
        groups: Dict[str, Dict[str, List[float]]] = {}
        for trial in self.trials:
            bucket = groups.setdefault(
                bucket_for(trial.min_speech_seconds), {"genuine": [], "impostor": []}
            )
            bucket["genuine" if trial.genuine else "impostor"].append(trial.score)
        return {
            label: {
                "genuine": summarise(values["genuine"]),
                "impostor": summarise(values["impostor"]),
            }
            for label, values in groups.items()
        }

    def to_dict(self) -> Dict[str, Any]:
        rate, threshold = equal_error_rate(self.points())
        return {
            "corpus_size": self.corpus_size,
            "speaker_count": self.speaker_count,
            "embedding_model": self.embedding_model,
            "trial_count": len(self.trials),
            "genuine": summarise(self.genuine_scores),
            "impostor": summarise(self.impostor_scores),
            "equal_error_rate": rate,
            "equal_error_threshold": threshold,
            "operating_points": [p.to_dict() for p in self.points()],
            "by_duration": self.by_duration(),
            "skipped": list(self.skipped),
            "note": (
                "These figures describe this corpus only. Operational thresholds "
                "should be chosen from recordings representative of the intended "
                "deployment - channel, language, noise and duration all move "
                "these numbers."
            ),
        }

    def summary_lines(self) -> List[str]:
        rate, threshold = equal_error_rate(self.points())
        genuine = summarise(self.genuine_scores)
        impostor = summarise(self.impostor_scores)
        lines = [
            f"Corpus: {self.corpus_size} recording(s), {self.speaker_count} speaker(s)",
            f"Embedding model: {self.embedding_model}",
            f"Trials: {len(self.trials)} "
            f"({genuine.get('count', 0)} genuine, {impostor.get('count', 0)} impostor)",
        ]
        if genuine.get("count"):
            lines.append(
                f"Genuine scores:  median {genuine['median']:.3f}  "
                f"p25 {genuine['p25']:.3f}  min {genuine['min']:.3f}"
            )
        if impostor.get("count"):
            lines.append(
                f"Impostor scores: median {impostor['median']:.3f}  "
                f"p95 {impostor['p95']:.3f}  max {impostor['max']:.3f}"
            )
        if rate is not None and threshold is not None:
            lines.append(
                f"Equal error rate: {rate * 100:.1f}% at threshold {threshold:.3f}"
            )
        lines.append("")
        lines.append("Candidate thresholds (false accept / false reject):")
        for point in _threshold_table(self.points()):
            lines.append(
                f"  {point.threshold:.2f}   FA {point.false_accept_rate * 100:5.1f}%  "
                f"FR {point.false_reject_rate * 100:5.1f}%"
            )
        return lines


def _threshold_table(
    points: Sequence[OperatingPoint], step: float = 0.05
) -> List[OperatingPoint]:
    """Thin the operating points to a readable table at regular thresholds."""
    if not points:
        return []
    table: List[OperatingPoint] = []
    target = 0.0
    for point in sorted(points, key=lambda p: p.threshold):
        if point.threshold >= target:
            table.append(point)
            while target <= point.threshold:
                target += step
    return table


def write_json(report: ValidationReport, path: PathLike) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out


def write_trials_csv(report: ValidationReport, path: PathLike) -> Path:
    """Every trial, so the raw scores can be re-analysed elsewhere."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "score",
                "genuine",
                "speaker_a",
                "speaker_b",
                "file_a",
                "file_b",
                "min_speech_seconds",
                "condition_a",
                "condition_b",
            ]
        )
        for trial in report.trials:
            writer.writerow(
                [
                    f"{trial.score:.6f}",
                    int(trial.genuine),
                    trial.speaker_a,
                    trial.speaker_b,
                    trial.file_a,
                    trial.file_b,
                    f"{trial.min_speech_seconds:.2f}",
                    trial.condition_a,
                    trial.condition_b,
                ]
            )
    return out


def write_roc_csv(report: ValidationReport, path: PathLike) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "threshold",
                "false_accept_rate",
                "false_reject_rate",
                "genuine_accept_rate",
            ]
        )
        for point in report.points():
            writer.writerow(
                [
                    f"{point.threshold:.4f}",
                    f"{point.false_accept_rate:.6f}",
                    f"{point.false_reject_rate:.6f}",
                    f"{point.genuine_accept_rate:.6f}",
                ]
            )
    return out


def maybe_plot(report: ValidationReport, path: PathLike) -> Optional[Path]:
    """Plot the ROC and score histograms when matplotlib is available.

    Optional by design: matplotlib is a developer convenience and is never a
    production-bundle dependency.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001 - plots are a nicety
        return None
    points = report.points()
    figure, (roc_ax, hist_ax) = plt.subplots(1, 2, figsize=(11, 4.5))
    roc_ax.plot(
        [p.false_accept_rate for p in points],
        [p.genuine_accept_rate for p in points],
    )
    roc_ax.set_xlabel("False accept rate")
    roc_ax.set_ylabel("Genuine accept rate")
    roc_ax.set_title("ROC")
    hist_ax.hist(report.impostor_scores, bins=30, alpha=0.6, label="impostor")
    hist_ax.hist(report.genuine_scores, bins=30, alpha=0.6, label="genuine")
    hist_ax.set_xlabel("Similarity score")
    hist_ax.set_title("Score distributions")
    hist_ax.legend()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(out, dpi=120)
    plt.close(figure)
    return out


# -- Embedding a corpus (needs the bundled model) --------------------------


def embed_corpus(
    items: Iterable[CorpusItem],
    *,
    embedder: Optional[Any] = None,
    min_speech_seconds: float = 3.0,
    progress: Optional[Any] = None,
) -> Tuple[List[EmbeddedItem], List[str]]:
    """Embed each recording once, skipping those without enough usable speech."""
    from .enrollment import prepare_source
    from .quality import analyse_span

    if embedder is None:
        from .voiceprints import SpeakerEmbedder

        embedder = SpeakerEmbedder()

    embedded: List[EmbeddedItem] = []
    skipped: List[str] = []
    for item in items:
        if progress is not None:
            progress(f"Embedding {item.path.name}…")
        wav = None
        temporary = False
        try:
            wav, _digest, temporary = prepare_source(item.path)
            report = analyse_span(wav)
            if report.voiced_seconds < min_speech_seconds:
                skipped.append(
                    f"{item.path.name}: only {report.voiced_seconds:.1f}s of speech."
                )
                continue
            vector = embedder.embed_span(wav, 0.0, report.duration_seconds)
            if not vector:
                skipped.append(f"{item.path.name}: no embedding produced.")
                continue
            embedded.append(
                EmbeddedItem(
                    item=item,
                    embedding=list(vector),
                    speech_seconds=report.voiced_seconds,
                    quality=report.assessment,
                )
            )
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop a run
            skipped.append(f"{item.path.name}: {exc}")
        finally:
            if temporary and wav is not None:
                try:
                    Path(wav).unlink()
                except OSError:
                    pass
    return embedded, skipped


def validate_corpus(
    corpus: PathLike,
    *,
    embedder: Optional[Any] = None,
    progress: Optional[Any] = None,
) -> ValidationReport:
    """Load, embed and score a whole corpus."""
    from .speaker_profiles import bundled_model_identity

    items = load_corpus(corpus)
    embedded, skipped = embed_corpus(items, embedder=embedder, progress=progress)
    identity = bundled_model_identity()
    return ValidationReport(
        trials=build_trials(embedded),
        corpus_size=len(items),
        speaker_count=len({item.speaker_id for item in items}),
        skipped=skipped,
        embedding_model=identity.describe() if identity else "unknown",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI: ``python -m whispr.validation <corpus> [--out DIR]``."""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Measure speaker-verification performance over a corpus of known "
            "speakers. Offline; needs the bundled speaker-embedding model."
        )
    )
    parser.add_argument("corpus", help="Corpus directory or .json/.csv manifest")
    parser.add_argument("--out", default="validation-results", help="Output directory")
    parser.add_argument(
        "--plot", action="store_true", help="Also write plots (needs matplotlib)"
    )
    args = parser.parse_args(argv)

    report = validate_corpus(args.corpus, progress=lambda msg: print(msg))
    out = Path(args.out)
    write_json(report, out / "validation.json")
    write_trials_csv(report, out / "trials.csv")
    write_roc_csv(report, out / "roc.csv")
    if args.plot:
        plotted = maybe_plot(report, out / "validation.png")
        if plotted is None:
            print("matplotlib is not installed; skipped plots.")
    print()
    print("\n".join(report.summary_lines()))
    print()
    print(f"Wrote results to {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
