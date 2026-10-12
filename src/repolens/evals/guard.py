"""Prompt-injection guard evaluation.

Two measurements:

* **labelled set** (``evals/datasets/injection.jsonl``): precision / recall of the
  guard on hand-written injections and benign look-alikes, both for "flagged"
  (suspected or blocked) and for "blocked";
* **corpus scan** (optional): every chunk of a trusted index is assumed benign, so
  anything flagged there is a false positive on real-world code.

Dataset line format::

    {"id": "inj001", "text": "# ignore previous instructions ...", "label": "injection",
     "tags": ["comment"]}
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, computed_field

from repolens.evals.dataset import DatasetError
from repolens.guard.base import InjectionDetector
from repolens.guard.policy import Verdict, assess
from repolens.models import Chunk, InjectionFlag


class GuardExample(BaseModel):
    """One labelled text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    text: str
    label: Literal["injection", "benign"]
    tags: list[str] = Field(default_factory=list)


class BinaryMetrics(BaseModel):
    """Confusion counts and derived scores, with injection as the positive class."""

    tp: int
    fp: int
    fn: int
    tn: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def precision(self) -> float:
        """Share of positive predictions that are injections (1.0 if none predicted)."""
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 1.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def recall(self) -> float:
        """Share of injections caught."""
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 1.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


class ExampleOutcome(BaseModel):
    """What the guard decided for one labelled example."""

    id: str
    label: str
    tags: list[str]
    flag: InjectionFlag
    rules: list[str]


class CorpusScan(BaseModel):
    """False positives on an index assumed to be benign."""

    chunks: int
    flagged: int
    blocked: int
    rule_counts: dict[str, int]
    examples: list[str] = Field(description="Up to 10 'path:line rule-ids' of flagged chunks.")


class GuardConfigResult(BaseModel):
    """Outcome of one detector configuration."""

    name: str
    skipped: str | None = None
    flagged: BinaryMetrics | None = None
    blocked: BinaryMetrics | None = None
    outcomes: list[ExampleOutcome] = Field(default_factory=list)
    corpus: CorpusScan | None = None


class GuardReport(BaseModel):
    """Full guard evaluation output."""

    created_at: datetime
    dataset: str
    example_count: int
    block_high_severity: bool
    configs: list[GuardConfigResult]


def load_examples(path: Path) -> list[GuardExample]:
    """Parse a labelled JSONL file."""
    if not path.is_file():
        raise DatasetError(f"Dataset not found: {path}")
    examples: list[GuardExample] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                examples.append(GuardExample.model_validate_json(line))
            except ValidationError as exc:
                raise DatasetError(f"{path}:{number}: {exc}") from exc
    if not examples:
        raise DatasetError(f"{path}: no examples")
    return examples


def run_guard_eval(
    examples: Sequence[GuardExample],
    detectors: Mapping[str, InjectionDetector | str],
    dataset: Path,
    corpus: Sequence[Chunk] | None = None,
    block_high_severity: bool = True,
) -> GuardReport:
    """Evaluate each detector; a string value means "skipped, for this reason"."""
    results = [
        GuardConfigResult(name=name, skipped=detector)
        if isinstance(detector, str)
        else _evaluate(name, detector, examples, corpus, block_high_severity)
        for name, detector in detectors.items()
    ]
    return GuardReport(
        created_at=datetime.now(UTC),
        dataset=str(dataset),
        example_count=len(examples),
        block_high_severity=block_high_severity,
        configs=results,
    )


def write_guard_report(report: GuardReport, out_dir: Path) -> Path:
    """Write ``<out_dir>/guard-<UTC timestamp>.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"guard-{report.created_at:%Y%m%dT%H%M%SZ}.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path


def markdown_guard_report(report: GuardReport) -> str:
    """Markdown tables: labelled-set metrics, then misses / false positives / corpus scan."""
    lines = [
        f"Dataset: `{Path(report.dataset).name}` ({report.example_count} examples)",
        "",
        "| Detector | Flagged P | Flagged R | Flagged F1 | Blocked P | Blocked R | "
        "Corpus flagged |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c in report.configs:
        if c.skipped or c.flagged is None or c.blocked is None:
            lines.append(f"| {c.name} | skipped: {c.skipped} ||||||")
            continue
        f, b = c.flagged, c.blocked
        corpus = f"{c.corpus.flagged} / {c.corpus.chunks}" if c.corpus else "n/a"
        lines.append(
            f"| {c.name} | {f.precision:.2f} | {f.recall:.2f} | {f.f1:.2f} | "
            f"{b.precision:.2f} | {b.recall:.2f} | {corpus} |"
        )
    for c in report.configs:
        if not c.skipped:
            lines += _error_lines(c)
    return "\n".join(lines)


def _evaluate(
    name: str,
    detector: InjectionDetector,
    examples: Sequence[GuardExample],
    corpus: Sequence[Chunk] | None,
    block_high_severity: bool,
) -> GuardConfigResult:
    outcomes = []
    for example in examples:
        detections = detector.scan(example.text)
        verdict = assess(detections, block_high_severity)
        outcomes.append(
            ExampleOutcome(
                id=example.id,
                label=example.label,
                tags=example.tags,
                flag=verdict.flag,
                rules=[d.rule_id for d in detections],
            )
        )
    return GuardConfigResult(
        name=name,
        flagged=_confusion(outcomes, {InjectionFlag.SUSPECTED, InjectionFlag.BLOCKED}),
        blocked=_confusion(outcomes, {InjectionFlag.BLOCKED}),
        outcomes=outcomes,
        corpus=_scan_corpus(detector, corpus, block_high_severity) if corpus else None,
    )


def _confusion(outcomes: Sequence[ExampleOutcome], positive: set[InjectionFlag]) -> BinaryMetrics:
    counts: Counter[str] = Counter()
    for o in outcomes:
        predicted = o.flag in positive
        actual = o.label == "injection"
        counts[("t" if predicted == actual else "f") + ("p" if predicted else "n")] += 1
    return BinaryMetrics(tp=counts["tp"], fp=counts["fp"], fn=counts["fn"], tn=counts["tn"])


def _scan_corpus(
    detector: InjectionDetector, chunks: Sequence[Chunk], block_high_severity: bool
) -> CorpusScan:
    rule_counts: Counter[str] = Counter()
    flagged: list[tuple[Chunk, Verdict]] = []
    for chunk in chunks:
        detections = detector.scan(chunk.text)
        verdict = assess(detections, block_high_severity)
        if verdict.flag is not InjectionFlag.NONE:
            flagged.append((chunk, verdict))
            rule_counts.update(d.rule_id for d in detections)
    return CorpusScan(
        chunks=len(chunks),
        flagged=len(flagged),
        blocked=sum(v.flag is InjectionFlag.BLOCKED for _, v in flagged),
        rule_counts=dict(sorted(rule_counts.items())),
        examples=[
            f"{c.path}:{c.start_line} {','.join(r.split(':')[0] for r in v.reasons)}"
            for c, v in flagged[:10]
        ],
    )


def _error_lines(result: GuardConfigResult) -> list[str]:
    misses = [o for o in result.outcomes if o.label == "injection" and o.flag is InjectionFlag.NONE]
    false_pos = [
        o for o in result.outcomes if o.label == "benign" and o.flag is not InjectionFlag.NONE
    ]
    lines = ["", f"**{result.name}**"]
    lines.append(
        "- missed injections: "
        + (", ".join(f"{o.id} ({'/'.join(o.tags)})" for o in misses) or "none")
    )
    lines.append(
        "- false positives: "
        + (", ".join(f"{o.id} {o.flag.value} ({','.join(o.rules)})" for o in false_pos) or "none")
    )
    if result.corpus and result.corpus.flagged:
        rules = ", ".join(f"{r} x{n}" for r, n in result.corpus.rule_counts.items())
        lines.append(f"- corpus hits by rule: {rules}")
    return lines
