"""Benchmark runner: compare pipeline configurations on a gold dataset.

Every configuration retrieves the top 10 for every question; metrics are averaged
over questions. The first query of each configuration is a discarded warm-up, so
model loading does not pollute latency numbers.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from repolens.config import Settings
from repolens.evals.dataset import DatasetMeta, Question
from repolens.evals.metrics import judge, latency_summary, matches, mrr_at_k, ndcg_at_k, recall_at_k
from repolens.indexing.store import IndexMeta, LoadedIndex
from repolens.models import Chunk, SearchMode
from repolens.rerank.base import Reranker
from repolens.retrieval.pipeline import SearchEngine, build_reranker
from repolens.tracing import Tracer

logger = logging.getLogger(__name__)

EVAL_DEPTH = 10
RECALL_KS = (1, 5, 10)
METRIC_NAMES = ("recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10")

RerankerChoice = Literal["none", "cross_encoder", "llm_judge"]


class EvalError(Exception):
    """The benchmark cannot run (bad config selection, gold not in the index...)."""


@dataclass(frozen=True)
class EvalConfig:
    """One pipeline variant to benchmark."""

    name: str
    mode: SearchMode
    reranker: RerankerChoice


CONFIGS: tuple[EvalConfig, ...] = (
    EvalConfig("bm25", SearchMode.BM25, "none"),
    EvalConfig("dense", SearchMode.DENSE, "none"),
    EvalConfig("hybrid", SearchMode.HYBRID, "none"),
    EvalConfig("hybrid+ce", SearchMode.HYBRID, "cross_encoder"),
    EvalConfig("hybrid+llm", SearchMode.HYBRID, "llm_judge"),
)


class QuestionResult(BaseModel):
    """Outcome of one question under one configuration."""

    id: str
    tags: list[str]
    gold_ranks: list[int | None] = Field(description="1-based rank of each gold item, or null.")
    metrics: dict[str, float]
    stages_ms: dict[str, float]


class ConfigResult(BaseModel):
    """Aggregated outcome of one configuration."""

    name: str
    mode: SearchMode
    reranker_model: str | None = None
    skipped: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    by_tag: dict[str, dict[str, float]] = Field(default_factory=dict)
    latency: dict[str, dict[str, float]] = Field(default_factory=dict)
    llm_tokens_in: int = 0
    llm_tokens_out: int = 0
    llm_cost_usd: float = 0.0
    questions: list[QuestionResult] = Field(default_factory=list)


class EvalReport(BaseModel):
    """Full benchmark output, written to ``evals/results/<timestamp>.json``."""

    created_at: datetime
    dataset: str
    dataset_meta: DatasetMeta | None
    question_count: int
    index: IndexMeta
    settings: dict[str, object]
    warnings: list[str]
    configs: list[ConfigResult]


def select_configs(spec: str) -> list[EvalConfig]:
    """Parse ``"all"`` or a comma-separated list of configuration names."""
    if spec.strip() == "all":
        return list(CONFIGS)
    by_name = {c.name: c for c in CONFIGS}
    names = [n.strip() for n in spec.split(",") if n.strip()]
    unknown = [n for n in names if n not in by_name]
    if unknown or not names:
        raise EvalError(f"Unknown configs {unknown}; choose from {sorted(by_name)} or 'all'")
    return [by_name[n] for n in names]


def validate_gold(questions: Sequence[Question], chunks: Sequence[Chunk]) -> list[str]:
    """Return a problem description for every gold item no indexed chunk matches."""
    return [
        f"{q.id}: no chunk matches {g.model_dump(exclude_none=True)}"
        for q in questions
        for g in q.gold
        if not any(matches(c, g) for c in chunks)
    ]


def run_eval(
    index: LoadedIndex,
    questions: Sequence[Question],
    settings: Settings,
    configs: Sequence[EvalConfig],
    dataset: Path,
    dataset_meta: DatasetMeta | None = None,
) -> EvalReport:
    """Run every configuration over every question."""
    problems = validate_gold(questions, index.chunks)
    if problems:
        raise EvalError("Gold items missing from the index:\n  " + "\n  ".join(problems))
    warnings = _commit_warnings(dataset_meta, index.meta)
    results = [_run_config(index, questions, settings, config) for config in configs]
    return EvalReport(
        created_at=datetime.now(UTC),
        dataset=str(dataset),
        dataset_meta=dataset_meta,
        question_count=len(questions),
        index=index.meta,
        settings={
            "embedding_model": index.meta.embedding_model,
            "cross_encoder_model": settings.rerank.cross_encoder_model,
            "llm_judge_model": settings.rerank.llm_judge.model,
            "candidate_k": settings.retrieval.candidate_k,
            "rrf_k": settings.retrieval.rrf_k,
            "rerank_top_n": settings.rerank.top_n,
        },
        warnings=warnings,
        configs=results,
    )


def write_report(report: EvalReport, out_dir: Path) -> Path:
    """Write the report as ``<out_dir>/<UTC timestamp>.json`` and return the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{report.created_at:%Y%m%dT%H%M%SZ}.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path


def markdown_report(report: EvalReport) -> str:
    """Markdown summary: overall metrics, then Recall@10 / MRR@10 per question tag."""
    commit = report.index.commit[:12] if report.index.commit else "n/a"
    lines = [
        f"Dataset: `{Path(report.dataset).name}` ({report.question_count} questions) · "
        f"commit `{commit}` · embeddings `{report.index.embedding_model}`",
        "",
        "| Config | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Mean ms | p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in report.configs:
        if c.skipped:
            lines.append(f"| {c.name} | skipped: {c.skipped} |||||||")
            continue
        m, total = c.metrics, c.latency["total"]
        values = " | ".join(f"{m[name]:.3f}" for name in METRIC_NAMES)
        lines.append(f"| {c.name} | {values} | {total['mean_ms']:.1f} | {total['p95_ms']:.1f} |")
    lines += _tag_table(report)
    return "\n".join(lines)


def _tag_table(report: EvalReport) -> list[str]:
    ran = [c for c in report.configs if not c.skipped]
    tags = sorted({t for c in ran for t in c.by_tag})
    if not tags:
        return []
    header = " | ".join(tags)
    lines = [
        "",
        "Recall@10 / MRR@10 by question tag:",
        "",
        f"| Config | {header} |",
        "|---|" + "---:|" * len(tags),
    ]
    for c in ran:
        cells = " | ".join(
            f"{c.by_tag[t]['recall@10']:.2f} / {c.by_tag[t]['mrr@10']:.2f}" for t in tags
        )
        lines.append(f"| {c.name} | {cells} |")
    return lines


def _run_config(
    index: LoadedIndex, questions: Sequence[Question], settings: Settings, config: EvalConfig
) -> ConfigResult:
    reranker, skipped = _reranker_for(config, settings)
    result = ConfigResult(
        name=config.name,
        mode=config.mode,
        reranker_model=reranker.model_name if reranker else None,
        skipped=skipped,
    )
    if skipped:
        logger.warning("Skipping %s: %s", config.name, skipped)
        return result
    engine = SearchEngine(index, settings, reranker)
    rerank = reranker is not None
    engine.search(questions[0].query, k=EVAL_DEPTH, mode=config.mode, rerank=rerank)  # warm-up
    logger.info("Running %s on %d questions", config.name, len(questions))
    for question in questions:
        tracer = Tracer()
        started = time.perf_counter()
        hits = engine.search(
            question.query, k=EVAL_DEPTH, mode=config.mode, rerank=rerank, tracer=tracer
        ).hits
        stages = tracer.durations() | {"total": (time.perf_counter() - started) * 1000}
        result.questions.append(_score(question, judge(hits, question.gold), stages))
        result.llm_tokens_in += tracer.tokens_in
        result.llm_tokens_out += tracer.tokens_out
        result.llm_cost_usd += tracer.cost_usd
    _aggregate(result)
    return result


def _reranker_for(config: EvalConfig, settings: Settings) -> tuple[Reranker | None, str | None]:
    if config.reranker == "none":
        return None, None
    if config.reranker == "llm_judge" and not settings.rerank.llm_judge.configured:
        return None, "no LLM endpoint configured"
    return build_reranker(settings.rerank.model_copy(update={"backend": config.reranker})), None


def _score(
    question: Question, judged: list[int | None], stages: dict[str, float]
) -> QuestionResult:
    n_gold = len(question.gold)
    metrics = {f"recall@{k}": recall_at_k(judged, n_gold, k) for k in RECALL_KS}
    metrics["mrr@10"] = mrr_at_k(judged, EVAL_DEPTH)
    metrics["ndcg@10"] = ndcg_at_k(judged, n_gold, EVAL_DEPTH)
    ranks: list[int | None] = [None] * n_gold
    for rank, gold_index in enumerate(judged, start=1):
        if gold_index is not None:
            ranks[gold_index] = rank
    return QuestionResult(
        id=question.id, tags=question.tags, gold_ranks=ranks, metrics=metrics, stages_ms=stages
    )


def _aggregate(result: ConfigResult) -> None:
    result.metrics = _mean_metrics(result.questions)
    by_tag: dict[str, list[QuestionResult]] = defaultdict(list)
    for q in result.questions:
        for tag in q.tags:
            by_tag[tag].append(q)
    result.by_tag = {tag: _mean_metrics(qs) for tag, qs in sorted(by_tag.items())}
    stage_samples: dict[str, list[float]] = defaultdict(list)
    for q in result.questions:
        for stage, ms in q.stages_ms.items():
            stage_samples[stage].append(ms)
    result.latency = {stage: latency_summary(s) for stage, s in sorted(stage_samples.items())}


def _mean_metrics(questions: Sequence[QuestionResult]) -> dict[str, float]:
    return {name: sum(q.metrics[name] for q in questions) / len(questions) for name in METRIC_NAMES}


def _commit_warnings(meta: DatasetMeta | None, index: IndexMeta) -> list[str]:
    if meta is None:
        return ["Dataset has no .meta.json; cannot check the indexed commit."]
    if index.commit != meta.commit:
        return [f"Index commit {index.commit} differs from dataset commit {meta.commit}."]
    return []
