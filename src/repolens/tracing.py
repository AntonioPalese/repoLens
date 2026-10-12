"""Per-request tracing: stage timings, candidate counts and LLM usage.

A :class:`Tracer` is created per request and passed down the pipeline::

    tracer = Tracer()
    with tracer.stage("dense") as stage:
        hits = dense.retrieve(query, k)
        stage.candidates = len(hits)

Nothing is sent anywhere; records are written locally by the caller.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from repolens.models import StageRecord


@dataclass
class StageHandle:
    """Mutable handle yielded by :meth:`Tracer.stage` to annotate the stage."""

    candidates: int | None = None


@dataclass(frozen=True)
class LLMUsage:
    """Token counts and estimated cost of one LLM call."""

    tokens_in: int
    tokens_out: int
    cost_usd: float = 0.0


class Tracer:
    """Collects the stages and LLM usage of one request."""

    def __init__(self) -> None:
        self.stages: list[StageRecord] = []
        self.tokens_in = 0
        self.tokens_out = 0
        self.cost_usd = 0.0

    @contextmanager
    def stage(self, name: str) -> Iterator[StageHandle]:
        """Time the enclosed block; the stage is recorded even if it raises."""
        handle = StageHandle()
        started = time.perf_counter()
        try:
            yield handle
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000
            self.stages.append(
                StageRecord(name=name, duration_ms=elapsed_ms, candidates=handle.candidates)
            )

    def add_llm_usage(self, usage: LLMUsage) -> None:
        """Accumulate token counts and cost."""
        self.tokens_in += usage.tokens_in
        self.tokens_out += usage.tokens_out
        self.cost_usd += usage.cost_usd

    def durations(self) -> dict[str, float]:
        """Total milliseconds per stage name."""
        totals: dict[str, float] = {}
        for record in self.stages:
            totals[record.name] = totals.get(record.name, 0.0) + record.duration_ms
        return totals
