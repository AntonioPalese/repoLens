"""Reranker protocol: re-scores a candidate list against the query."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from repolens.models import ScoredChunk


class Reranker(Protocol):
    """Reorders first-stage candidates using a more expensive relevance model."""

    @property
    def model_name(self) -> str:
        """Model identifier recorded in traces."""
        ...

    def rerank(self, query: str, candidates: Sequence[ScoredChunk], k: int) -> list[ScoredChunk]:
        """Return the ``k`` most relevant candidates with reranker scores, best first."""
        ...
