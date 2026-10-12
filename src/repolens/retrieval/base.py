"""Retriever protocol shared by BM25, dense and hybrid retrieval."""

from __future__ import annotations

from typing import Protocol

from repolens.models import ScoredChunk


class Retriever(Protocol):
    """Returns the ``k`` best chunks for a query, best first."""

    @property
    def name(self) -> str:
        """Short identifier used in traces and eval reports."""
        ...

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        """Return up to ``k`` chunks sorted by descending score."""
        ...
