"""Retriever protocol and helpers shared by BM25, dense and hybrid retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from repolens.models import Chunk, ScoredChunk


class Retriever(Protocol):
    """Returns the ``k`` best chunks for a query, best first."""

    @property
    def name(self) -> str:
        """Short identifier used in traces and eval reports."""
        ...

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        """Return up to ``k`` chunks sorted by descending score."""
        ...


def document_text(chunk: Chunk) -> str:
    """Text a chunk is indexed under: its location and qualified name, then its source.

    The header gives methods their class context (``Client.send``) and lets queries
    match file names.
    """
    header = f"{chunk.path} {chunk.symbol}" if chunk.symbol else chunk.path
    return f"{header}\n{chunk.text}"


def top_k(
    chunks: Sequence[Chunk],
    positions: NDArray[np.int64],
    scores: NDArray[np.float32],
    k: int,
) -> list[ScoredChunk]:
    """Pick the ``k`` best ``(position, score)`` pairs.

    Ties are broken by chunk position (i.e. path, then line), so results are
    deterministic.
    """
    order = np.lexsort((positions, -scores))[:k]
    return [ScoredChunk(chunk=chunks[int(positions[i])], score=float(scores[i])) for i in order]
