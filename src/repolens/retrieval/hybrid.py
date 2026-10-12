"""Hybrid retrieval: Reciprocal Rank Fusion of several first-stage rankings."""

from __future__ import annotations

from collections.abc import Sequence

from repolens.models import Chunk, ScoredChunk
from repolens.retrieval.base import Retriever


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ScoredChunk]], rrf_k: int = 60
) -> list[ScoredChunk]:
    """Fuse rankings with RRF: ``score(d) = sum over lists of 1 / (rrf_k + rank(d))``.

    Ranks are 1-based. Only ranks are used, so retrievers with incomparable score
    scales (BM25 vs cosine) can be combined. A chunk appearing twice in one list
    counts once, at its best rank. Ties are broken by file path and line, so the
    output is deterministic.
    """
    fused: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, scored in enumerate(ranking, start=1):
            chunk_id = scored.chunk.id
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            chunks[chunk_id] = scored.chunk
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    order = sorted(fused, key=lambda cid: (-fused[cid], *_position(chunks[cid])))
    return [ScoredChunk(chunk=chunks[cid], score=fused[cid]) for cid in order]


class HybridRetriever:
    """Runs several retrievers and fuses their rankings with RRF."""

    def __init__(
        self, retrievers: Sequence[Retriever], candidate_k: int = 50, rrf_k: int = 60
    ) -> None:
        if not retrievers:
            raise ValueError("HybridRetriever needs at least one retriever")
        self._retrievers = list(retrievers)
        self._candidate_k = candidate_k
        self._rrf_k = rrf_k

    @property
    def name(self) -> str:
        """Retriever identifier."""
        return "hybrid"

    @property
    def retrievers(self) -> list[Retriever]:
        """The fused retrievers (the pipeline times them individually)."""
        return list(self._retrievers)

    def depth(self, k: int) -> int:
        """How many candidates to request from each retriever for a top-``k`` result."""
        return max(self._candidate_k, k)

    def fuse(self, rankings: Sequence[Sequence[ScoredChunk]], k: int) -> list[ScoredChunk]:
        """Fuse per-retriever rankings and keep the best ``k``."""
        return reciprocal_rank_fusion(rankings, self._rrf_k)[:k]

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        """Fuse the top ``depth(k)`` of every retriever; return the best ``k``."""
        rankings = [retriever.retrieve(query, self.depth(k)) for retriever in self._retrievers]
        return self.fuse(rankings, k)


def _position(chunk: Chunk) -> tuple[str, int, int, str]:
    return (chunk.path, chunk.start_line, chunk.end_line, chunk.id)
