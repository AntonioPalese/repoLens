"""Search pipeline: retrieve -> (rerank) -> (guard) -> SearchResult.

This module is the single place where index, embedder, retrievers and reranker
are wired together. The injection guard (milestone 6) plugs in here.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from repolens.config import RerankSettings, Settings
from repolens.embeddings import build_embedder
from repolens.indexing.store import LoadedIndex, load_index
from repolens.models import ScoredChunk, SearchHit, SearchMode, SearchResult
from repolens.rerank.base import Reranker
from repolens.rerank.cross_encoder import CrossEncoderReranker
from repolens.retrieval.base import Retriever
from repolens.retrieval.hybrid import HybridRetriever

TRUNCATION_MARKER = "\n… [truncated]"


class SearchEngine:
    """Answers queries against one loaded index."""

    def __init__(
        self, index: LoadedIndex, settings: Settings, reranker: Reranker | None = None
    ) -> None:
        self._index = index
        self._settings = settings
        self._reranker = reranker
        retrieval = settings.retrieval
        self._retrievers: dict[SearchMode, Retriever] = {
            SearchMode.BM25: index.bm25,
            SearchMode.DENSE: index.dense,
            SearchMode.HYBRID: HybridRetriever(
                [index.bm25, index.dense],
                candidate_k=retrieval.candidate_k,
                rrf_k=retrieval.rrf_k,
            ),
        }

    @classmethod
    def open(cls, index_dir: Path, settings: Settings) -> SearchEngine:
        """Load the index at ``index_dir``; raises ``IndexLoadError`` on mismatch.

        Models are not loaded here: they load on the first query that needs them.
        """
        embedder = build_embedder(settings.embedding)
        index = load_index(index_dir, settings, embedder)
        return cls(index, settings, build_reranker(settings.rerank))

    @property
    def index(self) -> LoadedIndex:
        """The loaded index (metadata, chunks, retrievers)."""
        return self._index

    @property
    def reranker(self) -> Reranker | None:
        """The configured reranker, or None when reranking is disabled."""
        return self._reranker

    def search(
        self, query: str, k: int = 8, mode: SearchMode = SearchMode.HYBRID, rerank: bool = True
    ) -> SearchResult:
        """Return the top ``k`` chunks for ``query``.

        With reranking, the first stage fetches ``max(k, rerank.top_n)`` candidates and
        the reranker picks the final ``k``. ``rerank=True`` is a no-op when no reranker
        is configured; ``SearchResult.reranked`` reports what actually happened.
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        reranker = self._reranker if rerank else None
        depth = max(k, self._settings.rerank.top_n) if reranker else k
        candidates = self._retrievers[mode].retrieve(query, depth)
        if reranker is not None:
            candidates = reranker.rerank(query, candidates, k)
        return SearchResult(
            request_id=uuid.uuid4().hex,
            query=query,
            mode=mode,
            reranked=reranker is not None,
            hits=[self._to_hit(c) for c in candidates],
        )

    def _to_hit(self, scored: ScoredChunk) -> SearchHit:
        chunk = scored.chunk
        return SearchHit(
            path=chunk.path,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            symbol=chunk.symbol,
            kind=chunk.kind,
            score=scored.score,
            snippet=truncate(chunk.text, self._settings.server.snippet_max_chars),
        )


def truncate(text: str, max_chars: int) -> str:
    """Cut ``text`` to at most ``max_chars`` characters, marking the cut."""
    if len(text) <= max_chars:
        return text
    keep = max(0, max_chars - len(TRUNCATION_MARKER))
    return text[:keep] + TRUNCATION_MARKER


def build_reranker(settings: RerankSettings) -> Reranker | None:
    """Create the reranker selected by configuration (None for ``backend = "none"``)."""
    if settings.backend == "none":
        return None
    if settings.backend == "llm_judge":
        raise NotImplementedError("The LLM-judge reranker is not implemented yet")
    return CrossEncoderReranker(
        settings.cross_encoder_model, device=settings.device, batch_size=settings.batch_size
    )
