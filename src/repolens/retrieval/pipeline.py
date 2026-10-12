"""Search pipeline: retrieve -> (rerank) -> (guard) -> SearchResult.

This module is the single place where index, embedder and retrievers are wired
together. Reranking (milestone 4) and the injection guard (milestone 6) plug in
here.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from repolens.config import Settings
from repolens.embeddings import build_embedder
from repolens.indexing.store import LoadedIndex, load_index
from repolens.models import ScoredChunk, SearchHit, SearchMode, SearchResult
from repolens.retrieval.base import Retriever

logger = logging.getLogger(__name__)

TRUNCATION_MARKER = "\n… [truncated]"


class SearchEngine:
    """Answers queries against one loaded index."""

    def __init__(self, index: LoadedIndex, settings: Settings) -> None:
        self._index = index
        self._settings = settings
        self._retrievers: dict[SearchMode, Retriever] = {
            SearchMode.BM25: index.bm25,
            SearchMode.DENSE: index.dense,
        }

    @classmethod
    def open(cls, index_dir: Path, settings: Settings) -> SearchEngine:
        """Load the index at ``index_dir``; raises ``IndexLoadError`` on mismatch."""
        embedder = build_embedder(settings.embedding)
        return cls(load_index(index_dir, settings, embedder), settings)

    @property
    def index(self) -> LoadedIndex:
        """The loaded index (metadata, chunks, retrievers)."""
        return self._index

    def search(
        self, query: str, k: int = 8, mode: SearchMode = SearchMode.HYBRID, rerank: bool = True
    ) -> SearchResult:
        """Return the top ``k`` chunks for ``query``."""
        if k < 1:
            raise ValueError("k must be >= 1")
        retriever = self._retrievers.get(mode)
        if retriever is None:
            raise NotImplementedError(f"{mode.value!r} search is not implemented yet")
        if rerank:
            logger.debug("Reranking is not implemented yet; returning first-stage ranking")
        candidates = retriever.retrieve(query, k)
        return SearchResult(
            request_id=uuid.uuid4().hex,
            query=query,
            mode=mode,
            reranked=False,
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
