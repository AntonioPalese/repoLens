"""Search pipeline: retrieve -> (rerank) -> (guard) -> SearchResult.

This module is the single place where index, embedder, retrievers and reranker
are wired together. The injection guard (milestone 6) plugs in here.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from repolens.config import LLMJudgeSettings, RerankSettings, Settings
from repolens.embeddings import build_embedder
from repolens.indexing.store import LoadedIndex, load_index
from repolens.models import ScoredChunk, SearchHit, SearchMode, SearchResult
from repolens.rerank.base import ReportsUsage, Reranker
from repolens.rerank.cross_encoder import CrossEncoderReranker
from repolens.rerank.llm_judge import LLMJudgeReranker
from repolens.retrieval.base import Retriever
from repolens.retrieval.hybrid import HybridRetriever
from repolens.tracing import Tracer

TRUNCATION_MARKER = "\n… [truncated]"


class SearchEngine:
    """Answers queries against one loaded index."""

    def __init__(
        self, index: LoadedIndex, settings: Settings, reranker: Reranker | None = None
    ) -> None:
        self._index = index
        self._settings = settings
        self._reranker = reranker
        self._single: dict[SearchMode, Retriever] = {
            SearchMode.BM25: index.bm25,
            SearchMode.DENSE: index.dense,
        }
        self._hybrid = HybridRetriever(
            [index.bm25, index.dense],
            candidate_k=settings.retrieval.candidate_k,
            rrf_k=settings.retrieval.rrf_k,
        )

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
        self,
        query: str,
        k: int = 8,
        mode: SearchMode | None = None,
        rerank: bool = True,
        tracer: Tracer | None = None,
    ) -> SearchResult:
        """Return the top ``k`` chunks for ``query``.

        With reranking, the first stage fetches ``max(k, rerank.top_n)`` candidates and
        the reranker picks the final ``k``. ``rerank=True`` is a no-op when no reranker
        is configured; ``SearchResult.reranked`` reports what actually happened.
        ``mode=None`` uses ``retrieval.default_mode``. Stage timings and LLM usage are
        recorded on ``tracer`` when given.
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        mode = mode if mode is not None else self._settings.retrieval.default_mode
        tracer = tracer if tracer is not None else Tracer()
        reranker = self._reranker if rerank else None
        depth = max(k, self._settings.rerank.top_n) if reranker else k
        candidates = self._first_stage(query, mode, depth, tracer)
        if reranker is not None:
            candidates = self._rerank(reranker, query, candidates, k, tracer)
        return SearchResult(
            request_id=uuid.uuid4().hex,
            query=query,
            mode=mode,
            reranked=reranker is not None,
            hits=[self._to_hit(c) for c in candidates],
        )

    def _first_stage(
        self, query: str, mode: SearchMode, depth: int, tracer: Tracer
    ) -> list[ScoredChunk]:
        if mode is not SearchMode.HYBRID:
            return _timed_retrieve(self._single[mode], query, depth, tracer)
        rankings = [
            _timed_retrieve(r, query, self._hybrid.depth(depth), tracer)
            for r in self._hybrid.retrievers
        ]
        with tracer.stage("fusion") as stage:
            fused = self._hybrid.fuse(rankings, depth)
            stage.candidates = len(fused)
        return fused

    @staticmethod
    def _rerank(
        reranker: Reranker, query: str, candidates: list[ScoredChunk], k: int, tracer: Tracer
    ) -> list[ScoredChunk]:
        with tracer.stage("rerank") as stage:
            reranked = reranker.rerank(query, candidates, k)
            stage.candidates = len(reranked)
        if isinstance(reranker, ReportsUsage) and reranker.last_usage is not None:
            tracer.add_llm_usage(reranker.last_usage)
        return reranked

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
        return build_llm_judge(settings.llm_judge)
    return CrossEncoderReranker(
        settings.cross_encoder_model, device=settings.device, batch_size=settings.batch_size
    )


def build_llm_judge(settings: LLMJudgeSettings) -> LLMJudgeReranker:
    """Create the LLM judge; requires an explicitly configured endpoint and model."""
    if settings.base_url is None or settings.model is None:
        raise ValueError(
            "The LLM-judge reranker needs rerank.llm_judge.base_url and rerank.llm_judge.model"
        )
    from openai import OpenAI

    api_key = settings.api_key.get_secret_value() if settings.api_key else "not-needed"
    client = OpenAI(base_url=settings.base_url, api_key=api_key, timeout=settings.timeout_s)
    return LLMJudgeReranker(
        client,
        settings.model,
        max_chars_per_candidate=settings.max_chars_per_candidate,
        price_in_per_mtok=settings.price_in_per_mtok,
        price_out_per_mtok=settings.price_out_per_mtok,
    )


def _timed_retrieve(retriever: Retriever, query: str, k: int, tracer: Tracer) -> list[ScoredChunk]:
    with tracer.stage(retriever.name) as stage:
        hits = retriever.retrieve(query, k)
        stage.candidates = len(hits)
    return hits
