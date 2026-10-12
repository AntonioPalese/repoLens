from __future__ import annotations

import pytest

from repolens.embeddings import HashingEmbedder
from repolens.indexing.tokenizer import CodeTokenizer
from repolens.models import Chunk, ChunkKind, ScoredChunk
from repolens.retrieval.base import Retriever
from repolens.retrieval.bm25 import BM25Retriever
from repolens.retrieval.dense import DenseRetriever
from repolens.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion


def _chunk(name: str, line: int = 1) -> Chunk:
    return Chunk(
        id=name,
        path=f"{name}.py",
        start_line=line,
        end_line=line,
        kind=ChunkKind.FUNCTION,
        symbol=name,
        language="python",
        text=f"def {name}(): ...",
    )


def _ranking(*names: str) -> list[ScoredChunk]:
    return [ScoredChunk(chunk=_chunk(n), score=100.0 - i) for i, n in enumerate(names)]


def _ids(ranked: list[ScoredChunk]) -> list[str]:
    return [r.chunk.id for r in ranked]


class StaticRetriever:
    """Returns a fixed ranking and records the depth it was asked for."""

    def __init__(self, name: str, ranking: list[ScoredChunk]) -> None:
        self._name = name
        self._ranking = ranking
        self.requested_k: list[int] = []

    @property
    def name(self) -> str:
        return self._name

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        self.requested_k.append(k)
        return self._ranking[:k]


class TestReciprocalRankFusion:
    def test_scores_follow_the_formula(self) -> None:
        fused = reciprocal_rank_fusion([_ranking("a", "b"), _ranking("b", "c")], rrf_k=10)
        scores = {r.chunk.id: r.score for r in fused}

        assert scores["b"] == pytest.approx(1 / 12 + 1 / 11)
        assert scores["a"] == pytest.approx(1 / 11)
        assert scores["c"] == pytest.approx(1 / 12)
        assert _ids(fused) == ["b", "a", "c"]

    def test_agreement_beats_a_single_first_place(self) -> None:
        fused = reciprocal_rank_fusion(
            [_ranking("solo", "shared"), _ranking("other", "shared")], rrf_k=60
        )

        assert _ids(fused)[0] == "shared"

    def test_ignores_raw_score_scales(self) -> None:
        tiny = [ScoredChunk(chunk=_chunk("a"), score=0.001)]
        huge = [ScoredChunk(chunk=_chunk("a"), score=1e6)]

        assert reciprocal_rank_fusion([tiny]) == reciprocal_rank_fusion([huge])

    def test_ties_are_broken_by_location(self) -> None:
        fused = reciprocal_rank_fusion([_ranking("zeta"), _ranking("alpha")])

        assert _ids(fused) == ["alpha", "zeta"]

    def test_duplicates_within_a_ranking_count_once(self) -> None:
        fused = reciprocal_rank_fusion([_ranking("a", "a", "b")], rrf_k=0)

        assert {r.chunk.id: r.score for r in fused} == {"a": 1.0, "b": pytest.approx(1 / 3)}

    def test_empty_inputs(self) -> None:
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []


class TestHybridRetriever:
    def test_fetches_candidate_depth_and_truncates(self) -> None:
        first = StaticRetriever("one", _ranking("a", "b", "c", "d"))
        second = StaticRetriever("two", _ranking("d", "c"))
        hybrid = HybridRetriever([first, second], candidate_k=3, rrf_k=60)

        result = hybrid.retrieve("q", k=2)

        assert first.requested_k == [3]
        assert second.requested_k == [3]
        assert _ids(result) == ["c", "a"]

    def test_depth_is_at_least_k(self) -> None:
        inner = StaticRetriever("one", _ranking("a"))

        HybridRetriever([inner], candidate_k=2).retrieve("q", k=5)

        assert inner.requested_k == [5]

    def test_requires_a_retriever(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            HybridRetriever([])

    def test_on_fixture_repo(self, sample_chunks: list[Chunk]) -> None:
        hybrid: Retriever = HybridRetriever(
            [
                BM25Retriever.build(sample_chunks, CodeTokenizer()),
                DenseRetriever.build(sample_chunks, HashingEmbedder()),
            ]
        )

        hits = hybrid.retrieve("parseAuthHeader", 3)

        assert hybrid.name == "hybrid"
        assert hits[0].chunk.symbol == "parseAuthHeader"
        assert len(hits) == 3
