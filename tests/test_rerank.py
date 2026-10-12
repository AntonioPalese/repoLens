from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from repolens.config import RerankSettings, Settings
from repolens.models import Chunk, ChunkKind, ScoredChunk, SearchMode
from repolens.rerank import cross_encoder
from repolens.rerank.base import Reranker
from repolens.rerank.cross_encoder import CrossEncoderReranker
from repolens.retrieval.pipeline import SearchEngine, build_reranker


def _candidate(name: str, text: str) -> ScoredChunk:
    chunk = Chunk(
        id=name,
        path="m.py",
        start_line=1,
        end_line=1,
        kind=ChunkKind.FUNCTION,
        symbol=name,
        language="python",
        text=text,
    )
    return ScoredChunk(chunk=chunk, score=0.0)


class FakeCrossEncoder:
    """Scores a pair by how many query words occur in the document."""

    def predict(self, pairs: list[tuple[str, str]], **_: object) -> list[float]:
        return [float(sum(w in doc for w in query.split())) for query, doc in pairs]


class RecordingReranker:
    """Reverses the candidate order and records what it received."""

    def __init__(self) -> None:
        self.received: list[int] = []

    @property
    def model_name(self) -> str:
        return "reverse"

    def rerank(self, query: str, candidates: Sequence[ScoredChunk], k: int) -> list[ScoredChunk]:
        self.received.append(len(candidates))
        return [ScoredChunk(chunk=c.chunk, score=float(i)) for i, c in enumerate(candidates)][::-1][
            :k
        ]


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cross_encoder, "_load_cross_encoder", lambda *_: FakeCrossEncoder())


class TestCrossEncoderReranker:
    def test_reorders_by_model_score(self, fake_model: None) -> None:
        reranker: Reranker = CrossEncoderReranker("fake")
        candidates = [
            _candidate("a", "nothing relevant"),
            _candidate("b", "retry backoff delay"),
            _candidate("c", "retry once"),
        ]

        ranked = reranker.rerank("retry backoff", candidates, k=2)

        assert [r.chunk.id for r in ranked] == ["b", "c"]
        assert [r.score for r in ranked] == [2.0, 1.0]
        assert reranker.model_name == "fake"

    def test_ties_keep_first_stage_order(self, fake_model: None) -> None:
        candidates = [_candidate(n, "same text") for n in ("x", "y", "z")]

        ranked = CrossEncoderReranker("fake").rerank("same", candidates, k=3)

        assert [r.chunk.id for r in ranked] == ["x", "y", "z"]

    def test_empty_candidates_do_not_load_the_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fail(*_: object) -> None:
            raise AssertionError("model loaded")

        monkeypatch.setattr(cross_encoder, "_load_cross_encoder", fail)

        assert CrossEncoderReranker("fake").rerank("q", [], k=3) == []

    @pytest.mark.slow
    def test_real_model_prefers_relevant_code(self) -> None:
        reranker = build_reranker(RerankSettings())
        assert reranker is not None
        candidates = [
            _candidate("parse_csv", "def parse_csv(path): return list(csv.reader(open(path)))"),
            _candidate("backoff", "def backoff(attempt): return 0.5 * 2 ** attempt"),
        ]

        ranked = reranker.rerank("exponential backoff delay between retries", candidates, k=2)

        assert ranked[0].chunk.id == "backoff"


class TestBuildReranker:
    def test_backends(self) -> None:
        assert build_reranker(RerankSettings(backend="none")) is None
        built = build_reranker(RerankSettings(cross_encoder_model="some/model"))
        assert isinstance(built, CrossEncoderReranker)
        assert built.model_name == "some/model"

    def test_llm_judge_not_available_yet(self) -> None:
        with pytest.raises(NotImplementedError):
            build_reranker(RerankSettings(backend="llm_judge"))


class TestPipelineReranking:
    @pytest.fixture
    def engine_and_reranker(
        self, sample_index: Path, offline_settings: Settings
    ) -> tuple[SearchEngine, RecordingReranker]:
        loaded = SearchEngine.open(sample_index, offline_settings).index
        reranker = RecordingReranker()
        settings = offline_settings.model_copy(update={"rerank": RerankSettings(top_n=6)})
        return SearchEngine(loaded, settings, reranker), reranker

    def test_reranker_sees_top_n_and_returns_k(
        self, engine_and_reranker: tuple[SearchEngine, RecordingReranker]
    ) -> None:
        engine, reranker = engine_and_reranker

        result = engine.search("client retry", k=2)

        assert reranker.received == [6]
        assert result.reranked is True
        assert result.mode is SearchMode.HYBRID
        assert len(result.hits) == 2

    def test_rerank_can_be_disabled_per_query(
        self, engine_and_reranker: tuple[SearchEngine, RecordingReranker]
    ) -> None:
        engine, reranker = engine_and_reranker

        result = engine.search("client retry", k=2, rerank=False)

        assert reranker.received == []
        assert result.reranked is False

    def test_no_configured_reranker_reports_not_reranked(
        self, sample_index: Path, offline_settings: Settings
    ) -> None:
        engine = SearchEngine.open(sample_index, offline_settings)

        result = engine.search("client retry", k=3)

        assert engine.reranker is None
        assert result.reranked is False
        assert len(result.hits) == 3
