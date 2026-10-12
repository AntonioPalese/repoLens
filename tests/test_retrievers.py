from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from repolens.config import EmbeddingSettings, ServerSettings, Settings
from repolens.embeddings import (
    HashingEmbedder,
    SentenceTransformerEmbedder,
    build_embedder,
)
from repolens.indexing.tokenizer import CodeTokenizer
from repolens.models import Chunk, ChunkKind, SearchMode
from repolens.retrieval.base import Retriever, document_text, top_k
from repolens.retrieval.bm25 import BM25Retriever
from repolens.retrieval.dense import DenseRetriever
from repolens.retrieval.pipeline import TRUNCATION_MARKER, SearchEngine, truncate


@pytest.fixture
def bm25(sample_chunks: list[Chunk]) -> BM25Retriever:
    return BM25Retriever.build(sample_chunks, CodeTokenizer())


@pytest.fixture
def dense(sample_chunks: list[Chunk]) -> DenseRetriever:
    return DenseRetriever.build(sample_chunks, HashingEmbedder())


def _symbols(retriever: Retriever, query: str, k: int = 3) -> list[str | None]:
    return [hit.chunk.symbol for hit in retriever.retrieve(query, k)]


def _chunk(position: int) -> Chunk:
    return Chunk(
        id=str(position),
        path="a.py",
        start_line=position + 1,
        end_line=position + 1,
        kind=ChunkKind.MODULE,
        language="python",
        text="",
    )


def test_retrievers_satisfy_protocol(bm25: BM25Retriever, dense: DenseRetriever) -> None:
    retrievers: list[Retriever] = [bm25, dense]
    assert [r.name for r in retrievers] == ["bm25", "dense"]


def test_document_text_includes_location_and_symbol(sample_chunks: list[Chunk]) -> None:
    send = next(c for c in sample_chunks if c.symbol == "Client.send")

    assert document_text(send).startswith("minihttp/client.py Client.send\n")


def test_top_k_breaks_ties_by_position() -> None:
    chunks = [_chunk(i) for i in range(4)]
    positions = np.array([3, 1, 2, 0], dtype=np.int64)
    scores = np.array([0.5, 0.9, 0.5, 0.5], dtype=np.float32)

    ranked = top_k(chunks, positions, scores, k=3)

    assert [r.chunk.id for r in ranked] == ["1", "0", "2"]


class TestBM25:
    def test_exact_identifier_ranks_definition_high(self, bm25: BM25Retriever) -> None:
        # Plain BM25 does not guarantee the definition is first: the shorter
        # BasicAuth.header, which calls build_auth_header, can outrank it.
        assert "build_auth_header" in _symbols(bm25, "build_auth_header", k=2)

    def test_unique_identifier_ranks_first(self, bm25: BM25Retriever) -> None:
        assert _symbols(bm25, "parseAuthHeader", k=1) == ["parseAuthHeader"]

    def test_identifier_parts_match_natural_language(self, bm25: BM25Retriever) -> None:
        assert "RetryPolicy.backoff" in _symbols(bm25, "retry policy backoff delay")

    def test_unknown_terms_return_nothing(self, bm25: BM25Retriever) -> None:
        assert bm25.retrieve("zzqx unknownterm", 5) == []
        assert bm25.retrieve("", 5) == []

    def test_scores_are_positive_descending_and_limited(self, bm25: BM25Retriever) -> None:
        hits = bm25.retrieve("client request", 4)

        assert len(hits) == 4
        assert all(h.score > 0 for h in hits)
        assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)

    def test_save_load_round_trip(
        self, bm25: BM25Retriever, sample_chunks: list[Chunk], tmp_path: Path
    ) -> None:
        bm25.save(tmp_path)
        loaded = BM25Retriever.load(tmp_path, sample_chunks, CodeTokenizer())

        assert loaded.retrieve("auth header", 5) == bm25.retrieve("auth header", 5)


class TestDense:
    def test_chunk_text_retrieves_itself(
        self, dense: DenseRetriever, sample_chunks: list[Chunk]
    ) -> None:
        target = next(c for c in sample_chunks if c.symbol == "parseAuthHeader")

        best = dense.retrieve(document_text(target), 1)[0]

        assert best.chunk == target
        assert best.score == pytest.approx(1.0, abs=1e-5)

    def test_k_larger_than_corpus(self, dense: DenseRetriever, sample_chunks: list[Chunk]) -> None:
        hits = dense.retrieve("client", len(sample_chunks) + 10)

        assert len(hits) == len(sample_chunks)
        assert len({h.chunk.id for h in hits}) == len(sample_chunks)

    def test_save_load_round_trip(
        self, dense: DenseRetriever, sample_chunks: list[Chunk], tmp_path: Path
    ) -> None:
        dense.save(tmp_path / "dense.faiss")
        loaded = DenseRetriever.load(tmp_path / "dense.faiss", sample_chunks, HashingEmbedder())

        assert loaded.retrieve("retry backoff", 5) == dense.retrieve("retry backoff", 5)

    def test_build_is_deterministic(self, sample_chunks: list[Chunk]) -> None:
        first = DenseRetriever.build(sample_chunks, HashingEmbedder())
        second = DenseRetriever.build(sample_chunks, HashingEmbedder())

        assert first.retrieve("send request", 10) == second.retrieve("send request", 10)


class TestEmbedders:
    def test_hashing_embedder_is_normalised_and_deterministic(self) -> None:
        embedder = HashingEmbedder(dim=32)
        vectors = embedder.embed_documents(["retry policy", "auth header", ""])

        assert vectors.shape == (3, 32)
        assert vectors.dtype == np.float32
        assert np.linalg.norm(vectors[:2], axis=1) == pytest.approx([1.0, 1.0])
        assert not vectors[2].any()
        assert np.array_equal(embedder.embed_query("retry policy"), vectors[0])

    def test_build_embedder_selects_backend(self) -> None:
        hashing = build_embedder(EmbeddingSettings(backend="hashing", hashing_dim=64))
        model = build_embedder(EmbeddingSettings(model_name="some/model"))

        assert hashing.model_id == "hashing:64"
        assert isinstance(model, SentenceTransformerEmbedder)
        assert model.model_id == "some/model"

    @pytest.mark.slow
    def test_sentence_transformer_semantics(self) -> None:
        embedder = build_embedder(EmbeddingSettings())
        docs = embedder.embed_documents(["def retry_with_backoff(): ...", "def parse_csv(): ..."])
        query = embedder.embed_query("exponential backoff between attempts")

        assert docs.shape[0] == 2
        assert np.linalg.norm(docs, axis=1) == pytest.approx([1.0, 1.0], abs=1e-4)
        assert float(docs[0] @ query) > float(docs[1] @ query)


class TestSearchEngine:
    def test_bm25_search_returns_hits(self, sample_index: Path, offline_settings: Settings) -> None:
        engine = SearchEngine.open(sample_index, offline_settings)

        result = engine.search("parseAuthHeader", k=2, mode=SearchMode.BM25)

        assert len(result.hits) == 2
        assert result.hits[0].symbol == "parseAuthHeader"
        assert result.hits[0].path == "minihttp/auth.py"
        assert result.reranked is False
        assert len(result.request_id) == 32

    def test_snippets_are_truncated(self, sample_index: Path, offline_settings: Settings) -> None:
        settings = offline_settings.model_copy(
            update={"server": ServerSettings(snippet_max_chars=40)}
        )
        engine = SearchEngine.open(sample_index, settings)

        hit = engine.search("build_url attempt", k=1, mode=SearchMode.BM25).hits[0]

        assert len(hit.snippet) <= 40
        assert hit.snippet.endswith(TRUNCATION_MARKER)

    def test_rejects_bad_requests(self, sample_index: Path, offline_settings: Settings) -> None:
        engine = SearchEngine.open(sample_index, offline_settings)

        with pytest.raises(ValueError, match="k must be"):
            engine.search("q", k=0, mode=SearchMode.BM25)


def test_truncate_keeps_short_text() -> None:
    assert truncate("short", 10) == "short"
