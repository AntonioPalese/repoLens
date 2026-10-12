"""Dense retrieval over a FAISS HNSW index of normalised embeddings."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import faiss
import numpy as np

from repolens.embeddings import Embedder
from repolens.models import Chunk, ScoredChunk
from repolens.retrieval.base import document_text, top_k


class DenseRetriever:
    """Approximate nearest-neighbour search with inner product (cosine) similarity."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        index: faiss.IndexHNSWFlat,
        embedder: Embedder,
        ef_search: int = 64,
    ) -> None:
        self._chunks = chunks
        self._index = index
        self._embedder = embedder
        self._ef_search = ef_search

    @property
    def name(self) -> str:
        """Retriever identifier."""
        return "dense"

    @property
    def dim(self) -> int:
        """Embedding dimension of the index."""
        return int(self._index.d)

    @classmethod
    def build(
        cls,
        chunks: Sequence[Chunk],
        embedder: Embedder,
        m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
    ) -> DenseRetriever:
        """Embed ``chunks`` and build the HNSW graph."""
        vectors = embedder.embed_documents([document_text(c) for c in chunks])
        index = faiss.IndexHNSWFlat(vectors.shape[1], m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        with _single_threaded():  # parallel HNSW insertion is order-dependent
            index.add(vectors)
        return cls(chunks, index, embedder, ef_search)

    def save(self, path: Path) -> None:
        """Write the FAISS index to ``path``."""
        faiss.write_index(self._index, str(path))

    @classmethod
    def load(
        cls, path: Path, chunks: Sequence[Chunk], embedder: Embedder, ef_search: int = 64
    ) -> DenseRetriever:
        """Read an index written by :meth:`save`."""
        index = faiss.read_index(str(path))
        if not isinstance(index, faiss.IndexHNSWFlat):
            raise TypeError(f"{path} is not an HNSW index")
        return cls(chunks, index, embedder, ef_search)

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        """Return up to ``k`` nearest chunks, best first."""
        query_vector = self._embedder.embed_query(query).reshape(1, -1)
        self._index.hnsw.efSearch = max(self._ef_search, k)
        scores, positions = self._index.search(query_vector, k)
        found = positions[0] >= 0  # FAISS pads with -1 when fewer than k vectors exist
        return top_k(
            self._chunks,
            positions[0][found].astype(np.int64),
            scores[0][found].astype(np.float32),
            k,
        )


@contextmanager
def _single_threaded() -> Iterator[None]:
    previous = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)
    try:
        yield
    finally:
        faiss.omp_set_num_threads(previous)
