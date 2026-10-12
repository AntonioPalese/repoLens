"""Text embedders for the dense retriever.

All embedders return L2-normalised ``float32`` vectors, so inner product equals
cosine similarity.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import NDArray

from repolens.config import EmbeddingSettings
from repolens.indexing.tokenizer import CodeTokenizer

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

Vectors = NDArray[np.float32]


class Embedder(Protocol):
    """Maps text to normalised vectors."""

    @property
    def model_id(self) -> str:
        """Identifier stored in the index; a mismatch at load time forces a re-index."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        """Embed documents; returns an array of shape ``(len(texts), dim)``."""
        ...

    def embed_query(self, text: str) -> Vectors:
        """Embed one query; returns an array of shape ``(dim,)``."""
        ...


class HashingEmbedder:
    """Feature-hashing bag of code tokens: deterministic, offline, no semantics.

    Used by tests and smoke runs; it behaves like a dense keyword matcher.
    """

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim
        self._tokenizer = CodeTokenizer()

    @property
    def model_id(self) -> str:
        """Backend and dimension, e.g. ``hashing:256``."""
        return f"hashing:{self._dim}"

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        """Embed each text as a signed, normalised token-count vector."""
        vectors = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in self._tokenizer.tokenize(text):
                bucket, sign = self._hash(token)
                vectors[row, bucket] += sign
        return _normalise(vectors)

    def embed_query(self, text: str) -> Vectors:
        """Embed a query exactly like a document."""
        return self.embed_documents([text]).reshape(-1)

    def _hash(self, token: str) -> tuple[int, float]:
        digest = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")
        return digest % self._dim, 1.0 if (digest >> 63) else -1.0


class SentenceTransformerEmbedder:
    """Embedder backed by a ``sentence-transformers`` model, loaded on first use."""

    def __init__(
        self, model_name: str, query_prefix: str = "", batch_size: int = 32, device: str = "cpu"
    ) -> None:
        self._model_name = model_name
        self._query_prefix = query_prefix
        self._batch_size = batch_size
        self._device = device

    @property
    def model_id(self) -> str:
        """The Hugging Face model name."""
        return self._model_name

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        """Embed documents in batches."""
        return self._encode(list(texts))

    def embed_query(self, text: str) -> Vectors:
        """Embed a query, prepending the model's query instruction if any."""
        return self._encode([self._query_prefix + text]).reshape(-1)

    def _encode(self, texts: list[str]) -> Vectors:
        model = _load_sentence_transformer(self._model_name, self._device)
        vectors = model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        result: Vectors = np.asarray(vectors, dtype=np.float32).reshape(len(texts), -1)
        return result


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str, device: str) -> SentenceTransformer:
    """Load a model once per process (imports torch lazily, too)."""
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model %s on %s", model_name, device)
    return SentenceTransformer(model_name, device=device)


def build_embedder(settings: EmbeddingSettings) -> Embedder:
    """Create the embedder selected by configuration."""
    if settings.backend == "hashing":
        return HashingEmbedder(settings.hashing_dim)
    return SentenceTransformerEmbedder(
        settings.model_name,
        query_prefix=settings.query_prefix,
        batch_size=settings.batch_size,
        device=settings.device,
    )


def _normalise(vectors: Vectors) -> Vectors:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalised: Vectors = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)
    return normalised
