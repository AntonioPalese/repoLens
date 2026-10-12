"""Sparse retrieval with BM25 over code-aware tokens."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import bm25s
import numpy as np
from numpy.typing import NDArray

from repolens.indexing.tokenizer import CodeTokenizer
from repolens.models import Chunk, ScoredChunk
from repolens.retrieval.base import document_text, top_k

_VOCAB_FILE = "repolens_vocab.json"


class BM25Retriever:
    """BM25 retriever; tokenization is ours, scoring is delegated to ``bm25s``."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        model: bm25s.BM25,
        vocab: dict[str, int],
        tokenizer: CodeTokenizer,
    ) -> None:
        self._chunks = chunks
        self._model = model
        self._vocab = vocab
        self._tokenizer = tokenizer

    @property
    def name(self) -> str:
        """Retriever identifier."""
        return "bm25"

    @classmethod
    def build(
        cls,
        chunks: Sequence[Chunk],
        tokenizer: CodeTokenizer,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> BM25Retriever:
        """Tokenize and index ``chunks``."""
        vocab: dict[str, int] = {}
        corpus_ids = [
            [vocab.setdefault(token, len(vocab)) for token in tokenizer.tokenize(document_text(c))]
            for c in chunks
        ]
        model = bm25s.BM25(k1=k1, b=b)
        model.index(bm25s.tokenization.Tokenized(ids=corpus_ids, vocab=vocab), show_progress=False)
        return cls(chunks, model, vocab, tokenizer)

    def save(self, directory: Path) -> None:
        """Persist the index and vocabulary into ``directory``."""
        self._model.save(str(directory))
        (directory / _VOCAB_FILE).write_text(json.dumps(self._vocab), encoding="utf-8")

    @classmethod
    def load(
        cls, directory: Path, chunks: Sequence[Chunk], tokenizer: CodeTokenizer
    ) -> BM25Retriever:
        """Load an index written by :meth:`save`."""
        model = bm25s.BM25.load(str(directory))
        vocab = json.loads((directory / _VOCAB_FILE).read_text(encoding="utf-8"))
        return cls(chunks, model, vocab, tokenizer)

    def scores(self, query: str) -> NDArray[np.float32]:
        """BM25 score of every chunk for ``query`` (zeros when no token is known)."""
        ids = [self._vocab[t] for t in self._tokenizer.tokenize(query) if t in self._vocab]
        if not ids:
            return np.zeros(len(self._chunks), dtype=np.float32)
        return np.asarray(self._model.get_scores(ids), dtype=np.float32)

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        """Return up to ``k`` chunks with a positive score, best first."""
        scores = self.scores(query)
        positions = np.flatnonzero(scores > 0).astype(np.int64)
        return top_k(self._chunks, positions, scores[positions], k)
