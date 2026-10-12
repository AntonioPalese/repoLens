"""Cross-encoder reranker (local ``sentence-transformers`` model)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from repolens.models import ScoredChunk
from repolens.retrieval.base import document_text

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Scores each ``(query, chunk)`` pair jointly; the model loads on first use.

    Scores are the model's raw relevance logits (higher is better). Inputs longer
    than the model's maximum sequence length are truncated by the model.
    """

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 32) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size

    @property
    def model_name(self) -> str:
        """Hugging Face model name."""
        return self._model_name

    def rerank(self, query: str, candidates: Sequence[ScoredChunk], k: int) -> list[ScoredChunk]:
        """Return the ``k`` best candidates by cross-encoder score.

        Equal scores keep their first-stage order, so results are deterministic.
        """
        if not candidates:
            return []
        model = _load_cross_encoder(self._model_name, self._device)
        pairs = [(query, document_text(c.chunk)) for c in candidates]
        raw = model.predict(pairs, batch_size=self._batch_size, show_progress_bar=False)
        scores = np.asarray(raw, dtype=np.float32).reshape(-1)
        order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))[:k]
        return [ScoredChunk(chunk=candidates[i].chunk, score=float(scores[i])) for i in order]


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str, device: str) -> CrossEncoder:
    """Load a cross-encoder once per process."""
    from sentence_transformers import CrossEncoder

    logger.info("Loading reranker %s on %s", model_name, device)
    model: CrossEncoder = CrossEncoder(model_name, device=device)
    return model
