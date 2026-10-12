"""Retrieval metrics with binary relevance against gold locations.

A ranked hit is *relevant* if it matches a gold item that no earlier hit already
matched, so a symbol split into several window chunks is counted once.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from repolens.evals.dataset import GoldItem


class Located(Protocol):
    """Anything with a location: chunks and search hits both qualify."""

    @property
    def path(self) -> str: ...
    @property
    def symbol(self) -> str | None: ...
    @property
    def start_line(self) -> int: ...
    @property
    def end_line(self) -> int: ...


def matches(hit: Located, gold: GoldItem) -> bool:
    """Same path and same symbol, or overlapping lines when the gold has no symbol."""
    if hit.path != gold.path:
        return False
    if gold.symbol is not None:
        return hit.symbol == gold.symbol
    assert gold.start_line is not None and gold.end_line is not None  # validated
    return hit.start_line <= gold.end_line and gold.start_line <= hit.end_line


def judge(hits: Sequence[Located], gold: Sequence[GoldItem]) -> list[int | None]:
    """For each hit, the index of the gold item it newly matches, else None."""
    found: set[int] = set()
    judged: list[int | None] = []
    for hit in hits:
        match = next(
            (i for i, g in enumerate(gold) if i not in found and matches(hit, g)),
            None,
        )
        if match is not None:
            found.add(match)
        judged.append(match)
    return judged


def recall_at_k(judged: Sequence[int | None], n_gold: int, k: int) -> float:
    """Fraction of gold items found in the top ``k``."""
    return sum(1 for g in judged[:k] if g is not None) / n_gold


def mrr_at_k(judged: Sequence[int | None], k: int) -> float:
    """Reciprocal rank of the first relevant hit in the top ``k`` (0 if none)."""
    for rank, g in enumerate(judged[:k], start=1):
        if g is not None:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(judged: Sequence[int | None], n_gold: int, k: int) -> float:
    """Normalised DCG with binary gains; the ideal ranks all gold items first."""
    dcg = sum(1.0 / math.log2(rank + 1) for rank, g in enumerate(judged[:k], 1) if g is not None)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(n_gold, k) + 1))
    return dcg / ideal


def latency_summary(samples_ms: Sequence[float]) -> dict[str, float]:
    """Mean and 95th percentile (linear interpolation) of latency samples."""
    if not samples_ms:
        return {"mean_ms": 0.0, "p95_ms": 0.0}
    values = np.asarray(samples_ms, dtype=np.float64)
    return {"mean_ms": float(values.mean()), "p95_ms": float(np.percentile(values, 95))}
