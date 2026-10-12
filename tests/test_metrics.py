from __future__ import annotations

import math

import pytest

from repolens.evals.dataset import GoldItem
from repolens.evals.metrics import (
    judge,
    latency_summary,
    matches,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from repolens.models import ChunkKind, SearchHit


def _hit(path: str, symbol: str | None = None, start: int = 1, end: int = 1) -> SearchHit:
    return SearchHit(
        path=path,
        start_line=start,
        end_line=end,
        symbol=symbol,
        kind=ChunkKind.FUNCTION,
        score=0.0,
        snippet="",
    )


class TestMatching:
    def test_symbol_match_requires_same_path_and_symbol(self) -> None:
        gold = GoldItem(path="a.py", symbol="Client.send")

        assert matches(_hit("a.py", "Client.send"), gold)
        assert not matches(_hit("b.py", "Client.send"), gold)
        assert not matches(_hit("a.py", "Client"), gold)

    @pytest.mark.parametrize(
        ("start", "end", "expected"),
        [(1, 4, False), (1, 5, True), (6, 8, True), (10, 20, True), (11, 20, False)],
    )
    def test_line_range_match_uses_overlap(self, start: int, end: int, expected: bool) -> None:
        gold = GoldItem(path="a.py", start_line=5, end_line=10)

        assert matches(_hit("a.py", None, start, end), gold) is expected

    def test_gold_without_symbol_needs_lines(self) -> None:
        with pytest.raises(ValueError, match="start_line"):
            GoldItem(path="a.py")

    def test_judge_counts_each_gold_once(self) -> None:
        gold = [GoldItem(path="a.py", symbol="f"), GoldItem(path="b.py", symbol="g")]
        hits = [_hit("x.py", "h"), _hit("a.py", "f"), _hit("a.py", "f"), _hit("b.py", "g")]

        assert judge(hits, gold) == [None, 0, None, 1]


class TestRankMetrics:
    # Toy ranking: relevant at ranks 2 and 4, three gold items in total.
    judged = (None, 0, None, 1, None)

    def test_recall(self) -> None:
        assert recall_at_k(self.judged, 3, 1) == 0.0
        assert recall_at_k(self.judged, 3, 2) == pytest.approx(1 / 3)
        assert recall_at_k(self.judged, 3, 5) == pytest.approx(2 / 3)

    def test_mrr(self) -> None:
        assert mrr_at_k(self.judged, 10) == 0.5
        assert mrr_at_k(self.judged, 1) == 0.0
        assert mrr_at_k((None, None), 10) == 0.0

    def test_ndcg(self) -> None:
        dcg = 1 / math.log2(3) + 1 / math.log2(5)
        ideal = 1 + 1 / math.log2(3) + 1 / math.log2(4)

        assert ndcg_at_k(self.judged, 3, 10) == pytest.approx(dcg / ideal)

    def test_perfect_ranking_scores_one(self) -> None:
        perfect = (0, 1, None)

        assert recall_at_k(perfect, 2, 10) == 1.0
        assert mrr_at_k(perfect, 10) == 1.0
        assert ndcg_at_k(perfect, 2, 10) == pytest.approx(1.0)

    def test_ndcg_ideal_is_capped_at_k(self) -> None:
        # 5 gold items but k=2: two relevant hits on top is already ideal.
        assert ndcg_at_k((0, 1), 5, 2) == pytest.approx(1.0)


def test_latency_summary() -> None:
    summary = latency_summary([float(v) for v in range(1, 101)])

    assert summary["mean_ms"] == pytest.approx(50.5)
    assert summary["p95_ms"] == pytest.approx(95.05)
    assert latency_summary([]) == {"mean_ms": 0.0, "p95_ms": 0.0}
