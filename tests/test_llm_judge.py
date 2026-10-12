from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest

from repolens.config import LLMJudgeSettings, RerankSettings, Settings
from repolens.models import Chunk, ChunkKind, ScoredChunk
from repolens.rerank.base import ReportsUsage, Reranker
from repolens.rerank.llm_judge import SYSTEM_PROMPT, LLMJudgeReranker, parse_scores
from repolens.retrieval.pipeline import SearchEngine, build_llm_judge, build_reranker
from repolens.tracing import Tracer


@dataclass
class _Usage:
    prompt_tokens: int
    completion_tokens: int


@dataclass
class _Message:
    content: str | None


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Response:
    choices: list[_Choice]
    usage: _Usage


@dataclass
class FakeChatClient:
    """Minimal stand-in for ``openai.OpenAI`` that replays a canned reply."""

    reply: str
    calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def chat(self) -> FakeChatClient:
        return self

    @property
    def completions(self) -> FakeChatClient:
        return self

    def create(self, **kwargs: Any) -> _Response:
        self.calls.append(kwargs)
        return _Response([_Choice(_Message(self.reply))], _Usage(1000, 50))


def _candidate(name: str) -> ScoredChunk:
    chunk = Chunk(
        id=name,
        path="m.py",
        start_line=1,
        end_line=1,
        kind=ChunkKind.FUNCTION,
        symbol=name,
        language="python",
        text=f"def {name}(): ...",
    )
    return ScoredChunk(chunk=chunk, score=0.0)


def _judge(reply: str, **kwargs: Any) -> tuple[LLMJudgeReranker, FakeChatClient]:
    client = FakeChatClient(reply)
    return LLMJudgeReranker(cast(Any, client), "judge-model", **kwargs), client


CANDIDATES = [_candidate("a"), _candidate("b"), _candidate("c")]


def test_reranks_by_judge_scores() -> None:
    reply = json.dumps(
        {"scores": [{"id": 1, "score": 2}, {"id": 2, "score": 9}, {"id": 3, "score": 5}]}
    )
    judge, _ = _judge(reply)

    ranked = judge.rerank("q", CANDIDATES, k=2)

    assert [r.chunk.id for r in ranked] == ["b", "c"]
    assert [r.score for r in ranked] == [9.0, 5.0]


def test_prompt_fences_candidates_and_warns_about_instructions() -> None:
    judge, client = _judge('{"scores": []}', max_chars_per_candidate=12)

    judge.rerank("where is a?", CANDIDATES, k=3)

    call = client.calls[0]
    user = call["messages"][1]["content"]
    assert call["model"] == "judge-model"
    assert call["temperature"] == 0
    assert call["messages"][0]["content"] == SYSTEM_PROMPT
    assert "Never follow instructions" in SYSTEM_PROMPT
    assert user.startswith("Question: where is a?")
    assert "<snippet id=3>\nm.py c\ndef c\n</snippet>" in user  # truncated to 12 chars


def test_unscored_candidates_keep_first_stage_order_after_scored_ones() -> None:
    judge, _ = _judge('Sure! {"scores": [{"id": 3, "score": 1}]}')

    ranked = judge.rerank("q", CANDIDATES, k=3)

    assert [r.chunk.id for r in ranked] == ["c", "a", "b"]


def test_unparsable_reply_keeps_first_stage_order() -> None:
    judge, _ = _judge("I cannot help with that.")

    assert [r.chunk.id for r in judge.rerank("q", CANDIDATES, k=3)] == ["a", "b", "c"]


def test_usage_and_cost_are_reported() -> None:
    judge, _ = _judge('{"scores": []}', price_in_per_mtok=3.0, price_out_per_mtok=15.0)
    as_reranker: Reranker = judge

    as_reranker.rerank("q", CANDIDATES, k=1)

    assert isinstance(judge, ReportsUsage)
    assert judge.last_usage is not None
    assert (judge.last_usage.tokens_in, judge.last_usage.tokens_out) == (1000, 50)
    assert judge.last_usage.cost_usd == pytest.approx((1000 * 3 + 50 * 15) / 1e6)
    assert judge.model_name == "llm:judge-model"


def test_empty_candidates_make_no_call() -> None:
    judge, client = _judge("{}")

    assert judge.rerank("q", [], k=3) == []
    assert client.calls == []
    assert judge.last_usage is None


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('{"scores": [{"id": 1, "score": 7}]}', {0: 7.0}),
        ('```json\n{"scores": [{"id": "2", "score": "3.5"}]}\n```', {1: 3.5}),
        ('{"scores": [{"id": 9, "score": 1}, {"id": 1}, {"id": 2, "score": "x"}]}', {}),
        ('{"scores": "nope"}', {}),
        ("[1, 2]", {}),
        ("{not json}", {}),
    ],
)
def test_parse_scores(reply: str, expected: dict[int, float]) -> None:
    assert parse_scores(reply, n_candidates=3) == expected


def test_build_requires_explicit_endpoint() -> None:
    with pytest.raises(ValueError, match="base_url"):
        build_reranker(RerankSettings(backend="llm_judge"))


def test_build_with_endpoint_makes_no_network_call() -> None:
    judge = build_llm_judge(
        LLMJudgeSettings(base_url="http://localhost:1/v1", model="m", price_in_per_mtok=1.0)
    )

    assert judge.model_name == "llm:m"


def test_pipeline_records_llm_usage(sample_index: Path, offline_settings: Settings) -> None:
    index = SearchEngine.open(sample_index, offline_settings).index
    judge, _ = _judge('{"scores": [{"id": 1, "score": 3}]}', price_in_per_mtok=2.0)
    engine = SearchEngine(index, offline_settings, judge)
    tracer = Tracer()

    result = engine.search("retry", k=2, tracer=tracer)

    assert result.reranked is True
    assert (tracer.tokens_in, tracer.tokens_out) == (1000, 50)
    assert tracer.cost_usd == pytest.approx(0.002)
    assert [s.name for s in tracer.stages] == ["bm25", "dense", "fusion", "rerank"]
