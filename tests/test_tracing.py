from __future__ import annotations

import pytest

from repolens.tracing import LLMUsage, Tracer


def test_stages_record_duration_and_candidates() -> None:
    tracer = Tracer()

    with tracer.stage("bm25") as stage:
        stage.candidates = 7
    with tracer.stage("dense"):
        pass

    assert [(s.name, s.candidates) for s in tracer.stages] == [("bm25", 7), ("dense", None)]
    assert all(s.duration_ms >= 0 for s in tracer.stages)


def test_stage_is_recorded_when_it_raises() -> None:
    tracer = Tracer()

    with pytest.raises(RuntimeError), tracer.stage("rerank"):
        raise RuntimeError("boom")

    assert [s.name for s in tracer.stages] == ["rerank"]


def test_durations_sum_repeated_stages() -> None:
    tracer = Tracer()
    for _ in range(2):
        with tracer.stage("dense"):
            pass

    assert set(tracer.durations()) == {"dense"}
    assert tracer.durations()["dense"] == pytest.approx(sum(s.duration_ms for s in tracer.stages))


def test_llm_usage_accumulates() -> None:
    tracer = Tracer()

    tracer.add_llm_usage(LLMUsage(tokens_in=10, tokens_out=2, cost_usd=0.5))
    tracer.add_llm_usage(LLMUsage(tokens_in=5, tokens_out=1))

    assert (tracer.tokens_in, tracer.tokens_out, tracer.cost_usd) == (15, 3, 0.5)
