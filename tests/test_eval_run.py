from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from repolens.cli import app
from repolens.config import Settings
from repolens.embeddings import HashingEmbedder
from repolens.evals.dataset import (
    DatasetError,
    DatasetMeta,
    GoldItem,
    Question,
    load_meta,
    load_questions,
)
from repolens.evals.run import (
    CONFIGS,
    EvalError,
    EvalReport,
    markdown_report,
    run_eval,
    select_configs,
    validate_gold,
    write_report,
)
from repolens.indexing.store import LoadedIndex, load_index
from repolens.rerank import cross_encoder


class FakeCrossEncoder:
    def predict(self, pairs: list[tuple[str, str]], **_: object) -> list[float]:
        return [float(len(doc)) for _, doc in pairs]


QUESTIONS = [
    Question(
        id="q1",
        query="parseAuthHeader",
        gold=[GoldItem(path="minihttp/auth.py", symbol="parseAuthHeader")],
        tags=["symbol"],
    ),
    Question(
        id="q2",
        query="how long to wait before retrying",
        gold=[
            GoldItem(path="minihttp/client.py", symbol="RetryPolicy.backoff"),
            GoldItem(path="minihttp/client.py", start_line=8, end_line=8),
        ],
        tags=["concept"],
    ),
]


@pytest.fixture
def loaded(sample_index: Path, offline_settings: Settings) -> LoadedIndex:
    return load_index(sample_index, offline_settings, HashingEmbedder())


def _write_dataset(path: Path, questions: list[Question]) -> Path:
    path.write_text("".join(q.model_dump_json() + "\n" for q in questions))
    return path


class TestDataset:
    def test_round_trip_and_meta(self, tmp_path: Path) -> None:
        dataset = _write_dataset(tmp_path / "mini.jsonl", QUESTIONS)
        meta = DatasetMeta(repo_url="https://example.com/r", commit="abc")
        (tmp_path / "mini.meta.json").write_text(meta.model_dump_json())

        assert load_questions(dataset) == QUESTIONS
        assert load_meta(dataset) == meta
        assert load_meta(tmp_path / "other.jsonl") is None

    @pytest.mark.parametrize(
        ("content", "message"),
        [
            ('{"id": "a", "query": "q", "gold": []}\n', "gold"),
            ('{"id": "a", "query": "q", "gold": [{"path": "x.py", "symbol": "f"}], "x": 1}\n', "x"),
            ("", "no questions"),
            (
                '{"id": "a", "query": "q", "gold": [{"path": "x.py", "symbol": "f"}]}\n' * 2,
                "duplicate",
            ),
        ],
    )
    def test_rejects_malformed_files(self, tmp_path: Path, content: str, message: str) -> None:
        dataset = tmp_path / "bad.jsonl"
        dataset.write_text(content)

        with pytest.raises(DatasetError, match=message):
            load_questions(dataset)

    def test_shipped_httpx_dataset_is_valid(self) -> None:
        dataset = Path(__file__).parents[1] / "evals" / "datasets" / "httpx.jsonl"

        questions = load_questions(dataset)
        meta = load_meta(dataset)

        assert len(questions) == 30
        assert meta is not None
        assert len(meta.commit) == 40
        assert {t for q in questions for t in q.tags} == {"symbol", "concept", "cross-file"}


class TestRunner:
    def test_select_configs(self) -> None:
        assert select_configs("all") == list(CONFIGS)
        assert [c.name for c in select_configs("bm25, hybrid+ce")] == ["bm25", "hybrid+ce"]
        with pytest.raises(EvalError, match="Unknown"):
            select_configs("bm25,nope")

    def test_validate_gold_reports_unknown_locations(self, loaded: LoadedIndex) -> None:
        bad = Question(id="x", query="q", gold=[GoldItem(path="minihttp/auth.py", symbol="nope")])

        assert validate_gold(QUESTIONS, loaded.chunks) == []
        assert validate_gold([bad], loaded.chunks) == [
            "x: no chunk matches {'path': 'minihttp/auth.py', 'symbol': 'nope'}"
        ]

    def test_run_rejects_unresolvable_gold(
        self, loaded: LoadedIndex, offline_settings: Settings, tmp_path: Path
    ) -> None:
        bad = Question(id="x", query="q", gold=[GoldItem(path="missing.py", symbol="f")])

        with pytest.raises(EvalError, match=r"missing\.py"):
            run_eval(loaded, [bad], offline_settings, select_configs("bm25"), tmp_path / "d")

    def test_report_contents(
        self,
        loaded: LoadedIndex,
        offline_settings: Settings,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # hybrid+ce forces the cross-encoder backend: swap in a fake model.
        monkeypatch.setattr(cross_encoder, "_load_cross_encoder", lambda *_: FakeCrossEncoder())
        meta = DatasetMeta(repo_url="u", commit="not-the-index-commit")

        report = run_eval(
            loaded, QUESTIONS, offline_settings, select_configs("all"), tmp_path / "d", meta
        )

        by_name = {c.name: c for c in report.configs}
        assert by_name["hybrid+llm"].skipped == "no LLM endpoint configured"
        bm25 = by_name["bm25"]
        assert bm25.metrics["recall@1"] >= 0.5  # q1 is an exact identifier
        assert bm25.questions[0].gold_ranks == [1]
        assert set(by_name["hybrid"].latency) == {"bm25", "dense", "fusion", "total"}
        assert set(bm25.by_tag) == {"concept", "symbol"}
        assert by_name["hybrid+ce"].reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert report.warnings and "differs" in report.warnings[0]

    def test_write_and_render(
        self, loaded: LoadedIndex, offline_settings: Settings, tmp_path: Path
    ) -> None:
        report = run_eval(
            loaded, QUESTIONS, offline_settings, select_configs("bm25,dense"), tmp_path
        )

        path = write_report(report, tmp_path / "results")
        table = markdown_report(report)

        assert EvalReport.model_validate_json(path.read_text()) == report
        assert "| bm25 |" in table
        assert "| dense |" in table
        assert "by question tag" in table


def test_eval_cli(sample_index: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REPOLENS_EMBEDDING__BACKEND", "hashing")
    dataset = _write_dataset(tmp_path / "mini.jsonl", QUESTIONS)
    out = tmp_path / "results"

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "--index",
            str(sample_index),
            "--dataset",
            str(dataset),
            "--configs",
            "bm25",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.stdout.lstrip().startswith("Dataset: `mini.jsonl` (2 questions)")
    assert len(list(out.glob("*.json"))) == 1
    assert json.loads(next(out.glob("*.json")).read_text())["question_count"] == 2
