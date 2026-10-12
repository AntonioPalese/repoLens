from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from repolens import __version__
from repolens.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_help_lists_all_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("index", "query", "serve", "eval"):
        assert command in result.stdout


def test_stub_command_fails_cleanly(tmp_path: Path) -> None:
    result = runner.invoke(app, ["serve", "--index", str(tmp_path)])

    assert result.exit_code == 1
    assert "not implemented" in result.stderr


def test_bad_config_exits_with_code_2(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["query", str(tmp_path), "q", "--config", str(tmp_path / "missing.toml")]
    )

    assert result.exit_code == 2
    assert "Invalid configuration" in result.stderr


def test_index_then_query(
    tmp_path: Path, sample_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REPOLENS_EMBEDDING__BACKEND", "hashing")
    out = tmp_path / "idx"

    indexed = runner.invoke(app, ["index", str(sample_repo), "--out", str(out)])
    queried = runner.invoke(app, ["query", str(out), "parseAuthHeader", "--k", "1"])

    assert indexed.exit_code == 0, indexed.output
    assert "Index built" in indexed.stdout
    assert queried.exit_code == 0, queried.output
    assert "minihttp/auth.py" in queried.stdout
    assert "parseAuthHeader" in queried.stdout
    assert "(dense)" in queried.stdout  # default mode, no reranker configured


def test_query_missing_index_fails_cleanly(tmp_path: Path) -> None:
    result = runner.invoke(app, ["query", str(tmp_path / "nope"), "q", "--mode", "bm25"])

    assert result.exit_code == 2
    assert "No index found" in result.stderr


def test_index_refuses_unrelated_directory(tmp_path: Path, sample_repo: Path) -> None:
    out = tmp_path / "precious"
    out.mkdir()
    (out / "notes.txt").write_text("keep me")

    result = runner.invoke(app, ["index", str(sample_repo), "--out", str(out)])

    assert result.exit_code == 2
    assert (out / "notes.txt").exists()
