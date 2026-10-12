from __future__ import annotations

from pathlib import Path

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
    result = runner.invoke(app, ["index", str(tmp_path), "--out", str(tmp_path / "idx")])

    assert result.exit_code == 1
    assert "not implemented" in result.stderr


def test_bad_config_exits_with_code_2(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["query", str(tmp_path), "q", "--config", str(tmp_path / "missing.toml")]
    )

    assert result.exit_code == 2
    assert "Invalid configuration" in result.stderr
