from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from repolens.config import Settings, load_settings


def test_defaults_keep_remote_calls_disabled() -> None:
    settings = load_settings()

    assert settings.rerank.backend == "none"
    assert settings.retrieval.default_mode == "dense"
    assert settings.rerank.llm_judge.configured is False
    assert settings.guard.classifier_enabled is False


def test_toml_file_is_loaded(tmp_path: Path) -> None:
    config = tmp_path / "custom.toml"
    config.write_text("seed = 7\n[retrieval]\nrrf_k = 30\n")

    settings = load_settings(config)

    assert settings.seed == 7
    assert settings.retrieval.rrf_k == 30
    assert settings.retrieval.candidate_k == 50  # untouched default


def test_default_toml_in_cwd_is_picked_up(tmp_path: Path) -> None:
    (tmp_path / "repolens.toml").write_text("[server]\nread_max_lines = 10\n")

    assert load_settings().server.read_max_lines == 10


def test_env_overrides_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "custom.toml"
    config.write_text("[retrieval]\nrrf_k = 30\ncandidate_k = 20\n")
    monkeypatch.setenv("REPOLENS_RETRIEVAL__RRF_K", "10")

    settings = load_settings(config)

    assert settings.retrieval.rrf_k == 10
    assert settings.retrieval.candidate_k == 20


def test_missing_explicit_config_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / "nope.toml")


def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    config = tmp_path / "typo.toml"
    config.write_text("[retrieval]\nrrf = 30\n")

    with pytest.raises(ValidationError):
        load_settings(config)


def test_llm_judge_configured_needs_url_and_model() -> None:
    settings = Settings(
        rerank={"llm_judge": {"base_url": "http://localhost:8000/v1", "model": "qwen"}}
    )

    assert settings.rerank.llm_judge.configured is True
