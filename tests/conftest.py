"""Shared pytest fixtures."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from repolens.config import Settings
from repolens.embeddings import build_embedder
from repolens.indexing.indexer import build_index, collect_chunks, default_chunkers, discover_files
from repolens.models import Chunk


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Keep tests independent of the developer's env vars and ./repolens.toml."""
    for name in list(os.environ):
        if name.startswith("REPOLENS_"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    yield


FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_REPO = FIXTURES / "sample_repo"


@pytest.fixture
def sample_repo() -> Path:
    """Path to the tiny fixture repository."""
    return SAMPLE_REPO


@pytest.fixture
def offline_settings() -> Settings:
    """Hashing embedder and no reranker: nothing is downloaded."""
    return Settings(embedding={"backend": "hashing"}, rerank={"backend": "none"})


@pytest.fixture
def sample_chunks(offline_settings: Settings) -> list[Chunk]:
    """All chunks of the fixture repository, in index order."""
    files = discover_files(SAMPLE_REPO, offline_settings.indexing)
    return collect_chunks(SAMPLE_REPO, files, default_chunkers(offline_settings), 1_000_000)


@pytest.fixture
def sample_index(tmp_path: Path, offline_settings: Settings) -> Path:
    """A hashing-embedder index of the fixture repository."""
    out = tmp_path / "index"
    build_index(SAMPLE_REPO, out, offline_settings, build_embedder(offline_settings.embedding))
    return out
