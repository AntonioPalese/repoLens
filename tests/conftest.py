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
def hashing_settings() -> Settings:
    """Settings using the offline hashing embedder (no model downloads)."""
    return Settings(embedding={"backend": "hashing"})


@pytest.fixture
def sample_chunks(hashing_settings: Settings) -> list[Chunk]:
    """All chunks of the fixture repository, in index order."""
    files = discover_files(SAMPLE_REPO, hashing_settings.indexing)
    return collect_chunks(SAMPLE_REPO, files, default_chunkers(hashing_settings), 1_000_000)


@pytest.fixture
def sample_index(tmp_path: Path, hashing_settings: Settings) -> Path:
    """A hashing-embedder index of the fixture repository."""
    out = tmp_path / "index"
    build_index(SAMPLE_REPO, out, hashing_settings, build_embedder(hashing_settings.embedding))
    return out
