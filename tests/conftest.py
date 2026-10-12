"""Shared pytest fixtures."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Keep tests independent of the developer's env vars and ./repolens.toml."""
    for name in list(os.environ):
        if name.startswith("REPOLENS_"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    yield
