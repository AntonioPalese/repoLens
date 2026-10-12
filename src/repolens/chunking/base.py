"""Chunker protocol: one implementation per language."""

from __future__ import annotations

import hashlib
from typing import Protocol

from repolens.models import Chunk


class Chunker(Protocol):
    """Turns the source of one file into chunks."""

    @property
    def language(self) -> str:
        """Language identifier stored on every produced chunk, e.g. ``"python"``."""
        ...

    def supports(self, path: str) -> bool:
        """Return True if this chunker can handle the file at ``path``."""
        ...

    def chunk(self, path: str, source: str) -> list[Chunk]:
        """Split ``source`` (the content of repo-relative ``path``) into chunks."""
        ...


def make_chunk_id(path: str, start_line: int, end_line: int, kind: str, symbol: str | None) -> str:
    """Return a stable id for a chunk, identical across re-indexing of unchanged code."""
    key = "\0".join((path, str(start_line), str(end_line), kind, symbol or ""))
    return hashlib.sha1(key.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
