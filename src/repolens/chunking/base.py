"""Chunker protocol: one implementation per language."""

from __future__ import annotations

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
