"""Guarded, path-safe reads of line ranges from the indexed repository.

Only files that are part of the index can be read: a client cannot use this to
fetch ``.env``, ``.git/config`` or anything outside the repository, including via
``..`` or symlinks. The returned text goes through the same injection guard as
search results, so a chunk that search withholds cannot be read raw by asking for
its lines.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from repolens.guard.base import InjectionDetector
from repolens.guard.policy import assess, render_snippet
from repolens.models import FileRange


class FileAccessError(ValueError):
    """The requested path or range cannot be served."""


class FileReader:
    """Serves line ranges of indexed files under ``repo_root``."""

    def __init__(
        self,
        repo_root: Path,
        indexed_paths: Iterable[str],
        max_lines: int = 400,
        detector: InjectionDetector | None = None,
        block_high_severity: bool = True,
    ) -> None:
        self._root = repo_root.resolve()
        self._indexed = frozenset(indexed_paths)
        self._max_lines = max_lines
        self._detector = detector
        self._block_high = block_high_severity

    def read(self, path: str, start_line: int, end_line: int) -> FileRange:
        """Return lines ``start_line..end_line`` (1-based, inclusive) of ``path``.

        ``end_line`` is clamped to the end of the file and to ``max_lines`` lines;
        ``truncated`` reports when that happened.
        """
        if start_line < 1 or end_line < start_line:
            raise FileAccessError(f"Invalid range {start_line}-{end_line}")
        file = self._resolve(path)
        lines = file.read_text(encoding="utf-8", errors="replace").split("\n")
        if lines and lines[-1] == "":
            lines.pop()  # a trailing newline ends the last line; it does not start a new one
        total = len(lines)
        if start_line > total:
            raise FileAccessError(f"{path} has only {total} lines")
        last = min(end_line, total, start_line + self._max_lines - 1)
        text = "\n".join(line.removesuffix("\r") for line in lines[start_line - 1 : last])
        detections = self._detector.scan(text) if self._detector else []
        verdict = assess(detections, self._block_high)
        return FileRange(
            path=path,
            start_line=start_line,
            end_line=last,
            total_lines=total,
            truncated=last < end_line,
            text=render_snippet(text, verdict),
            injection_flag=verdict.flag,
            injection_reasons=verdict.reasons,
        )

    def _resolve(self, path: str) -> Path:
        relative = PurePosixPath(path)
        if relative.is_absolute() or ".." in relative.parts or path != relative.as_posix():
            raise FileAccessError(f"Path must be relative to the repository root: {path!r}")
        if path not in self._indexed:
            raise FileAccessError(f"Not an indexed file: {path!r}")
        file = (self._root / relative).resolve()
        if not file.is_relative_to(self._root) or not file.is_file():
            raise FileAccessError(f"File not available: {path!r}")
        return file
