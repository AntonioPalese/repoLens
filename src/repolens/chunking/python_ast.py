"""AST-aware Python chunker built on tree-sitter.

Chunking rules (rows are 0-based internally, lines 1-based in :class:`Chunk`):

* Every top-level function becomes a ``function`` chunk; every function directly
  inside a class body becomes a ``method`` chunk. Nested functions stay inside
  their parent's chunk.
* A class produces a ``class`` chunk covering its header (decorators, signature,
  docstring and the statements before its first method), then one chunk per
  method. Class-level statements found after a method become extra ``class``
  chunks. Nested classes recurse, giving qualified names like ``Outer.Inner.m``.
* Consecutive module-level statements between definitions are grouped into
  ``module`` chunks (``symbol`` is ``None``).
* Comments directly above a definition (no blank line in between) are attached
  to that definition.
* Any chunk longer than ``max_chunk_lines`` is split into consecutive windows
  that keep the same kind and symbol.

Chunk text is always the exact source lines of its range, so a snippet and a
``read_file_range`` call over the same lines agree. ``symbol`` is qualified
within the module (``Client.send``); the module itself is identified by ``path``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath

import tree_sitter_python
from tree_sitter import Language, Node, Parser

from repolens.chunking.base import make_chunk_id
from repolens.models import Chunk, ChunkKind

_LANGUAGE = "python"
_SUFFIXES = frozenset({".py", ".pyi"})


@dataclass
class _FileContext:
    """Per-file state shared by the walk: source lines and collected chunks."""

    path: str
    lines: list[str]
    max_lines: int
    chunks: list[Chunk] = field(default_factory=list)

    def emit(self, kind: ChunkKind, symbol: str | None, start_row: int, end_row: int) -> None:
        """Add a chunk for rows ``start_row..end_row``, split into windows if too long."""
        for window_start in range(start_row, end_row + 1, self.max_lines):
            window_end = min(window_start + self.max_lines - 1, end_row)
            self._append(kind, symbol, window_start + 1, window_end + 1)

    def _append(self, kind: ChunkKind, symbol: str | None, start_line: int, end_line: int) -> None:
        self.chunks.append(
            Chunk(
                id=make_chunk_id(self.path, start_line, end_line, kind, symbol),
                path=self.path,
                start_line=start_line,
                end_line=end_line,
                kind=kind,
                symbol=symbol,
                language=_LANGUAGE,
                text="\n".join(self.lines[start_line - 1 : end_line]),
            )
        )


@dataclass
class _Scope:
    """Where a block of statements lives: the module, or a class body."""

    qualname: tuple[str, ...] = ()
    # Rows of the enclosing class header, prepended to the first loose group.
    header: tuple[int, int] | None = None

    @property
    def symbol(self) -> str | None:
        return ".".join(self.qualname) or None

    @property
    def loose_kind(self) -> ChunkKind:
        return ChunkKind.CLASS if self.qualname else ChunkKind.MODULE


class PythonChunker:
    """Chunker for Python source files (see module docstring for the rules)."""

    def __init__(self, max_chunk_lines: int = 200) -> None:
        if max_chunk_lines < 1:
            raise ValueError("max_chunk_lines must be >= 1")
        self._max_lines = max_chunk_lines
        self._parser = Parser(Language(tree_sitter_python.language()))

    @property
    def language(self) -> str:
        """Language identifier stored on every chunk."""
        return _LANGUAGE

    def supports(self, path: str) -> bool:
        """Return True for ``.py`` and ``.pyi`` files."""
        return PurePosixPath(path).suffix in _SUFFIXES

    def chunk(self, path: str, source: str) -> list[Chunk]:
        """Split ``source`` into chunks in document order.

        Syntax errors are tolerated: tree-sitter recovers locally and unparsable
        regions end up in ``module`` (or ``class``) chunks.
        """
        lines = [line.removesuffix("\r") for line in source.split("\n")]
        ctx = _FileContext(path=path, lines=lines, max_lines=self._max_lines)
        tree = self._parser.parse(source.encode("utf-8"))
        self._walk_block(tree.root_node.children, _Scope(), ctx)
        return ctx.chunks

    def _walk_block(self, nodes: Sequence[Node], scope: _Scope, ctx: _FileContext) -> None:
        """Emit chunks for the statements of a module or class body."""
        loose: list[Node] = []
        for node in nodes:
            definition = _definition_of(node)
            if definition is None:
                loose.append(node)
                continue
            start_row = _pop_leading_comments(loose, node.start_point.row)
            self._flush_loose(loose, scope, ctx)
            loose = []
            self._emit_definition(definition, start_row, node.end_point.row, scope, ctx)
        self._flush_loose(loose, scope, ctx)

    def _flush_loose(self, loose: list[Node], scope: _Scope, ctx: _FileContext) -> None:
        """Emit one chunk for a run of non-definition statements (and the class header)."""
        rows = [(n.start_point.row, n.end_point.row) for n in loose]
        if scope.header is not None:
            rows.append(scope.header)
            scope.header = None
        if rows:
            start = min(r[0] for r in rows)
            end = max(r[1] for r in rows)
            ctx.emit(scope.loose_kind, scope.symbol, start, end)

    def _emit_definition(
        self, definition: Node, start_row: int, end_row: int, scope: _Scope, ctx: _FileContext
    ) -> None:
        name = _name_of(definition)
        qualname = (*scope.qualname, name)
        if definition.type == "function_definition":
            kind = ChunkKind.METHOD if scope.qualname else ChunkKind.FUNCTION
            ctx.emit(kind, ".".join(qualname), start_row, end_row)
            return

        body = definition.child_by_field_name("body")
        body_nodes = body.children if body is not None else []
        body_row = body.start_point.row if body is not None else end_row + 1
        # Header = decorators/comments + signature lines up to (not including) the body,
        # or just the first line when the body sits on the same line as `class`.
        header_end = max(start_row, body_row - 1, definition.start_point.row)
        class_scope = _Scope(qualname=qualname, header=(start_row, header_end))
        self._walk_block(body_nodes, class_scope, ctx)


def _definition_of(node: Node) -> Node | None:
    """Return the function/class definition a top-level node holds, if any."""
    if node.type == "decorated_definition":
        inner = node.child_by_field_name("definition")
        node = inner if inner is not None else node
    if node.type in ("function_definition", "class_definition") and _has_name(node):
        return node
    return None


def _has_name(definition: Node) -> bool:
    name = definition.child_by_field_name("name")
    return name is not None and bool(name.text)


def _name_of(definition: Node) -> str:
    name = definition.child_by_field_name("name")
    assert name is not None and name.text is not None  # guaranteed by _definition_of
    return name.text.decode("utf-8", errors="replace")


def _pop_leading_comments(loose: list[Node], start_row: int) -> int:
    """Detach comments directly above a definition from ``loose``; return the new start row."""
    while loose and loose[-1].type == "comment" and loose[-1].end_point.row == start_row - 1:
        start_row = loose.pop().start_point.row
    return start_row
