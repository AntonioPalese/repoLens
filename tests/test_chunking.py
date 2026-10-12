from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from repolens.chunking.base import Chunker
from repolens.chunking.python_ast import PythonChunker
from repolens.models import Chunk, ChunkKind


@pytest.fixture
def chunker() -> PythonChunker:
    return PythonChunker()


def _chunk_text(chunker: PythonChunker, source: str) -> list[Chunk]:
    return chunker.chunk("mod.py", textwrap.dedent(source))


def _by_symbol(chunks: list[Chunk]) -> dict[str | None, Chunk]:
    return {c.symbol: c for c in chunks}


def _fixture_files(sample_repo: Path) -> list[Path]:
    return sorted(sample_repo.rglob("*.py"))


def test_satisfies_chunker_protocol(chunker: PythonChunker) -> None:
    as_protocol: Chunker = chunker
    assert as_protocol.language == "python"


@pytest.mark.parametrize(
    ("path", "expected"),
    [("a/b.py", True), ("stubs/x.pyi", True), ("README.md", False), ("py", False)],
)
def test_supports(chunker: PythonChunker, path: str, expected: bool) -> None:
    assert chunker.supports(path) is expected


def test_client_fixture_structure(chunker: PythonChunker, sample_repo: Path) -> None:
    path = "minihttp/client.py"
    chunks = chunker.chunk(path, (sample_repo / path).read_text())

    summary = [(c.kind, c.symbol) for c in chunks]

    assert summary == [
        (ChunkKind.MODULE, None),
        (ChunkKind.CLASS, "RetryPolicy"),
        (ChunkKind.METHOD, "RetryPolicy.should_retry"),
        (ChunkKind.METHOD, "RetryPolicy.backoff"),
        (ChunkKind.CLASS, "Client"),
        (ChunkKind.METHOD, "Client.__init__"),
        (ChunkKind.METHOD, "Client.base_url"),
        (ChunkKind.METHOD, "Client.send"),
        (ChunkKind.METHOD, "Client.get"),
        (ChunkKind.METHOD, "Client.aget"),
        (ChunkKind.METHOD, "Client._transport"),
        (ChunkKind.CLASS, "Client.Config"),
        (ChunkKind.METHOD, "Client.Config.as_dict"),
        (ChunkKind.CLASS, "Client"),  # class attribute after the methods
    ]


def test_top_level_functions_and_module_code(chunker: PythonChunker) -> None:
    chunks = _chunk_text(
        chunker,
        '''
        """Doc."""
        import os

        def f():
            return os.sep

        X = 1
        ''',
    )

    assert [(c.kind, c.symbol, c.start_line, c.end_line) for c in chunks] == [
        (ChunkKind.MODULE, None, 2, 3),
        (ChunkKind.FUNCTION, "f", 5, 6),
        (ChunkKind.MODULE, None, 8, 8),
    ]


def test_decorators_and_leading_comments_are_attached(chunker: PythonChunker) -> None:
    chunks = _chunk_text(
        chunker,
        """
        X = 1

        # Explains f.
        # More about f.
        @decorator
        @other(arg=1)
        def f():
            pass
        """,
    )

    f = _by_symbol(chunks)["f"]
    assert (f.start_line, f.end_line) == (4, 9)
    assert f.text.startswith("# Explains f.")


def test_comment_separated_by_blank_line_stays_in_module(chunker: PythonChunker) -> None:
    chunks = _chunk_text(
        chunker,
        """
        # Section header.

        def f():
            pass
        """,
    )

    assert _by_symbol(chunks)[None].text == "# Section header."
    assert _by_symbol(chunks)["f"].start_line == 4


def test_nested_functions_stay_in_parent(chunker: PythonChunker) -> None:
    chunks = _chunk_text(
        chunker,
        """
        def outer():
            def inner():
                class Local:
                    pass
            return inner
        """,
    )

    assert [c.symbol for c in chunks] == ["outer"]


def test_class_header_includes_docstring_and_attributes(chunker: PythonChunker) -> None:
    chunks = _chunk_text(
        chunker,
        """
        class A(
            Base,
        ):
            \"\"\"Doc.\"\"\"
            x = 1

            def m(self):
                pass
        """,
    )

    header = _by_symbol(chunks)["A"]
    assert (header.kind, header.start_line, header.end_line) == (ChunkKind.CLASS, 2, 6)
    assert _by_symbol(chunks)["A.m"].kind is ChunkKind.METHOD


def test_class_without_methods_is_one_chunk(chunker: PythonChunker) -> None:
    chunks = _chunk_text(chunker, "class A: pass\n\nclass B:\n    x = 1\n")

    assert [(c.symbol, c.start_line, c.end_line) for c in chunks] == [
        ("A", 1, 1),
        ("B", 3, 4),
    ]


def test_class_with_only_methods_gets_header_chunk(chunker: PythonChunker) -> None:
    chunks = _chunk_text(chunker, "class A:\n    def m(self):\n        pass\n")

    assert [(c.symbol, c.start_line, c.end_line) for c in chunks] == [
        ("A", 1, 1),
        ("A.m", 2, 3),
    ]


def test_oversized_definitions_are_split(chunker: PythonChunker) -> None:
    body = "\n".join(f"    x{i} = {i}" for i in range(9))
    small = PythonChunker(max_chunk_lines=4)

    chunks = small.chunk("mod.py", f"def big():\n{body}\n")

    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 4), (5, 8), (9, 10)]
    assert {c.symbol for c in chunks} == {"big"}
    assert len({c.id for c in chunks}) == 3


def test_rejects_non_positive_max_lines() -> None:
    with pytest.raises(ValueError, match="max_chunk_lines"):
        PythonChunker(max_chunk_lines=0)


@pytest.mark.parametrize("source", ["", "\n\n   \n"])
def test_empty_files_produce_no_chunks(chunker: PythonChunker, source: str) -> None:
    assert chunker.chunk("empty.py", source) == []


def test_syntax_errors_do_not_lose_code(chunker: PythonChunker) -> None:
    source = "def ok():\n    return 1\n\ndef broken(:\n    pass\n\nx = (\n"

    chunks = chunker.chunk("bad.py", source)

    assert "ok" in _by_symbol(chunks)
    _assert_covers_non_blank_lines(chunks, source)


def test_crlf_line_endings(chunker: PythonChunker) -> None:
    chunks = chunker.chunk("win.py", "def f():\r\n    pass\r\n")

    assert chunks[0].text == "def f():\n    pass"


def test_chunk_ids_are_stable_and_unique(chunker: PythonChunker, sample_repo: Path) -> None:
    for file in _fixture_files(sample_repo):
        source = file.read_text()
        first = chunker.chunk(file.name, source)
        second = PythonChunker().chunk(file.name, source)

        assert [c.id for c in first] == [c.id for c in second]
        assert len({c.id for c in first}) == len(first)


def test_text_matches_source_lines(chunker: PythonChunker, sample_repo: Path) -> None:
    for file in _fixture_files(sample_repo):
        lines = file.read_text().split("\n")
        for c in chunker.chunk(file.name, file.read_text()):
            assert c.text == "\n".join(lines[c.start_line - 1 : c.end_line])
            assert c.language == "python"


def test_every_non_blank_line_is_covered(chunker: PythonChunker, sample_repo: Path) -> None:
    for file in _fixture_files(sample_repo):
        source = file.read_text()
        _assert_covers_non_blank_lines(chunker.chunk(file.name, source), source)


def test_injection_payloads_land_in_chunks(chunker: PythonChunker, sample_repo: Path) -> None:
    source = (sample_repo / "minihttp/plugins.py").read_text()
    chunks = _by_symbol(chunker.chunk("minihttp/plugins.py", source))

    assert "ignore all previous instructions" in (chunks[None].text)
    assert "developer mode" in chunks["register"].text
    assert "\u200b" in chunks["unregister"].text


def _assert_covers_non_blank_lines(chunks: list[Chunk], source: str) -> None:
    covered = {line for c in chunks for line in range(c.start_line, c.end_line + 1)}
    non_blank = {i for i, line in enumerate(source.split("\n"), start=1) if line.strip()}
    assert non_blank <= covered, f"uncovered lines: {sorted(non_blank - covered)}"


def test_chunks_real_world_sources(chunker: PythonChunker) -> None:
    # Regression: tree-sitter 0.26.0 corrupted memory (segfault) on ordinary
    # mid-size files that the tiny fixtures never triggered. Chunk our own package.
    package = Path(__file__).parents[1] / "src" / "repolens"
    files = sorted(package.rglob("*.py"))

    for file in files:
        source = file.read_text()
        _assert_covers_non_blank_lines(chunker.chunk(file.name, source), source)
    assert len(files) > 20
