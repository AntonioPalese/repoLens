"""Full indexing run: discover files -> chunk -> build BM25 and dense indexes -> persist."""

from __future__ import annotations

import logging
import re
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from repolens.chunking.base import Chunker
from repolens.chunking.python_ast import PythonChunker
from repolens.config import IndexingSettings, Settings
from repolens.embeddings import Embedder
from repolens.indexing.store import FORMAT_VERSION, IndexMeta, save_index
from repolens.indexing.tokenizer import TOKENIZER_VERSION, CodeTokenizer
from repolens.models import Chunk
from repolens.retrieval.bm25 import BM25Retriever
from repolens.retrieval.dense import DenseRetriever

logger = logging.getLogger(__name__)


class IndexBuildError(Exception):
    """Indexing could not produce a usable index."""


def build_index(
    repo_path: Path,
    out_dir: Path,
    settings: Settings,
    embedder: Embedder,
    chunkers: Sequence[Chunker] | None = None,
) -> IndexMeta:
    """Index ``repo_path`` from scratch and write the result to ``out_dir``.

    TODO(incremental): v0.1 always re-indexes everything. An incremental mode would
    diff ``discover_files`` + file hashes against the previous ``chunks.jsonl`` here.
    """
    repo = repo_path.resolve()
    if not repo.is_dir():
        raise IndexBuildError(f"{repo} is not a directory")
    chunkers = chunkers or default_chunkers(settings)

    files = discover_files(repo, settings.indexing)
    chunks = collect_chunks(repo, files, chunkers, settings.indexing.max_file_bytes)
    if not chunks:
        raise IndexBuildError(f"No indexable source files found in {repo}")
    logger.info("Chunked %d files into %d chunks", len(files), len(chunks))

    retrieval = settings.retrieval
    bm25 = BM25Retriever.build(chunks, CodeTokenizer(), retrieval.bm25_k1, retrieval.bm25_b)
    logger.info("Embedding %d chunks with %s", len(chunks), embedder.model_id)
    dense = DenseRetriever.build(
        chunks,
        embedder,
        m=retrieval.hnsw_m,
        ef_construction=retrieval.hnsw_ef_construction,
        ef_search=retrieval.hnsw_ef_search,
    )
    meta = IndexMeta(
        format_version=FORMAT_VERSION,
        repo_path=str(repo),
        commit=git_commit(repo),
        created_at=datetime.now(UTC),
        embedding_model=embedder.model_id,
        embedding_dim=dense.dim,
        tokenizer=TOKENIZER_VERSION,
        file_count=len({c.path for c in chunks}),
        chunk_count=len(chunks),
    )
    save_index(out_dir, meta, chunks, bm25, dense)
    return meta


def default_chunkers(settings: Settings) -> list[Chunker]:
    """Chunkers for all supported languages (Python only in v0.1)."""
    return [PythonChunker(max_chunk_lines=settings.chunking.max_chunk_lines)]


def discover_files(repo: Path, settings: IndexingSettings) -> list[str]:
    """Return sorted repo-relative POSIX paths matching include and not exclude.

    In a Git work tree, files ignored by ``.gitignore`` are skipped. Symlinks are
    never followed, so indexed content always lives inside ``repo``.
    """
    repo = repo.resolve()
    candidates = _git_files(repo)
    if candidates is None:
        candidates = [p.relative_to(repo).as_posix() for p in repo.rglob("*")]
    include = [glob_to_regex(p) for p in settings.include]
    exclude = [glob_to_regex(p) for p in settings.exclude]
    return sorted(
        rel
        for rel in set(candidates)
        if any(p.match(rel) for p in include)
        and not any(p.match(rel) for p in exclude)
        and _is_regular_file_inside(repo, rel)
    )


def collect_chunks(
    repo: Path, files: Sequence[str], chunkers: Sequence[Chunker], max_file_bytes: int
) -> list[Chunk]:
    """Chunk every file with the first chunker that supports it."""
    chunks: list[Chunk] = []
    for rel in files:
        chunker = next((c for c in chunkers if c.supports(rel)), None)
        source = read_source(repo / rel, max_file_bytes) if chunker else None
        if chunker is not None and source is not None:
            chunks.extend(chunker.chunk(rel, source))
    return chunks


def read_source(path: Path, max_bytes: int) -> str | None:
    """Read a text file, or return None for oversized or binary files."""
    if path.stat().st_size > max_bytes:
        logger.warning("Skipping %s: larger than %d bytes", path, max_bytes)
        return None
    data = path.read_bytes()
    if b"\0" in data:
        logger.warning("Skipping %s: looks binary", path)
        return None
    return data.decode("utf-8", errors="replace")


def git_commit(repo: Path) -> str | None:
    """HEAD commit of the work tree containing ``repo``, or None."""
    output = _git(repo, "rev-parse", "HEAD")
    return output.strip() if output else None


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Compile a glob where ``*`` stays within a path segment and ``**`` spans segments.

    ``**/`` also matches zero directories, so ``**/*.py`` matches ``setup.py``.
    """
    parts: list[str] = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            parts.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            parts.append(".*")
            i += 2
        else:
            char = pattern[i]
            parts.append({"*": "[^/]*", "?": "[^/]"}.get(char, re.escape(char)))
            i += 1
    return re.compile("".join(parts) + r"\Z")


def _git_files(repo: Path) -> list[str] | None:
    output = _git(repo, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
    if output is None:
        return None
    return [rel for rel in output.split("\0") if rel]


def _git(repo: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout


def _is_regular_file_inside(repo: Path, rel: str) -> bool:
    """True for a regular file that is not, and is not reached through, a symlink."""
    path = repo / rel
    return path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(repo)
