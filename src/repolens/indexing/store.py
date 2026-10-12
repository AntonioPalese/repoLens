"""On-disk index layout, persistence and validated loading.

Layout of an index directory::

    meta.json      IndexMeta (format version, repo, commit, models, counts)
    chunks.jsonl   one Chunk per line; line number == position in both indexes
    bm25/          bm25s index + our token vocabulary
    dense.faiss    FAISS HNSW index
    traces/        per-request JSONL traces (written by the server)
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ValidationError

from repolens.config import Settings
from repolens.embeddings import Embedder
from repolens.indexing.tokenizer import TOKENIZER_VERSION, CodeTokenizer
from repolens.models import Chunk
from repolens.retrieval.bm25 import BM25Retriever
from repolens.retrieval.dense import DenseRetriever

FORMAT_VERSION = 1


class IndexLoadError(Exception):
    """The index is missing, corrupt, or incompatible with the current configuration."""


class IndexMeta(BaseModel):
    """Metadata written alongside every index."""

    format_version: int
    repo_path: str
    commit: str | None
    created_at: datetime
    embedding_model: str
    embedding_dim: int
    tokenizer: str
    file_count: int
    chunk_count: int


@dataclass(frozen=True)
class IndexLayout:
    """Paths inside an index directory."""

    root: Path

    @property
    def meta(self) -> Path:
        return self.root / "meta.json"

    @property
    def chunks(self) -> Path:
        return self.root / "chunks.jsonl"

    @property
    def bm25_dir(self) -> Path:
        return self.root / "bm25"

    @property
    def dense(self) -> Path:
        return self.root / "dense.faiss"

    @property
    def traces_dir(self) -> Path:
        return self.root / "traces"


@dataclass(frozen=True)
class LoadedIndex:
    """Everything needed to serve queries from one index."""

    root: Path
    meta: IndexMeta
    chunks: list[Chunk]
    bm25: BM25Retriever
    dense: DenseRetriever


def save_index(
    out_dir: Path,
    meta: IndexMeta,
    chunks: Sequence[Chunk],
    bm25: BM25Retriever,
    dense: DenseRetriever,
) -> None:
    """Write an index atomically: build in a sibling temp dir, then swap it in.

    An existing ``out_dir`` is replaced only if it is empty or holds an index, so a
    mistyped ``--out`` cannot wipe an unrelated directory.
    """
    _check_replaceable(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.tmp-", dir=out_dir.parent))
    try:
        layout = IndexLayout(staging)
        layout.bm25_dir.mkdir()
        layout.chunks.write_text(
            "".join(c.model_dump_json() + "\n" for c in chunks), encoding="utf-8"
        )
        bm25.save(layout.bm25_dir)
        dense.save(layout.dense)
        layout.meta.write_text(meta.model_dump_json(indent=2), encoding="utf-8")
        _swap_in(staging, out_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def read_meta(index_dir: Path) -> IndexMeta:
    """Read and validate ``meta.json``."""
    layout = IndexLayout(index_dir)
    if not layout.meta.is_file():
        raise IndexLoadError(
            f"No index found at {index_dir}. Build one with "
            f"`repolens index <repo> --out {index_dir}`."
        )
    try:
        meta = IndexMeta.model_validate_json(layout.meta.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise IndexLoadError(f"Corrupt index metadata in {layout.meta}: {exc}") from exc
    if meta.format_version != FORMAT_VERSION:
        raise IndexLoadError(
            f"Index format v{meta.format_version} is not supported (expected "
            f"v{FORMAT_VERSION}); re-index the repository."
        )
    return meta


def load_index(index_dir: Path, settings: Settings, embedder: Embedder) -> LoadedIndex:
    """Load an index, failing fast if it does not match the current configuration."""
    meta = read_meta(index_dir)
    if meta.embedding_model != embedder.model_id:
        raise IndexLoadError(
            f"Index was built with embedding model {meta.embedding_model!r} but the "
            f"configuration selects {embedder.model_id!r}. Re-index or change the config."
        )
    if meta.tokenizer != TOKENIZER_VERSION:
        raise IndexLoadError(
            f"Index was built with tokenizer {meta.tokenizer!r}, this version uses "
            f"{TOKENIZER_VERSION!r}; re-index the repository."
        )
    layout = IndexLayout(index_dir)
    chunks = _read_chunks(layout.chunks)
    if len(chunks) != meta.chunk_count:
        raise IndexLoadError(
            f"{layout.chunks} has {len(chunks)} chunks, expected {meta.chunk_count}"
        )
    return LoadedIndex(
        root=index_dir,
        meta=meta,
        chunks=chunks,
        bm25=BM25Retriever.load(layout.bm25_dir, chunks, CodeTokenizer()),
        dense=DenseRetriever.load(
            layout.dense, chunks, embedder, ef_search=settings.retrieval.hnsw_ef_search
        ),
    )


def load_chunks(index_dir: Path) -> list[Chunk]:
    """Read only the chunks of an index (no models are loaded)."""
    meta = read_meta(index_dir)
    chunks = _read_chunks(IndexLayout(index_dir).chunks)
    if len(chunks) != meta.chunk_count:
        raise IndexLoadError(f"{index_dir} has {len(chunks)} chunks, expected {meta.chunk_count}")
    return chunks


def _read_chunks(path: Path) -> list[Chunk]:
    try:
        with path.open(encoding="utf-8") as lines:
            return [Chunk.model_validate_json(line) for line in lines if line.strip()]
    except (OSError, ValidationError) as exc:
        raise IndexLoadError(f"Cannot read chunks from {path}: {exc}") from exc


def _check_replaceable(out_dir: Path) -> None:
    if not out_dir.exists():
        return
    if not out_dir.is_dir():
        raise FileExistsError(f"{out_dir} exists and is not a directory")
    if any(out_dir.iterdir()) and not IndexLayout(out_dir).meta.is_file():
        raise FileExistsError(f"{out_dir} is not empty and does not contain a repoLens index")


def _swap_in(staging: Path, out_dir: Path) -> None:
    if not out_dir.exists():
        staging.rename(out_dir)
        return
    old_traces = IndexLayout(out_dir).traces_dir
    if old_traces.is_dir():  # traces are history, not derived data: keep them
        old_traces.rename(IndexLayout(staging).traces_dir)
    backup = out_dir.with_name(f".{out_dir.name}.old-{staging.name.rsplit('-', 1)[-1]}")
    out_dir.rename(backup)
    staging.rename(out_dir)
    shutil.rmtree(backup, ignore_errors=True)
