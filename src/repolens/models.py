"""Domain models shared across indexing, retrieval, the MCP server and tracing."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChunkKind(StrEnum):
    """Syntactic unit a chunk was cut from."""

    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    MODULE = "module"


class InjectionFlag(StrEnum):
    """Outcome of the prompt-injection guard for one chunk."""

    NONE = "none"
    SUSPECTED = "suspected"
    BLOCKED = "blocked"


class SearchMode(StrEnum):
    """First-stage retrieval strategy."""

    HYBRID = "hybrid"
    BM25 = "bm25"
    DENSE = "dense"


class Chunk(BaseModel):
    """A contiguous, indexable span of a source file.

    ``path`` is POSIX-style and relative to the repository root; lines are 1-based
    and inclusive.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    kind: ChunkKind
    symbol: str | None = Field(default=None, description="Qualified name, e.g. 'Client.send'.")
    language: str
    text: str

    @model_validator(mode="after")
    def _check_line_range(self) -> Self:
        if self.end_line < self.start_line:
            raise ValueError(f"end_line ({self.end_line}) < start_line ({self.start_line})")
        return self


class ScoredChunk(BaseModel):
    """A chunk with the score assigned by a retriever or reranker (higher is better)."""

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float


class SearchHit(BaseModel):
    """One result as returned to MCP clients."""

    path: str
    start_line: int
    end_line: int
    symbol: str | None
    kind: ChunkKind
    score: float
    snippet: str
    injection_flag: InjectionFlag = InjectionFlag.NONE
    injection_reasons: list[str] = Field(default_factory=list)


class SearchResult(BaseModel):
    """Ranked evidence for a query."""

    request_id: str
    query: str
    mode: SearchMode
    reranked: bool
    hits: list[SearchHit]


class IndexStatus(BaseModel):
    """Metadata describing a persisted index."""

    repo_path: str
    commit: str | None = Field(description="HEAD at index time; None if not a git repo.")
    chunk_count: int = Field(ge=0)
    embedding_model: str
    rerank_model: str | None
    created_at: datetime
    age_seconds: float = Field(ge=0)


class StageRecord(BaseModel):
    """Timing and fan-out of one pipeline stage."""

    name: str
    duration_ms: float = Field(ge=0)
    candidates: int | None = None


class TraceRecord(BaseModel):
    """One JSONL line written per tool call."""

    request_id: str
    timestamp: datetime
    tool: str
    query: str | None = None
    stages: list[StageRecord] = Field(default_factory=list)
    rerank_model: str | None = None
    llm_tokens_in: int = Field(default=0, ge=0)
    llm_tokens_out: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0)


class FileRange(BaseModel):
    """Lines read from an indexed file, after the injection guard."""

    path: str
    start_line: int
    end_line: int = Field(description="Last line returned (clamped to the file and the limit).")
    total_lines: int
    truncated: bool = Field(description="True if fewer lines than requested were returned.")
    text: str
    injection_flag: InjectionFlag = InjectionFlag.NONE
    injection_reasons: list[str] = Field(default_factory=list)
