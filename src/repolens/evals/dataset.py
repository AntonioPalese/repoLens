"""Benchmark dataset format.

A dataset is a JSONL file, one question per line::

    {"id": "q001", "query": "where are HTTP retries configured?",
     "gold": [{"path": "httpx/_transports/default.py", "symbol": "HTTPTransport.__init__"}],
     "tags": ["concept"]}

A gold item names a chunk by ``path`` + ``symbol`` (qualified within the module, as
produced by the chunker), or by ``path`` + line range when ``symbol`` is null.

The repository and commit the questions were written against live in a sidecar
``<name>.meta.json`` file, so the JSONL lines keep the documented format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class DatasetError(Exception):
    """The dataset file is missing or malformed."""


class GoldItem(BaseModel):
    """One relevant location for a question."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    symbol: str | None = None
    start_line: int | None = Field(default=None, ge=1)
    end_line: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _needs_symbol_or_lines(self) -> Self:
        if self.symbol is None and (self.start_line is None or self.end_line is None):
            raise ValueError("a gold item without symbol needs start_line and end_line")
        return self


class Question(BaseModel):
    """A benchmark query and its gold locations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    query: str
    gold: list[GoldItem] = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)


class DatasetMeta(BaseModel):
    """Repository the dataset was written against."""

    repo_url: str
    commit: str
    description: str = ""


def load_questions(path: Path) -> list[Question]:
    """Parse a JSONL dataset; ids must be unique."""
    if not path.is_file():
        raise DatasetError(f"Dataset not found: {path}")
    questions: list[Question] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            questions.append(Question.model_validate_json(line))
        except ValidationError as exc:
            raise DatasetError(f"{path}:{number}: {exc}") from exc
    ids = [q.id for q in questions]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise DatasetError(f"{path}: duplicate question ids {duplicates}")
    if not questions:
        raise DatasetError(f"{path}: no questions")
    return questions


def meta_path(dataset: Path) -> Path:
    """Sidecar metadata path: ``httpx.jsonl`` -> ``httpx.meta.json``."""
    return dataset.with_suffix(".meta.json")


def load_meta(dataset: Path) -> DatasetMeta | None:
    """Read the sidecar metadata if present."""
    path = meta_path(dataset)
    if not path.is_file():
        return None
    try:
        return DatasetMeta.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise DatasetError(f"{path}: {exc}") from exc
