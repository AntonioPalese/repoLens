from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from repolens.models import Chunk, ChunkKind, InjectionFlag, SearchHit


def _chunk(start: int, end: int) -> Chunk:
    return Chunk(
        id="c1",
        path="pkg/mod.py",
        start_line=start,
        end_line=end,
        kind=ChunkKind.FUNCTION,
        symbol="pkg.mod.f",
        language="python",
        text="def f(): ...",
    )


def test_chunk_accepts_single_line_range() -> None:
    assert _chunk(3, 3).end_line == 3


def test_chunk_rejects_inverted_range() -> None:
    with pytest.raises(ValidationError, match="end_line"):
        _chunk(5, 4)


def test_search_hit_serialises_enums_as_strings() -> None:
    hit = SearchHit(
        path="a.py",
        start_line=1,
        end_line=2,
        symbol=None,
        kind=ChunkKind.MODULE,
        score=0.5,
        snippet="x = 1",
    )

    payload = json.loads(hit.model_dump_json())

    assert payload["kind"] == "module"
    assert payload["injection_flag"] == "none"
    assert payload["injection_reasons"] == []
    assert SearchHit.model_validate(payload).injection_flag is InjectionFlag.NONE
