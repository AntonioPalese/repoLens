"""Code-aware tokenizer for BM25.

Identifiers are emitted twice: once whole (so an exact name like
``ignore_previous`` matches strongly) and once split into their parts
(``snake_case``, ``camelCase``, ``HTTPClientError`` -> ``http client error``),
so natural-language queries reach code. Everything is case-folded. The same
tokenizer must be applied to documents and queries.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# Bump whenever tokenization output changes: persisted BM25 indexes depend on it.
TOKENIZER_VERSION = "code-v1"

# A word: starts with a Unicode letter or underscore, continues with word chars.
_WORD_RE = re.compile(r"[^\W\d]\w*")
# Sub-words inside an identifier: acronyms before a capitalised word (HTTP|Client),
# capitalised or lower-case runs, remaining acronyms; trailing digits stick to the word
# (HTTP2, utf8, sha256). Underscores are separators because they never match.
_PART_RE = re.compile(r"[A-Z]+\d*(?=[A-Z][a-z])|[A-Z]?[a-z]+\d*|[A-Z]+\d*|\d+")

DEFAULT_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for", "from",
        "how", "if", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
        "what", "when", "where", "which", "with",
    }
)  # fmt: skip


class CodeTokenizer:
    """Splits text into lower-case word and identifier-part tokens."""

    def __init__(self, min_length: int = 2, stopwords: Iterable[str] = DEFAULT_STOPWORDS) -> None:
        self._min_length = min_length
        self._stopwords = frozenset(word.casefold() for word in stopwords)

    def tokenize(self, text: str) -> list[str]:
        """Return tokens in order of appearance, duplicates kept (BM25 needs term counts)."""
        tokens: list[str] = []
        for word in _WORD_RE.findall(text):
            parts = split_identifier(word)
            candidates = [word] if parts == [word] else [word, *parts]
            tokens.extend(self._normalise(candidates))
        return tokens

    def _normalise(self, candidates: list[str]) -> list[str]:
        folded = (c.casefold() for c in candidates)
        return [
            t
            for t in folded
            if len(t) >= self._min_length
            and t not in self._stopwords
            and any(ch.isalnum() for ch in t)
        ]


def split_identifier(identifier: str) -> list[str]:
    """Split an identifier into its snake_case / camelCase parts, preserving case.

    >>> split_identifier("getHTTP2Response_now")
    ['get', 'HTTP2', 'Response', 'now']
    """
    parts: list[str] = []
    for piece in identifier.split("_"):
        if not piece:
            continue
        found = _PART_RE.findall(piece)
        # Non-ASCII identifiers do not match the ASCII-only case rules: keep them whole.
        parts.extend(found if "".join(found) == piece else [piece])
    return parts
