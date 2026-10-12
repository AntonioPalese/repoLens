from __future__ import annotations

import pytest

from repolens.indexing.tokenizer import CodeTokenizer, split_identifier


@pytest.fixture
def tokenizer() -> CodeTokenizer:
    return CodeTokenizer()


@pytest.mark.parametrize(
    ("identifier", "parts"),
    [
        ("snake_case_name", ["snake", "case", "name"]),
        ("camelCaseName", ["camel", "Case", "Name"]),
        ("HTTPClientError", ["HTTP", "Client", "Error"]),
        ("HTTP2Connection", ["HTTP2", "Connection"]),
        ("getHTTP2Response_now", ["get", "HTTP2", "Response", "now"]),
        ("utf8_decode", ["utf8", "decode"]),
        ("__init__", ["init"]),
        ("CONSTANT_VALUE", ["CONSTANT", "VALUE"]),
        ("x", ["x"]),
        ("_", []),
        ("größe_berechnen", ["größe", "berechnen"]),
    ],
)
def test_split_identifier(identifier: str, parts: list[str]) -> None:
    assert split_identifier(identifier) == parts


def test_emits_whole_identifier_and_parts(tokenizer: CodeTokenizer) -> None:
    assert tokenizer.tokenize("RetryPolicy") == ["retrypolicy", "retry", "policy"]
    assert tokenizer.tokenize("ignore_previous") == ["ignore_previous", "ignore", "previous"]


def test_simple_words_are_not_duplicated(tokenizer: CodeTokenizer) -> None:
    assert tokenizer.tokenize("Client timeout") == ["client", "timeout"]


def test_dunder_names_also_yield_bare_name(tokenizer: CodeTokenizer) -> None:
    assert tokenizer.tokenize("def __init__(self)") == ["def", "__init__", "init", "self"]


def test_punctuation_numbers_and_operators_are_dropped(tokenizer: CodeTokenizer) -> None:
    assert tokenizer.tokenize("x = foo(1, 2.5) + bar[0] # 404 !!") == ["foo", "bar"]


def test_stopwords_and_short_tokens_are_dropped(tokenizer: CodeTokenizer) -> None:
    assert tokenizer.tokenize("Where are the HTTP retries configured?") == [
        "http",
        "retries",
        "configured",
    ]


def test_duplicates_are_kept_for_term_frequency(tokenizer: CodeTokenizer) -> None:
    assert tokenizer.tokenize("retry retry") == ["retry", "retry"]


def test_custom_settings() -> None:
    tokenizer = CodeTokenizer(min_length=1, stopwords=["Foo"])

    assert tokenizer.tokenize("foo x FOO bar") == ["x", "bar"]


def test_query_reaches_code_tokens(tokenizer: CodeTokenizer) -> None:
    doc = set(tokenizer.tokenize("def build_auth_header(username): ..."))
    query = tokenizer.tokenize("how is the auth header built")

    assert {"auth", "header"} <= doc & set(query)


def test_zero_width_characters_split_words(tokenizer: CodeTokenizer) -> None:
    assert tokenizer.tokenize("ig\u200bnore") == ["ig", "nore"]
