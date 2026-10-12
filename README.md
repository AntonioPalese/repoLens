# repoLens

An MCP server for codebase exploration: it indexes a local Git repository, answers
code-search questions with hybrid retrieval (BM25 + dense vectors) and reranking,
flags prompt injection hidden in repository content, and ships with an evaluation
harness that measures retrieval quality.

> **Status:** early scaffolding (v0.1 in progress). The commands below exist but are
> not implemented yet. The full README, with eval results, comes in the final milestone.

## Development

```bash
uv sync
uv run ruff check && uv run ruff format --check
uv run mypy
uv run pytest -m "not slow"
```
