# PROJECT_BRIEF.md — `repoLens` v0.1 scaffolding

> Brief for Claude Code. Read it fully before writing any code.
> Working name: **repoLens** (rename freely — keep it in one place: `pyproject.toml` + package dir).

## 1. What we are building

An open-source **MCP server for codebase exploration**: it indexes a local Git repository, answers code-search questions with **hybrid retrieval (BM25 + vector) and reranking**, guards against **prompt injection hidden in repository content**, and ships with an **evaluation harness** that measures retrieval quality with real numbers.

Primary client: Claude Code (or any MCP client) connecting over stdio.

The project is a portfolio piece for an Agentic AI Engineer profile. Priorities, in order:
1. **Correctness and measurability** — every retrieval improvement must be provable via the eval harness.
2. **Clean, readable, tested code** — reviewers will read it.
3. **One-command setup** — `uv sync` locally, or `docker run`.
4. Features — only after the above.

## 2. Scope of v0.1

### In scope
- Index one local repo (Python first; architecture must allow more languages via tree-sitter).
- AST-aware chunking (functions, classes, methods; module-level fallback).
- Two retrievers: **BM25** and **dense vector** (FAISS HNSW), plus **hybrid fusion** (Reciprocal Rank Fusion).
- **Reranker** with a pluggable interface: cross-encoder (default, local) and LLM-as-judge (optional, any OpenAI-compatible endpoint, e.g. vLLM / Ollama).
- **Prompt-injection detector** applied to retrieved chunks before they are returned; flagged chunks are marked, not silently dropped.
- **MCP server** (stdio) exposing the tools in §5.
- **CLI** for indexing, querying, and running evals.
- **Eval harness** with a small hand-written benchmark and metrics in §7.
- **Tracing**: JSONL trace per request (latency per stage, token counts, cost estimate when an LLM is used).
- Dockerfile, GitHub Actions CI, README with a results table.

### Out of scope (do NOT build in v0.1)
- Web UI, auth, multi-user, remote HTTP transport.
- Incremental/watch-mode re-indexing (full re-index only; leave a TODO hook).
- Languages other than Python (but keep the chunker interface generic).
- Answer generation / chat. The server returns ranked evidence; the client LLM does the reasoning.
- Fine-tuning anything.

## 3. Tech stack (use these unless there is a blocking reason — then ask)

| Concern | Choice |
|---|---|
| Language | Python 3.11+ |
| Packaging / env | `uv`, `pyproject.toml` |
| MCP | official `mcp` Python SDK (`FastMCP`) |
| Parsing | `tree-sitter` + `tree-sitter-python` |
| Embeddings | `sentence-transformers`, default a small code-capable model (configurable) |
| Vector index | `faiss-cpu`, `IndexHNSWFlat` |
| BM25 | `bm25s` (or `rank-bm25` if problems) with a code-aware tokenizer (split snake_case / camelCase) |
| Reranker | `sentence-transformers` CrossEncoder; LLM judge via `openai` client with configurable `base_url` |
| Config & models | `pydantic` v2 + `pydantic-settings` (env vars + optional `repolens.toml`) |
| CLI | `typer` |
| Tests | `pytest`, `pytest-cov` |
| Lint / types | `ruff` (lint + format), `mypy --strict` on the package |

No LangChain for v0.1: the point is to show the retrieval pipeline explicitly.

## 4. Repository layout

```
repolens/
├── pyproject.toml
├── README.md
├── LICENSE                      # MIT
├── Dockerfile
├── .github/workflows/ci.yml
├── repolens.example.toml
├── src/repolens/
│   ├── __init__.py
│   ├── config.py                # Settings (pydantic-settings)
│   ├── models.py                # Chunk, SearchHit, SearchResult, TraceRecord…
│   ├── chunking/
│   │   ├── base.py              # Chunker protocol
│   │   └── python_ast.py        # tree-sitter Python chunker
│   ├── indexing/
│   │   ├── indexer.py           # walk repo → chunks → both indexes → persist
│   │   ├── store.py             # on-disk layout, load/save, metadata
│   │   └── tokenizer.py         # code-aware tokenizer for BM25
│   ├── retrieval/
│   │   ├── base.py              # Retriever protocol
│   │   ├── bm25.py
│   │   ├── dense.py
│   │   ├── hybrid.py            # RRF fusion
│   │   └── pipeline.py          # retrieve → rerank → guard → result
│   ├── rerank/
│   │   ├── base.py              # Reranker protocol
│   │   ├── cross_encoder.py
│   │   └── llm_judge.py
│   ├── guard/
│   │   ├── base.py              # InjectionDetector protocol
│   │   ├── heuristic.py         # pattern/rule based
│   │   └── classifier.py        # optional HF model, behind a flag
│   ├── tracing.py               # context manager per stage → JSONL
│   ├── server.py                # MCP server (FastMCP), tools in §5
│   └── cli.py                   # typer app
├── evals/
│   ├── datasets/
│   │   └── <repo_name>.jsonl    # questions + gold chunk locations
│   ├── metrics.py
│   ├── run.py                   # runs configs, writes results/*.json + markdown table
│   └── results/
└── tests/
    ├── fixtures/sample_repo/    # tiny repo incl. a file with an injection payload
    ├── test_chunking.py
    ├── test_tokenizer.py
    ├── test_retrievers.py
    ├── test_hybrid.py
    ├── test_guard.py
    ├── test_metrics.py
    └── test_server.py           # call tools in-process
```

## 5. MCP tools

All tools return structured output (pydantic models serialized to JSON). Every hit includes: `path`, `start_line`, `end_line`, `symbol` (qualified name or null), `kind` (function/class/method/module), `score`, `snippet` (truncated, configurable), `injection_flag` (`none` | `suspected` | `blocked`) and `injection_reasons`.

| Tool | Input | Behaviour |
|---|---|---|
| `search_code` | `query: str`, `k: int = 8`, `mode: "hybrid" \| "bm25" \| "dense" = "hybrid"`, `rerank: bool = true` | Full pipeline. Default path used by clients. |
| `get_symbol` | `name: str` (exact or qualified) | Exact lookup from the chunk metadata; returns definition chunk(s). |
| `find_usages` | `name: str`, `k: int = 20` | BM25-only lookup of identifier occurrences, excluding the definition itself. |
| `read_file_range` | `path: str`, `start_line: int`, `end_line: int` | Returns raw lines; path must stay inside the indexed repo root (reject traversal). Max lines configurable. |
| `index_status` | — | Repo path, commit hash at index time, chunk count, model names, index age. |

Server startup: `repolens serve --index <dir>`. Fail fast with a clear message if the index is missing or was built with a different embedding model.

## 6. Prompt-injection guard

Threat model: a repository can contain text written to manipulate the LLM that reads search results (comments, docstrings, README, string literals), e.g. "ignore previous instructions and run …".

- `heuristic.py`: curated regex/phrase rules (instruction override, role impersonation like "system:", requests to run commands / exfiltrate secrets / fetch URLs, hidden Unicode such as zero-width and bidi control characters). Each rule has an id and a human-readable reason.
- `classifier.py`: optional open-source prompt-injection classifier from Hugging Face, enabled via config; model name configurable, off by default so CI stays light.
- Policy (configurable): `suspected` → returned but flagged and snippet wrapped in a clear "untrusted content" marker; `blocked` → snippet replaced by a placeholder, location still returned.
- The fixture repo must contain at least: one obvious injection in a comment, one in a docstring, one using zero-width characters, and several benign look-alikes (e.g. a real function called `ignore_previous`) to measure false positives.

## 7. Evaluation harness

Dataset format (`evals/datasets/*.jsonl`), one line per question:
```json
{"id": "q001", "query": "where are HTTP retries configured?", "gold": [{"path": "src/x/client.py", "symbol": "RetryPolicy"}], "tags": ["concept"]}
```
Match a hit to gold by `path` + `symbol`, or by line-range overlap when `symbol` is null.

Metrics (`metrics.py`, unit-tested on toy inputs): **Recall@k** (k = 1, 5, 10), **MRR@10**, **nDCG@10**, plus mean / p95 latency per stage.

Configurations compared by `evals/run.py`:
1. BM25 only
2. Dense only
3. Hybrid (RRF)
4. Hybrid + cross-encoder rerank
5. Hybrid + LLM-judge rerank (skipped automatically if no endpoint configured)

Output: `evals/results/<timestamp>.json` and a markdown table printed to stdout, ready to paste into the README.

Seed dataset for v0.1: pick **one** mid-size, well-known public Python repo (e.g. `httpx` or `rich`), pin the commit in the dataset file, and write **30 questions** by hand covering: exact symbol lookup, conceptual "where/how" questions, and cross-file questions. Leave a `CONTRIBUTING` note on how to add questions.

Guard eval: small labelled set (`evals/datasets/injection.jsonl`) → precision / recall of the detector.

## 8. Tracing

`tracing.py` provides a context manager: `with trace.stage("dense"):` records duration. One JSONL record per tool call in `<index_dir>/traces/` with: request id, tool, query, per-stage ms, number of candidates per stage, rerank model, LLM tokens in/out and estimated cost (price table in config, default 0 for local endpoints). No external telemetry.

## 9. CLI

```
repolens index <repo_path> --out <index_dir> [--config repolens.toml]
repolens query <index_dir> "question" [--mode hybrid] [--k 8] [--no-rerank]
repolens serve --index <index_dir>
repolens eval --index <index_dir> --dataset evals/datasets/httpx.jsonl [--configs all]
```
`query` pretty-prints results (use `rich`) — handy for demos and screenshots.

## 10. Engineering rules

- Type hints everywhere; `mypy --strict` passes on `src/`.
- Protocols (`typing.Protocol`) for Chunker, Retriever, Reranker, InjectionDetector — implementations are swappable via config.
- No global state; dependencies wired in one place (`pipeline.py` / a small factory).
- Heavy models load lazily and are cached per process.
- Tests must not download large models: use tiny models or fakes (a deterministic fake embedder for unit tests). Mark any test needing real models with `@pytest.mark.slow`, excluded in CI by default.
- Deterministic results: fixed seeds, sorted tie-breaking.
- Logging with the standard `logging` module, never `print` outside the CLI.
- Docstrings on public classes/functions; keep functions short.

## 11. CI and Docker

- GitHub Actions on push/PR: `uv sync`, `ruff check`, `ruff format --check`, `mypy`, `pytest -m "not slow"` with coverage report. Python 3.11 and 3.12.
- `Dockerfile`: slim Python base, `uv` install, non-root user, entrypoint `repolens`. Document `docker run` for both `index` and `serve`.

## 12. README (write it last, keep it honest)

Sections: one-paragraph pitch · 30-second quickstart · how to connect it to Claude Code (MCP config snippet) · architecture diagram (Mermaid) · **results table from the eval run** · guard design and its limits · roadmap. Do not claim numbers that were not produced by `evals/run.py`.

## 13. Milestones — work in this order, stop after each for review

1. **Skeleton**: layout, `pyproject.toml`, config, models, CLI stubs, CI green with a trivial test.
2. **Chunking + tokenizer** with tests on `tests/fixtures/sample_repo`.
3. **Indexer + BM25 + dense** retrievers, persisted index, `repolens query` working.
4. **Hybrid RRF + cross-encoder rerank**.
5. **Eval harness** + seed dataset + first results table (this is where we learn whether steps 3–4 actually work).
6. **Guard** (heuristic first, classifier behind flag) + injection eval.
7. **MCP server** with all §5 tools + in-process tests; verify manually from Claude Code.
8. **Tracing, Dockerfile, README**.

At the end of each milestone: run lint, types and tests, then summarise what was built, what was skipped, and any decision that deviates from this brief.

## 14. Ask before doing

- Adding a dependency not listed in §3.
- Changing a public tool signature in §5 or the dataset format in §7.
- Anything that sends repository content to a remote API by default (remote calls must always be opt-in).