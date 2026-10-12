<p align="center">
  <img src="docs/assets/banner.png" alt="repoLens: MCP server for codebase search. Hybrid retrieval, reranking, prompt-injection guard, measured with evals." width="100%">
</p>

# repoLens

An MCP server for codebase exploration. It indexes a local Git repository, answers
code-search questions with hybrid retrieval (BM25 + dense vectors) and reranking,
flags prompt injection hidden in repository content, and ships with an evaluation
harness that measures retrieval quality with real numbers.

> **Status: v0.1 in progress.** Indexing, search and the eval harness work today.
> Still to come: the MCP server (`repolens serve`), the prompt-injection guard,
> request tracing and Docker. This README will get its final form with them.

## Quickstart

```bash
uv sync

# Build an index (Python files; respects .gitignore)
uv run repolens index path/to/repo --out .repolens/myrepo

# Search it
uv run repolens query .repolens/myrepo "where are HTTP retries configured?"
uv run repolens query .repolens/myrepo "Client.send" --mode bm25 --k 3 --no-rerank
```

Models are downloaded from Hugging Face on first use and then cached. Everything
runs locally; the optional LLM-judge reranker is the only component that can call
a remote endpoint, and only when you configure one.

## How it works

1. **Chunking:** tree-sitter splits Python files into functions, methods, class
   headers and module-level code, each with a qualified name (`Client.send`).
2. **Retrieval:** BM25 over code-aware tokens (`HTTPClientError` → `http client
   error`) and dense search (FAISS HNSW over `BAAI/bge-small-en-v1.5` embeddings),
   fused with Reciprocal Rank Fusion.
3. **Reranking (optional):** a local cross-encoder, or an LLM judge on any
   OpenAI-compatible endpoint (vLLM, Ollama, …).

Search modes: `hybrid` (default), `bm25`, `dense`. Configuration lives in
`repolens.toml` (see [`repolens.example.toml`](repolens.example.toml)) or in
`REPOLENS_*` environment variables.

## Evaluation

`repolens eval` runs a hand-written benchmark against a pinned repository and
reports Recall@k, MRR@10, nDCG@10 and latency per pipeline configuration.
The current seed set has 30 questions on [httpx](https://github.com/encode/httpx)
0.28.1: exact symbol lookups, conceptual "where/how" questions and cross-file questions.

| Config | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Mean ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 0.406 | 0.617 | 0.689 | 0.564 | 0.555 | 0.1 | 0.1 |
| dense | 0.556 | 0.700 | 0.828 | 0.733 | 0.721 | 10.0 | 10.6 |
| hybrid | 0.539 | 0.739 | 0.756 | 0.721 | 0.687 | 10.4 | 11.0 |
| hybrid + cross-encoder | 0.378 | 0.622 | 0.756 | 0.540 | 0.576 | 924.2 | 974.1 |
| hybrid + LLM judge | not run yet (no endpoint configured) | | | | | | |

Produced by `repolens eval` on CPU; full report in
[`evals/results/20261012T133306Z.json`](evals/results/20261012T133306Z.json).
What these first numbers say, honestly:

* **Dense retrieval is currently the best configuration.** Fusing in BM25 helps
  Recall@5 but hurts cross-file questions.
* **The default cross-encoder (`ms-marco-MiniLM-L-6-v2`) makes ranking worse** and
  costs ~0.9 s per query on CPU. It was trained on web passages, not code.
* With 30 questions, differences below ~0.05 are within noise.

Reproduce, or add questions: see [`evals/CONTRIBUTING.md`](evals/CONTRIBUTING.md).

```bash
git clone https://github.com/encode/httpx evals/repos/httpx
git -C evals/repos/httpx checkout 26d48e0634e6ee9cdc0533996db289ce4b430177
uv run repolens index evals/repos/httpx --out .repolens/httpx
uv run repolens eval --index .repolens/httpx --dataset evals/datasets/httpx.jsonl
```

## Development

```bash
uv sync
uv run ruff check && uv run ruff format --check
uv run mypy
uv run pytest -m "not slow"   # fast suite: no model downloads
uv run pytest -m slow         # real embedding / reranker models
```

## License

[MIT](LICENSE)
