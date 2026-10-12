# Contributing benchmark questions

Datasets live in `evals/datasets/<repo>.jsonl` (one question per line), with the
repository and pinned commit in `evals/datasets/<repo>.meta.json`.

## Set up the benchmark repository

```bash
git clone https://github.com/encode/httpx evals/repos/httpx
git -C evals/repos/httpx checkout 26d48e0634e6ee9cdc0533996db289ce4b430177
uv run repolens index evals/repos/httpx --out .repolens/httpx
uv run repolens eval --index .repolens/httpx --dataset evals/datasets/httpx.jsonl
```

`evals/repos/` is git-ignored.

## Add a question

```json
{"id": "q031", "query": "how are redirects limited?", "gold": [{"path": "httpx/_client.py", "symbol": "Client._send_handling_redirects"}], "tags": ["concept"]}
```

* **`id`**: unique, never reused. Don't renumber old questions; results are compared by id.
* **`query`**: write it the way a developer would ask. For `concept` and `cross-file`
  questions, avoid copying identifiers from the code. Otherwise you're testing BM25
  rather than understanding.
* **`gold`**: every location that answers the question, and nothing that merely mentions it.
  * `symbol` is the name qualified *within the module*, exactly as the chunker emits it:
    `Client.send`, `URLPattern.matches`, `Cookies._CookieCompatRequest`.
  * For module-level code (constants, tables), set `"symbol": null` and give
    `start_line` / `end_line`; a hit matches when its lines overlap.
  * When sync and async twins both implement the logic and the question doesn't say
    which one, list both.
* **`tags`**: one of `symbol` (exact lookup), `concept` (where/how, usually one place),
  `cross-file` (the answer spans several files).

`repolens eval` refuses to run if any gold item doesn't resolve to an indexed chunk,
so typos and stale line numbers fail loudly. If you bump the pinned commit, re-check
every line-range gold item.
