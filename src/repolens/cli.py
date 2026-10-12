"""Command-line interface: ``repolens index | query | serve | eval``."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from pydantic import ValidationError
from rich.console import Console, Group
from rich.logging import RichHandler
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from repolens import __version__
from repolens.config import Settings, load_settings
from repolens.embeddings import build_embedder
from repolens.evals.dataset import DatasetError, load_meta, load_questions
from repolens.evals.guard import (
    load_examples,
    markdown_guard_report,
    run_guard_eval,
    write_guard_report,
)
from repolens.evals.run import EvalError, markdown_report, run_eval, select_configs, write_report
from repolens.guard.base import CompositeDetector, InjectionDetector
from repolens.guard.classifier import ClassifierDetector
from repolens.guard.heuristic import HeuristicDetector
from repolens.indexing.indexer import IndexBuildError, build_index
from repolens.indexing.store import IndexLoadError, IndexMeta, load_chunks, load_index
from repolens.models import InjectionFlag, SearchHit, SearchMode, SearchResult
from repolens.retrieval.pipeline import SearchEngine

app = typer.Typer(
    name="repolens",
    help="Index a Git repository and search it with hybrid retrieval.",
    no_args_is_help=True,
    add_completion=False,
)
# Third-party loggers that are chatty at INFO/DEBUG; shown only with --verbose.
_NOISY_LOGGERS = ("bm25s", "sentence_transformers", "transformers", "httpx", "huggingface_hub")
console = Console()
err_console = Console(stderr=True)

ConfigOption = Annotated[
    Path | None,
    typer.Option("--config", help="TOML config file (default: ./repolens.toml if present)."),
]
IndexDirArg = Annotated[
    Path, typer.Argument(help="Index directory created by `repolens index`.", file_okay=False)
]


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"repolens {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
) -> None:
    """Index a Git repository and search it with hybrid retrieval."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=err_console, show_path=False)],
    )
    if not verbose:
        for noisy in _NOISY_LOGGERS:
            logging.getLogger(noisy).setLevel(logging.WARNING)


def _settings(config: Path | None) -> Settings:
    """Load settings, turning config errors into a clean CLI failure."""
    try:
        return load_settings(config)
    except (FileNotFoundError, ValidationError) as exc:
        err_console.print(f"[red]Invalid configuration:[/red] {exc}")
        raise typer.Exit(code=2) from exc


def _fail(message: str, code: int = 2) -> NoReturn:
    err_console.print(f"[red]Error:[/red] {message}")
    raise typer.Exit(code=code)


def _not_implemented(command: str, milestone: int) -> NoReturn:
    err_console.print(
        f"[yellow]`repolens {command}` is not implemented yet "
        f"(planned for milestone {milestone}).[/yellow]"
    )
    raise typer.Exit(code=1)


@app.command()
def index(
    repo_path: Annotated[
        Path,
        typer.Argument(
            help="Repository to index.", exists=True, file_okay=False, resolve_path=True
        ),
    ],
    out: Annotated[Path, typer.Option("--out", help="Directory to write the index to.")],
    config: ConfigOption = None,
) -> None:
    """Chunk a repository and build the BM25 and vector indexes."""
    settings = _settings(config)
    started = time.perf_counter()
    try:
        meta = build_index(repo_path, out, settings, build_embedder(settings.embedding))
    except (IndexBuildError, FileExistsError) as exc:
        _fail(str(exc))
    _print_index_summary(meta, out, time.perf_counter() - started)


@app.command()
def query(
    index_dir: IndexDirArg,
    question: Annotated[str, typer.Argument(help="Natural-language or identifier query.")],
    mode: Annotated[
        SearchMode | None,
        typer.Option(help="Retrieval mode [default: retrieval.default_mode, i.e. dense]."),
    ] = None,
    k: Annotated[int, typer.Option(min=1, help="Number of results.")] = 8,
    rerank: Annotated[
        bool,
        typer.Option("--rerank/--no-rerank", help="Apply the configured reranker, if any."),
    ] = True,
    config: ConfigOption = None,
) -> None:
    """Search an index and pretty-print the results."""
    settings = _settings(config)
    try:
        engine = SearchEngine.open(index_dir, settings)
        result = engine.search(question, k=k, mode=mode, rerank=rerank)
    except (IndexLoadError, ValueError) as exc:
        _fail(str(exc))
    _print_result(result)


@app.command()
def serve(
    index_dir: Annotated[
        Path, typer.Option("--index", help="Index directory to serve.", file_okay=False)
    ],
    config: ConfigOption = None,
) -> None:
    """Run the MCP server over stdio."""
    _settings(config)
    _not_implemented("serve", milestone=7)


@app.command("eval")
def eval_(
    index_dir: Annotated[Path, typer.Option("--index", help="Index to evaluate.", file_okay=False)],
    dataset: Annotated[
        Path, typer.Option("--dataset", help="JSONL benchmark file.", exists=True, dir_okay=False)
    ],
    configs: Annotated[
        str,
        typer.Option(
            help="Comma-separated configs (bm25,dense,hybrid,hybrid+ce,hybrid+llm) or 'all'."
        ),
    ] = "all",
    out: Annotated[
        Path, typer.Option("--out", help="Directory for the JSON report.", file_okay=False)
    ] = Path("evals/results"),
    config: ConfigOption = None,
) -> None:
    """Run the retrieval benchmark and print a markdown results table."""
    settings = _settings(config)
    try:
        selected = select_configs(configs)
        questions = load_questions(dataset)
        index = load_index(index_dir, settings, build_embedder(settings.embedding))
        report = run_eval(index, questions, settings, selected, dataset, load_meta(dataset))
    except (EvalError, DatasetError, IndexLoadError, ValueError) as exc:
        _fail(str(exc))
    for warning in report.warnings:
        err_console.print(f"[yellow]Warning:[/yellow] {warning}")
    err_console.print(f"Report written to {write_report(report, out)}")
    print(markdown_report(report))


@app.command("eval-guard")
def eval_guard(
    dataset: Annotated[
        Path,
        typer.Option("--dataset", help="Labelled injection JSONL.", exists=True, dir_okay=False),
    ] = Path("evals/datasets/injection.jsonl"),
    index_dir: Annotated[
        Path | None,
        typer.Option("--index", help="Also scan this (trusted) index for false positives."),
    ] = None,
    classifier: Annotated[
        bool, typer.Option("--classifier", help="Also evaluate the HF classifier (downloads it).")
    ] = False,
    out: Annotated[
        Path, typer.Option("--out", help="Directory for the JSON report.", file_okay=False)
    ] = Path("evals/results"),
    config: ConfigOption = None,
) -> None:
    """Measure the prompt-injection guard and print a markdown table."""
    settings = _settings(config)
    try:
        examples = load_examples(dataset)
        corpus = load_chunks(index_dir) if index_dir else None
    except (DatasetError, IndexLoadError) as exc:
        _fail(str(exc))
    report = run_guard_eval(
        examples,
        _guard_configs(settings, classifier),
        dataset,
        corpus,
        settings.guard.block_high_severity,
    )
    err_console.print(f"Report written to {write_guard_report(report, out)}")
    print(markdown_guard_report(report))


def _guard_configs(settings: Settings, with_classifier: bool) -> dict[str, InjectionDetector | str]:
    heuristic = HeuristicDetector()
    if not with_classifier:
        reason = "pass --classifier to download and run it"
        return {"heuristic": heuristic, "classifier": reason, "heuristic+classifier": reason}
    guard = settings.guard
    model = ClassifierDetector(
        guard.classifier_model,
        threshold=guard.classifier_threshold,
        label=guard.classifier_label,
        device=guard.classifier_device,
    )
    return {
        "heuristic": heuristic,
        "classifier": model,
        "heuristic+classifier": CompositeDetector([heuristic, model]),
    }


def _print_index_summary(meta: IndexMeta, out: Path, seconds: float) -> None:
    table = Table(show_header=False, box=None)
    table.add_row("Index", str(out))
    table.add_row("Repository", meta.repo_path)
    table.add_row("Commit", meta.commit or "(not a git repository)")
    table.add_row("Files / chunks", f"{meta.file_count} / {meta.chunk_count}")
    table.add_row("Embedding model", f"{meta.embedding_model} (dim {meta.embedding_dim})")
    table.add_row("Took", f"{seconds:.1f}s")
    console.print(Panel(table, title="[green]Index built[/green]", expand=False))


def _print_result(result: SearchResult) -> None:
    if not result.hits:
        console.print(f"No results for [bold]{result.query}[/bold] ({result.mode.value}).")
        return
    reranked = ", reranked" if result.reranked else ""
    console.print(f"[bold]{result.query}[/bold]  ({result.mode.value}{reranked})\n")
    for rank, hit in enumerate(result.hits, start=1):
        console.print(_hit_panel(rank, hit))


def _hit_panel(rank: int, hit: SearchHit) -> Panel:
    title = f"#{rank} [cyan]{hit.path}[/cyan]:{hit.start_line}-{hit.end_line}"
    if hit.symbol:
        title += f"  [bold]{hit.symbol}[/bold]"
    lexer = Syntax.guess_lexer(hit.path, code=hit.snippet)
    body = Syntax(hit.snippet, lexer, line_numbers=True, start_line=hit.start_line)
    subtitle = f"{hit.kind.value} · score {hit.score:.4f}"
    if hit.injection_flag is InjectionFlag.NONE:
        return Panel(
            body, title=title, title_align="left", subtitle=subtitle, subtitle_align="right"
        )
    colour = "red" if hit.injection_flag is InjectionFlag.BLOCKED else "yellow"
    title += f"  [{colour}]⚠ {hit.injection_flag.value}[/{colour}]"
    reasons = Text("\n".join(f"• {r}" for r in hit.injection_reasons), style=colour)
    return Panel(
        Group(reasons, Text(""), body),
        title=title,
        title_align="left",
        subtitle=subtitle,
        subtitle_align="right",
        border_style=colour,
    )
