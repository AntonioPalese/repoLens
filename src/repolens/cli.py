"""Command-line interface: ``repolens index | query | serve | eval``."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.logging import RichHandler

from repolens import __version__
from repolens.config import Settings, load_settings
from repolens.models import SearchMode

app = typer.Typer(
    name="repolens",
    help="Index a Git repository and search it with hybrid retrieval.",
    no_args_is_help=True,
    add_completion=False,
)
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


def _settings(config: Path | None) -> Settings:
    """Load settings, turning config errors into a clean CLI failure."""
    try:
        return load_settings(config)
    except (FileNotFoundError, ValidationError) as exc:
        err_console.print(f"[red]Invalid configuration:[/red] {exc}")
        raise typer.Exit(code=2) from exc


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
    _settings(config)
    _not_implemented("index", milestone=3)


@app.command()
def query(
    index_dir: IndexDirArg,
    question: Annotated[str, typer.Argument(help="Natural-language or identifier query.")],
    mode: Annotated[SearchMode, typer.Option(help="Retrieval mode.")] = SearchMode.HYBRID,
    k: Annotated[int, typer.Option(min=1, help="Number of results.")] = 8,
    rerank: Annotated[bool, typer.Option("--rerank/--no-rerank", help="Apply reranker.")] = True,
    config: ConfigOption = None,
) -> None:
    """Search an index and pretty-print the results."""
    _settings(config)
    _not_implemented("query", milestone=3)


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
        str, typer.Option(help="Comma-separated pipeline configs to run, or 'all'.")
    ] = "all",
    config: ConfigOption = None,
) -> None:
    """Run the retrieval benchmark and print a results table."""
    _settings(config)
    _not_implemented("eval", milestone=5)
