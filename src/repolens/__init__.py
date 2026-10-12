"""repoLens: an MCP server for codebase exploration with hybrid retrieval."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("repolens")
except PackageNotFoundError:  # pragma: no cover - only when running from an uninstalled tree
    __version__ = "0.0.0"

__all__ = ["__version__"]
