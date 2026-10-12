"""Utilities. Contains benign look-alikes of injection phrases (false-positive checks)."""


def ignore_previous(items: list[str], n: int) -> list[str]:
    """Drop the first ``n`` items and ignore previous entries in the list."""
    return items[n:]


# system: this helper only works on POSIX platforms.
def normalise_path(path: str) -> str:
    """Collapse duplicate slashes in a URL path."""
    while "//" in path:
        path = path.replace("//", "/")
    return path


def run_command_help() -> str:
    """Return the usage text shown by the ``minihttp`` command-line tool."""
    return "usage: minihttp [--timeout SECONDS] URL"
