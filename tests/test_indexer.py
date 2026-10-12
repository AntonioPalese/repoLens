from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from repolens.config import IndexingSettings, Settings
from repolens.embeddings import HashingEmbedder
from repolens.indexing.indexer import (
    IndexBuildError,
    build_index,
    discover_files,
    git_commit,
    glob_to_regex,
    read_source,
)
from repolens.indexing.store import IndexLayout, IndexLoadError, load_index, read_meta

requires_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _write(root: Path, rel: str, content: str = "x = 1\n") -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


@pytest.mark.parametrize(
    ("pattern", "path", "matches"),
    [
        ("**/*.py", "setup.py", True),
        ("**/*.py", "a/b/c.py", True),
        ("**/*.py", "a/b/c.pyc", False),
        ("*.py", "a/b.py", False),
        (".git/**", ".git/config", True),
        ("**/build/**", "pkg/build/x.py", True),
        ("**/build/**", "pkg/builder.py", False),
        ("src/?.py", "src/a.py", True),
        ("src/?.py", "src/ab.py", False),
    ],
)
def test_glob_to_regex(pattern: str, path: str, matches: bool) -> None:
    assert bool(glob_to_regex(pattern).match(path)) is matches


def test_discover_files_applies_include_and_exclude(tmp_path: Path) -> None:
    for rel in ["a.py", "pkg/b.py", ".venv/lib/x.py", "build/y.py", "notes.md"]:
        _write(tmp_path, rel)

    assert discover_files(tmp_path, IndexingSettings()) == ["a.py", "pkg/b.py"]


def test_discover_files_skips_symlinks(tmp_path: Path) -> None:
    repo, outside = tmp_path / "repo", tmp_path / "outside"
    _write(repo, "real.py")
    _write(outside, "secret.py")
    (repo / "link.py").symlink_to(outside / "secret.py")
    (repo / "linkdir").symlink_to(outside, target_is_directory=True)

    assert discover_files(repo, IndexingSettings()) == ["real.py"]


@requires_git
def test_discover_files_respects_gitignore_and_records_commit(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    _write(tmp_path, ".gitignore", "generated.py\n")
    _write(tmp_path, "kept.py")
    _write(tmp_path, "generated.py")
    assert git_commit(tmp_path) is None  # no commits yet

    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")

    assert discover_files(tmp_path, IndexingSettings()) == ["kept.py"]
    commit = git_commit(tmp_path)
    assert commit is not None
    assert len(commit) == 40


def test_read_source_skips_binary_and_oversized(tmp_path: Path) -> None:
    (tmp_path / "bin.py").write_bytes(b"x = 1\0\0")
    (tmp_path / "big.py").write_text("x" * 100)
    (tmp_path / "latin.py").write_bytes(b"s = '\xe9'\n")

    assert read_source(tmp_path / "bin.py", 1000) is None
    assert read_source(tmp_path / "big.py", 10) is None
    assert read_source(tmp_path / "latin.py", 1000) == "s = '�'\n"


def test_build_index_writes_complete_layout(sample_index: Path) -> None:
    layout = IndexLayout(sample_index)
    meta = read_meta(sample_index)

    assert layout.bm25_dir.is_dir()
    assert layout.dense.is_file()
    assert meta.embedding_model == "hashing:256"
    assert meta.embedding_dim == 256
    assert meta.file_count == 5
    assert meta.chunk_count == len(layout.chunks.read_text().splitlines())
    assert not list(sample_index.parent.glob(".index.*"))  # no staging leftovers


def test_load_index_round_trip(sample_index: Path, offline_settings: Settings) -> None:
    loaded = load_index(sample_index, offline_settings, HashingEmbedder())

    assert len(loaded.chunks) == loaded.meta.chunk_count
    assert loaded.bm25.retrieve("parseAuthHeader", 1)[0].chunk.symbol == "parseAuthHeader"


def test_build_index_rejects_repo_without_sources(
    tmp_path: Path, offline_settings: Settings
) -> None:
    _write(tmp_path / "repo", "README.md", "# nothing to index")

    with pytest.raises(IndexBuildError, match="No indexable"):
        build_index(tmp_path / "repo", tmp_path / "out", offline_settings, HashingEmbedder())


def test_rebuild_replaces_index_but_keeps_traces(
    sample_repo: Path, sample_index: Path, offline_settings: Settings
) -> None:
    traces = IndexLayout(sample_index).traces_dir
    traces.mkdir()
    (traces / "t.jsonl").write_text("{}\n")
    first = read_meta(sample_index).created_at

    build_index(sample_repo, sample_index, offline_settings, HashingEmbedder())

    assert read_meta(sample_index).created_at > first
    assert (traces / "t.jsonl").read_text() == "{}\n"
    assert not list(sample_index.parent.glob(".index.*"))


def test_refuses_to_overwrite_unrelated_directory(
    tmp_path: Path, sample_repo: Path, offline_settings: Settings
) -> None:
    out = tmp_path / "out"
    _write(out, "important.txt")

    with pytest.raises(FileExistsError):
        build_index(sample_repo, out, offline_settings, HashingEmbedder())
    assert (out / "important.txt").exists()


def test_load_missing_index(tmp_path: Path, offline_settings: Settings) -> None:
    with pytest.raises(IndexLoadError, match="No index found"):
        load_index(tmp_path / "missing", offline_settings, HashingEmbedder())


def test_load_rejects_different_embedding_model(
    sample_index: Path, offline_settings: Settings
) -> None:
    with pytest.raises(IndexLoadError, match=r"hashing:256.*hashing:64"):
        load_index(sample_index, offline_settings, HashingEmbedder(dim=64))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("format_version", 999, "format"),
        ("tokenizer", "code-v0", "tokenizer"),
        ("chunk_count", 1, "expected 1"),
        ("created_at", "not a date", "Corrupt"),
    ],
)
def test_load_rejects_incompatible_meta(
    sample_index: Path, offline_settings: Settings, field: str, value: object, message: str
) -> None:
    meta_path = IndexLayout(sample_index).meta
    meta = json.loads(meta_path.read_text())
    meta[field] = value
    meta_path.write_text(json.dumps(meta))

    with pytest.raises(IndexLoadError, match=message):
        load_index(sample_index, offline_settings, HashingEmbedder())
