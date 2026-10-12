"""Runtime configuration.

Settings are resolved from, in decreasing priority:

1. keyword arguments passed to :class:`Settings` (used by tests),
2. environment variables prefixed with ``REPOLENS_`` (nested fields use ``__``,
   e.g. ``REPOLENS_RETRIEVAL__RRF_K=30``),
3. a TOML file (``repolens.toml`` in the working directory, or ``--config``),
4. the defaults declared below.

Anything that would send repository content to a remote service is disabled by
default and must be configured explicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

DEFAULT_CONFIG_FILE = Path("repolens.toml")
ENV_PREFIX = "REPOLENS_"


class _Section(BaseModel):
    """Base for config sections: unknown keys are errors, so typos surface early."""

    model_config = ConfigDict(extra="forbid")


class IndexingSettings(_Section):
    """Which files are indexed."""

    include: list[str] = Field(default_factory=lambda: ["**/*.py"])
    exclude: list[str] = Field(
        default_factory=lambda: [
            ".git/**",
            "**/.venv/**",
            "**/venv/**",
            "**/node_modules/**",
            "**/__pycache__/**",
            "**/build/**",
            "**/dist/**",
        ]
    )
    max_file_bytes: int = Field(default=1_000_000, gt=0)


class ChunkingSettings(_Section):
    """How source files are split into chunks."""

    max_chunk_lines: int = Field(default=200, gt=0)


class EmbeddingSettings(_Section):
    """Dense embedding model. Changing the backend or model requires a re-index.

    The ``hashing`` backend is a dependency-free feature-hashing embedder: useless for
    semantic search, but deterministic and instant, so it is used for tests and offline
    smoke runs.
    """

    backend: Literal["sentence_transformers", "hashing"] = "sentence_transformers"
    model_name: str = "BAAI/bge-small-en-v1.5"
    # bge models expect this instruction on queries (not on documents).
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    batch_size: int = Field(default=32, gt=0)
    device: str = "cpu"
    hashing_dim: int = Field(default=256, gt=0)


class RetrievalSettings(_Section):
    """First-stage retrieval and fusion parameters."""

    candidate_k: int = Field(default=50, gt=0, description="Candidates per retriever.")
    bm25_k1: float = Field(default=1.5, ge=0)
    bm25_b: float = Field(default=0.75, ge=0, le=1)
    rrf_k: int = Field(default=60, gt=0, description="RRF damping constant.")
    hnsw_m: int = Field(default=32, gt=0)
    hnsw_ef_construction: int = Field(default=200, gt=0)
    hnsw_ef_search: int = Field(default=64, gt=0)


class LLMJudgeSettings(_Section):
    """OpenAI-compatible endpoint for the LLM-as-judge reranker (opt-in)."""

    base_url: str | None = None
    model: str | None = None
    api_key: SecretStr | None = None
    timeout_s: float = Field(default=30.0, gt=0)
    price_in_per_mtok: float = Field(default=0.0, ge=0, description="USD per 1M input tokens.")
    price_out_per_mtok: float = Field(default=0.0, ge=0, description="USD per 1M output tokens.")

    @property
    def configured(self) -> bool:
        """True when both an endpoint and a model name are set."""
        return self.base_url is not None and self.model is not None


class RerankSettings(_Section):
    """Second-stage reranking."""

    backend: Literal["cross_encoder", "llm_judge", "none"] = "cross_encoder"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = Field(default=30, gt=0, description="How many fused candidates to rerank.")
    llm_judge: LLMJudgeSettings = Field(default_factory=LLMJudgeSettings)


class GuardSettings(_Section):
    """Prompt-injection detection applied to returned chunks."""

    enabled: bool = True
    classifier_enabled: bool = False
    classifier_model: str = "protectai/deberta-v3-base-prompt-injection-v2"
    classifier_threshold: float = Field(default=0.5, ge=0, le=1)


class ServerSettings(_Section):
    """Limits applied to MCP tool output."""

    snippet_max_chars: int = Field(default=1200, gt=0)
    read_max_lines: int = Field(default=400, gt=0)


class TracingSettings(_Section):
    """Local JSONL tracing; nothing is sent anywhere."""

    enabled: bool = True
    dir_name: str = "traces"


class Settings(BaseSettings):
    """Top-level repoLens configuration."""

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter="__",
        extra="forbid",
        toml_file=DEFAULT_CONFIG_FILE,
    )

    seed: int = 42
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    indexing: IndexingSettings = Field(default_factory=IndexingSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    rerank: RerankSettings = Field(default_factory=RerankSettings)
    guard: GuardSettings = Field(default_factory=GuardSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    tracing: TracingSettings = Field(default_factory=TracingSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Use init kwargs, then env vars, then the TOML file."""
        return (init_settings, env_settings, TomlConfigSettingsSource(settings_cls))


def load_settings(config_file: Path | None = None) -> Settings:
    """Load settings, optionally from an explicit TOML file.

    Without ``config_file`` the default ``repolens.toml`` is read if present.
    An explicit path that does not exist is an error rather than silently ignored.
    """
    if config_file is None:
        return Settings()
    if not config_file.is_file():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    class _FileSettings(Settings):
        model_config = SettingsConfigDict(toml_file=config_file)

    return _FileSettings()
