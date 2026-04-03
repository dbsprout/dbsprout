"""Configuration models for DBSprout.

All models are frozen Pydantic v2 models with ``extra="forbid"`` to
reject unknown TOML keys with clear validation errors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from pathlib import Path


class SchemaConfig(BaseModel):
    """Schema input settings from ``[schema]`` section."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dialect: str | None = None
    source: str | None = None
    snapshot: str | None = None


class GenerationConfig(BaseModel):
    """Generation settings from ``[generation]`` section."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    default_rows: int = Field(default=100, ge=1)
    seed: int = Field(default=42, ge=0)
    engine: Literal["heuristic", "spec", "statistical", "finetuned"] = "heuristic"
    output_format: Literal["sql", "csv", "json", "jsonl", "parquet"] = "sql"
    output_dir: str = "./seeds"


class TableOverride(BaseModel):
    """Per-table overrides from ``[tables.<name>]`` sections."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rows: int | None = Field(default=None, ge=1)
    exclude: bool = False


class PrivacyConfig(BaseModel):
    """Privacy settings from ``[privacy]`` section."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tier: Literal["local", "redacted", "cloud"] = "local"


class DBSproutConfig(BaseModel):
    """Root configuration loaded from ``dbsprout.toml``."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    schema_: SchemaConfig = Field(default_factory=SchemaConfig, alias="schema")
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    tables: dict[str, TableOverride] = Field(default_factory=dict)

    @classmethod
    def from_toml(cls, path: Path | None = None) -> DBSproutConfig:
        """Load config from a TOML file. Returns defaults if file is missing."""
        from dbsprout.config.loader import load_config  # noqa: PLC0415

        return load_config(path)
