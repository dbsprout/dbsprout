"""Configuration models for DBSprout.

All models are frozen Pydantic v2 models with ``extra="forbid"`` to
reject unknown TOML keys with clear validation errors.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


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
    seed: int = 42
    engine: str = "heuristic"
    output_format: str = "sql"
    output_dir: str = "./seeds"


class TableOverride(BaseModel):
    """Per-table overrides from ``[tables.<name>]`` sections."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rows: int | None = Field(default=None, ge=1)
    exclude: bool = False


class DBSproutConfig(BaseModel):
    """Root configuration loaded from ``dbsprout.toml``."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    schema_: SchemaConfig = Field(default_factory=SchemaConfig, alias="schema")
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    tables: dict[str, TableOverride] = Field(default_factory=dict)
