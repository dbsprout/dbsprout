"""Pydantic v2 schemas for the sample extractor."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 - runtime use by Pydantic field annotations
from pathlib import Path  # noqa: TC003 - runtime use by Pydantic field annotations

from pydantic import BaseModel, ConfigDict, Field


class ExtractorConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    db_url: str
    sample_rows: int = Field(gt=0)
    output_dir: Path
    seed: int = 0
    min_per_table: int = Field(default=10, ge=0)
    max_per_table: int | None = None
    quiet: bool = False
    fk_closure_max_iterations: int | None = None


class SampleAllocation(BaseModel):
    model_config = ConfigDict(frozen=True)

    table: str
    row_count: int
    target: int
    floor_clamped: bool
    ceiling_clamped: bool


class TableExtractionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    table: str
    target: int
    sampled: int
    fk_closure_added: int = 0
    parquet_path: Path


class SampleResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    output_dir: Path
    manifest_path: Path
    tables: tuple[TableExtractionResult, ...]
    duration_seconds: float


class SampleManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    manifest_version: int = 1
    dbsprout_version: str
    extracted_at: datetime
    dialect: str
    schema_hash: str
    seed: int
    requested_budget: int
    effective_budget: int
    tables: tuple[TableExtractionResult, ...]
    fk_closure_iterations: int
    fk_unresolved_per_table: dict[str, int]
    duration_seconds: float


class ClosureReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    iterations: int
    additions: dict[str, int]
    unresolved_per_table: dict[str, int]
