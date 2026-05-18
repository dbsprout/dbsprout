"""Pydantic v2 schemas for the sample extractor."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 - runtime use by Pydantic field annotations
from enum import Enum
from pathlib import Path  # noqa: TC003 - runtime use by Pydantic field annotations

from pydantic import BaseModel, ConfigDict, Field


class ExtractorConfig(BaseModel):
    """User-supplied configuration for one ``SampleExtractor.extract`` call."""

    model_config = ConfigDict(frozen=True)

    sample_rows: int = Field(gt=0)
    output_dir: Path
    seed: int = 0
    min_per_table: int = Field(default=10, ge=0)
    max_per_table: int | None = None
    quiet: bool = False
    fk_closure_max_iterations: int | None = None


class SampleAllocation(BaseModel):
    """Per-table extraction target derived by the stratified allocator.

    ``floor_clamped`` / ``ceiling_clamped`` reflect only the user-configured
    ``min_per_table`` / ``max_per_table`` bounds — not the implicit per-table
    row-count cap.
    """

    model_config = ConfigDict(frozen=True)

    table: str
    row_count: int
    target: int
    floor_clamped: bool
    ceiling_clamped: bool


class TableExtractionResult(BaseModel):
    """Outcome of extracting one table.

    ``sampled`` counts rows obtained by the random sampler; ``fk_closure_added``
    counts rows the FK closure pass appended afterwards. The Parquet file
    contains both (sampled + closure-added).
    """

    model_config = ConfigDict(frozen=True)

    table: str
    target: int = Field(ge=0)
    sampled: int = Field(ge=0)
    fk_closure_added: int = Field(default=0, ge=0)
    parquet_path: Path


class SampleResult(BaseModel):
    """Aggregate return value of ``SampleExtractor.extract``."""

    model_config = ConfigDict(frozen=True)

    output_dir: Path
    manifest_path: Path
    tables: tuple[TableExtractionResult, ...]
    duration_seconds: float


class SampleManifest(BaseModel):
    """On-disk metadata written next to the per-table Parquet files.

    ``effective_budget`` is the actual total row count written
    (``sampled + fk_closure_added`` summed across tables) — typically larger
    than ``requested_budget`` once FK closure pulls in missing parent rows.
    ``fk_unresolved_total`` is a convenience sum over ``fk_unresolved_per_table``.
    """

    model_config = ConfigDict(frozen=True)

    manifest_version: int = 1
    dbsprout_version: str
    extracted_at: datetime
    dialect: str
    schema_hash: str
    seed: int
    requested_budget: int = Field(ge=0)
    effective_budget: int = Field(ge=0)
    tables: tuple[TableExtractionResult, ...]
    fk_closure_iterations: int = Field(ge=0)
    fk_unresolved_per_table: dict[str, int]
    fk_unresolved_total: int = Field(default=0, ge=0)
    duration_seconds: float = Field(ge=0)


class ClosureReport(BaseModel):
    """Internal return value of ``close_fk_graph``."""

    model_config = ConfigDict(frozen=True)

    iterations: int
    additions: dict[str, int]
    unresolved_per_table: dict[str, int]


class NullPolicy(str, Enum):
    """How NULL cells are rendered in a GReaT-style serialized row.

    ``SKIP`` omits the ``column is value`` clause entirely (the GReaT paper's
    canonical behaviour — absent attributes are simply not mentioned).
    ``LITERAL`` renders the clause as ``column is NULL``.
    """

    SKIP = "skip"
    LITERAL = "literal"


class SerializerConfig(BaseModel):
    """User-supplied configuration for one ``DataPreparer.prepare`` call."""

    model_config = ConfigDict(frozen=True)

    input_dir: Path
    output_path: Path
    seed: int = 0
    null_policy: NullPolicy = NullPolicy.SKIP
    quiet: bool = False


class TableSerializationResult(BaseModel):
    """Per-table outcome of GReaT serialization.

    ``rows_serialized`` counts emitted JSONL lines for the table;
    ``nulls_skipped`` counts NULL cells dropped under ``NullPolicy.SKIP``.
    """

    model_config = ConfigDict(frozen=True)

    table: str
    rows_serialized: int = Field(ge=0)
    nulls_skipped: int = Field(default=0, ge=0)


class SerializationResult(BaseModel):
    """Aggregate return value of ``DataPreparer.prepare``."""

    model_config = ConfigDict(frozen=True)

    output_path: Path
    tables: tuple[TableSerializationResult, ...]
    total_rows: int = Field(ge=0)
    duration_seconds: float = Field(ge=0)
