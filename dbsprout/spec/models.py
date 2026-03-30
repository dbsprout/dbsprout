"""Spec generation models — generator mappings, DataSpec, and related types.

``GeneratorMapping`` is the Sprint 2 heuristic output (column → generator).
``GeneratorConfig`` through ``DataSpec`` are the richer LLM/spec output used
by the spec-driven generation engine (Sprint 3+).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class GeneratorMapping(BaseModel):
    """Maps a column to a specific data generator with confidence scoring.

    Used by the heuristic engine (S-012/S-013). For the richer spec-driven
    engine, see ``GeneratorConfig``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    generator_name: str
    provider: str  # "mimesis", "faker", "numpy", "builtin"
    confidence: float = Field(ge=0.0, le=1.0)
    params: dict[str, Any] = Field(default_factory=dict)


# ── DataSpec models (Sprint 3+) ─────────────────────────────────────


class GeneratorConfig(BaseModel):
    """Per-column generation instructions produced by LLM or spec analyzer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str
    method: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    distribution: str | None = None
    distribution_params: dict[str, float] = Field(default_factory=dict)
    min_value: float | None = None
    max_value: float | None = None
    enum_values: list[str] | None = None
    format_pattern: str | None = None
    unique: bool = False
    nullable_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    vectorized: bool = False


class DerivedColumn(BaseModel):
    """Expression-based column derived from other columns."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    column: str
    expression: str
    depends_on: list[str]


class CorrelationRule(BaseModel):
    """Multi-column coherence rule (e.g., city/state/zip lookup)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    columns: list[str]
    lookup_table: str | None = None
    strategy: str = "lookup"


class TableSpec(BaseModel):
    """Per-table generation spec with column configs and rules."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    table_name: str
    row_count: int = Field(default=100, ge=1)
    columns: dict[str, GeneratorConfig]
    derived: list[DerivedColumn] = Field(default_factory=list)
    correlations: list[CorrelationRule] = Field(default_factory=list)
    cardinality: dict[str, Any] | None = None


class DataSpec(BaseModel):
    """Top-level spec: the contract between spec generation and data generation.

    Produced once by LLM or heuristics, cached as JSON, and interpreted
    by the spec-driven generation engine.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = "1.0"
    tables: list[TableSpec]
    global_seed: int = Field(default=42, ge=0)
    schema_hash: str = ""
    model_used: str | None = None
    created_at: str | None = None

    def get_table_spec(self, name: str) -> TableSpec | None:
        """Find a table spec by name, or ``None`` if not found."""
        for ts in self.tables:
            if ts.table_name == name:
                return ts
        return None
