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
    """Per-column generation instructions produced by LLM or spec analyzer.

    Each field carries a ``description=`` string that the Studio UI surfaces
    as an inline tooltip (S-123). The descriptions are the *single source of
    truth* — no parallel hand-maintained list. The :func:`field_descriptions`
    helper exposes the same data as a plain ``dict[str, str]`` for templates
    that prefer pre-resolved access over reaching into ``model_fields``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = Field(
        ...,
        description=(
            "Generator provider namespace (e.g. 'mimesis', 'faker', 'builtin', "
            "'numpy'). Determines which library produces values for this column."
        ),
    )
    method: str | None = Field(
        default=None,
        description=(
            "Method name within the provider (e.g. 'email', 'random_int'). "
            "When None the engine falls back to a type-driven default."
        ),
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Keyword arguments passed to the generator method "
            "(e.g. {'min': 0, 'max': 100} for random_int)."
        ),
    )
    distribution: str | None = Field(
        default=None,
        description=(
            "Statistical distribution shaping numeric output "
            "(e.g. 'uniform', 'normal', 'zipf'). Leave None for the provider default."
        ),
    )
    distribution_params: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Distribution parameters (e.g. mean/std for 'normal', "
            "s for 'zipf'). Ignored when distribution is None."
        ),
    )
    min_value: float | None = Field(
        default=None,
        description=(
            "Inclusive minimum value for numeric/temporal columns. "
            "Overrides any method-level default."
        ),
    )
    max_value: float | None = Field(
        default=None,
        description=(
            "Inclusive maximum value for numeric/temporal columns. "
            "Overrides any method-level default."
        ),
    )
    enum_values: list[str] | None = Field(
        default=None,
        description=(
            "Closed set of allowed string values. Required for the "
            "'random_choice' method; ignored otherwise."
        ),
    )
    format_pattern: str | None = Field(
        default=None,
        description=(
            "Regex / format pattern values must match "
            "(e.g. '###-##-####' for SSN). Provider-specific."
        ),
    )
    unique: bool = Field(
        default=False,
        description=(
            "When true, the engine refuses duplicate values within the column. "
            "Costs extra memory; required for PK / unique-constrained columns."
        ),
    )
    nullable_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of rows that should be NULL (0.0 to 1.0). Only valid for "
            "nullable columns; defaults to 0.0 (never NULL)."
        ),
    )
    vectorized: bool = Field(
        default=False,
        description=(
            "When true, prefer the NumPy-vectorized engine for this column. "
            "Falls back automatically when the method has no vectorized impl."
        ),
    )


def field_descriptions() -> dict[str, str]:
    """Return ``{field_name: description}`` for every ``GeneratorConfig`` field.

    The mapping is a fresh dict copy on each call — callers may mutate freely
    without affecting the underlying Pydantic ``FieldInfo`` objects. Used by
    the Studio templates to render inline tooltips next to each config field.
    """
    return {name: info.description or "" for name, info in GeneratorConfig.model_fields.items()}


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
