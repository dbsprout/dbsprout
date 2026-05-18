"""State-layer Pydantic models.

The CLI writes generation-run telemetry to ``.dbsprout/state.db``; visual
surfaces (HTML report, TUI, web dashboard) read from it. These frozen
Pydantic v2 models are the contract for that state layer.

``id`` / ``run_id`` are assigned by SQLite (autoincrement primary keys /
foreign keys) and are therefore optional on inbound records.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 - runtime use by Pydantic fields

from pydantic import BaseModel, ConfigDict, Field


class TableStats(BaseModel):
    """Per-table generation statistics for a single run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: int | None = None
    run_id: int | None = None
    table_name: str
    row_count: int = 0
    generation_ms: int = 0
    rows_per_sec: float = 0.0
    errors: int = 0


class QualityResult(BaseModel):
    """A single quality-metric result for a run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: int | None = None
    run_id: int | None = None
    metric_type: str
    metric_name: str
    score: float = 0.0
    passed: bool = False
    details_json: str | None = None


class LLMCall(BaseModel):
    """A single LLM invocation recorded during a run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: int | None = None
    run_id: int | None = None
    timestamp: datetime
    provider: str
    model: str
    tokens_sent: int = 0
    tokens_received: int = 0
    cost_usd: float = 0.0
    privacy_tier: str | None = None
    cached: bool = False


class RunRecord(BaseModel):
    """A complete generation run plus its associated child records."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: int | None = None
    started_at: datetime
    completed_at: datetime | None = None
    duration_ms: int | None = None
    engine: str
    llm_provider: str | None = None
    llm_model: str | None = None
    total_rows: int = 0
    total_tables: int = 0
    seed: int | None = None
    config_json: str | None = None
    table_stats: list[TableStats] = Field(default_factory=list)
    quality_results: list[QualityResult] = Field(default_factory=list)
    llm_calls: list[LLMCall] = Field(default_factory=list)
