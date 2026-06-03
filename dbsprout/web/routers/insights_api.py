"""JSON state-telemetry API for the React Workbench (P1c-4).

Three read-only JSON endpoints surface the SQLite state layer
(``.dbsprout/state.db``) to the SPA's "Runs & Quality" panels:

* ``GET /api/runs?page=`` — paginated run history (wraps
  :func:`dbsprout.web.views.insights.paginate_runs`).
* ``GET /api/quality?run_id=`` — per-metric pass/fail/warn table for one run
  (wraps :func:`dbsprout.report.quality_table.build_quality_table`); the latest
  run when ``run_id`` is omitted.
* ``GET /api/costs`` — LLM cost summary (wraps
  :func:`dbsprout.web.views.insights.build_cost_summary`).

These are the JSON twins of the HTML ``views/insights.py`` views, which stay
until the P1c-5 cutover removes them. The endpoints **wrap** the existing pure
builders — they never re-derive the aggregations.

State-data policy
-----------------
The state DB stores *telemetry only* (runs, table_stats, quality_results,
llm_calls), never generated rows, and never secrets — ``config_json`` and any
connection target are deliberately excluded from every response model. When the
CLI has never run, every endpoint returns an honest empty payload with ``200``
(never ``500``): ``/api/runs`` → zero rows, ``/api/quality`` → ``found=false``,
``/api/costs`` → zero totals.

The router accesses the request-scoped :class:`~dbsprout.state.db.StateDB` via
``app.state.get_state_db`` (the same accessor ``views/insights.py`` uses), so a
fresh WAL connection is opened per request and the ``dbsprout serve``
lazy-import contract is preserved (no eager generation imports).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, ConfigDict

from dbsprout.report.quality_table import build_quality_table
from dbsprout.web.views.insights import (
    RUNS_PER_PAGE,
    build_cost_summary,
    paginate_runs,
)

if TYPE_CHECKING:
    from dbsprout.state.db import StateDB
    from dbsprout.state.models import RunRecord

insights_api_router = APIRouter()


# ── response models ──────────────────────────────────────────────────────────


class RunRow(BaseModel):
    """One row of run history. Telemetry only — no ``config_json``/secrets."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: int | None
    started_at: str
    engine: str
    provider: str | None
    total_rows: int
    total_tables: int
    duration_ms: int | None
    cost: float


class RunsResponse(BaseModel):
    """Paginated run history with navigation metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rows: list[RunRow]
    page: int
    total_pages: int
    total_runs: int
    has_prev: bool
    has_next: bool


class QualityRow(BaseModel):
    """One classified quality metric (``status`` ∈ pass/fail/warn)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric_type: str
    metric_name: str
    score: float
    passed: bool
    status: str
    details_json: str | None


class QualityResponse(BaseModel):
    """Quality table for the selected (or latest) run.

    ``found`` is ``False`` (with empty ``rows`` and ``run_id=None``) when no run
    exists or the requested ``run_id`` is unknown — an honest empty-state.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    found: bool
    run_id: int | None
    rows: list[QualityRow]


class ProviderCost(BaseModel):
    """Per-provider LLM cost rollup."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str
    cost: float
    tokens: int
    calls: int


class CostsResponse(BaseModel):
    """LLM cost summary (totals + per-provider). No Plotly spec — the SPA charts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_cost: float
    total_tokens: int
    total_calls: int
    avg_cost_per_run: float
    per_provider: list[ProviderCost]


# ── shared accessors (mirror dbsprout.web.views.insights) ────────────────────


def _state_db(request: Request) -> StateDB:
    factory = cast("Any", request.app.state.get_state_db)
    return cast("StateDB", factory())


# ── pure row builder ─────────────────────────────────────────────────────────


def build_run_row(run: RunRecord) -> RunRow:
    """Shape a :class:`RunRecord` into a :class:`RunRow` (cost summed from calls)."""
    cost = round(sum(call.cost_usd for call in run.llm_calls), 6)
    return RunRow(
        id=run.id,
        started_at=run.started_at.isoformat(),
        engine=run.engine,
        provider=run.llm_provider,
        total_rows=run.total_rows,
        total_tables=run.total_tables,
        duration_ms=run.duration_ms,
        cost=cost,
    )


# ── routes ───────────────────────────────────────────────────────────────────


@insights_api_router.get("/api/runs")
async def get_runs(request: Request, page: int = Query(default=1)) -> RunsResponse:
    """Paginated run history (10/page). Out-of-range ``page`` clamps to 1."""
    runs = _state_db(request).get_runs()
    pagination = paginate_runs(runs, page=page, per_page=RUNS_PER_PAGE)
    rows = [build_run_row(run) for run in pagination["runs"]]
    return RunsResponse(
        rows=rows,
        page=pagination["page"],
        total_pages=pagination["total_pages"],
        total_runs=pagination["total_runs"],
        has_prev=pagination["has_prev"],
        has_next=pagination["has_next"],
    )


@insights_api_router.get("/api/quality")
async def get_quality(
    request: Request,
    run_id: int | None = Query(default=None),
) -> QualityResponse:
    """Quality table for *run_id* (latest when omitted). Honest empty-state."""
    runs = _state_db(request).get_runs()
    if run_id is None:
        run = runs[0] if runs else None
    else:
        run = next((r for r in runs if r.id == run_id), None)
    if run is None:
        return QualityResponse(found=False, run_id=None, rows=[])
    rows = [QualityRow(**row) for row in build_quality_table(run)]
    return QualityResponse(found=True, run_id=run.id, rows=rows)


@insights_api_router.get("/api/costs")
async def get_costs(request: Request) -> CostsResponse:
    """LLM cost summary across all runs (totals + per-provider)."""
    summary = build_cost_summary(_state_db(request).get_runs())
    per_provider = [ProviderCost(**row) for row in summary["per_provider"]]
    return CostsResponse(
        total_cost=summary["total_cost"],
        total_tokens=summary["total_tokens"],
        total_calls=summary["total_calls"],
        avg_cost_per_run=summary["avg_cost_per_run"],
        per_provider=per_provider,
    )


__all__ = ["insights_api_router"]
