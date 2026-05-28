"""Insights views for the DBSprout web dashboard (S-093).

Four data-backed views over the read-only SQLite state layer
(``.dbsprout/state.db``, S-079/S-080): quality metrics, data preview, LLM cost
tracking, and run history. They expose :data:`insights_router`, registered in
:func:`dbsprout.web.app.create_app` inside the ``S-093 views region`` block.
S-093 removed the S-090 ``/quality`` placeholder from ``routes.py`` so the real
view here is the sole handler for that path (FastAPI matches the first registered
route for a path).

State data sourcing — the state DB stores *telemetry only* (runs, table_stats,
quality_results, llm_calls), never the generated sample rows. Hence:

* **/quality** reuses :func:`dbsprout.report.quality_table.build_quality_table`
  (S-083) to classify each metric ``pass``/``fail``/``warn``.
* **/preview** lists the latest run's tables (a selector dropdown populated from
  ``table_stats``) and shows per-table stats; sample rows are not persisted in
  the state DB, so the view renders an honest note rather than fabricating data.
* **/costs** aggregates ``llm_calls`` (total cost/tokens, avg cost per run,
  per-provider breakdown) and builds a Plotly cost-over-time spec following the
  :mod:`dbsprout.report.charts` convention (Plotly.js is JS-only, loaded via CDN).
* **/history** paginates over all runs (10/page), with cost-per-run summed from
  each run's ``llm_calls``.

The pure builders (:func:`build_cost_summary`, :func:`paginate_runs`) take plain
:class:`~dbsprout.state.models.RunRecord` lists and return template-friendly
dicts, keeping the route handlers and templates dumb (and ``autoescape`` safe).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from fastapi import APIRouter, Request
from fastapi.responses import Response

from dbsprout.report.quality_table import build_quality_table

if TYPE_CHECKING:
    from fastapi.templating import Jinja2Templates

    from dbsprout.state.db import StateDB
    from dbsprout.state.models import RunRecord

#: Runs shown per history page (AC: pagination, 10 runs per page).
RUNS_PER_PAGE = 10

#: Sample-row cap mirrored from the report layer; documented for the preview note.
PREVIEW_ROW_LIMIT = 10

insights_router = APIRouter()


# ── shared accessors (mirror dbsprout.web.routes) ──────────────────────────


def _templates(request: Request) -> Jinja2Templates:
    return cast("Jinja2Templates", request.app.state.templates)


def _state_db(request: Request) -> StateDB:
    factory = cast("Any", request.app.state.get_state_db)
    return cast("StateDB", factory())


def _latest_run(request: Request) -> RunRecord | None:
    """The most recent run (``get_runs`` returns newest-first), or ``None``."""
    runs = _state_db(request).get_runs()
    return runs[0] if runs else None


# ── pure view-model builders ───────────────────────────────────────────────


def build_cost_summary(runs: list[RunRecord]) -> dict[str, Any]:
    """Aggregate LLM cost telemetry across *runs* into a view-model.

    Returns total cost/tokens/calls, average cost per run, a per-provider
    breakdown (sorted by descending cost), and a Plotly cost-over-time chart
    spec (one point per run that recorded calls) — or ``chart=None`` when no
    LLM calls exist anywhere (offline/heuristic runs).
    """
    total_cost = 0.0
    total_tokens = 0
    total_calls = 0
    per_provider: dict[str, dict[str, float | int]] = {}
    series_x: list[str] = []
    series_y: list[float] = []

    for run in runs:
        run_cost = 0.0
        for call in run.llm_calls:
            total_cost += call.cost_usd
            run_cost += call.cost_usd
            total_tokens += call.tokens_sent + call.tokens_received
            total_calls += 1
            bucket = per_provider.setdefault(call.provider, {"cost": 0.0, "tokens": 0, "calls": 0})
            call_tokens = call.tokens_sent + call.tokens_received
            bucket["cost"] = cast("float", bucket["cost"]) + call.cost_usd
            bucket["tokens"] = cast("int", bucket["tokens"]) + call_tokens
            bucket["calls"] = cast("int", bucket["calls"]) + 1
        if run.llm_calls:
            series_x.append(run.started_at.isoformat())
            series_y.append(round(run_cost, 6))

    num_runs = len(runs)
    avg_cost_per_run = total_cost / num_runs if num_runs else 0.0

    provider_rows = sorted(
        (
            {
                "provider": name,
                "cost": round(cast("float", vals["cost"]), 6),
                "tokens": cast("int", vals["tokens"]),
                "calls": cast("int", vals["calls"]),
            }
            for name, vals in per_provider.items()
        ),
        key=lambda row: cast("float", row["cost"]),
        reverse=True,
    )

    chart = _cost_chart(series_x, series_y) if series_x else None

    return {
        "total_cost": round(total_cost, 6),
        "total_tokens": total_tokens,
        "total_calls": total_calls,
        "avg_cost_per_run": round(avg_cost_per_run, 6),
        "per_provider": provider_rows,
        "chart": chart,
    }


def _cost_chart(x_values: list[str], y_values: list[float]) -> dict[str, Any]:
    """Plotly cost-over-time spec (same convention as ``report.charts``)."""
    return {
        "data": [
            {
                "type": "scatter",
                "mode": "lines+markers",
                "x": x_values,
                "y": y_values,
                "name": "Cost per run (USD)",
            }
        ],
        "layout": {
            "title": {"text": "LLM cost over time"},
            "margin": {"t": 40, "r": 16, "b": 48, "l": 56},
            "yaxis": {"title": {"text": "USD"}},
        },
        "config": {"displaylogo": False, "responsive": True},
    }


def paginate_runs(runs: list[RunRecord], *, page: int, per_page: int) -> dict[str, Any]:
    """Slice *runs* into one page plus navigation metadata.

    ``page`` is clamped to ``[1, total_pages]`` (out-of-range or non-positive
    values fall back to page 1). An empty list yields ``total_pages == 1`` so the
    template never divides by zero.
    """
    total = len(runs)
    total_pages = max(1, -(-total // per_page))  # ceil division
    safe_page = page if 1 <= page <= total_pages else 1
    start = (safe_page - 1) * per_page
    return {
        "runs": runs[start : start + per_page],
        "page": safe_page,
        "total_pages": total_pages,
        "total_runs": total,
        "has_prev": safe_page > 1,
        "has_next": safe_page < total_pages,
    }


def _run_history_row(run: RunRecord) -> dict[str, Any]:
    """Shape a run into a history-table row, with cost summed from llm_calls."""
    cost = round(sum(call.cost_usd for call in run.llm_calls), 6)
    return {
        "id": run.id,
        "started_at": run.started_at.isoformat(),
        "engine": run.engine,
        "provider": run.llm_provider or "—",
        "total_rows": run.total_rows,
        "total_tables": run.total_tables,
        "duration_ms": run.duration_ms,
        "cost": cost,
    }


# ── routes ─────────────────────────────────────────────────────────────────


@insights_router.get("/quality", response_class=Response)
async def quality(request: Request) -> Response:
    """Quality-metrics dashboard with pass/fail/warn badges (replaces S-090 stub)."""
    run = _latest_run(request)
    rows = build_quality_table(run) if run is not None else []
    return _templates(request).TemplateResponse(
        request,
        "quality.html",
        {"active": "quality", "rows": rows, "has_run": run is not None},
    )


@insights_router.get("/preview", response_class=Response)
async def preview_index(request: Request) -> Response:
    """Data-preview landing: table selector populated from the latest run."""
    return _preview_response(request, table=None)


@insights_router.get("/preview/{table}", response_class=Response)
async def preview_table(request: Request, table: str) -> Response:
    """Per-table preview: stats from ``table_stats`` + honest sample-rows note."""
    return _preview_response(request, table=table)


def _preview_response(request: Request, *, table: str | None) -> Response:
    run = _latest_run(request)
    tables = [
        {
            "name": s.table_name,
            "row_count": s.row_count,
            "generation_ms": s.generation_ms,
            "rows_per_sec": round(s.rows_per_sec, 2),
            "errors": s.errors,
        }
        for s in (run.table_stats if run is not None else [])
    ]
    selected = next((t for t in tables if t["name"] == table), None) if table else None
    return _templates(request).TemplateResponse(
        request,
        "preview.html",
        {
            "active": "preview",
            "has_run": run is not None,
            "tables": tables,
            "selected_name": table,
            "selected": selected,
            "preview_row_limit": PREVIEW_ROW_LIMIT,
        },
    )


@insights_router.get("/costs", response_class=Response)
async def costs(request: Request) -> Response:
    """LLM cost tracking: summary cards, per-provider breakdown, cost-over-time."""
    summary = build_cost_summary(_state_db(request).get_runs())
    return _templates(request).TemplateResponse(
        request,
        "costs.html",
        {"active": "costs", "summary": summary},
    )


@insights_router.get("/history", response_class=Response)
async def history(request: Request, page: int = 1) -> Response:
    """Paginated run history (10/page) with cost per run."""
    runs = _state_db(request).get_runs()
    pagination = paginate_runs(runs, page=page, per_page=RUNS_PER_PAGE)
    rows = [_run_history_row(run) for run in pagination["runs"]]
    return _templates(request).TemplateResponse(
        request,
        "history.html",
        {"active": "history", "rows": rows, "pagination": pagination},
    )
