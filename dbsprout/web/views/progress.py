"""Web SSE progress view (S-092).

Streams real-time generation progress to the dashboard via Server-Sent Events.
The dashboard is a *read-only* surface over the SQLite state layer
(``.dbsprout/state.db``, S-079/S-080); this view derives progress from the
**latest run** returned by :meth:`StateDB.get_runs`.

Because the state layer has no per-table *target* row count and no separate
"live partial write" contract, progress is modelled as repeated snapshots of the
latest run:

* overall percent = completed-tables / ``total_tables``;
* a run with a non-null ``completed_at`` is *complete*, otherwise *running*;
* no runs at all → *idle* (page shows "No generation in progress").

**Bounded generator (critical).** :func:`iter_progress_events` polls at most
``max_polls`` times, terminates early when the latest run is complete or absent,
and **always** emits a terminal ``event: complete`` sentinel. There is no
unbounded ``while True`` — clients (and tests) consume a finite number of frames
and the stream closes itself. The endpoint uses FastAPI's native
``StreamingResponse`` (``text/event-stream``); no extra dependency is required.

This module is registered as its own router (``progress_router``) in
:func:`dbsprout.web.app.create_app`.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any, cast

from fastapi import APIRouter, Request
from fastapi.responses import Response, StreamingResponse

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from fastapi.templating import Jinja2Templates

    from dbsprout.state.db import StateDB
    from dbsprout.state.models import RunRecord

#: Default poll cadence (seconds) and cap. ``max_polls`` bounds the stream:
#: 240 polls x 0.5 s ~= 2 minutes before the generator gives up and closes itself.
DEFAULT_INTERVAL = 0.5
DEFAULT_MAX_POLLS = 240

progress_router = APIRouter()


def _templates(request: Request) -> Jinja2Templates:
    """Typed accessor for the shared Jinja2 environment wired in ``app.py``."""
    return cast("Jinja2Templates", request.app.state.templates)


def _state_db(request: Request) -> StateDB:
    """Open a fresh state-DB connection via the factory wired in ``app.py``."""
    factory = cast("Any", request.app.state.get_state_db)
    return cast("StateDB", factory())


def build_snapshot(runs: list[RunRecord]) -> dict[str, Any]:
    """Shape the latest run into a template/SSE-friendly progress snapshot.

    *runs* is newest-first (as returned by :meth:`StateDB.get_runs`). Returns an
    ``idle`` snapshot when empty. Overall percent is completed-tables /
    ``total_tables``; per-table rows surface ``row_count``, ``rows_per_sec`` and
    ``generation_ms`` (elapsed). ``eta_ms`` is a proportional estimate, only
    meaningful while *running* with progress > 0.
    """
    if not runs:
        return {
            "status": "idle",
            "overall_percent": 0,
            "total_rows": 0,
            "total_tables": 0,
            "completed_tables": 0,
            "current_table": None,
            "elapsed_ms": 0,
            "eta_ms": None,
            "tables": [],
        }

    run = runs[0]
    is_complete = run.completed_at is not None
    completed_tables = len(run.table_stats)
    total_tables = run.total_tables or completed_tables
    overall_percent = 100 if is_complete else _percent(completed_tables, total_tables)
    elapsed_ms = _elapsed_ms(run)
    eta_ms = None if is_complete else _eta_ms(elapsed_ms, overall_percent)
    current_table = run.table_stats[-1].table_name if run.table_stats else None

    return {
        "status": "complete" if is_complete else "running",
        "overall_percent": overall_percent,
        "total_rows": run.total_rows or sum(t.row_count for t in run.table_stats),
        "total_tables": total_tables,
        "completed_tables": completed_tables,
        "current_table": current_table,
        "elapsed_ms": elapsed_ms,
        "eta_ms": eta_ms,
        "tables": [
            {
                "table_name": t.table_name,
                "row_count": t.row_count,
                "rows_per_sec": round(t.rows_per_sec, 2),
                "generation_ms": t.generation_ms,
                "errors": t.errors,
                "status": "done",
            }
            for t in run.table_stats
        ],
    }


def _percent(done: int, total: int) -> int:
    if total <= 0:
        return 0
    return min(100, round(done / total * 100))


def _elapsed_ms(run: RunRecord) -> int:
    if run.duration_ms is not None:
        return run.duration_ms
    if run.completed_at is not None:
        return int((run.completed_at - run.started_at).total_seconds() * 1000)
    return sum(t.generation_ms for t in run.table_stats)


def _eta_ms(elapsed_ms: int, percent: int) -> int | None:
    if percent <= 0:
        return None
    total_estimate = elapsed_ms / (percent / 100)
    return max(0, int(total_estimate - elapsed_ms))


def iter_progress_events(
    get_runs: Callable[[], list[RunRecord]],
    *,
    max_polls: int = DEFAULT_MAX_POLLS,
    interval: float = DEFAULT_INTERVAL,
) -> Iterator[str]:
    """Yield a **bounded** sequence of SSE frames for the latest run.

    Polls *get_runs* up to *max_polls* times, emitting a ``data: <json>\\n\\n``
    snapshot each iteration and sleeping *interval* seconds between polls. The
    loop breaks early as soon as the latest run is *complete* or *idle* (no
    runs). A terminal ``event: complete\\ndata: {}\\n\\n`` frame is **always**
    emitted last, so the stream — and any client/test consuming it — terminates.
    """
    polls = max(1, max_polls)
    for poll in range(polls):
        snapshot = build_snapshot(get_runs())
        yield f"data: {json.dumps(snapshot)}\n\n"
        if snapshot["status"] in ("complete", "idle"):
            break
        if poll < polls - 1 and interval > 0:
            time.sleep(interval)
    yield "event: complete\ndata: {}\n\n"


@progress_router.get("/progress", response_class=Response)
async def progress_page(request: Request) -> Response:
    """Render the live progress page (subscribes to ``/progress/stream``)."""
    snapshot = build_snapshot(_state_db(request).get_runs())
    return _templates(request).TemplateResponse(
        request,
        "progress.html",
        {"active": "progress", "snapshot": snapshot},
    )


@progress_router.get("/progress/stream")
async def progress_stream(request: Request) -> StreamingResponse:
    """SSE endpoint streaming a bounded sequence of progress snapshots."""
    state_db = _state_db(request)
    return StreamingResponse(
        iter_progress_events(state_db.get_runs),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
