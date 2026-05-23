"""Shared route table for the DBSprout web dashboard (S-090).

This module is the **extension seam** for sibling stories. ``app.py`` builds
the :class:`~fastapi.FastAPI` instance and calls ``app.include_router(router)``;
new views are added by appending handlers to this ``router`` inside a
region-delimited block, e.g.::

    # ── S-091 ERD routes ──────────────────────────────────────────────
    @router.get("/schema-erd")
    async def schema_erd(request: Request) -> Response: ...


    # ── end S-091 ─────────────────────────────────────────────────────

Each handler renders a Jinja2 template via :func:`_templates` and reads telemetry
through :func:`_state_db` — both pull typed objects off ``app.state`` (wired in
``app.py``), so handlers stay import-light and siblings reuse the same plumbing.

The home dashboard is the only data-backed view in this skeleton; ``/schema``,
``/progress`` and ``/quality`` are intentionally "Coming soon" placeholders that
S-091, S-092 and S-093 replace with real views.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

if TYPE_CHECKING:
    from fastapi.templating import Jinja2Templates

    from dbsprout.state.db import StateDB
    from dbsprout.state.models import RunRecord

router = APIRouter()


def _templates(request: Request) -> Jinja2Templates:
    """Typed accessor for the shared Jinja2 environment wired in ``app.py``."""
    return cast("Jinja2Templates", request.app.state.templates)


def _state_db(request: Request) -> StateDB:
    """Open a fresh state-DB connection via the factory wired in ``app.py``."""
    factory = cast("Any", request.app.state.get_state_db)
    return cast("StateDB", factory())


def _run_summary(run: RunRecord) -> dict[str, Any]:
    """Shape a state-layer run into a template-friendly summary dict."""
    return {
        "engine": run.engine,
        "total_rows": run.total_rows,
        "total_tables": run.total_tables,
        "seed": run.seed,
        "started_at": run.started_at.isoformat(),
        "duration_ms": run.duration_ms,
    }


@router.get("/", response_class=Response)
async def index(request: Request) -> Response:
    """Dashboard home: run summary + navigation.

    Reads from ``.dbsprout/state.db`` and renders even when no run exists yet
    (and even when the CLI has never created the file).
    """
    runs = [_run_summary(run) for run in _state_db(request).get_runs()]
    return _templates(request).TemplateResponse(
        request,
        "index.html",
        {"runs": runs, "active": "home"},
    )


@router.get("/health", response_class=JSONResponse)
async def health() -> JSONResponse:
    """Liveness probe — cheap JSON, no template."""
    return JSONResponse({"status": "ok"})


def _placeholder(request: Request, *, active: str, title: str) -> Response:
    return _templates(request).TemplateResponse(
        request,
        "placeholder.html",
        {"active": active, "title": title},
    )


# ── sibling-view placeholders (extension seam — replace in S-091/092/093) ──


@router.get("/schema", response_class=Response)
async def schema(request: Request) -> Response:
    """Schema ERD placeholder — replaced by S-091."""
    return _placeholder(request, active="schema", title="Schema")


@router.get("/progress", response_class=Response)
async def progress(request: Request) -> Response:
    """Generation progress placeholder — replaced by S-092."""
    return _placeholder(request, active="progress", title="Progress")


# NOTE: the ``/quality`` placeholder was removed in S-093; the real view now
# lives in ``dbsprout.web.views.insights`` (FastAPI matches the first registered
# route for a path, so the placeholder had to go rather than be shadowed).


# ── end sibling-view placeholders ─────────────────────────────────────
