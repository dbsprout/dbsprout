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

The home dashboard is the data-backed view in this skeleton; ``/schema`` and
``/quality`` remain "Coming soon" placeholders that S-091 and S-093 replace with
real views. ``/progress`` is served by the real S-092 router
(:mod:`dbsprout.web.views.progress`), registered separately in ``app.py``.
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


# Sibling-view routes (/schema, /progress, /quality) are owned by their own
# router modules under ``dbsprout.web.views`` (S-091/S-092/S-093), each
# registered via a region-delimited ``include_router`` line in ``app.py``.
# The original "Coming soon" placeholder mechanism has been fully retired now
# that every sibling view is real (FastAPI matches the first registered route
# for a path, so a placeholder would shadow the real view).
