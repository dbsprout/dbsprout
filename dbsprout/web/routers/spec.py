"""``GET /api/spec`` — DataSpec read endpoint over the session workspace (S-118).

After a user connects (``POST /api/connect``, S-112) or uploads
(``POST /api/schema/load``, S-113) a schema, it lives on the in-memory
:class:`~dbsprout.web.workspace.Workspace` (``app.state.workspace``, S-111).
This module surfaces the *DataSpec* derived from that schema for review:

* ``GET /api/spec`` — the active ``DataSpec`` as JSON (per-table row count and
  per-column generator config). If no spec has been cached on the workspace
  yet, the handler builds one *heuristically* via the existing
  :func:`dbsprout.spec.analyzer.heuristic_fallback` (Sprint-2 patterns; no LLM,
  no cloud, deterministic) and caches it via ``Workspace.set_spec``. A
  ``409 Conflict`` JSON envelope (``{"code": "NO_SCHEMA", "message": ...}``)
  is raised when no schema is loaded — distinct from ``404`` ("URL does not
  exist") so the client can prompt the user to connect first.
* Same URL with ``Accept: text/html`` — renders ``spec_grid.html`` as an HTMX
  fragment: a row per column with a method pill (``<button data-method=...
  data-provider=...>``) carrying the click target shape that S-119 will wire
  up. The HTML branch returns ``200`` with an empty-state body when no schema
  is loaded so an HTMX swap shows a friendly message rather than a 4xx.

Reuse, not reimplementation
---------------------------
The DataSpec build is **not** re-derived here. ``heuristic_fallback`` already
turns a :class:`~dbsprout.schema.models.DatabaseSchema` into a fully-typed
:class:`~dbsprout.spec.models.DataSpec` via
:func:`dbsprout.spec.heuristics.map_columns`. The router is responsible only
for: workspace lookup, lazy import of the builder, cache write, and response
shaping.

This module is imported only by :mod:`dbsprout.web.app` (itself lazy-imported
by ``dbsprout serve``); it stays import-light — the spec analyzer is lazy-
imported inside the build helper so importing the router stays cheap. The
edit endpoint (S-119) will live in a sibling router; this module only surfaces
the read shape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from fastapi import APIRouter, HTTPException, Request, status

if TYPE_CHECKING:
    from fastapi.responses import Response
    from fastapi.templating import Jinja2Templates

    from dbsprout.spec.models import DataSpec
    from dbsprout.web.workspace import Workspace

spec_router = APIRouter()

#: Body for the ``409 Conflict`` envelope raised when no schema is loaded.
_NO_SCHEMA_DETAIL: dict[str, str] = {
    "code": "NO_SCHEMA",
    "message": ("No schema is loaded. Connect to a database or upload a schema file first."),
}

#: Friendly empty-state message echoed inside the HTML fragment when no
#: schema is loaded. Kept terse so the Studio grid panel surfaces a tidy nudge
#: rather than a wall of copy.
_HTML_NO_SCHEMA_MESSAGE = "Connect to a database or upload a schema file to see the spec grid."


def _workspace(request: Request) -> Workspace:
    """Typed accessor for the shared session workspace wired in ``app.py``."""
    return cast("Workspace", request.app.state.workspace)


def _templates(request: Request) -> Jinja2Templates:
    """Typed accessor for the shared Jinja2 environment wired in ``app.py``."""
    return cast("Jinja2Templates", request.app.state.templates)


def _wants_html(request: Request) -> bool:
    """Return ``True`` when the client prefers ``text/html`` over JSON.

    Content-negotiation heuristic: any explicit ``text/html`` in the ``Accept``
    header that appears *before* ``application/json`` (or with JSON absent)
    selects the HTML branch. Empty / missing ``Accept`` defaults to JSON.
    """
    accept = request.headers.get("accept", "").lower()
    if not accept:
        return False
    html_idx = accept.find("text/html")
    if html_idx == -1:
        return False
    json_idx = accept.find("application/json")
    if json_idx == -1:
        return True
    return html_idx < json_idx


def _build_or_get_spec(workspace: Workspace) -> DataSpec:
    """Return the cached spec, or build one heuristically and cache it.

    Reuses :func:`dbsprout.spec.analyzer.heuristic_fallback` — the same builder
    that powers the offline / no-LLM spec path elsewhere — so the JSON shape
    matches what the rest of the pipeline already understands. The builder is
    lazy-imported to preserve the router's import-light contract.
    """
    cached = workspace.get_spec()
    if cached is not None:
        return cached
    from dbsprout.spec.analyzer import heuristic_fallback  # noqa: PLC0415

    schema = workspace.get_schema()
    # The handler guards against ``schema is None`` before calling us, so this
    # branch is unreachable; we narrow the type without ``assert`` (which ruff
    # would flag as a leftover test idiom in production code).
    if schema is None:  # pragma: no cover — defensive invariant
        msg = "_build_or_get_spec called without a loaded schema"
        raise RuntimeError(msg)
    spec = heuristic_fallback(schema)
    workspace.set_spec(spec)
    return spec


@spec_router.get("/api/spec", response_model=None)
async def get_spec(request: Request) -> Response | dict[str, Any]:
    """Return the active workspace ``DataSpec`` (JSON) or its grid fragment (HTML).

    JSON branch (default ``Accept``): raises ``409`` with
    ``{"code": "NO_SCHEMA", "message": ...}`` when no schema is loaded; else
    returns ``spec.model_dump(mode="json")``.

    HTML branch (``Accept: text/html``): returns ``200`` with the spec grid
    template; on missing schema returns the empty-state body (still ``200``)
    so HTMX swaps cleanly.
    """
    workspace = _workspace(request)
    schema = workspace.get_schema()
    wants_html = _wants_html(request)

    if schema is None:
        if wants_html:
            return _templates(request).TemplateResponse(
                request,
                "spec_grid.html",
                {"spec": None, "empty_message": _HTML_NO_SCHEMA_MESSAGE},
            )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=_NO_SCHEMA_DETAIL,
        )

    spec = _build_or_get_spec(workspace)
    if wants_html:
        return _templates(request).TemplateResponse(
            request,
            "spec_grid.html",
            {"spec": spec, "empty_message": None},
        )
    return spec.model_dump(mode="json")
