"""Web schema ERD view (S-091).

Renders the latest ``DatabaseSchema`` snapshot as an interactive Entity-
Relationship Diagram in the browser. The diagram source is the Mermaid
``erDiagram`` text produced by :func:`dbsprout.report.erd.build_erd_mermaid`
(S-082, reused verbatim — no duplicated ERD logic); Mermaid.js is loaded from a
CDN and renders the SVG **client-side** (no server-side image generation, per AC).

This module owns its own :class:`~fastapi.APIRouter` (``erd_router``) which
``dbsprout.web.app.create_app`` registers inside a delimited region. The schema
snapshot is read through a factory wired onto ``app.state`` in ``app.py`` so the
handler stays import-light and tests can point it at a temporary directory.

When no snapshot exists yet the view renders a graceful empty state with HTTP
200 — it never raises, so the dashboard stays usable before the first
``dbsprout init`` / ``generate`` run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from fastapi import APIRouter, Request
from fastapi.responses import Response

from dbsprout.report.erd import build_erd_mermaid

if TYPE_CHECKING:
    from fastapi.templating import Jinja2Templates

    from dbsprout.migrate.snapshot import SnapshotStore

erd_router = APIRouter()


def _templates(request: Request) -> Jinja2Templates:
    """Typed accessor for the shared Jinja2 environment wired in ``app.py``."""
    return cast("Jinja2Templates", request.app.state.templates)


def _snapshot_store(request: Request) -> SnapshotStore:
    """Open the snapshot store via the factory wired in ``app.py``."""
    factory = cast("Any", request.app.state.get_snapshot_store)
    return cast("SnapshotStore", factory())


@erd_router.get("/schema", response_class=Response)
async def schema_erd(request: Request) -> Response:
    """Render the schema ERD from the latest snapshot.

    Reads the most recent ``DatabaseSchema`` snapshot, converts it to a Mermaid
    ``erDiagram`` source string, and hands it to ``schema.html`` for client-side
    rendering. Returns HTTP 200 with an empty-state message when no snapshot
    exists.
    """
    schema = _snapshot_store(request).load_latest()
    erd_mermaid = build_erd_mermaid(schema) if schema is not None else None
    return _templates(request).TemplateResponse(
        request,
        "schema.html",
        {"active": "schema", "erd_mermaid": erd_mermaid},
    )
