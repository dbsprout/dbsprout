"""Web schema ERD view (S-091 / S-091-F1).

Renders the latest ``DatabaseSchema`` snapshot as an interactive Entity-
Relationship Diagram in the browser. The diagram source is the Mermaid
``erDiagram`` text produced by :func:`dbsprout.report.erd.build_erd_mermaid`
(S-082, reused verbatim — no duplicated ERD logic); Mermaid.js is loaded from a
CDN and renders the SVG **client-side** (no server-side image generation, per AC).

S-091-F1 adds:
* SVG pan/zoom via the ``svg-pan-zoom`` CDN library wired after Mermaid render.
* Per-table Mermaid ``click`` directives so clicking a table node fires a JS
  handler that shows its column/constraint detail panel.
* A ``<script id="erd-table-data" type="application/json">`` blob with full
  column + FK metadata so the click handler can populate the detail panel from
  the already-loaded page (no extra network round-trip).
* Filter-by-name input (``id="erd-filter-input"``) and show/hide-columns toggle
  (``id="erd-columns-toggle"``) for large-schema ergonomics.

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
    from dbsprout.schema.models import DatabaseSchema

erd_router = APIRouter()


def _templates(request: Request) -> Jinja2Templates:
    """Typed accessor for the shared Jinja2 environment wired in ``app.py``."""
    return cast("Jinja2Templates", request.app.state.templates)


def _snapshot_store(request: Request) -> SnapshotStore:
    """Open the snapshot store via the factory wired in ``app.py``."""
    factory = cast("Any", request.app.state.get_snapshot_store)
    return cast("SnapshotStore", factory())


def _build_erd_with_clicks(schema: DatabaseSchema) -> str:
    """Return the Mermaid ``erDiagram`` source for *schema*.

    Thin alias around :func:`dbsprout.report.erd.build_erd_mermaid` kept here so
    that ``dbsprout.web.routers.schema`` (S-115) can import a single helper from
    this module. Mermaid 10.9.x ``erDiagram`` rejects per-node ``click``
    directives, so click-to-detail is bound in JS post-render against the
    embedded JSON metadata (see ``_build_table_details``).
    """
    return build_erd_mermaid(schema)


def _build_table_details(schema: DatabaseSchema) -> dict[str, Any]:
    """Build a per-table column/FK detail dict suitable for JSON serialisation.

    The result is embedded in the page as ``<script id="erd-table-data"
    type="application/json">`` so the client-side click handler can populate
    the detail panel without a network round-trip.

    Returns a mapping of ``table_name -> {columns: [...], foreign_keys: [...]}``.
    Each column dict has: name, data_type, nullable, primary_key, unique.
    Each FK dict has: columns, ref_table, ref_columns, on_delete.
    """
    result: dict[str, Any] = {}
    for table in schema.tables:
        columns = [
            {
                "name": col.name,
                "data_type": col.data_type.value,
                "nullable": col.nullable,
                "primary_key": col.primary_key,
                "unique": col.unique,
            }
            for col in table.columns
        ]
        fks = [
            {
                "columns": list(fk.columns),
                "ref_table": fk.ref_table,
                "ref_columns": list(fk.ref_columns),
                "on_delete": fk.on_delete,
            }
            for fk in table.foreign_keys
        ]
        result[table.name] = {"columns": columns, "foreign_keys": fks}
    return result


@erd_router.get("/schema", response_class=Response)
async def schema_erd(request: Request) -> Response:
    """Render the schema ERD from the latest snapshot.

    Reads the most recent ``DatabaseSchema`` snapshot, converts it to a Mermaid
    ``erDiagram`` source string, builds a per-table detail JSON blob, and hands
    everything to ``schema.html`` for client-side rendering. Click-to-detail is
    bound in JS post-render (Mermaid 10.9.x erDiagram rejects ``click``
    directives). Returns HTTP 200 with an empty-state message when no snapshot
    exists.
    """
    import json  # noqa: PLC0415 — stdlib, lazy for startup speed

    schema = _snapshot_store(request).load_latest()
    erd_mermaid: str | None = None
    table_details_json: str | None = None

    if schema is not None:
        # Plain erDiagram source — no Mermaid ``click`` directives: erDiagram in
        # Mermaid 10.9.x rejects them ("Syntax error in text"). Click-to-detail
        # is bound in JS post-render (schema.html) against the embedded JSON.
        erd_mermaid = build_erd_mermaid(schema)
        table_details_json = json.dumps(_build_table_details(schema), separators=(",", ":"))

    return _templates(request).TemplateResponse(
        request,
        "schema.html",
        {
            "active": "schema",
            "erd_mermaid": erd_mermaid,
            "table_details_json": table_details_json,
        },
    )
