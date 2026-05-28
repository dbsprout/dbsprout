"""Read-only schema-review endpoints over the session workspace (S-115).

After a user connects (``POST /api/connect``, S-112) or uploads
(``POST /api/schema/load``, S-113) a schema, it lives on the in-memory
:class:`~dbsprout.web.workspace.Workspace` (``app.state.workspace``, S-111).
This module surfaces that loaded schema for *review*:

* ``GET /api/schema`` — the schema as a tree-shaped JSON body (each table with
  its columns/types, table-level primary key, and foreign keys), suitable for a
  client-side tree view. A friendly ``404`` JSON envelope when none is loaded.
* ``GET /api/schema/erd`` — an HTMX ERD *fragment* rendering the workspace
  schema as a Mermaid ``erDiagram``. A graceful ``200`` empty-state fragment when
  none is loaded.

Reuse, not reimplementation
---------------------------
The ERD is **not** re-derived here. The tree body reuses
:func:`dbsprout.web.views.erd._build_table_details` (so the tree and the ERD
click-detail panel show identical column metadata), and the ERD fragment reuses
S-082 :func:`dbsprout.report.erd.build_erd_mermaid` directly (PR #143 dropped the
intermediate ``_build_erd_with_clicks`` helper — Mermaid 10.9.x erDiagram does
not accept ``click`` directives; clicks are bound in JS post-render). The only
new behaviour here is reading the schema from the *workspace* rather than the
SnapshotStore.

Two read surfaces, one builder
------------------------------
The pre-existing ``GET /schema`` Jinja page (:mod:`dbsprout.web.views.erd`,
S-091) renders the ERD from the persisted **SnapshotStore**
(``.dbsprout/snapshots/``). These endpoints render from the **live workspace**
(connect/load). Different lifecycles → distinct routes; the snapshot view is left
untouched. The full Studio layout that arranges the tree + ERD into one page is
the later S-117 — hence the ERD endpoint returns a *fragment* (no ``base.html``)
that S-117 can ``hx-get`` into a panel.

This module is imported only by :mod:`dbsprout.web.app` (itself lazy-imported by
``dbsprout serve``); it stays import-light — the reused ERD helpers and
``build_erd_mermaid`` are pure (Pydantic types + stdlib + regex) and pull no
``core``/``generate.orchestrator``/CLI code. ``json`` is imported lazily in the
fragment handler (mirrors :mod:`dbsprout.web.views.erd`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from dbsprout.report.erd import build_erd_mermaid
from dbsprout.web.views.erd import _build_table_details

if TYPE_CHECKING:
    from fastapi.templating import Jinja2Templates

    from dbsprout.schema.models import DatabaseSchema
    from dbsprout.web.workspace import Workspace

schema_router = APIRouter()

#: Shown when ``GET /api/schema`` is hit before any schema is loaded.
_EMPTY_DETAIL = "No schema loaded. Connect to a database or upload a schema file first."


def _workspace(request: Request) -> Workspace:
    """Typed accessor for the shared session workspace wired in ``app.py``."""
    return cast("Workspace", request.app.state.workspace)


def _templates(request: Request) -> Jinja2Templates:
    """Typed accessor for the shared Jinja2 environment wired in ``app.py``."""
    return cast("Jinja2Templates", request.app.state.templates)


def _schema_tree(schema: DatabaseSchema, source: str | None) -> dict[str, Any]:
    """Shape *schema* into a tree-friendly JSON body.

    Per-table columns + foreign keys come from
    :func:`dbsprout.web.views.erd._build_table_details` (the same metadata the
    ERD click-detail panel uses), augmented with the table name and the
    table-level primary-key column list. The top-level envelope carries
    ``table_count``, ``dialect`` and the (already-redacted) ``source``.
    """
    details = _build_table_details(schema)
    tables = [
        {"name": table.name, "primary_key": list(table.primary_key), **details[table.name]}
        for table in schema.tables
    ]
    return {
        "table_count": len(schema.tables),
        "dialect": schema.dialect,
        "source": source,
        "tables": tables,
    }


@schema_router.get("/api/schema")
async def get_schema(request: Request) -> dict[str, Any]:
    """Return the loaded workspace schema as a tree-shaped JSON body.

    Raises a friendly ``404`` (``{"detail": ...}``) when no schema is loaded, so
    the client can prompt the user to connect or upload first — never a
    traceback.
    """
    workspace = _workspace(request)
    schema = workspace.get_schema()
    if schema is None:
        raise HTTPException(status_code=404, detail=_EMPTY_DETAIL)
    return _schema_tree(schema, workspace.get_source())


@schema_router.get("/api/schema/erd", response_class=Response)
async def schema_erd_fragment(request: Request) -> Response:
    """Render the loaded workspace schema as an HTMX ERD fragment.

    Reuses :func:`dbsprout.report.erd.build_erd_mermaid` and
    :func:`dbsprout.web.views.erd._build_table_details` so ERD generation is not
    duplicated. Returns ``200`` with a graceful empty-state fragment when no
    schema is loaded (so an HTMX swap shows a message rather than an error).
    """
    import json  # noqa: PLC0415 — stdlib, lazy for startup speed

    schema = _workspace(request).get_schema()
    erd_mermaid: str | None = None
    table_details_json: str | None = None

    if schema is not None:
        erd_mermaid = build_erd_mermaid(schema)
        table_details_json = json.dumps(_build_table_details(schema), separators=(",", ":"))

    return _templates(request).TemplateResponse(
        request,
        "schema_fragment.html",
        {"erd_mermaid": erd_mermaid, "table_details_json": table_details_json},
    )
