"""Read-only schema-review endpoints over the session workspace (S-115).

After a user connects (``POST /api/connect``, S-112) or uploads
(``POST /api/schema/load``, S-113) a schema, it lives on the in-memory
:class:`~dbsprout.web.workspace.Workspace` (``app.state.workspace``, S-111).
This module surfaces that loaded schema for *review*:

* ``GET /api/schema`` — the schema as a tree-shaped JSON body (each table with
  its columns/types, table-level primary key, and foreign keys), suitable for a
  client-side tree view. A friendly ``404`` JSON envelope when none is loaded.
* ``GET /api/schema/erd`` — the workspace schema as JSON
  ``{"mermaid": <erDiagram source | null>, "table_details": {...} | null}`` for
  the SPA to render client-side (Mermaid.js). A graceful ``200`` empty-state
  (``{"mermaid": null, "table_details": null}``) when none is loaded.

Reuse, not reimplementation
---------------------------
The ERD is **not** re-derived here. Both surfaces share the local
:func:`_build_table_details` helper (so the tree and the ERD click-detail panel
show identical column metadata), and the ERD ``mermaid`` string reuses S-082
:func:`dbsprout.report.erd.build_erd_mermaid` directly. The schema is read from
the *live workspace* (connect/load).

This module is imported only by :mod:`dbsprout.web.app` (itself lazy-imported by
``dbsprout serve``); it stays import-light — ``build_erd_mermaid`` and
``_build_table_details`` are pure (Pydantic types + stdlib + regex) and pull no
``core``/``generate.orchestrator``/CLI code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from fastapi import APIRouter, HTTPException, Request

from dbsprout.report.erd import build_erd_mermaid

if TYPE_CHECKING:
    from dbsprout.schema.models import DatabaseSchema
    from dbsprout.web.workspace import Workspace

schema_router = APIRouter()


def _build_table_details(schema: DatabaseSchema) -> dict[str, Any]:
    """Build a per-table column/FK detail dict suitable for JSON serialisation.

    Relocated from the former ``dbsprout.web.views.erd`` module (removed in the
    P1c-5 cutover); this is now the sole home of the helper.

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


#: Shown when ``GET /api/schema`` is hit before any schema is loaded.
_EMPTY_DETAIL = "No schema loaded. Connect to a database or upload a schema file first."


def _workspace(request: Request) -> Workspace:
    """Typed accessor for the shared session workspace wired in ``app.py``."""
    return cast("Workspace", request.app.state.workspace)


def _schema_tree(schema: DatabaseSchema, source: str | None) -> dict[str, Any]:
    """Shape *schema* into a tree-friendly JSON body.

    Per-table columns + foreign keys come from the local
    :func:`_build_table_details` (the same metadata the ERD click-detail panel
    uses), augmented with the table name and the table-level primary-key column
    list. The top-level envelope carries ``table_count``, ``dialect`` and the
    (already-redacted) ``source``.
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


@schema_router.get("/api/schema/erd")
async def get_schema_erd(request: Request) -> dict[str, Any]:
    """Return the loaded workspace schema as ERD JSON for client-side rendering.

    Reuses :func:`dbsprout.report.erd.build_erd_mermaid` and the local
    :func:`_build_table_details` so ERD generation is not duplicated. The SPA
    renders the ``mermaid`` string with Mermaid.js and binds the click-detail
    panel from ``table_details``. Returns ``200`` with
    ``{"mermaid": null, "table_details": null}`` when no schema is loaded.
    """
    schema = _workspace(request).get_schema()
    if schema is None:
        return {"mermaid": None, "table_details": None}
    return {
        "mermaid": build_erd_mermaid(schema),
        "table_details": _build_table_details(schema),
    }
