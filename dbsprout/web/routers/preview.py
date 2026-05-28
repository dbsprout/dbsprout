"""``GET /api/preview/{table}`` — bounded JSON sample of generated rows (S-147).

After ``POST /api/generate`` (S-124) populates the per-session
:class:`~dbsprout.web.workspace.Workspace` (``app.state.workspace``, S-111) with a
:class:`~dbsprout.generate.orchestrator.GenerateResult`, the Studio grid needs a
fast way to render a small slice of the generated data for one table — the
preview/regen feedback loop has to stay quick (FR-024).

This module exposes a single read-only endpoint:

* ``GET /api/preview/{table}?limit=N`` returns up to ``N`` rows from
  ``workspace.last_result.tables_data[table]`` as JSON, with ``limit`` bounded
  ``[1, 1000]``. The envelope echoes ``table``, the requested ``limit``, the
  ``total`` rows available on the server (so a client can show "showing X of Y"
  without a second request) and ``rows`` (the bounded slice).

Empty state policy
------------------
Both "no generation has run yet" (``workspace.last_result is None``) and "the
table is not in the last result" surface as a **friendly 404** with a string
``detail`` — symmetric with :mod:`dbsprout.web.routers.schema` (S-115). The two
cases use distinct messages so the client can disambiguate them, but neither
ever leaks a Python traceback. ``limit`` validation lives on the FastAPI
``Query`` declaration (``ge=1, le=1000``) — out-of-range or non-integer values
return ``422`` automatically.

This module is imported only by :mod:`dbsprout.web.app` (itself lazy-imported by
``dbsprout serve``). It stays import-light — only ``fastapi`` and stdlib typing.
The :class:`~dbsprout.web.workspace.Workspace` and the
:class:`~dbsprout.generate.orchestrator.GenerateResult` are accessed via
``app.state`` at request time, never imported at module level (kept under
:data:`typing.TYPE_CHECKING`), preserving the ``dbsprout serve`` lazy-import
contract — see ``test_preview_router_no_eager_generation_import``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from fastapi import APIRouter, HTTPException, Query, Request

if TYPE_CHECKING:
    from dbsprout.web.workspace import Workspace

preview_router = APIRouter()

#: Sane cap on the requested sample size — the preview is for human-eyeballing
#: in the Studio grid, not for bulk export (use the output writers for that).
_MAX_PREVIEW_LIMIT: int = 1000

#: Default sample size when ``?limit`` is omitted — matches the AC's "up to 100".
_DEFAULT_PREVIEW_LIMIT: int = 100

#: Surfaced as the ``detail`` of the 404 when no generation has run yet.
_NO_RESULT_DETAIL = "No generation result available; run POST /api/generate before previewing."


def _workspace(request: Request) -> Workspace:
    """Typed accessor for the shared session workspace wired in ``app.py``."""
    return cast("Workspace", request.app.state.workspace)


def _table_not_found_detail(table: str, available: list[str]) -> str:
    """Build a friendly 404 detail naming the table and listing the available ones."""
    if available:
        names = ", ".join(sorted(available))
        return f"Table {table!r} is not in the last generation result. Available tables: {names}."
    return f"Table {table!r} is not in the last generation result (the result contains no tables)."


@preview_router.get("/api/preview/{table}")
async def preview_table(
    request: Request,
    table: str,
    limit: int = Query(
        default=_DEFAULT_PREVIEW_LIMIT,
        ge=1,
        le=_MAX_PREVIEW_LIMIT,
        description="Maximum number of rows to return (1-1000).",
    ),
) -> dict[str, Any]:
    """Return up to *limit* rows of generated data for *table* as JSON.

    ``404`` (friendly JSON) when no generation has run yet or *table* is unknown.
    ``422`` (auto from Pydantic) when *limit* is outside ``[1, 1000]`` or not an
    integer. Never leaks a Python traceback.
    """
    workspace = _workspace(request)
    last_result = workspace.get_last_result()
    if last_result is None:
        raise HTTPException(status_code=404, detail=_NO_RESULT_DETAIL)

    tables_data = last_result.tables_data
    if table not in tables_data:
        raise HTTPException(
            status_code=404,
            detail=_table_not_found_detail(table, list(tables_data.keys())),
        )

    rows = tables_data[table]
    return {
        "table": table,
        "limit": limit,
        "total": len(rows),
        "rows": rows[:limit],
    }
