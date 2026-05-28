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
from pydantic import ValidationError

from dbsprout.spec.models import GeneratorConfig

if TYPE_CHECKING:
    from fastapi import FastAPI
    from fastapi.responses import Response
    from fastapi.templating import Jinja2Templates

    from dbsprout.schema.models import DatabaseSchema
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
            {
                "spec": spec,
                "empty_message": None,
                # S-120: ship a ``{table: {column: dtype_name}}`` map so the
                # spec_row.html pill can carry ``data-dtype`` for the method
                # picker. ``schema`` is the loaded ``DatabaseSchema``; the
                # template handles a missing entry as ''.
                "column_dtypes": _column_dtypes_map(schema),
            },
        )
    return spec.model_dump(mode="json")


def _column_dtypes_map(schema: DatabaseSchema) -> dict[str, dict[str, str]]:
    """Return a ``{table_name: {column_name: dtype_name}}`` map.

    Sourced from the loaded :class:`~dbsprout.schema.models.DatabaseSchema`;
    ``dtype_name`` is the upper-case :class:`ColumnType.name` so the
    method-picker can match it against ``GET /api/generators?dtype=...``
    1:1 without a client-side lookup.
    """
    return {
        table.name: {col.name: col.data_type.name for col in table.columns}
        for table in schema.tables
    }


# region: PUT table row_count (S-121)
#
# ``PUT /api/spec/tables/{table_name}`` — set the row count for a single
# table on the workspace spec. The route lives in its own region block so the
# sibling S-119 (column PUT) can be union-merged without conflicts: it touches
# only this region + the matching ``Workspace.update_table_row_count`` helper.
#
# Validation:
#   * body must be JSON ``{"row_count": <int>}``,
#   * ``row_count`` ∈ [1, upper_bound] — anything else → 422 with
#     ``{"code": "INVALID_ROW_COUNT", "message": …}``.
#
# Upper bound resolution: ``app.state.config.generation.max_rows_per_table``
# when present and non-None, else ``_DEFAULT_UPPER_BOUND`` (10_000_000). The
# bound is intentionally not surfaced to callers as a separate field — the
# message embeds it.
#
# Content-negotiation: HTMX (``HX-Request: true``) or ``Accept: text/html``
# returns the re-rendered ``_spec_table_header.html`` partial (200, text/html);
# default JSON branch returns ``{"table_name": …, "row_count": <new>}``.
#
# No cache persistence here — S-122 owns that.

#: Default upper bound for a single table's row_count. Keep in sync with the
#: PRD (FR-014) and any explicit config override (see ``_resolve_upper_bound``).
_DEFAULT_UPPER_BOUND: int = 10_000_000

#: Detail body for the 422 envelope when ``row_count`` is missing / wrong type /
#: out of bounds. The ``message`` is filled in per call; ``code`` is stable.
_INVALID_ROW_COUNT_HINT = 'Send {"row_count": <int>} with 1 ≤ value ≤ upper bound.'


def _resolve_upper_bound(app: FastAPI) -> int:
    """Return the configured upper bound for table row_count.

    Prefers ``app.state.config.generation.max_rows_per_table`` when the host
    wires a :class:`~dbsprout.config.models.DBSproutConfig` onto the app; falls
    back to :data:`_DEFAULT_UPPER_BOUND` otherwise. The fallback path keeps the
    route working in the default localhost flow where no TOML config has been
    loaded — the bound is a defence-in-depth limit, not a feature gate.
    """
    config = getattr(app.state, "config", None)
    if config is None:
        return _DEFAULT_UPPER_BOUND
    generation = getattr(config, "generation", None)
    if generation is None:
        return _DEFAULT_UPPER_BOUND
    bound = getattr(generation, "max_rows_per_table", None)
    if bound is None:
        return _DEFAULT_UPPER_BOUND
    return int(bound)


def _wants_html_response(request: Request) -> bool:
    """Return ``True`` when an HTMX request OR ``Accept: text/html`` was sent."""
    if request.headers.get("hx-request", "").lower() == "true":
        return True
    return _wants_html(request)


def _invalid_row_count(message: str) -> HTTPException:
    """Build the 422 envelope for any row_count validation failure."""
    # 422 Unprocessable Content (FastAPI/Starlette renamed the constant in
    # newer versions; use the literal to stay forward-compatible).
    return HTTPException(
        status_code=422,
        detail={
            "code": "INVALID_ROW_COUNT",
            "message": message,
            "hint": _INVALID_ROW_COUNT_HINT,
        },
    )


def _validate_row_count(payload: Any, upper_bound: int) -> int:
    """Validate the parsed JSON payload and return the row_count int.

    Raises :class:`fastapi.HTTPException` (422 envelope) for any failure:
    non-dict body, missing key, non-int / bool / float value, or value outside
    ``[1, upper_bound]``.
    """
    if not isinstance(payload, dict):
        raise _invalid_row_count("Request body must be a JSON object.")
    if "row_count" not in payload:
        raise _invalid_row_count("Missing required field 'row_count'.")
    raw_value: object = payload["row_count"]
    # ``bool`` is a subclass of ``int`` — reject it explicitly so True/False
    # cannot be smuggled in as 1/0.
    if isinstance(raw_value, bool) or not isinstance(raw_value, int):
        raise _invalid_row_count("Field 'row_count' must be an integer.")
    value: int = raw_value
    if value < 1:
        raise _invalid_row_count(
            f"Field 'row_count' must be ≥ 1 (got {value}).",
        )
    if value > upper_bound:
        raise _invalid_row_count(
            f"Field 'row_count' must be ≤ {upper_bound} (got {value}).",
        )
    return value


@spec_router.put("/api/spec/tables/{table_name}", response_model=None)
async def put_table_row_count(
    request: Request,
    table_name: str,
) -> Response | dict[str, Any]:
    """Set the ``row_count`` for one table on the workspace spec (S-121).

    See the region header above for the full contract. On the HTMX / HTML
    branch, the response body is the ``_spec_table_header.html`` partial so
    the Studio grid can swap the new value into ``[data-row-count]`` cleanly.
    """
    workspace = _workspace(request)
    schema = workspace.get_schema()
    if schema is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=_NO_SCHEMA_DETAIL,
        )

    # Parse + validate body BEFORE touching the spec — bad input must not
    # cause a heuristic build as a side effect.
    try:
        payload = await request.json()
    except (ValueError, TypeError) as exc:
        raise _invalid_row_count("Request body must be valid JSON.") from exc
    upper_bound = _resolve_upper_bound(request.app)
    row_count = _validate_row_count(payload, upper_bound)

    # Lazily build/load the spec (same path as GET /api/spec) so the PUT
    # works even if the client edits before reading.
    _build_or_get_spec(workspace)
    try:
        new_value = workspace.update_table_row_count(table_name, row_count)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "UNKNOWN_TABLE",
                "message": f"Table {table_name!r} not found in spec.",
            },
        ) from exc

    if _wants_html_response(request):
        return _templates(request).TemplateResponse(
            request,
            "_spec_table_header.html",
            {"table_name": table_name, "row_count": new_value},
        )
    return {"table_name": table_name, "row_count": new_value}


# endregion: PUT table row_count (S-121)


# region: PUT column (S-119) -------------------------------------------------
# Sibling S-121 owns the row-count PUT and edits the same module — keep this
# block self-contained so the wave merge can union the two regions cleanly.


def _wants_html_or_htmx(request: Request) -> bool:
    """Return ``True`` when the client carries ``Accept: text/html`` *or* ``HX-Request: true``.

    HTMX-driven edits commonly send ``Accept: */*`` and signal their intent via
    the ``HX-Request`` header instead; we accept both.
    """
    if _wants_html(request):
        return True
    hx = request.headers.get("hx-request", "").lower()
    return hx == "true"


@spec_router.put(
    "/api/spec/tables/{table}/columns/{column}",
    response_model=None,
)
async def put_column_config(
    table: str,
    column: str,
    request: Request,
) -> Response | dict[str, Any]:
    """Replace one column's ``GeneratorConfig`` on the workspace spec.

    Request body is validated against :class:`~dbsprout.spec.models.GeneratorConfig`
    via :meth:`pydantic.BaseModel.model_validate`; malformed input yields ``422``
    with Pydantic-native field-level errors. A referential-integrity guard
    (:func:`dbsprout.spec.constraints.check_column_update`) rejects PK / FK-
    target downgrades with ``409 CONSTRAINT_VIOLATION``. Missing schema is
    ``409 NO_SCHEMA`` (the same envelope ``GET /api/spec`` raises). Unknown
    table / column is ``404 NOT_FOUND``.

    Response shape:

    * JSON branch (default ``Accept``) — the new ``GeneratorConfig`` as JSON.
    * HTML branch (``Accept: text/html`` or ``HX-Request: true``) — the
      single-row HTMX fragment from ``spec_row.html`` so the studio grid can
      hot-swap one row without rebuilding the whole grid.
    """
    workspace = _workspace(request)
    schema = workspace.get_schema()
    if schema is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=_NO_SCHEMA_DETAIL,
        )

    # 1. Pydantic validation at the boundary — closed by ``extra='forbid'``.
    try:
        raw = await request.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=[{"loc": ["body"], "msg": str(exc), "type": "value_error.jsondecode"}],
        ) from exc
    try:
        new_config = GeneratorConfig.model_validate(raw)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=exc.errors(),
        ) from exc

    # 2. Referential-integrity guard against the *loaded schema*.
    from dbsprout.spec.constraints import check_column_update  # noqa: PLC0415

    reason = check_column_update(schema, table, column, new_config)
    if reason is not None:
        # Distinguish "you can't change this" (409) from "this doesn't exist"
        # (404). When the column / table doesn't exist on the *schema* but the
        # spec also won't know about it, surface 404; otherwise 409.
        table_obj = schema.get_table(table)
        if table_obj is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "NOT_FOUND",
                    "message": reason,
                    "table": table,
                    "column": column,
                },
            )
        if table_obj.get_column(column) is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "NOT_FOUND",
                    "message": reason,
                    "table": table,
                    "column": column,
                },
            )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "CONSTRAINT_VIOLATION",
                "message": reason,
                "table": table,
                "column": column,
            },
        )

    # 3. Apply the immutable swap on the workspace; LookupError → 404.
    if workspace.get_spec() is None:
        _build_or_get_spec(workspace)
    try:
        stored = workspace.update_column(table, column, new_config)
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "NOT_FOUND",
                "message": str(exc),
                "table": table,
                "column": column,
            },
        ) from exc

    # 4. Shape the response — HTMX fragment vs. JSON.
    if _wants_html_or_htmx(request):
        # S-120: resolve the dtype for the pill so the rendered fragment
        # still carries ``data-dtype`` after the swap. The schema lookup
        # is already done above (``check_column_update``) so this is free.
        col_dtype = ""
        table_obj = schema.get_table(table)
        if table_obj is not None:
            col_obj = table_obj.get_column(column)
            if col_obj is not None:
                col_dtype = col_obj.data_type.name
        return _templates(request).TemplateResponse(
            request,
            "spec_row.html",
            {
                "table_name": table,
                "col_name": column,
                "col_cfg": stored,
                "col_dtype": col_dtype,
            },
        )
    return stored.model_dump(mode="json")


# endregion -----------------------------------------------------------------
