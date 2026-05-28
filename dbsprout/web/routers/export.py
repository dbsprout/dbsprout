"""``POST /api/export`` — stream the workspace last run as a file download (S-140).

The Studio user clicks **Export → SQL / CSV / JSON / Parquet** in the
dashboard; the handler validates the request body, reads the loaded
:class:`~dbsprout.web.workspace.Workspace` (S-111, wired on
``app.state.workspace``), resolves the writer via the existing
``dbsprout/plugins/dispatch::resolve_writer`` path (the same one the CLI
uses through :mod:`dbsprout.core.service`), writes per-table files into a
per-request temporary directory, and streams them back as the response
body. The temporary directory is cleaned up when the streaming generator
exits — see :func:`_stream_export`.

Multi-table policy (Brainstorm decision)
----------------------------------------
* ``sql`` — concatenate per-table files in ``insertion_order`` with a
  ``\\n-- next table --\\n`` separator (each per-table file is wrapped in
  ``BEGIN;...COMMIT;`` so the concat is a valid script).
* ``json`` — stream a hand-written top-level
  ``{"insertion_order": [...], "tables": {table: [rows], ...}}`` envelope
  directly from ``result.tables_data`` (the per-table writer output is a
  bare array; the wrap is what makes a single downloadable file useful).
* ``csv`` / ``parquet`` — no portable single-file representation for
  multi-table runs; require a single-element ``tables: [name]`` subset
  (422 ``EXPORT_MULTI_TABLE_UNSUPPORTED`` otherwise).

Guards (all return typed envelopes via :mod:`dbsprout.web.errors`)
------------------------------------------------------------------
* No run on the workspace ⇒ ``409 NO_RUN``.
* Unknown / missing format ⇒ ``422`` (Pydantic body validation).
* Unknown table in ``tables[]`` ⇒ ``404 NOT_FOUND``.
* Multi-table CSV / Parquet ⇒ ``422 EXPORT_MULTI_TABLE_UNSUPPORTED``.
* Parquet without ``[data]`` extra installed ⇒
  ``422 EXPORT_DEPENDENCY_MISSING``.

Lazy-import contract
--------------------
Module imports only stdlib + FastAPI + Pydantic at module level. The
writer, the workspace, and the dispatch helper are all lazy-imported
inside the handler / streaming generator (mirrors
``dbsprout/web/routers/insert.py``).
"""

from __future__ import annotations

import json as _json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Iterator

    from dbsprout.web.workspace import Workspace


export_router = APIRouter()


ExportFormat = Literal["sql", "csv", "json", "parquet"]


class ExportRequest(BaseModel):
    """Request body for ``POST /api/export``.

    ``format`` is restricted to the four file-format writers shipped with
    DBSprout. ``tables`` is optional — ``None`` or ``[]`` means *every
    table in the last run*; otherwise the export is scoped to the listed
    tables (FK-safe order is preserved by filtering, never re-sorting).
    """

    model_config = ConfigDict(extra="forbid")

    format: ExportFormat
    tables: list[str] | None = Field(default=None)


def _resolve_scope(
    insertion_order: list[str],
    tables: list[str] | None,
) -> tuple[list[str], list[str]]:
    """Resolve the FK-safe export scope and report any missing tables.

    Returns ``(scope_in_order, missing_names)``.

    * ``tables=None`` or empty ⇒ scope is the full ``insertion_order``.
    * ``tables=[...]`` ⇒ scope is ``insertion_order`` filtered to the
      requested set (preserves dependency order — never re-sorts);
      ``missing`` lists names not present in ``insertion_order``.
    """
    if not tables:
        return list(insertion_order), []
    requested = set(tables)
    known = set(insertion_order)
    missing = sorted(requested - known)
    scope = [t for t in insertion_order if t in requested]
    return scope, missing


# ── per-format metadata (Content-Type / extension / multi-table policy) ─

_PER_FORMAT: dict[str, dict[str, Any]] = {
    "sql": {
        "content_type": "application/sql",
        "extension": "sql",
        "supports_multi_table": True,
    },
    "csv": {
        "content_type": "text/csv; charset=utf-8",
        "extension": "csv",
        "supports_multi_table": False,
    },
    "json": {
        "content_type": "application/json",
        "extension": "json",
        "supports_multi_table": True,
    },
    "parquet": {
        "content_type": "application/vnd.apache.parquet",
        "extension": "parquet",
        "supports_multi_table": False,
    },
}


def _workspace(request: Request) -> Workspace:
    """Typed accessor for the shared session wired in ``app.py``."""
    return cast("Workspace", request.app.state.workspace)


def _stream_json(
    scope: list[str],
    result_tables_data: dict[str, list[dict[str, Any]]],
) -> Iterator[bytes]:
    """Stream JSON directly from ``result_tables_data`` — no writer call needed.

    Single table: emit the bare array (matches the writer's file shape).
    Multi-table: emit ``{"insertion_order": [...], "tables": {...}}``.
    """
    if len(scope) == 1:
        table = scope[0]
        rows = result_tables_data.get(table, [])
        payload = _json.dumps(rows, default=str, ensure_ascii=False)
        yield payload.encode("utf-8")
        return
    envelope = {
        "insertion_order": list(scope),
        "tables": {t: result_tables_data.get(t, []) for t in scope},
    }
    yield _json.dumps(envelope, default=str, ensure_ascii=False).encode("utf-8")


def _stream_via_writer(
    *,
    fmt: str,
    scope: list[str],
    result_tables_data: dict[str, list[dict[str, Any]]],
    schema: object,
    separator: bytes,
) -> Iterator[bytes]:
    """Run the dispatch-resolved writer into a temp dir, stream its files in order.

    The ``TemporaryDirectory`` lifetime is bound to the iterator — FastAPI
    exhausts the iterator while serving the response, then the ``with`` block
    exits and the dir is removed.
    """
    from dbsprout.plugins.dispatch import resolve_writer  # noqa: PLC0415

    writer = resolve_writer(fmt)
    with tempfile.TemporaryDirectory(prefix="dbsprout-export-") as tmp_dir:  # nosec B108
        tmp_path = Path(tmp_dir)
        writer.write(result_tables_data, schema, scope, tmp_path)
        files = sorted(tmp_path.iterdir())
        first = True
        for f in files:
            if not first and separator:
                yield separator
            first = False
            with f.open("rb") as fh:
                while True:
                    chunk = fh.read(64 * 1024)
                    if not chunk:
                        break
                    yield chunk


def _stream_export(
    *,
    fmt: str,
    scope: list[str],
    result_tables_data: dict[str, list[dict[str, Any]]],
    schema: object,
) -> Iterator[bytes]:
    """Return a byte iterator that owns a temp dir for the writer's per-table files.

    Cleanup runs when the iterator's ``__exit__`` fires (after FastAPI
    exhausts it). Multi-table policy is enforced upstream in ``post_export``;
    this generator assumes the scope is already format-safe.
    """
    if fmt == "json":
        yield from _stream_json(scope, result_tables_data)
        return
    if fmt == "sql":
        yield from _stream_via_writer(
            fmt="sql",
            scope=scope,
            result_tables_data=result_tables_data,
            schema=schema,
            separator=b"\n-- next table --\n",
        )
        return
    # csv / parquet — single-table only (guarded upstream).
    yield from _stream_via_writer(
        fmt=fmt,
        scope=scope,
        result_tables_data=result_tables_data,
        schema=schema,
        separator=b"",
    )


def _materialise_stream(
    *,
    fmt: str,
    scope: list[str],
    result_tables_data: dict[str, list[dict[str, Any]]],
    schema: object,
) -> Iterator[bytes]:
    """Probe optional deps synchronously, then return the streaming iterator.

    The parquet writer raises ``ImportError`` if ``polars`` is absent; that
    has to surface BEFORE :class:`fastapi.responses.StreamingResponse`
    starts serving (otherwise the response is already 200 by the time the
    iterator runs). We probe by importing the writer here.
    """
    if fmt == "parquet":
        from dbsprout.output import parquet_writer as _pw  # noqa: PLC0415

        if getattr(_pw, "pl", None) is None:
            msg = "polars is required for Parquet output"
            raise ImportError(msg)
    return _stream_export(
        fmt=fmt,
        scope=scope,
        result_tables_data=result_tables_data,
        schema=schema,
    )


@export_router.post("/api/export")
def post_export(body: ExportRequest, request: Request) -> StreamingResponse:
    """Stream the workspace's last generation result as a file download."""
    from dbsprout.web.errors import (  # noqa: PLC0415
        raise_web_error,
        web_error_export_dependency_missing,
        web_error_export_multi_table_unsupported,
        web_error_no_run,
        web_error_not_found_tables,
    )

    workspace = _workspace(request)
    result = workspace.get_last_result()
    if result is None or not result.tables_data:
        raise_web_error(request, web_error_no_run())
    assert result is not None  # narrowed for type-checker

    scope, missing = _resolve_scope(result.insertion_order, body.tables)
    if missing:
        raise_web_error(request, web_error_not_found_tables(missing))

    meta = _PER_FORMAT[body.format]
    if not meta["supports_multi_table"] and len(scope) > 1:
        raise_web_error(request, web_error_export_multi_table_unsupported(body.format))

    try:
        stream = _materialise_stream(
            fmt=body.format,
            scope=scope,
            result_tables_data=result.tables_data,
            schema=workspace.get_schema(),
        )
    except ImportError:
        raise_web_error(
            request,
            web_error_export_dependency_missing(body.format, "data"),
        )

    extension = cast("str", meta["extension"])
    media_type = cast("str", meta["content_type"])
    filename = f"{scope[0]}.{extension}" if len(scope) == 1 else f"dbsprout-export.{extension}"

    return StreamingResponse(
        stream,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


__all__ = [
    "_PER_FORMAT",
    "ExportRequest",
    "_materialise_stream",
    "_resolve_scope",
    "_stream_export",
    "export_router",
    "post_export",
]
