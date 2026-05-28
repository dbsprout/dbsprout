"""Column-update writer — patch a single column on an existing dataset.

S-138 (Sprint 4, Web UI v1.2.0). After a Studio user re-rolls one column via
``regenerate_column`` (S-129), this writer pushes the patched values back
into the live database **without re-inserting every row**.

Design notes
------------

* **Pure core module** — no imports from ``dbsprout.web``, ``dbsprout.cli``,
  or ``dbsprout.tui``. The HTTP route (S-139) will sit on top of this.
* **Parameterised SQL only.** Values are bound, never interpolated, on every
  dialect.
* **PK-less tables are refused.** The writer raises
  :class:`ColumnUpdateError` with code ``"no_primary_key"`` before any SQL
  is issued. The HTTP wrapper (S-139) maps that code to ``409 Conflict``.
* **Single-column PK** → bulk path via ``executemany``.
* **Composite PK** → falls back to one statement per row (documented
  limitation for v1.2.0). Each row must arrive as ``(pk_tuple, new_value)``
  with ``pk_tuple`` matching the table's ``primary_key`` arity.
* **Identifier safety** — table, column, and PK column names are validated
  with the same allow-list (``^[A-Za-z_][A-Za-z0-9_]*$``) used by
  :mod:`dbsprout.output.sa_batch`. Anything else raises
  :class:`ColumnUpdateError` with code ``"unsafe_identifier"``.
* **Transactional** — each call runs inside an explicit transaction. If the
  caller passes an :class:`sqlalchemy.Engine`, the writer opens its own
  ``BEGIN…COMMIT/ROLLBACK``. If the caller passes a live
  :class:`sqlalchemy.Connection`, the writer uses ``begin_nested`` (when a
  transaction is already active) or ``begin``.

Public surface
--------------

* :func:`update_column` — the core function. Returns
  :class:`~dbsprout.output.models.ColumnUpdateResult`.
* :class:`ColumnUpdateWriter` — thin :class:`OutputWriter`-shaped wrapper
  that lets the registry expose this module as a plugin
  (``format = "column_update"``).
* :class:`ColumnUpdateError` — typed error with a string ``code``
  discriminator (``"no_primary_key"``, ``"unknown_table"``,
  ``"unknown_column"``, ``"pk_column_update"``, ``"pk_arity_mismatch"``,
  ``"unsafe_identifier"``, ``"invalid_connection"``, ``"update_failed"``).
"""

from __future__ import annotations

import re
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa

from dbsprout.output.models import ColumnUpdateResult

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from dbsprout.schema.models import DatabaseSchema


# ---------------------------------------------------------------------------
# Identifier guard (same allow-list as sa_batch)
# ---------------------------------------------------------------------------

_SAFE_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _safe_ident(name: str) -> str:
    """Return *name* if it is a safe SQL identifier, else raise."""
    if not _SAFE_IDENT_RE.match(name):
        raise ColumnUpdateError(
            "unsafe_identifier", table=name, detail=f"identifier {name!r} rejected"
        )
    return f'"{name}"'


# ---------------------------------------------------------------------------
# Typed error
# ---------------------------------------------------------------------------


class ColumnUpdateError(Exception):
    """Typed error raised by :func:`update_column`.

    Parameters
    ----------
    code:
        Machine-readable discriminator. Stable across dialects so callers
        (especially the S-139 HTTP route) can map it to status codes.
    table:
        Name of the offending table — included for log/triage clarity.
    detail:
        Optional human-readable detail (e.g. the underlying DBAPI
        exception class name).
    """

    __slots__ = ("code", "detail", "table")

    def __init__(self, code: str, *, table: str, detail: str | None = None) -> None:
        self.code = code
        self.table = table
        self.detail = detail
        parts = [f"[{code}] table={table!r}"]
        if detail:
            parts.append(f"detail={detail}")
        super().__init__(" ".join(parts))


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------


@contextmanager
def _txn_scope(
    connection: sa.Connection | sa.Engine,
) -> Iterator[sa.Connection]:
    """Yield a connection inside an explicit transaction.

    * If *connection* is an :class:`sa.Engine` we open a short-lived
      connection and use ``begin()`` — auto-commit on success, rollback on
      exception.
    * If *connection* is a live :class:`sa.Connection` and a transaction is
      already in flight we use ``begin_nested()`` (SAVEPOINT semantics).
      Otherwise we call ``begin()``.

    Anything else raises :class:`ColumnUpdateError` (``"invalid_connection"``).
    """
    if isinstance(connection, sa.Engine):
        with connection.connect() as conn, conn.begin():
            yield conn
        return
    if isinstance(connection, sa.Connection):
        if connection.in_transaction():
            with connection.begin_nested():
                yield connection
        else:
            with connection.begin():
                yield connection
        return
    raise ColumnUpdateError(
        "invalid_connection",
        table="<unknown>",
        detail=f"expected sa.Engine or sa.Connection, got {type(connection).__name__}",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def update_column(
    *,
    connection: sa.Connection | sa.Engine,
    schema: DatabaseSchema,
    table: str,
    column: str,
    rows: Iterable[tuple[Any, Any]],
) -> ColumnUpdateResult:
    """Patch a single column on an existing dataset.

    Parameters
    ----------
    connection:
        Either a SQLAlchemy :class:`Engine` (the writer manages the
        transaction) or a live :class:`Connection` (the writer wraps the
        work in a nested transaction).
    schema:
        The unified :class:`DatabaseSchema` describing the target dataset.
        Used to look up primary-key columns and to validate that *column*
        exists on *table*.
    table:
        Target table name.
    column:
        Target column name. **Must not be a primary-key column.**
    rows:
        Iterable of ``(pk, new_value)`` pairs.

        * For tables with a **single-column primary key**, ``pk`` is the
          scalar PK value.
        * For tables with a **composite primary key**, ``pk`` must be a
          tuple matching the table's ``primary_key`` arity.

    Returns
    -------
    ColumnUpdateResult
        Carries ``rows_updated`` (the count of rows the writer touched, not
        the number of DB rows that actually changed value) and
        ``duration_seconds``.

    Raises
    ------
    ColumnUpdateError
        With a stable string ``code`` describing the failure reason.
    """
    table_schema = schema.get_table(table)
    if table_schema is None:
        raise ColumnUpdateError("unknown_table", table=table)

    pk_cols: list[str] = list(table_schema.primary_key)
    if not pk_cols:
        raise ColumnUpdateError("no_primary_key", table=table)

    column_names = {c.name for c in table_schema.columns}
    if column not in column_names:
        raise ColumnUpdateError("unknown_column", table=table, detail=column)

    if column in pk_cols:
        raise ColumnUpdateError("pk_column_update", table=table, detail=column)

    # Validate identifiers up-front so unsafe names fail before we open a txn.
    quoted_table = _safe_ident(table)
    quoted_column = _safe_ident(column)
    quoted_pk_cols = [_safe_ident(c) for c in pk_cols]

    composite = len(pk_cols) > 1
    materialised = list(rows)

    start = time.perf_counter()
    if not materialised:
        return ColumnUpdateResult(rows_updated=0, duration_seconds=0.0)

    try:
        with _txn_scope(connection) as conn:
            if composite:
                rows_updated = _execute_composite(
                    conn,
                    quoted_table=quoted_table,
                    quoted_column=quoted_column,
                    quoted_pk_cols=quoted_pk_cols,
                    pk_cols=pk_cols,
                    pairs=materialised,
                    table=table,
                )
            else:
                rows_updated = _execute_single(
                    conn,
                    quoted_table=quoted_table,
                    quoted_column=quoted_column,
                    quoted_pk_col=quoted_pk_cols[0],
                    pairs=materialised,
                )
    except ColumnUpdateError:
        raise
    except Exception as exc:
        raise ColumnUpdateError("update_failed", table=table, detail=type(exc).__name__) from exc

    duration = time.perf_counter() - start
    return ColumnUpdateResult(rows_updated=rows_updated, duration_seconds=duration)


# ---------------------------------------------------------------------------
# Execution paths
# ---------------------------------------------------------------------------


def _execute_single(
    conn: sa.Connection,
    *,
    quoted_table: str,
    quoted_column: str,
    quoted_pk_col: str,
    pairs: list[tuple[Any, Any]],
) -> int:
    """Bulk path — one ``executemany`` for the whole batch."""
    stmt = sa.text(
        f"UPDATE {quoted_table} SET {quoted_column} = :v WHERE {quoted_pk_col} = :pk"  # noqa: S608  # nosec B608 — identifiers validated by _safe_ident
    )
    params = [{"v": value, "pk": pk} for pk, value in pairs]
    conn.execute(stmt, params)
    return len(params)


def _execute_composite(  # noqa: PLR0913 — one keyword arg per validated identifier; tuple-bag would obscure intent
    conn: sa.Connection,
    *,
    quoted_table: str,
    quoted_column: str,
    quoted_pk_cols: list[str],
    pk_cols: list[str],
    pairs: list[tuple[Any, Any]],
    table: str,
) -> int:
    """Fallback path — one statement per row for composite-PK tables."""
    where_clause = " AND ".join(
        f"{quoted} = :pk_{name}" for quoted, name in zip(quoted_pk_cols, pk_cols, strict=True)
    )
    stmt = sa.text(
        f"UPDATE {quoted_table} SET {quoted_column} = :v WHERE {where_clause}"  # noqa: S608  # nosec B608 — identifiers validated by _safe_ident
    )
    count = 0
    for pk_value, new_value in pairs:
        if not isinstance(pk_value, tuple) or len(pk_value) != len(pk_cols):
            raise ColumnUpdateError(
                "pk_arity_mismatch",
                table=table,
                detail=(f"expected tuple of length {len(pk_cols)}, got {pk_value!r}"),
            )
        bind: dict[str, Any] = {"v": new_value}
        for name, value in zip(pk_cols, pk_value, strict=True):
            bind[f"pk_{name}"] = value
        conn.execute(stmt, bind)
        count += 1
    return count


# ---------------------------------------------------------------------------
# Plugin wrapper — registered via [project.entry-points."dbsprout.outputs"]
# ---------------------------------------------------------------------------


class ColumnUpdateWriter:
    """Thin :class:`OutputWriter`-shaped wrapper around :func:`update_column`.

    Exists so the writer can register under ``dbsprout.outputs`` alongside
    the insert writers (``sql``, ``csv``, ``pg_copy`` …) without inventing a
    new Protocol. The semantics are very different from an insert writer —
    this one patches existing rows — but the registry only cares about the
    ``format`` attribute and a callable ``write``.
    """

    format: str = "column_update"

    def write(self, *args: Any, **kwargs: Any) -> ColumnUpdateResult:
        """Forward to :func:`update_column`. Kwargs-only by convention."""
        if args:
            # Defensive: refuse positional args so we don't silently re-order
            # parameters relative to update_column().
            raise TypeError("ColumnUpdateWriter.write() takes only keyword arguments")
        return update_column(**kwargs)


__all__ = [
    "ColumnUpdateError",
    "ColumnUpdateWriter",
    "update_column",
]
