"""Generic SQLAlchemy batch writer -- dialect-aware fallback for direct insertion."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa

from dbsprout.output.models import InsertResult

if TYPE_CHECKING:
    from dbsprout.schema.models import DatabaseSchema

_MSSQL_PARAM_LIMIT = 2100


class SaBatchWriter:
    """Generic batch INSERT writer using SQLAlchemy text() + executemany."""

    def write(
        self,
        tables_data: dict[str, list[dict[str, Any]]],
        schema: DatabaseSchema,  # noqa: ARG002
        insertion_order: list[str],
        db_url: str,
        batch_size: int = 10_000,
        *,
        _engine_override: sa.Engine | None = None,
    ) -> InsertResult:
        """Insert rows into a database using parameterised batch INSERTs.

        Parameters
        ----------
        tables_data:
            Mapping of table name to list of row dicts.
        schema:
            The unified database schema (used for type info by callers).
        insertion_order:
            Table names in FK-safe insertion order.
        db_url:
            SQLAlchemy connection URL.
        batch_size:
            Maximum rows per ``executemany`` call (auto-capped for MSSQL).
        _engine_override:
            **Test-only.** When provided, the writer reuses this engine
            instead of creating one from *db_url*.
        """
        if not tables_data:
            return InsertResult(tables_inserted=0, total_rows=0, duration_seconds=0.0)

        owns_engine = _engine_override is None
        engine = _engine_override or sa.create_engine(db_url)
        dialect = engine.dialect.name
        start = time.perf_counter()
        tables_inserted = 0
        total_rows = 0

        try:
            with engine.connect() as conn:
                self._apply_pragmas(conn, dialect, db_url)

                for table_name in insertion_order:
                    rows = tables_data.get(table_name, [])
                    if not rows:
                        continue

                    columns = list(rows[0].keys())
                    n_cols = len(columns)
                    effective_batch = self._compute_batch_size(dialect, n_cols, batch_size)

                    col_list = ", ".join(columns)
                    param_list = ", ".join(f":{c}" for c in columns)
                    insert_sql = sa.text(
                        f"INSERT INTO {table_name} ({col_list}) "  # noqa: S608  # nosec B608
                        f"VALUES ({param_list})"
                    )

                    for i in range(0, len(rows), effective_batch):
                        batch = rows[i : i + effective_batch]
                        conn.execute(insert_sql, batch)

                    tables_inserted += 1
                    total_rows += len(rows)

                conn.commit()
        finally:
            if owns_engine:
                engine.dispose()

        duration = time.perf_counter() - start
        return InsertResult(
            tables_inserted=tables_inserted,
            total_rows=total_rows,
            duration_seconds=duration,
        )

    @staticmethod
    def _compute_batch_size(dialect: str, n_cols: int, user_batch: int) -> int:
        """Return effective batch size, respecting MSSQL parameter limits."""
        if dialect == "mssql":
            return min(user_batch, _MSSQL_PARAM_LIMIT // max(n_cols, 1))
        return user_batch

    @staticmethod
    def _apply_pragmas(conn: sa.Connection, dialect: str, db_url: str) -> None:
        """Apply performance pragmas for SQLite file-based databases."""
        if dialect != "sqlite":
            return
        lower_url = db_url.lower()
        if ":memory:" in lower_url or "mode=memory" in lower_url:
            return
        conn.execute(sa.text("PRAGMA journal_mode=WAL"))
        conn.execute(sa.text("PRAGMA synchronous=OFF"))
        conn.commit()
