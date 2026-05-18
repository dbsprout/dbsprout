"""PostgreSQL COPY output writer — direct insertion via psycopg3.

Formats data for PostgreSQL COPY FROM STDIN (text format) and inserts
directly into a PostgreSQL database at 100K+ rows/sec.
"""

from __future__ import annotations

import json
import math
import time as time_mod
import uuid
from datetime import date, datetime, time
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from dbsprout.output.models import InsertResult

if TYPE_CHECKING:
    from types import ModuleType

    from dbsprout.schema.models import DatabaseSchema

try:
    import psycopg
except ImportError:  # pragma: no cover
    psycopg: ModuleType | None = None  # type: ignore[no-redef]


def format_copy_value(value: Any) -> str:  # noqa: PLR0911
    """Format a Python value for PostgreSQL COPY text format.

    COPY text format rules:
    - NULL → ``\\N``
    - Bool → ``t`` / ``f``
    - Numeric NaN/Inf → ``\\N``
    - Strings: escape ``\\``, tab, newline, carriage return
    - bytes → ``\\\\x`` + hex (bytea hex format)
    - dict/list → JSON string with COPY escaping
    """
    if value is None:
        return "\\N"
    if isinstance(value, bool):
        return "t" if value else "f"
    if isinstance(value, (int, float, Decimal)):
        return _format_numeric(value)
    if isinstance(value, datetime):
        return _escape_copy_str(str(value))
    if isinstance(value, date):
        return str(value)
    if isinstance(value, time):
        return str(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, bytes):
        return f"\\\\x{value.hex()}"
    if isinstance(value, (dict, list)):
        return _escape_copy_str(json.dumps(value, default=str))
    return _escape_copy_str(str(value))


def _format_numeric(value: int | float | Decimal) -> str:
    """Format numeric, converting NaN/Inf to NULL."""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "\\N"
    if isinstance(value, Decimal) and (value.is_nan() or value.is_infinite()):
        return "\\N"
    return str(value)


def _escape_copy_str(value: str) -> str:
    """Escape a string for COPY text format.

    Backslash is escaped first to avoid double-escaping. The ``\\.`` end-of-data
    marker cannot appear because any literal ``\\`` becomes ``\\\\\\\\``.
    """
    return (
        value.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")
    )


def build_copy_data(columns: list[str], rows: list[dict[str, Any]]) -> str:
    """Build tab-delimited text block for COPY FROM STDIN.

    Each row is tab-delimited and newline-terminated.
    Returns empty string for empty rows.
    """
    if not rows:
        return ""
    lines: list[str] = []
    for row in rows:
        vals = "\t".join(format_copy_value(row.get(col)) for col in columns)
        lines.append(vals)
    return "\n".join(lines) + "\n"


_STAGING_PREFIX = "_dbsprout_stg_"


def _staging_name(table_name: str) -> str:
    """Deterministic staging table name for a target table."""
    return f"{_STAGING_PREFIX}{table_name}"


def _build_create_staging(target: str, staging: str) -> Any:
    """CREATE TEMP TABLE <staging> mirroring <target>, auto-dropped at commit."""
    return psycopg.sql.SQL(
        "CREATE TEMP TABLE {stg} (LIKE {tgt} INCLUDING DEFAULTS) ON COMMIT DROP"
    ).format(
        stg=psycopg.sql.Identifier(staging),
        tgt=psycopg.sql.Identifier(target),
    )


def _build_pg_merge_sql(
    target: str, staging: str, columns: list[str], pk_columns: list[str]
) -> Any:
    """INSERT INTO target SELECT ... FROM staging ON CONFLICT (pk) DO UPDATE.

    All-PK tables (no updatable columns) use DO NOTHING.
    """
    col_idents = psycopg.sql.SQL(", ").join(psycopg.sql.Identifier(c) for c in columns)
    conflict_cols = psycopg.sql.SQL(", ").join(psycopg.sql.Identifier(c) for c in pk_columns)
    update_cols = [c for c in columns if c not in pk_columns]
    if update_cols:
        sets = psycopg.sql.SQL(", ").join(
            psycopg.sql.SQL("{c} = EXCLUDED.{c}").format(c=psycopg.sql.Identifier(c))
            for c in update_cols
        )
        action = psycopg.sql.SQL("DO UPDATE SET {sets}").format(sets=sets)
    else:
        action = psycopg.sql.SQL("DO NOTHING")
    return psycopg.sql.SQL(
        "INSERT INTO {tgt} ({cols}) SELECT {cols} FROM {stg} ON CONFLICT ({conflict}) {action}"
    ).format(
        tgt=psycopg.sql.Identifier(target),
        cols=col_idents,
        stg=psycopg.sql.Identifier(staging),
        conflict=conflict_cols,
        action=action,
    )


__all__ = [
    "InsertResult",
    "PgCopyWriter",
    "build_copy_data",
    "format_copy_value",
]


class PgCopyWriter:
    """Write generated data directly to PostgreSQL via COPY FROM STDIN."""

    format: str = "pg_copy"

    def write(  # noqa: PLR0913
        self,
        tables_data: dict[str, list[dict[str, Any]]],
        schema: DatabaseSchema,
        insertion_order: list[str],
        db_url: str,
        batch_size: int = 10_000,
        upsert: bool = False,
    ) -> InsertResult:
        """Insert data via COPY for each table in topological order.

        When ``upsert`` is True and a table has a primary key, data is COPYed
        into a session-scoped ``TEMP`` staging table (auto-dropped at commit),
        then merged into the target via ``INSERT ... SELECT ... ON CONFLICT``.
        Tables without a primary key fall back to a plain COPY into the target,
        matching the SQL writer's ``build_upsert`` behaviour.

        Returns an InsertResult with counts and duration.
        """
        if psycopg is None:
            msg = (
                "psycopg3 is required for direct PostgreSQL insertion. "
                'Install it with: pip install "dbsprout[pg]"'
            )
            raise ImportError(msg)

        start = time_mod.monotonic()
        tables_inserted = 0
        total_rows = 0

        console = Console()
        try:
            with (
                psycopg.connect(db_url) as conn,
                conn.transaction(),
                conn.cursor() as cur,
                Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]{task.description}"),
                    TextColumn("{task.fields[rows]} rows"),
                    TimeElapsedColumn(),
                    console=console,
                    disable=not console.is_terminal,
                ) as progress,
            ):
                for table_name in insertion_order:
                    rows = tables_data.get(table_name, [])
                    if not rows:
                        continue

                    task_id = progress.add_task(
                        description=f"Inserting {table_name}",
                        total=1,
                        rows=len(rows),
                    )

                    table_schema = schema.get_table(table_name)
                    columns = (
                        [col.name for col in table_schema.columns]
                        if table_schema
                        else list(rows[0].keys())
                    )
                    pk_columns = (
                        list(table_schema.primary_key)
                        if upsert and table_schema and table_schema.primary_key
                        else []
                    )

                    _copy_into(cur, table_name, columns, rows, batch_size, pk_columns)

                    tables_inserted += 1
                    total_rows += len(rows)
                    progress.update(task_id, completed=1)

                _reset_sequences(cur, tables_data, schema, insertion_order)
        except ImportError:
            raise
        except Exception as exc:
            msg = (
                f"Database insertion failed: {type(exc).__name__}. "
                "Verify the --db URL and ensure the server is reachable."
            )
            raise RuntimeError(msg) from exc

        duration = time_mod.monotonic() - start
        return InsertResult(
            tables_inserted=tables_inserted,
            total_rows=total_rows,
            duration_seconds=duration,
        )


def _copy_batches(
    cur: Any, dest: str, columns: list[str], rows: list[dict[str, Any]], batch_size: int
) -> None:
    """COPY all batches of ``rows`` into ``dest`` (target or staging table)."""
    copy_sql = psycopg.sql.SQL("COPY {} ({}) FROM STDIN").format(
        psycopg.sql.Identifier(dest),
        psycopg.sql.SQL(", ").join(psycopg.sql.Identifier(c) for c in columns),
    )
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        data = build_copy_data(columns, batch)
        with cur.copy(copy_sql) as copy:
            copy.write(data.encode("utf-8"))


def _copy_into(  # noqa: PLR0913
    cur: Any,
    table_name: str,
    columns: list[str],
    rows: list[dict[str, Any]],
    batch_size: int,
    pk_columns: list[str],
) -> None:
    """COPY rows into ``table_name``.

    With ``pk_columns`` (upsert), COPY into a TEMP staging table then merge via
    ``ON CONFLICT``. Without a PK, COPY straight into the target.
    """
    if not pk_columns:
        _copy_batches(cur, table_name, columns, rows, batch_size)
        return

    staging = _staging_name(table_name)
    cur.execute(_build_create_staging(table_name, staging))
    _copy_batches(cur, staging, columns, rows, batch_size)
    cur.execute(_build_pg_merge_sql(table_name, staging, columns, pk_columns))


def _reset_sequences(
    cur: Any,
    tables_data: dict[str, list[dict[str, Any]]],
    schema: DatabaseSchema,
    insertion_order: list[str],
) -> None:
    """Reset PostgreSQL sequences for autoincrement columns after COPY."""
    for table_name in insertion_order:
        rows = tables_data.get(table_name, [])
        if not rows:
            continue
        table_schema = schema.get_table(table_name)
        if table_schema is None:
            continue
        for col in table_schema.columns:
            if col.autoincrement and col.primary_key:
                int_vals = [v for v in (row.get(col.name) for row in rows) if isinstance(v, int)]
                if not int_vals:
                    continue
                max_val = max(int_vals)
                cur.execute(
                    psycopg.sql.SQL(
                        "SELECT setval(pg_get_serial_sequence({table}, {col}), {val})"
                    ).format(
                        table=psycopg.sql.Literal(table_name),
                        col=psycopg.sql.Literal(col.name),
                        val=psycopg.sql.Literal(max_val),
                    )
                )
