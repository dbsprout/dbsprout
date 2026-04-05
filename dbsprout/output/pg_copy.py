"""PostgreSQL COPY output writer — direct insertion via psycopg3.

Formats data for PostgreSQL COPY FROM STDIN (text format) and inserts
directly into a PostgreSQL database at 100K+ rows/sec.
"""

from __future__ import annotations

import json
import math
import time as time_mod
import uuid
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from typing import TYPE_CHECKING, Any

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
    """Escape a string for COPY text format."""
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


@dataclass(frozen=True)
class InsertResult:
    """Result of a direct database insertion."""

    tables_inserted: int
    total_rows: int
    duration_seconds: float


class PgCopyWriter:
    """Write generated data directly to PostgreSQL via COPY FROM STDIN."""

    def write(
        self,
        tables_data: dict[str, list[dict[str, Any]]],
        schema: DatabaseSchema,
        insertion_order: list[str],
        db_url: str,
        batch_size: int = 10_000,
    ) -> InsertResult:
        """Insert data via COPY for each table in topological order.

        Returns an InsertResult with counts and duration.
        """
        if psycopg is None:
            msg = (
                "psycopg3 is required for direct PostgreSQL insertion. "
                "Install it with: pip install dbsprout[pg]"
            )
            raise ImportError(msg)

        start = time_mod.monotonic()
        tables_inserted = 0
        total_rows = 0

        conn = psycopg.connect(db_url)
        try:
            with conn.transaction(), conn.cursor() as cur:
                for table_name in insertion_order:
                    rows = tables_data.get(table_name, [])
                    if not rows:
                        continue

                    table_schema = schema.get_table(table_name)
                    columns = (
                        [col.name for col in table_schema.columns]
                        if table_schema
                        else list(rows[0].keys())
                    )

                    quoted_cols = ", ".join(f'"{c}"' for c in columns)
                    copy_sql = f'COPY "{table_name}" ({quoted_cols}) FROM STDIN'

                    for i in range(0, len(rows), batch_size):
                        batch = rows[i : i + batch_size]
                        data = build_copy_data(columns, batch)
                        with cur.copy(copy_sql) as copy:
                            copy.write(data.encode("utf-8"))

                    tables_inserted += 1
                    total_rows += len(rows)

                # Reset sequences for autoincrement PKs
                _reset_sequences(cur, tables_data, schema, insertion_order)
        finally:
            conn.close()

        duration = time_mod.monotonic() - start
        return InsertResult(
            tables_inserted=tables_inserted,
            total_rows=total_rows,
            duration_seconds=duration,
        )


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
                max_val = max(row.get(col.name, 0) for row in rows)
                cur.execute(
                    f"SELECT setval(pg_get_serial_sequence('{table_name}', "
                    f"'{col.name}'), {max_val})"
                )
