"""MySQL LOAD DATA output writer — direct insertion via pymysql.

Formats data for MySQL LOAD DATA LOCAL INFILE and inserts directly
into a MySQL database at 80K+ rows/sec.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import tempfile
import time as time_mod
import uuid
from datetime import date, datetime, time
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from types import ModuleType

    from dbsprout.schema.models import DatabaseSchema

try:
    import pymysql  # type: ignore[import-not-found,import-untyped,unused-ignore]
except ImportError:  # pragma: no cover
    pymysql: ModuleType | None = None  # type: ignore[no-redef]


def format_load_data_value(value: Any) -> str:  # noqa: PLR0911
    """Format a Python value for MySQL LOAD DATA text format.

    Similar to PG COPY but with MySQL-specific boolean formatting (1/0).
    """
    if value is None:
        return "\\N"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float, Decimal)):
        return _format_numeric(value)
    if isinstance(value, datetime):
        return _escape_str(str(value))
    if isinstance(value, date):
        return str(value)
    if isinstance(value, time):
        return str(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, (dict, list)):
        return _escape_str(json.dumps(value, default=str))
    return _escape_str(str(value))


def _format_numeric(value: int | float | Decimal) -> str:
    """Format numeric, converting NaN/Inf to NULL."""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "\\N"
    if isinstance(value, Decimal) and (value.is_nan() or value.is_infinite()):
        return "\\N"
    return str(value)


def _escape_str(value: str) -> str:
    """Escape a string for MySQL LOAD DATA text format."""
    return (
        value.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")
    )


def build_load_data_content(columns: list[str], rows: list[dict[str, Any]]) -> str:
    """Build tab-delimited text block for LOAD DATA LOCAL INFILE.

    Each row is tab-delimited and newline-terminated.
    Returns empty string for empty rows.
    """
    if not rows:
        return ""
    lines: list[str] = []
    for row in rows:
        vals = "\t".join(format_load_data_value(row.get(col)) for col in columns)
        lines.append(vals)
    return "\n".join(lines) + "\n"


def _parse_mysql_url(url: str) -> dict[str, Any]:
    """Parse a MySQL connection URL into pymysql connect kwargs."""
    parsed = urlparse(url)
    params: dict[str, Any] = {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 3306,
        "database": parsed.path.lstrip("/") if parsed.path else "",
        "local_infile": True,
    }
    if parsed.username:
        params["user"] = parsed.username
    if parsed.password:
        params["password"] = parsed.password
    return params


def _quote_mysql_identifier(name: str) -> str:
    """Quote a MySQL identifier, escaping embedded backticks."""
    return f"`{name.replace('`', '``')}`"


def _write_temp_file(content: str) -> str:
    """Write content to a temp .tsv file and return its path."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".tsv", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(content)
        return tmp.name


def _cleanup_temp_files(paths: list[str]) -> None:
    """Remove temporary files, ignoring errors."""
    for f in paths:
        with contextlib.suppress(OSError):
            os.unlink(f)


_LOCAL_INFILE_ERROR_CODES = frozenset({1148, 3948})


def _is_local_infile_error(exc: Exception) -> bool:
    """Check if an exception is a MySQL local_infile disabled error."""
    if pymysql is not None:
        try:
            if not isinstance(exc, pymysql.err.OperationalError):
                return False
        except TypeError:
            pass  # pymysql may be mocked in tests
    if not (hasattr(exc, "args") and exc.args and isinstance(exc.args[0], int)):
        return False
    return exc.args[0] in _LOCAL_INFILE_ERROR_CODES


class MysqlLoadDataWriter:
    """Write generated data directly to MySQL via LOAD DATA LOCAL INFILE."""

    format: str = "mysql_load_data"

    def write(
        self,
        tables_data: dict[str, list[dict[str, Any]]],
        schema: DatabaseSchema,
        insertion_order: list[str],
        db_url: str,
        batch_size: int = 10_000,
    ) -> Any:
        """Insert data via LOAD DATA for each table in topological order.

        Returns an InsertResult with counts and duration.
        """
        from dbsprout.output.models import InsertResult  # noqa: PLC0415

        if pymysql is None:
            msg = (
                "pymysql is required for direct MySQL insertion. "
                "Install it with: pip install dbsprout[db]"
            )
            raise ImportError(msg)

        start = time_mod.monotonic()
        tables_inserted = 0
        total_rows = 0
        temp_files: list[str] = []

        try:
            conn_params = _parse_mysql_url(db_url)
            try:
                conn = pymysql.connect(**conn_params)
            except Exception as exc:
                msg = (
                    f"MySQL insertion failed: {type(exc).__name__}. "
                    "Verify the --db URL and ensure the server is reachable."
                )
                raise RuntimeError(msg) from exc

            try:
                cur = conn.cursor()
                try:
                    cur.execute("SET FOREIGN_KEY_CHECKS=0")
                    tables_inserted, total_rows = self._load_tables(
                        cur, tables_data, schema, insertion_order, batch_size, temp_files
                    )
                    conn.commit()
                except RuntimeError:
                    raise
                except Exception as exc:
                    conn.rollback()
                    if _is_local_infile_error(exc):
                        msg = (
                            "MySQL LOAD DATA LOCAL INFILE is disabled. "
                            "Enable local_infile on the server with: "
                            "SET GLOBAL local_infile = 1; "
                            "and ensure the client connection uses local_infile=True."
                        )
                        raise RuntimeError(msg) from exc
                    msg = (
                        f"MySQL insertion failed: {type(exc).__name__}. "
                        "Verify the --db URL and ensure the server is reachable."
                    )
                    raise RuntimeError(msg) from exc
                finally:
                    with contextlib.suppress(Exception):
                        cur.execute("SET FOREIGN_KEY_CHECKS=1")
                    cur.close()
            finally:
                conn.close()
        finally:
            _cleanup_temp_files(temp_files)

        duration = time_mod.monotonic() - start
        return InsertResult(
            tables_inserted=tables_inserted,
            total_rows=total_rows,
            duration_seconds=duration,
        )

    def _load_tables(  # noqa: PLR0913
        self,
        cur: Any,
        tables_data: dict[str, list[dict[str, Any]]],
        schema: DatabaseSchema,
        insertion_order: list[str],
        batch_size: int,
        temp_files: list[str],
    ) -> tuple[int, int]:
        """Load all tables via LOAD DATA, returning (tables_inserted, total_rows)."""
        tables_inserted = 0
        total_rows = 0

        for table_name in insertion_order:
            rows = tables_data.get(table_name, [])
            if not rows:
                continue

            table_schema = schema.get_table(table_name)
            columns = (
                [col.name for col in table_schema.columns] if table_schema else list(rows[0].keys())
            )

            for i in range(0, len(rows), batch_size):
                batch = rows[i : i + batch_size]
                content = build_load_data_content(columns, batch)
                tmp_path = _write_temp_file(content)
                temp_files.append(tmp_path)

                quoted_table = _quote_mysql_identifier(table_name)
                quoted_cols = ", ".join(_quote_mysql_identifier(c) for c in columns)
                safe_path = tmp_path.replace("'", "\\'")
                load_sql = (
                    f"LOAD DATA LOCAL INFILE '{safe_path}' "
                    f"INTO TABLE {quoted_table} "
                    "FIELDS TERMINATED BY '\\t' "
                    "LINES TERMINATED BY '\\n' "
                    f"({quoted_cols})"
                )
                cur.execute(load_sql)

            tables_inserted += 1
            total_rows += len(rows)

        return tables_inserted, total_rows
