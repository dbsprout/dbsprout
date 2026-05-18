"""Parquet output writer via Polars.

Writes generated data as `.parquet` files with explicit schema
mapping and Snappy compression.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal

from dbsprout.output._perms import restrict_file_permissions
from dbsprout.schema.models import ColumnType

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType

    from dbsprout.schema.models import DatabaseSchema, TableSchema

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl: ModuleType | None = None  # type: ignore[no-redef]

_COMPRESSION: Literal["lz4", "uncompressed", "snappy", "gzip", "brotli", "zstd"] = "snappy"
_SAFE_TABLE_NAME = re.compile(r"^[A-Za-z0-9_.\- ]+$")

_COLUMN_TYPE_MAP: dict[ColumnType, Any] = {
    ColumnType.INTEGER: "Int64",
    ColumnType.BIGINT: "Int64",
    ColumnType.SMALLINT: "Int16",
    ColumnType.FLOAT: "Float64",
    ColumnType.DECIMAL: "Float64",
    ColumnType.BOOLEAN: "Boolean",
    ColumnType.VARCHAR: "String",
    ColumnType.TEXT: "String",
    ColumnType.DATE: "Date",
    ColumnType.DATETIME: "Datetime",
    ColumnType.TIMESTAMP: "Datetime",
    ColumnType.TIME: "Time",
    ColumnType.UUID: "String",
    ColumnType.JSON: "String",
    ColumnType.BINARY: "Binary",
    ColumnType.ENUM: "String",
    ColumnType.ARRAY: "String",
    ColumnType.UNKNOWN: "String",
}


_DT_TYPES = frozenset({ColumnType.DATETIME, ColumnType.TIMESTAMP})


def _polars_dtype(column_type: ColumnType, *, tz: str | None = None) -> Any:
    """Map a ColumnType to a Polars data type.

    Must only be called when ``pl`` is not None (i.e. after the import guard).
    A non-empty ``tz`` makes a Datetime column timezone-aware so tz-aware
    inputs round-trip without being silently flattened to naive ``us``.
    """
    name = _COLUMN_TYPE_MAP[column_type]
    if name == "Datetime":
        return pl.Datetime("us", time_zone=tz) if tz else pl.Datetime("us")
    return getattr(pl, name)


def _detect_tz(rows: list[dict[str, Any]], column: str) -> str | None:
    """Return the tzinfo name of the first tz-aware datetime in *column*."""
    for row in rows:
        value = row.get(column)
        if isinstance(value, datetime) and value.tzinfo is not None:
            return str(value.tzinfo)
    return None


def _build_schema(table_schema: TableSchema, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a Polars schema dict from a TableSchema.

    Datetime/timestamp columns whose generated values are tz-aware get a
    timezone-qualified ``pl.Datetime`` so the timezone is preserved on disk.
    """
    out: dict[str, Any] = {}
    for col in table_schema.columns:
        tz = _detect_tz(rows, col.name) if col.data_type in _DT_TYPES else None
        out[col.name] = _polars_dtype(col.data_type, tz=tz)
    return out


def _is_nan_or_inf(value: Any) -> bool:
    """Check if a numeric value is NaN or Inf."""
    if isinstance(value, float):
        return math.isnan(value) or math.isinf(value)
    if isinstance(value, Decimal):
        return value.is_nan() or value.is_infinite()
    return False


def _sanitize_value(value: Any) -> Any:
    """Sanitize a Python value for Polars DataFrame construction."""
    if value is None or _is_nan_or_inf(value):
        return None
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (dict, list, set, frozenset)):
        serializable = sorted(value, key=str) if isinstance(value, (set, frozenset)) else value
        try:
            return json.dumps(serializable, default=str)
        except (ValueError, TypeError):
            return None
    if isinstance(value, Decimal):
        return float(value)
    return value


def _sanitize_rows(
    rows: list[dict[str, Any]],
    schema: dict[str, Any],
) -> dict[str, list[Any]]:
    """Convert list-of-dicts to column-oriented dict with sanitized values."""
    return {col: [_sanitize_value(row.get(col)) for row in rows] for col in schema}


def _build_dataframe(
    col_data: dict[str, list[Any]],
    polars_schema: dict[str, Any],
    table_name: str,
) -> Any:
    """Build a Polars DataFrame, scrubbing cell values from any error.

    Polars surfaces offending cell values in type-mismatch ``TypeError`` /
    ``OverflowError`` messages (e.g. a SMALLINT overflow echoes the raw
    integer). Re-raise with only table / column / dtype so generated data
    never leaks into logs or tracebacks.
    """
    try:
        return pl.DataFrame(col_data, schema=polars_schema)
    except (TypeError, OverflowError) as exc:
        for name, dtype in polars_schema.items():
            try:
                pl.DataFrame({name: col_data[name]}, schema={name: dtype})
            except (TypeError, OverflowError, ValueError):
                msg = (
                    f"Failed to build Parquet column {table_name!r}.{name!r} "
                    f"as {dtype!r}: a value does not fit the column type "
                    f"({type(exc).__name__})."
                )
                raise ValueError(msg) from None
        msg = f"Failed to build Parquet table {table_name!r} ({type(exc).__name__})."
        raise ValueError(msg) from None


class ParquetWriter:
    """Write generated data as Parquet files via Polars."""

    format: str = "parquet"

    def write(
        self,
        tables_data: dict[str, list[dict[str, Any]]],
        schema: DatabaseSchema,
        insertion_order: list[str],
        output_dir: Path,
    ) -> list[Path]:
        """Write Parquet files for each table.

        Returns list of written file paths.
        """
        if pl is None:
            msg = (
                "polars is required for Parquet output. Install it with: pip install dbsprout[data]"
            )
            raise ImportError(msg)

        output_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []

        for idx, table_name in enumerate(insertion_order):
            rows = tables_data.get(table_name, [])
            table_schema = schema.get_table(table_name)

            if table_schema is None:
                continue

            safe_name = table_name
            if not _SAFE_TABLE_NAME.match(table_name):
                safe_name = re.sub(r"[^\w]", "_", table_name)
            polars_schema = _build_schema(table_schema, rows)
            filename = f"{idx + 1:03d}_{safe_name}.parquet"
            filepath = output_dir / filename

            if not filepath.resolve().is_relative_to(output_dir.resolve()):
                continue

            if not rows:
                df = pl.DataFrame(schema=polars_schema)
            else:
                col_data = _sanitize_rows(rows, polars_schema)
                df = _build_dataframe(col_data, polars_schema, table_name)

            df.write_parquet(filepath, compression=_COMPRESSION)
            restrict_file_permissions(filepath)
            written.append(filepath)

        return written
