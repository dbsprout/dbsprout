"""Parquet output writer via Polars.

Writes generated data as `.parquet` files with explicit schema
mapping and Snappy compression.
"""

from __future__ import annotations

import json
import math
import uuid
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal

from dbsprout.schema.models import ColumnType

if TYPE_CHECKING:
    from pathlib import Path

    from dbsprout.schema.models import DatabaseSchema, TableSchema

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]

_COMPRESSION: Literal["lz4", "uncompressed", "snappy", "gzip", "brotli", "zstd"] = "snappy"

_COLUMN_TYPE_MAP: dict[ColumnType, Any] = {
    ColumnType.INTEGER: "Int64",
    ColumnType.BIGINT: "Int64",
    ColumnType.SMALLINT: "Int16",
    ColumnType.FLOAT: "Float64",
    ColumnType.DECIMAL: "Float64",
    ColumnType.BOOLEAN: "Boolean",
    ColumnType.VARCHAR: "Utf8",
    ColumnType.TEXT: "Utf8",
    ColumnType.DATE: "Date",
    ColumnType.DATETIME: "Datetime",
    ColumnType.TIMESTAMP: "Datetime",
    ColumnType.TIME: "Time",
    ColumnType.UUID: "Utf8",
    ColumnType.JSON: "Utf8",
    ColumnType.BINARY: "Binary",
    ColumnType.ENUM: "Utf8",
    ColumnType.ARRAY: "Utf8",
    ColumnType.UNKNOWN: "Utf8",
}


def _polars_dtype(column_type: ColumnType) -> Any:
    """Map a ColumnType to a Polars data type."""
    name = _COLUMN_TYPE_MAP.get(column_type, "Utf8")
    if name == "Datetime":
        return pl.Datetime("us")
    return getattr(pl, name)


def _build_schema(table_schema: TableSchema) -> dict[str, Any]:
    """Build a Polars schema dict from a TableSchema."""
    return {col.name: _polars_dtype(col.data_type) for col in table_schema.columns}


def _is_nan_or_inf(value: Any) -> bool:
    """Check if a numeric value is NaN or Inf."""
    if isinstance(value, float):
        return math.isnan(value) or math.isinf(value)
    if isinstance(value, Decimal):
        return value.is_nan() or value.is_infinite()
    return False


def _sanitize_value(value: Any) -> Any:
    """Sanitize a Python value for Polars DataFrame construction."""
    if value is None:
        return None
    if _is_nan_or_inf(value):
        return None
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, default=str)
    if isinstance(value, Decimal):
        return float(value)
    return value


def _sanitize_rows(
    rows: list[dict[str, Any]],
    schema: dict[str, Any],
) -> dict[str, list[Any]]:
    """Convert list-of-dicts to column-oriented dict with sanitized values."""
    columns: dict[str, list[Any]] = {col: [] for col in schema}
    for row in rows:
        for col in schema:
            columns[col].append(_sanitize_value(row.get(col)))
    return columns


class ParquetWriter:
    """Write generated data as Parquet files via Polars."""

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

            polars_schema = _build_schema(table_schema)
            filename = f"{idx + 1:03d}_{table_name}.parquet"
            filepath = output_dir / filename

            if not rows:
                df = pl.DataFrame(schema=polars_schema)
            else:
                col_data = _sanitize_rows(rows, polars_schema)
                df = pl.DataFrame(col_data, schema=polars_schema)

            df.write_parquet(filepath, compression=_COMPRESSION)
            written.append(filepath)

        return written
