"""SQL INSERT output writer — PostgreSQL, MySQL, SQLite dialect-aware.

Writes generated data as `.sql` files with batch INSERT statements,
transaction wrapping, and correct per-dialect quoting/escaping.
"""

from __future__ import annotations

import json
import math
import uuid
from datetime import date, datetime, time
from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from dbsprout.schema.models import DatabaseSchema

_DIALECT_CONFIGS: dict[str, dict[str, str]] = {
    "postgresql": {
        "quote": '"',
        "bool_true": "TRUE",
        "bool_false": "FALSE",
        "escape": "standard",
        "upsert_style": "on_conflict",
        "excluded_prefix": "EXCLUDED",
    },
    "mysql": {
        "quote": "`",
        "bool_true": "1",
        "bool_false": "0",
        "escape": "backslash",
        "upsert_style": "on_duplicate_key",
        "excluded_prefix": "",
    },
    "sqlite": {
        "quote": '"',
        "bool_true": "1",
        "bool_false": "0",
        "escape": "standard",
        "upsert_style": "on_conflict",
        "excluded_prefix": "excluded",
    },
    "mssql": {
        "quote": "[",
        "bool_true": "1",
        "bool_false": "0",
        "escape": "standard",
        "upsert_style": "merge",
        "excluded_prefix": "src",
    },
}


def get_dialect_config(dialect: str) -> dict[str, str]:
    """Get formatting config for a SQL dialect."""
    if dialect not in _DIALECT_CONFIGS:
        supported = ", ".join(sorted(_DIALECT_CONFIGS))
        msg = f"Unsupported dialect: {dialect}. Must be one of: {supported}"
        raise ValueError(msg)
    return _DIALECT_CONFIGS[dialect]


def format_value(value: Any, config: dict[str, str]) -> str:
    """Format a Python value as a SQL literal."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return config["bool_true"] if value else config["bool_false"]
    if isinstance(value, (int, float, Decimal)):
        return _format_numeric(value)
    if isinstance(value, (datetime, date, time)):
        return _quote_string(str(value), config)
    if isinstance(value, bytes):
        return f"X'{value.hex()}'"
    return _format_complex(value, config)


def _format_numeric(value: int | float | Decimal) -> str:
    """Format numeric values, converting NaN/Inf to NULL."""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "NULL"
    if isinstance(value, Decimal) and (value.is_nan() or value.is_infinite()):
        return "NULL"
    return str(value)


def _format_complex(value: Any, config: dict[str, str]) -> str:
    """Format UUID, JSON, and other complex types."""
    if isinstance(value, uuid.UUID):
        return _quote_string(str(value), config)
    if isinstance(value, (dict, list)):
        json_str = json.dumps(value, default=str)
        return _quote_string(json_str, config)
    return _quote_string(str(value), config)


def _quote_string(value: str, config: dict[str, str]) -> str:
    """Quote a string value with dialect-appropriate escaping."""
    if config["escape"] == "backslash":
        escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    else:
        escaped = value.replace("'", "''")
    return f"'{escaped}'"


def quote_identifier(name: str, config: dict[str, str]) -> str:
    """Quote a SQL identifier (table/column name)."""
    q = config["quote"]
    if q == "[":
        escaped = name.replace("]", "]]")
        return f"[{escaped}]"
    escaped = name.replace(q, q + q)
    return f"{q}{escaped}{q}"


def _format_value_rows(
    columns: list[str],
    rows: list[dict[str, Any]],
    config: dict[str, str],
) -> str:
    """Format rows as a comma-separated VALUES list."""
    parts = [
        "(" + ", ".join(format_value(row.get(c), config) for c in columns) + ")" for row in rows
    ]
    return ",\n".join(parts)


def build_insert(
    table_name: str,
    columns: list[str],
    rows: list[dict[str, Any]],
    config: dict[str, str],
) -> str:
    """Build a batch INSERT statement."""
    quoted_table = quote_identifier(table_name, config)
    quoted_cols = ", ".join(quote_identifier(c, config) for c in columns)
    values_str = _format_value_rows(columns, rows, config)
    return f"INSERT INTO {quoted_table} ({quoted_cols}) VALUES\n{values_str};\n"  # nosec B608


def build_upsert(
    table_name: str,
    columns: list[str],
    rows: list[dict[str, Any]],
    config: dict[str, str],
    pk_columns: list[str],
) -> str:
    """Build a dialect-aware UPSERT statement.

    Falls back to plain INSERT when ``pk_columns`` is empty.
    """
    if not pk_columns:
        return build_insert(table_name, columns, rows, config)

    style = config.get("upsert_style", "on_conflict")

    if style == "merge":
        return _build_merge(table_name, columns, rows, config, pk_columns)
    if style == "on_duplicate_key":
        return _build_on_duplicate_key(table_name, columns, rows, config, pk_columns)
    return _build_on_conflict(table_name, columns, rows, config, pk_columns)


def _build_on_conflict(
    table_name: str,
    columns: list[str],
    rows: list[dict[str, Any]],
    config: dict[str, str],
    pk_columns: list[str],
) -> str:
    """PostgreSQL / SQLite ON CONFLICT upsert."""
    quoted_table = quote_identifier(table_name, config)
    quoted_cols = ", ".join(quote_identifier(c, config) for c in columns)
    excluded = config.get("excluded_prefix", "EXCLUDED")
    values_str = _format_value_rows(columns, rows, config)
    conflict_cols = ", ".join(quote_identifier(c, config) for c in pk_columns)
    update_cols = [c for c in columns if c not in pk_columns]

    if not update_cols:
        conflict_action = "DO NOTHING"
    else:
        sets = ", ".join(
            f"{quote_identifier(c, config)} = {excluded}.{quote_identifier(c, config)}"
            for c in update_cols
        )
        conflict_action = f"DO UPDATE SET {sets}"  # nosec B608

    return (  # nosec B608
        f"INSERT INTO {quoted_table} ({quoted_cols}) VALUES\n{values_str}\n"  # nosec B608
        f"ON CONFLICT ({conflict_cols}) {conflict_action};\n"
    )


def _build_on_duplicate_key(
    table_name: str,
    columns: list[str],
    rows: list[dict[str, Any]],
    config: dict[str, str],
    pk_columns: list[str],
) -> str:
    """MySQL ON DUPLICATE KEY UPDATE upsert."""
    quoted_table = quote_identifier(table_name, config)
    quoted_cols = ", ".join(quote_identifier(c, config) for c in columns)
    values_str = _format_value_rows(columns, rows, config)
    update_cols = [c for c in columns if c not in pk_columns]

    if not update_cols:
        return (  # nosec B608
            f"INSERT IGNORE INTO {quoted_table} ({quoted_cols}) VALUES\n{values_str};\n"
        )

    sets = ", ".join(
        f"{quote_identifier(c, config)} = VALUES({quote_identifier(c, config)})"
        for c in update_cols
    )
    return (  # nosec B608
        f"INSERT INTO {quoted_table} ({quoted_cols}) VALUES\n{values_str}\n"  # nosec B608
        f"ON DUPLICATE KEY UPDATE {sets};\n"
    )


def _build_merge(
    table_name: str,
    columns: list[str],
    rows: list[dict[str, Any]],
    config: dict[str, str],
    pk_columns: list[str],
) -> str:
    """SQL Server MERGE upsert."""
    quoted_table = quote_identifier(table_name, config)
    quoted_cols = ", ".join(quote_identifier(c, config) for c in columns)
    update_cols = [c for c in columns if c not in pk_columns]
    values_str = _format_value_rows(columns, rows, config)
    on_clause = " AND ".join(
        f"target.{quote_identifier(c, config)} = src.{quote_identifier(c, config)}"
        for c in pk_columns
    )

    parts: list[str] = [
        f"MERGE {quoted_table} AS target",
        f"USING (VALUES\n{values_str}\n) AS src ({quoted_cols})",
        f"ON {on_clause}",
    ]

    if update_cols:
        update_sets = ", ".join(
            f"target.{quote_identifier(c, config)} = src.{quote_identifier(c, config)}"
            for c in update_cols
        )
        parts.append(f"WHEN MATCHED THEN UPDATE SET {update_sets}")  # nosec B608

    src_cols = ", ".join(f"src.{quote_identifier(c, config)}" for c in columns)
    insert_clause = f"WHEN NOT MATCHED THEN INSERT ({quoted_cols}) VALUES ({src_cols});"  # noqa: S608  # nosec B608
    parts.append(insert_clause)

    return "\n".join(parts) + "\n"


class SQLWriter:
    """Write generated data as SQL INSERT files."""

    def write(  # noqa: PLR0913
        self,
        tables_data: dict[str, list[dict[str, Any]]],
        schema: DatabaseSchema,
        insertion_order: list[str],
        output_dir: Path,
        dialect: str = "postgresql",
        batch_size: int = 1000,
        upsert: bool = False,
    ) -> list[Path]:
        """Write SQL INSERT (or UPSERT) files for each table.

        Returns list of written file paths.
        """
        config = get_dialect_config(dialect)
        output_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []

        for idx, table_name in enumerate(insertion_order):
            rows = tables_data.get(table_name, [])
            if not rows:
                continue

            table_schema = schema.get_table(table_name)
            columns = (
                [col.name for col in table_schema.columns] if table_schema else list(rows[0].keys())
            )
            pk_columns = table_schema.primary_key if table_schema and upsert else []

            filename = f"{idx + 1:03d}_{table_name}.sql"
            filepath = output_dir / filename

            content = _build_file(table_name, columns, rows, config, batch_size, pk_columns)
            filepath.write_text(content, encoding="utf-8")
            written.append(filepath)

        return written


def _build_file(  # noqa: PLR0913
    table_name: str,
    columns: list[str],
    rows: list[dict[str, Any]],
    config: dict[str, str],
    batch_size: int,
    pk_columns: list[str] | None = None,
) -> str:
    """Build the full SQL file content with transaction wrapping."""
    safe_name = table_name.replace("\n", " ").replace("\r", " ")
    parts: list[str] = [
        "-- Generated by dbsprout",
        f"-- Table: {safe_name} ({len(rows)} rows)",
        "",
        "BEGIN;",
        "",
    ]

    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        if pk_columns:
            parts.append(build_upsert(table_name, columns, batch, config, pk_columns))
        else:
            parts.append(build_insert(table_name, columns, batch, config))

    parts.append("COMMIT;")
    parts.append("")
    return "\n".join(parts)
