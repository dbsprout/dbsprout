"""YAML schema parser — example dbsprout plugin.

Parses a minimal YAML schema document into a :class:`DatabaseSchema`.
The format is intentionally small (a mapping of table name to columns /
primary_key / foreign_keys); it exists to demonstrate the
``dbsprout.plugins.protocols.SchemaParser`` Protocol, not to be a
production schema language.
"""

from __future__ import annotations

from typing import Any

import yaml

from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    TableSchema,
)

_TYPE_MAP: dict[str, ColumnType] = {
    "int": ColumnType.INTEGER,
    "integer": ColumnType.INTEGER,
    "bigint": ColumnType.BIGINT,
    "smallint": ColumnType.SMALLINT,
    "float": ColumnType.FLOAT,
    "double": ColumnType.FLOAT,
    "decimal": ColumnType.DECIMAL,
    "numeric": ColumnType.DECIMAL,
    "boolean": ColumnType.BOOLEAN,
    "bool": ColumnType.BOOLEAN,
    "varchar": ColumnType.VARCHAR,
    "char": ColumnType.VARCHAR,
    "string": ColumnType.VARCHAR,
    "text": ColumnType.TEXT,
    "date": ColumnType.DATE,
    "datetime": ColumnType.DATETIME,
    "timestamp": ColumnType.TIMESTAMP,
    "time": ColumnType.TIME,
    "uuid": ColumnType.UUID,
    "json": ColumnType.JSON,
    "jsonb": ColumnType.JSON,
}


def _load_mapping(text: str) -> dict[str, Any] | None:
    """Parse *text* as YAML; return the mapping or ``None`` if not a schema."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict) or "tables" not in data:
        return None
    return data


def _build_column(name: str, spec: dict[str, Any]) -> ColumnSchema:
    raw = str(spec.get("type", "")).lower()
    return ColumnSchema(
        name=name,
        data_type=_TYPE_MAP.get(raw, ColumnType.UNKNOWN),
        raw_type=str(spec.get("type", "")),
        nullable=bool(spec.get("nullable", True)),
        primary_key=bool(spec.get("primary_key", False)),
        unique=bool(spec.get("unique", False)),
    )


def _build_table(name: str, spec: dict[str, Any]) -> TableSchema:
    columns = [
        _build_column(col_name, col_spec or {})
        for col_name, col_spec in (spec.get("columns") or {}).items()
    ]
    fks = [
        ForeignKeySchema(
            columns=list(fk["columns"]),
            ref_table=str(fk["ref_table"]),
            ref_columns=list(fk["ref_columns"]),
        )
        for fk in (spec.get("foreign_keys") or [])
    ]
    return TableSchema(
        name=name,
        columns=columns,
        primary_key=list(spec.get("primary_key") or []),
        foreign_keys=fks,
    )


def build_schema(text: str, *, source_file: str | None = None) -> DatabaseSchema:
    """Parse YAML *text* into a :class:`DatabaseSchema`.

    Raises:
        ValueError: the text is not valid YAML or not a mapping.
    """
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        msg = f"invalid YAML: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(data, dict):
        msg = "YAML schema must be a mapping with a 'tables' key"
        raise ValueError(msg)
    tables = [_build_table(name, spec or {}) for name, spec in (data.get("tables") or {}).items()]
    return DatabaseSchema(
        tables=tables,
        dialect=data.get("dialect"),
        source_file=source_file,
    )


class YamlParser:
    """Example ``SchemaParser`` plugin parsing a minimal YAML schema."""

    suffixes: tuple[str, ...] = (".yaml", ".yml")

    def can_parse(self, text: str) -> bool:
        return _load_mapping(text) is not None

    def parse(self, text: str, *, source_file: str | None = None) -> DatabaseSchema:
        return build_schema(text, source_file=source_file)


__all__ = ["YamlParser", "build_schema"]
