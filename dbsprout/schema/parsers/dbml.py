"""DBML file parser — converts DBML (dbdiagram.io) to DatabaseSchema.

Uses ``pydbml`` to parse DBML syntax, then translates the AST
into the unified ``DatabaseSchema`` Pydantic model.
"""

from __future__ import annotations

import logging
from typing import Any

from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    TableSchema,
)

logger = logging.getLogger(__name__)

_TYPE_MAP: dict[str, ColumnType] = {
    "int": ColumnType.INTEGER,
    "integer": ColumnType.INTEGER,
    "bigint": ColumnType.BIGINT,
    "smallint": ColumnType.SMALLINT,
    "float": ColumnType.FLOAT,
    "double": ColumnType.FLOAT,
    "real": ColumnType.FLOAT,
    "decimal": ColumnType.DECIMAL,
    "numeric": ColumnType.DECIMAL,
    "boolean": ColumnType.BOOLEAN,
    "bool": ColumnType.BOOLEAN,
    "varchar": ColumnType.VARCHAR,
    "char": ColumnType.VARCHAR,
    "text": ColumnType.TEXT,
    "string": ColumnType.VARCHAR,
    "date": ColumnType.DATE,
    "datetime": ColumnType.DATETIME,
    "timestamp": ColumnType.TIMESTAMP,
    "time": ColumnType.TIME,
    "uuid": ColumnType.UUID,
    "json": ColumnType.JSON,
    "jsonb": ColumnType.JSON,
    "blob": ColumnType.BINARY,
    "binary": ColumnType.BINARY,
    "bytea": ColumnType.BINARY,
}


def can_parse_dbml(source: str) -> bool:
    """Check if the source is a DBML file or contains DBML syntax."""
    if source.strip().lower().endswith(".dbml"):
        return True
    # Check for DBML keywords in content
    lower = source.lower()
    return "table " in lower and "{" in lower


def parse_dbml(
    text: str,
    source_file: str | None = None,
) -> DatabaseSchema:
    """Parse a DBML string into a ``DatabaseSchema``.

    Raises ``ValueError`` on malformed DBML.
    """
    try:
        from pydbml import PyDBML  # type: ignore[import-untyped]  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        msg = "pydbml is required for DBML parsing. Install it with: pip install dbsprout"
        raise ImportError(msg) from None

    try:
        parsed = PyDBML(text)
    except Exception as exc:
        msg = f"Failed to parse DBML: {exc}"
        raise ValueError(msg) from exc

    # Extract enums (preserve original case in dict, build lowercase lookup)
    enums: dict[str, list[str]] = {}
    enum_lower: dict[str, list[str]] = {}
    for enum in parsed.enums:
        values = [item.name for item in enum.items]
        enums[enum.name] = values
        enum_lower[enum.name.lower()] = values

    # Convert tables
    tables: list[TableSchema] = []
    for dbml_table in parsed.tables:
        tables.append(_convert_table(dbml_table, enum_lower))

    # Convert refs to FKs
    tables = _apply_refs(parsed.refs, tables)

    return DatabaseSchema(
        tables=tables,
        enums=enums,
        source_file=source_file,
    )


def _convert_table(
    dbml_table: Any,
    enums: dict[str, list[str]],
) -> TableSchema:
    """Convert a pydbml Table to a TableSchema."""
    columns: list[ColumnSchema] = []
    primary_key: list[str] = []

    for col in dbml_table.columns:
        raw_type = str(col.type).lower().strip()
        col_type = _normalize_type(raw_type)
        enum_values = enums.get(raw_type)

        is_pk = bool(col.pk)
        is_autoincrement = _is_autoincrement(col)

        columns.append(
            ColumnSchema(
                name=col.name,
                data_type=ColumnType.ENUM if enum_values else col_type,
                raw_type=str(col.type),
                nullable=not col.not_null and not is_pk,
                primary_key=is_pk,
                unique=bool(col.unique),
                autoincrement=is_autoincrement,
                default=str(col.default) if col.default is not None else None,
                enum_values=enum_values,
                comment=str(col.note) if hasattr(col, "note") and col.note else None,
            )
        )

        if is_pk:
            primary_key.append(col.name)

    note = None
    if hasattr(dbml_table, "note") and dbml_table.note:
        note = str(dbml_table.note)

    return TableSchema(
        name=dbml_table.name,
        columns=columns,
        primary_key=primary_key,
        comment=note,
    )


def _normalize_type(raw_type: str) -> ColumnType:
    """Normalize a DBML type string to ColumnType."""
    # Strip parenthesized params: varchar(255) → varchar
    base = raw_type.split("(", maxsplit=1)[0].strip().lower()
    return _TYPE_MAP.get(base, ColumnType.UNKNOWN)


def _is_autoincrement(col: Any) -> bool:
    """Check if a column has autoincrement/increment setting."""
    return bool(getattr(col, "autoinc", False))


def _apply_refs(
    refs: Any,
    tables: list[TableSchema],
) -> list[TableSchema]:
    """Convert pydbml refs to ForeignKeySchema and return updated tables."""
    table_map = {t.name: t for t in tables}

    for ref in refs:
        from_cols = ref.col1
        to_cols = ref.col2

        if ref.type == "<":
            from_cols, to_cols = to_cols, from_cols
        elif ref.type == "<>":
            logger.warning(
                "Many-to-many ref (%s <> %s) skipped — junction tables not auto-created",
                from_cols[0].table.name if from_cols else "?",
                to_cols[0].table.name if to_cols else "?",
            )
            continue

        if not from_cols or not to_cols:
            continue

        from_table_name = from_cols[0].table.name
        to_table_name = to_cols[0].table.name

        fk = ForeignKeySchema(
            columns=[c.name for c in from_cols],
            ref_table=to_table_name,
            ref_columns=[c.name for c in to_cols],
        )

        source = table_map.get(from_table_name)
        if source is not None:
            table_map[from_table_name] = source.model_copy(
                update={"foreign_keys": [*source.foreign_keys, fk]},
            )

    return [table_map[t.name] for t in tables]
