"""Prisma schema parser — converts .prisma files to DatabaseSchema.

Parses Prisma model blocks, scalar fields, relations, enums,
and field attributes (@id, @unique, @default, @relation).
"""

from __future__ import annotations

import re

from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    TableSchema,
)

_PRISMA_TYPE_MAP: dict[str, ColumnType] = {
    "string": ColumnType.VARCHAR,
    "int": ColumnType.INTEGER,
    "bigint": ColumnType.BIGINT,
    "float": ColumnType.FLOAT,
    "decimal": ColumnType.DECIMAL,
    "boolean": ColumnType.BOOLEAN,
    "datetime": ColumnType.TIMESTAMP,
    "json": ColumnType.JSON,
    "bytes": ColumnType.BINARY,
}

_RE_MODEL = re.compile(r"model\s+(\w+)\s*\{([^}]*)\}", re.DOTALL)
_RE_ENUM = re.compile(r"enum\s+(\w+)\s*\{([^}]*)\}")
_RE_FIELD = re.compile(r"^\s*(\w+)\s+(\w+)(\?)?(\[\])?(.*?)$", re.MULTILINE)
_RE_RELATION = re.compile(
    r"@relation\(\s*fields:\s*\[([^\]]*)\]\s*,\s*references:\s*\[([^\]]*)\]\s*\)"
)


def can_parse_prisma(source: str) -> bool:
    """Check if source is a Prisma schema file."""
    lower = source.strip().lower()
    if lower.endswith(".prisma"):
        return True
    return bool(re.search(r"\bmodel\s+\w+\s*\{", source))


def parse_prisma(
    text: str,
    source_file: str | None = None,
) -> DatabaseSchema:
    """Parse a Prisma schema into a DatabaseSchema.

    Raises ``ValueError`` if no model blocks found.
    """
    enums = _parse_enums(text)
    model_names = {m.group(1) for m in _RE_MODEL.finditer(text)}
    models = _parse_models(text, enums, model_names)

    if not models:
        msg = "No model blocks found in Prisma schema"
        raise ValueError(msg)

    return DatabaseSchema(
        tables=models,
        enums=enums,
        source_file=source_file,
    )


def _parse_enums(text: str) -> dict[str, list[str]]:
    """Extract enum blocks."""
    enums: dict[str, list[str]] = {}
    for match in _RE_ENUM.finditer(text):
        name = match.group(1)
        body = match.group(2)
        values = [
            line.strip()
            for line in body.strip().splitlines()
            if line.strip() and not line.strip().startswith("//")
        ]
        enums[name] = values
    return enums


def _parse_models(
    text: str,
    enums: dict[str, list[str]],
    model_names: set[str],
) -> list[TableSchema]:
    """Extract model blocks and build TableSchema objects."""
    tables: list[TableSchema] = []

    for match in _RE_MODEL.finditer(text):
        model_name = match.group(1)
        body = match.group(2)

        columns: list[ColumnSchema] = []
        primary_key: list[str] = []
        foreign_keys: list[ForeignKeySchema] = []

        for field_match in _RE_FIELD.finditer(body):
            field_name = field_match.group(1)
            field_type = field_match.group(2)
            optional = field_match.group(3) == "?"
            is_array = field_match.group(4) == "[]"
            attrs = field_match.group(5) or ""

            # Skip relation fields (type is another model name or array)
            if field_type in model_names or is_array:
                # Check for @relation to extract FK
                rel_match = _RE_RELATION.search(attrs)
                if rel_match:
                    fk_cols = [c.strip() for c in rel_match.group(1).split(",")]
                    ref_cols = [c.strip() for c in rel_match.group(2).split(",")]
                    foreign_keys.append(
                        ForeignKeySchema(
                            columns=fk_cols,
                            ref_table=field_type.lower(),
                            ref_columns=ref_cols,
                        )
                    )
                continue

            # Scalar field → column
            col_type, enum_values = _resolve_type(field_type, enums)
            is_id = "@id" in attrs
            is_unique = "@unique" in attrs
            is_autoincrement = "autoincrement()" in attrs
            default = _extract_default(attrs)

            columns.append(
                ColumnSchema(
                    name=field_name,
                    data_type=col_type,
                    raw_type=field_type,
                    nullable=optional,
                    primary_key=is_id,
                    unique=is_unique,
                    autoincrement=is_autoincrement,
                    default=default,
                    enum_values=enum_values,
                )
            )

            if is_id:
                primary_key.append(field_name)

        tables.append(
            TableSchema(
                name=model_name.lower(),
                columns=columns,
                primary_key=primary_key,
                foreign_keys=foreign_keys,
            )
        )

    return tables


def _resolve_type(
    field_type: str,
    enums: dict[str, list[str]],
) -> tuple[ColumnType, list[str] | None]:
    """Resolve a Prisma type to ColumnType + optional enum values."""
    if field_type in enums:
        return ColumnType.ENUM, enums[field_type]
    col_type = _PRISMA_TYPE_MAP.get(field_type.lower(), ColumnType.UNKNOWN)
    return col_type, None


def _extract_default(attrs: str) -> str | None:
    """Extract @default(...) value from attributes."""
    match = re.search(r"@default\(([^)]*)\)", attrs)
    if match:
        val = match.group(1).strip()
        if val == "autoincrement()":
            return None  # Handled via autoincrement flag
        return val
    return None
