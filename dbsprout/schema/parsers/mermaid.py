"""Mermaid ERD parser — converts erDiagram blocks to DatabaseSchema.

Uses custom regex parsing since no robust Python Mermaid ERD
parser exists. Handles entity blocks, relationships, PK/FK markers.
"""

from __future__ import annotations

import logging
import re
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
    "decimal": ColumnType.DECIMAL,
    "numeric": ColumnType.DECIMAL,
    "boolean": ColumnType.BOOLEAN,
    "bool": ColumnType.BOOLEAN,
    "string": ColumnType.VARCHAR,
    "varchar": ColumnType.VARCHAR,
    "text": ColumnType.TEXT,
    "date": ColumnType.DATE,
    "datetime": ColumnType.DATETIME,
    "timestamp": ColumnType.TIMESTAMP,
    "time": ColumnType.TIME,
    "uuid": ColumnType.UUID,
    "json": ColumnType.JSON,
    "blob": ColumnType.BINARY,
    "binary": ColumnType.BINARY,
}

# Entity block: ENTITY_NAME { ... }
_RE_ENTITY = re.compile(
    r"(\w+)\s*\{([^}]*)\}",
    re.MULTILINE,
)

# Column line inside entity: type name [PK|FK|UK]
_RE_COLUMN = re.compile(
    r"^\s*(\w+)\s+(\w+)(?:\s+(PK|FK|UK))?\s*$",
    re.MULTILINE,
)

# Relationship: ENTITY1 <cardinality> ENTITY2 : "label"
_RE_RELATIONSHIP = re.compile(
    r"(\w+)\s+([|}{o.]+--[|}{o.]+)\s+(\w+)\s*(?::\s*\"?[^\"]*\"?)?",
)

_MERMAID_EXTENSIONS = frozenset({".mermaid", ".mmd"})


def can_parse_mermaid(source: str) -> bool:
    """Check if the source is a Mermaid ERD file or contains erDiagram."""
    lower = source.strip().lower()
    for ext in _MERMAID_EXTENSIONS:
        if lower.endswith(ext):
            return True
    return "erdiagram" in lower


def parse_mermaid(
    text: str,
    source_file: str | None = None,
) -> DatabaseSchema:
    """Parse a Mermaid ERD diagram into a ``DatabaseSchema``.

    Raises ``ValueError`` if the content does not contain ``erDiagram``.
    """
    erd_text = _extract_erdiagram(text)

    entities = _parse_entities(erd_text)
    tables = [_build_table(name, cols) for name, cols in entities.items()]
    tables = _apply_relationships(erd_text, tables)

    return DatabaseSchema(
        tables=tables,
        source_file=source_file,
    )


def _extract_erdiagram(text: str) -> str:
    """Extract erDiagram block from text (handles markdown fences)."""
    # Check for markdown code fence
    fence_match = re.search(
        r"```mermaid\s*\n(.*?)```",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if fence_match:
        text = fence_match.group(1)

    if "erdiagram" not in text.lower():
        msg = "Content does not contain an erDiagram block"
        raise ValueError(msg)

    # Strip the erDiagram keyword line
    return re.sub(r"(?i)erDiagram\s*\n?", "", text)


def _parse_entities(
    erd_text: str,
) -> dict[str, list[dict[str, Any]]]:
    """Extract entity blocks and their columns."""
    entities: dict[str, list[dict[str, Any]]] = {}

    for match in _RE_ENTITY.finditer(erd_text):
        entity_name = match.group(1).lower()
        body = match.group(2)
        columns: list[dict[str, Any]] = []

        for col_match in _RE_COLUMN.finditer(body):
            col_type = col_match.group(1)
            col_name = col_match.group(2)
            marker = col_match.group(3)  # PK, FK, UK, or None

            columns.append(
                {
                    "name": col_name,
                    "type": col_type,
                    "pk": marker == "PK",
                    "fk": marker == "FK",
                    "unique": marker == "UK",
                }
            )

        entities[entity_name] = columns

    return entities


def _build_table(
    name: str,
    columns: list[dict[str, Any]],
) -> TableSchema:
    """Build a TableSchema from parsed entity data."""
    col_schemas: list[ColumnSchema] = []
    primary_key: list[str] = []

    for col in columns:
        col_type = _normalize_type(col["type"])
        is_pk = col["pk"]

        col_schemas.append(
            ColumnSchema(
                name=col["name"],
                data_type=col_type,
                raw_type=col["type"],
                nullable=not is_pk,
                primary_key=is_pk,
                unique=col["unique"],
            )
        )

        if is_pk:
            primary_key.append(col["name"])

    return TableSchema(
        name=name,
        columns=col_schemas,
        primary_key=primary_key,
    )


def _normalize_type(type_str: str) -> ColumnType:
    """Normalize a Mermaid type string to ColumnType."""
    return _TYPE_MAP.get(type_str.lower(), ColumnType.UNKNOWN)


def _apply_relationships(
    erd_text: str,
    tables: list[TableSchema],
) -> list[TableSchema]:
    """Parse relationship lines and apply FKs to tables."""
    table_map = {t.name: t for t in tables}

    for match in _RE_RELATIONSHIP.finditer(erd_text):
        left_name = match.group(1).lower()
        cardinality = match.group(2)
        right_name = match.group(3).lower()

        # Determine which side has the FK (the "many" side)
        fk_table, ref_table = _resolve_fk_direction(
            left_name,
            right_name,
            cardinality,
        )
        if fk_table is None or ref_table is None:
            continue

        source = table_map.get(fk_table)
        if source is None:  # pragma: no cover
            continue

        # Find FK column: look for FK-marked column, or infer {ref_table}_id
        fk_col = _find_fk_column(source, ref_table)
        if fk_col is None:
            continue

        # Find PK of referenced table
        ref = table_map.get(ref_table)
        if ref is None or not ref.primary_key:  # pragma: no cover
            continue

        fk = ForeignKeySchema(
            columns=[fk_col],
            ref_table=ref_table,
            ref_columns=ref.primary_key[:1],
        )

        table_map[fk_table] = source.model_copy(
            update={"foreign_keys": [*source.foreign_keys, fk]},
        )

    return [table_map[t.name] for t in tables]


def _resolve_fk_direction(
    left: str,
    right: str,
    cardinality: str,
) -> tuple[str | None, str | None]:
    """Determine FK table and referenced table from cardinality."""
    # The "many" side (with { or }) gets the FK
    parts = cardinality.split("--", maxsplit=1)
    left_part = parts[0]
    right_part = parts[-1]
    left_many = "{" in left_part
    right_many = "{" in right_part or "}" in right_part

    if right_many and not left_many:
        return right, left  # right is many, FK on right
    if left_many and not right_many:
        return left, right  # left is many, FK on left
    # Both many (many-to-many) — skip
    if left_many and right_many:  # pragma: no cover
        logger.warning("Many-to-many relationship %s <-> %s skipped", left, right)
        return None, None
    # One-to-one: FK on right by convention
    return right, left


def _find_fk_column(
    table: TableSchema,
    ref_table: str,
) -> str | None:
    """Find the FK column name in a table referencing another table."""
    lower_ref = ref_table.lower()
    # Try exact: {ref_table}_id, and singular: {ref_table[:-1]}_id
    candidates = [f"{lower_ref}_id"]
    if lower_ref.endswith("s"):
        candidates.append(f"{lower_ref[:-1]}_id")

    for col in table.columns:
        col_lower = col.name.lower()
        if col_lower in candidates:
            return col.name
        if col_lower.endswith("_id") and lower_ref in col_lower:  # pragma: no cover
            return col.name

    return None
