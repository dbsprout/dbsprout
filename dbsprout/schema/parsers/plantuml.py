"""PlantUML ERD parser — converts PlantUML entity diagrams to DatabaseSchema.

Uses custom regex parsing for PlantUML entity-relationship syntax.
Handles entity blocks, stereotypes (<<PK>>, <<FK>>), and relationships.
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

_RE_ENTITY = re.compile(
    r'entity\s+"?(\w+)"?\s*(?:as\s+(\w+))?\s*\{([^}]*)\}',
    re.IGNORECASE | re.DOTALL,
)

# Column: *name : type <<STEREOTYPE>>
_RE_COLUMN = re.compile(
    r"^\s*(\*?)(\w+)\s*:\s*(\w+)(?:\s*<<(\w+)>>)?",
    re.MULTILINE,
)

_RE_RELATIONSHIP = re.compile(
    r"(\w+)\s+([|}{o.]+--[|}{o.]+)\s+(\w+)",
)

_PLANTUML_EXTENSIONS = frozenset({".puml", ".plantuml", ".pu"})


def can_parse_plantuml(source: str) -> bool:
    """Check if source is a PlantUML file or contains entity definitions."""
    lower = source.strip().lower()
    for ext in _PLANTUML_EXTENSIONS:
        if lower.endswith(ext):
            return True
    return "@startuml" in lower


def parse_plantuml(
    text: str,
    source_file: str | None = None,
) -> DatabaseSchema:
    """Parse PlantUML entity-relationship diagram into DatabaseSchema.

    Raises ``ValueError`` if no entity blocks found.
    """
    content = _strip_wrappers(text)

    entities = _parse_entities(content)
    if not entities:
        msg = "No entity blocks found in PlantUML content"
        raise ValueError(msg)

    tables = [_build_table(name, cols) for name, cols in entities.items()]
    tables = _apply_relationships(content, tables)

    return DatabaseSchema(
        tables=tables,
        source_file=source_file,
    )


def _strip_wrappers(text: str) -> str:
    """Remove @startuml/@enduml wrappers."""
    result = re.sub(r"@startuml\b[^\n]*\n?", "", text, flags=re.IGNORECASE)
    return re.sub(r"@enduml\b[^\n]*\n?", "", result, flags=re.IGNORECASE)


def _parse_entities(text: str) -> dict[str, list[dict[str, Any]]]:
    """Extract entity blocks and their columns."""
    entities: dict[str, list[dict[str, Any]]] = {}

    for match in _RE_ENTITY.finditer(text):
        entity_name = match.group(1).lower()
        body = match.group(3)
        columns: list[dict[str, Any]] = []

        for col_match in _RE_COLUMN.finditer(body):
            required = col_match.group(1) == "*"
            col_name = col_match.group(2)
            col_type = col_match.group(3)
            stereotype = col_match.group(4)  # PK, FK, UK, or None

            columns.append(
                {
                    "name": col_name,
                    "type": col_type,
                    "required": required,
                    "pk": stereotype == "PK" if stereotype else False,
                    "fk": stereotype == "FK" if stereotype else False,
                    "unique": stereotype == "UK" if stereotype else False,
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
                nullable=not col["required"] and not is_pk,
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
    """Normalize a PlantUML type string to ColumnType."""
    return _TYPE_MAP.get(type_str.lower(), ColumnType.UNKNOWN)


def _apply_relationships(
    text: str,
    tables: list[TableSchema],
) -> list[TableSchema]:
    """Parse relationship lines and apply FKs."""
    table_map = {t.name: t for t in tables}

    for match in _RE_RELATIONSHIP.finditer(text):
        left = match.group(1).lower()
        cardinality = match.group(2)
        right = match.group(3).lower()

        fk_table, ref_table = _resolve_fk_direction(left, right, cardinality)
        if fk_table is None or ref_table is None:  # pragma: no cover
            continue

        source = table_map.get(fk_table)
        ref = table_map.get(ref_table)
        if source is None or ref is None or not ref.primary_key:  # pragma: no cover
            continue

        fk_col = _find_fk_column(source, ref_table)
        if fk_col is None:  # pragma: no cover
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
    """Determine FK table from cardinality markers."""
    parts = cardinality.split("--", maxsplit=1)
    left_part = parts[0]
    right_part = parts[-1]
    left_many = "{" in left_part or "}" in left_part
    right_many = "{" in right_part or "}" in right_part

    if right_many and not left_many:
        return right, left
    if left_many and not right_many:
        return left, right
    if left_many and right_many:  # pragma: no cover
        return None, None
    return right, left


def _find_fk_column(table: TableSchema, ref_table: str) -> str | None:
    """Find FK column in table referencing ref_table."""
    lower_ref = ref_table.lower()
    candidates = [f"{lower_ref}_id"]
    if lower_ref.endswith("s"):
        candidates.append(f"{lower_ref[:-1]}_id")

    for col in table.columns:
        if col.name.lower() in candidates:
            return col.name

    return None
