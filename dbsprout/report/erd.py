"""Build a Mermaid ``erDiagram`` source string from a ``DatabaseSchema``.

Pure, side-effect-free: ``DatabaseSchema`` in, Mermaid source ``str`` out.
The report template (S-081 ``erd`` block) embeds the returned text inside a
``<pre class="mermaid">`` element so the diagram is both machine-renderable
and human-readable, with no external dependencies.

Mermaid ``erDiagram`` lexical rules respected here:

* Entity and attribute names must match ``[A-Za-z0-9_]`` — non-conforming
  characters are replaced with ``_`` (Mermaid does not support quoting
  entity names).
* Each entity gets a ``NAME { ... }`` block of ``<type> <name> [PK|FK]``.
* Relationships use cardinality tokens: ``||--o{`` (1:m), ``||--||`` (1:1),
  ``}o--o{`` (m:m). Labels are quoted free text.
"""

from __future__ import annotations

import re
from itertools import pairwise
from typing import TYPE_CHECKING

from dbsprout.schema.models import ColumnType

if TYPE_CHECKING:
    from dbsprout.schema.models import (
        DatabaseSchema,
        ForeignKeySchema,
        TableSchema,
    )

#: Cap on attribute rows rendered per entity. Large schemas (20+ tables)
#: stay legible — Mermaid auto-lays-out tables, and limiting label volume
#: keeps the SVG free of overlapping text.
MAX_COLUMNS_PER_ENTITY = 15

_UNSAFE_IDENT_RE = re.compile(r"[^A-Za-z0-9_]")

#: ``ColumnType`` -> Mermaid attribute-type token. Every enum value is
#: already a safe ``[a-z]+`` identifier; this map is explicit so a new
#: ``ColumnType`` member fails the exhaustiveness test loudly.
_TYPE_TOKENS: dict[ColumnType, str] = {
    ColumnType.INTEGER: "integer",
    ColumnType.BIGINT: "bigint",
    ColumnType.SMALLINT: "smallint",
    ColumnType.FLOAT: "float",
    ColumnType.DECIMAL: "decimal",
    ColumnType.BOOLEAN: "boolean",
    ColumnType.VARCHAR: "varchar",
    ColumnType.TEXT: "text",
    ColumnType.DATE: "date",
    ColumnType.DATETIME: "datetime",
    ColumnType.TIMESTAMP: "timestamp",
    ColumnType.TIME: "time",
    ColumnType.UUID: "uuid",
    ColumnType.JSON: "json",
    ColumnType.BINARY: "binary",
    ColumnType.ENUM: "enum",
    ColumnType.ARRAY: "array",
    ColumnType.UNKNOWN: "unknown",
}


def _mermaid_type(col_type: ColumnType) -> str:
    """Map a :class:`ColumnType` to a Mermaid-safe attribute-type token."""
    return _TYPE_TOKENS.get(col_type, "unknown")


def _safe_ident(name: str) -> str:
    """Sanitise an identifier to Mermaid's ``[A-Za-z0-9_]`` charset."""
    cleaned = _UNSAFE_IDENT_RE.sub("_", name.strip())
    return cleaned or "_"


def _fk_columns(table: TableSchema) -> set[str]:
    return {col for fk in table.foreign_keys for col in fk.columns}


def _attribute_line(name: str, type_token: str, marker: str) -> str:
    suffix = f" {marker}" if marker else ""
    return f"    {type_token} {_safe_ident(name)}{suffix}"


def _entity_block(table: TableSchema) -> list[str]:
    """Render one ``NAME { ... }`` entity block (column-capped)."""
    pk_cols = set(table.primary_key)
    fk_cols = _fk_columns(table)
    lines = [f"  {_safe_ident(table.name)} {{"]

    shown = table.columns[:MAX_COLUMNS_PER_ENTITY]
    for col in shown:
        if col.primary_key or col.name in pk_cols:
            marker = "PK"
        elif col.name in fk_cols:
            marker = "FK"
        else:
            marker = ""
        lines.append(_attribute_line(col.name, _mermaid_type(col.data_type), marker))

    hidden = len(table.columns) - len(shown)
    if hidden > 0:
        lines.append(f'    string _more "(+{hidden} more)"')
    lines.append("  }")
    return lines


def _cardinality(table: TableSchema, fk: ForeignKeySchema) -> str:
    """Pick a Mermaid cardinality token for a non-junction FK edge.

    Junction tables are handled in :func:`_relationship_lines` (m:m) and
    never reach here.

    * unique / single-col-PK FK column -> ``||--||`` (one-to-one)
    * otherwise -> ``||--o{`` (one-to-many)
    """
    fk_col = table.get_column(fk.columns[0]) if len(fk.columns) == 1 else None
    if fk_col is not None:
        single_col_pk = table.primary_key == [fk_col.name]
        if fk_col.unique or single_col_pk:
            return "||--||"
    return "||--o{"


def _relationship_lines(schema: DatabaseSchema) -> list[str]:
    lines: list[str] = []
    seen_m2m: set[tuple[str, str]] = set()
    for table in schema.tables:
        if table.is_junction_table:
            parents = sorted(_safe_ident(p) for p in table.fk_parent_tables)
            for left, right in pairwise(parents):
                key = (left, right)
                if key in seen_m2m:
                    continue
                seen_m2m.add(key)
                lines.append(f'  {left} }}o--o{{ {right} : "{_safe_ident(table.name)}"')
            continue
        for fk in table.foreign_keys:
            parent = _safe_ident(fk.ref_table)
            child = _safe_ident(table.name)
            token = _cardinality(table, fk)
            label = _safe_ident(fk.name) if fk.name else "references"
            lines.append(f'  {parent} {token} {child} : "{label}"')
    return lines


def build_erd_mermaid(schema: DatabaseSchema) -> str:
    """Return a Mermaid ``erDiagram`` source string for ``schema``."""
    lines: list[str] = ["erDiagram"]
    for table in schema.tables:
        lines.extend(_entity_block(table))
    lines.extend(_relationship_lines(schema))
    return "\n".join(lines)
