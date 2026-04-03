"""Schema redaction for the ``redacted`` privacy tier.

Replaces table and column names with deterministic SHA-256 hashes,
strips comments, and preserves all structural information (types,
constraints, FK references).
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dbsprout.schema.models import DatabaseSchema, TableSchema


def redact_schema(schema: DatabaseSchema) -> DatabaseSchema:
    """Return a redacted copy of *schema* with hashed names and no comments.

    - Table names → ``tbl_<hash8>``
    - Column names → ``col_<hash8>``
    - Table/column comments → ``None``
    - FK references updated to use consistent hashed names
    - All other fields (types, nullable, PK, precision, etc.) preserved

    The hash is ``sha256(name)[:12]``, deterministic but reversible for common
    names via dictionary attack. Salted hashing is planned for S-040.
    """
    table_map = {t.name: _hash_table(t.name) for t in schema.tables}
    col_maps: dict[str, dict[str, str]] = {}
    for table in schema.tables:
        col_maps[table.name] = {c.name: _hash_column(c.name) for c in table.columns}

    redacted_tables = [_redact_table(table, table_map, col_maps) for table in schema.tables]

    return schema.model_copy(
        update={
            "tables": redacted_tables,
        },
    )


def _hash_name(name: str, prefix: str) -> str:
    """Hash a name with SHA-256 and prepend a prefix."""
    digest = hashlib.sha256(name.encode()).hexdigest()[:12]
    return f"{prefix}{digest}"


def _hash_table(name: str) -> str:
    return _hash_name(name, "tbl_")


def _hash_column(name: str) -> str:
    return _hash_name(name, "col_")


def _redact_table(
    table: TableSchema,
    table_map: dict[str, str],
    col_maps: dict[str, dict[str, str]],
) -> TableSchema:
    """Redact a single table: hash names, strip comments, update FKs."""
    from dbsprout.schema.models import ForeignKeySchema  # noqa: PLC0415

    col_map = col_maps[table.name]

    redacted_columns = [
        col.model_copy(
            update={
                "name": col_map[col.name],
                "comment": None,
            },
        )
        for col in table.columns
    ]

    redacted_pk = [col_map[pk] for pk in table.primary_key]

    redacted_fks = [
        ForeignKeySchema(
            name=fk.name,
            columns=[col_map[c] for c in fk.columns],
            ref_table=table_map[fk.ref_table],
            ref_columns=[col_maps[fk.ref_table][c] for c in fk.ref_columns],
            on_delete=fk.on_delete,
            on_update=fk.on_update,
            deferrable=fk.deferrable,
            initially=fk.initially,
        )
        for fk in table.foreign_keys
    ]

    redacted_indexes = [
        idx.model_copy(
            update={
                "columns": [col_map[c] for c in idx.columns],
            },
        )
        for idx in table.indexes
    ]

    return table.model_copy(
        update={
            "name": table_map[table.name],
            "columns": redacted_columns,
            "primary_key": redacted_pk,
            "foreign_keys": redacted_fks,
            "indexes": redacted_indexes,
            "comment": None,
        },
    )
