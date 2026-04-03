"""Schema redaction for the ``redacted`` privacy tier.

Replaces table and column names with HMAC-SHA256 hashes (salted),
strips comments and sensitive metadata, and preserves structural
information (types, constraints, FK references).
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dbsprout.schema.models import DatabaseSchema, TableSchema
    from dbsprout.spec.models import DataSpec


class RedactionMap(BaseModel):
    """Stores original→hashed name mappings for de-redaction.

    Returned alongside the redacted schema so that LLM responses
    using hashed names can be mapped back to original column names.

    **Security:** This object contains the HMAC salt and the full
    forward mapping (original names → hashed names). It must NEVER
    be sent to a cloud provider, persisted alongside cloud-bound data,
    or serialized to shared storage. Keep it in local memory only.
    """

    model_config = ConfigDict(frozen=True)

    salt: bytes
    table_map: dict[str, str]
    column_maps: dict[str, dict[str, str]]


def redact_schema(
    schema: DatabaseSchema,
    *,
    salt: bytes | None = None,
) -> tuple[DatabaseSchema, RedactionMap]:
    """Return a redacted copy of *schema* and a mapping for de-redaction.

    Parameters
    ----------
    schema:
        The database schema to redact.
    salt:
        HMAC salt for hashing. If ``None``, a random 16-byte salt is generated.

    Returns
    -------
    tuple[DatabaseSchema, RedactionMap]
        The redacted schema and the mapping needed to reverse hashed names.
    """
    if salt is None:
        salt = os.urandom(16)

    table_map = {t.name: _hash_table(t.name, salt) for t in schema.tables}
    col_maps: dict[str, dict[str, str]] = {}
    for table in schema.tables:
        col_maps[table.name] = {c.name: _hash_column(c.name, salt) for c in table.columns}

    redacted_tables = [_redact_table(table, table_map, col_maps, salt) for table in schema.tables]

    # Redact DatabaseSchema.enums dict (hash keys and values)
    redacted_enums: dict[str, list[str]] = {}
    for enum_name, enum_values in schema.enums.items():
        hashed_key = _hash_name(enum_name, "enum_", salt)
        redacted_enums[hashed_key] = [_hash_name(v, "val_", salt) for v in enum_values]

    redacted = schema.model_copy(
        update={
            "tables": redacted_tables,
            "enums": redacted_enums,
        },
    )

    redaction_map = RedactionMap(
        salt=salt,
        table_map=table_map,
        column_maps=col_maps,
    )

    return redacted, redaction_map


def de_redact_spec(spec: DataSpec, redaction_map: RedactionMap) -> DataSpec:
    """Reverse hashed names in a DataSpec using the redaction mapping.

    Replaces hashed table names and column names in the spec with
    their original values so downstream generation uses real names.
    """
    from dbsprout.spec.models import TableSpec  # noqa: PLC0415, TC001

    reverse_table = {v: k for k, v in redaction_map.table_map.items()}
    reverse_columns: dict[str, dict[str, str]] = {}
    for table_name, col_map in redaction_map.column_maps.items():
        reverse_columns[table_name] = {v: k for k, v in col_map.items()}

    new_tables: list[TableSpec] = []
    for table_spec in spec.tables:
        original_table = reverse_table.get(table_spec.table_name, table_spec.table_name)
        if table_spec.table_name not in reverse_table:
            logger.warning("De-redaction: unmapped table name %r", table_spec.table_name)
        col_reverse = reverse_columns.get(original_table, {})
        new_columns = {
            col_reverse.get(col_name, col_name): config
            for col_name, config in table_spec.columns.items()
        }
        new_tables.append(
            table_spec.model_copy(
                update={
                    "table_name": original_table,
                    "columns": new_columns,
                },
            )
        )

    return spec.model_copy(update={"tables": new_tables})


def _hash_name(name: str, prefix: str, salt: bytes) -> str:
    """Hash a name with HMAC-SHA256 and prepend a prefix."""
    digest = hmac.new(salt, name.encode(), hashlib.sha256).hexdigest()[:12]
    return f"{prefix}{digest}"


def _hash_table(name: str, salt: bytes) -> str:
    return _hash_name(name, "tbl_", salt)


def _hash_column(name: str, salt: bytes) -> str:
    return _hash_name(name, "col_", salt)


def _redact_table(
    table: TableSchema,
    table_map: dict[str, str],
    col_maps: dict[str, dict[str, str]],
    salt: bytes,
) -> TableSchema:
    """Redact a single table: hash names, strip metadata, update FKs."""
    from dbsprout.schema.models import ForeignKeySchema  # noqa: PLC0415

    col_map = col_maps[table.name]

    redacted_columns = [
        col.model_copy(
            update={
                "name": col_map[col.name],
                "comment": None,
                "default": None,
                "check_constraint": None,
                "enum_values": None,
            },
        )
        for col in table.columns
    ]

    redacted_pk = [col_map[pk] for pk in table.primary_key]

    redacted_fks = [
        ForeignKeySchema(
            name=_hash_name(fk.name, "fk_", salt) if fk.name else None,
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
                "name": _hash_name(idx.name, "idx_", salt) if idx.name else None,
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
