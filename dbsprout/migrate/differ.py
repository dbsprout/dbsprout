"""Schema diff algorithm for EPIC-006 Migration Awareness.

Compares two ``DatabaseSchema`` snapshots and returns a list of
``SchemaChange`` objects describing structural differences at the table,
column, foreign-key, index, and enum levels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dbsprout.migrate.models import SchemaChange, SchemaChangeType

if TYPE_CHECKING:
    from dbsprout.schema.models import (
        ColumnSchema,
        DatabaseSchema,
        ForeignKeySchema,
        IndexSchema,
        TableSchema,
    )

# Type alias for FK structural identity (ignoring name, on_delete, etc.)
_FkIdentity = tuple[frozenset[str], str, frozenset[str]]

# Type alias for Index structural identity (ignoring name)
_IdxIdentity = tuple[frozenset[str], bool]

# Sentinel table name for enum-level changes (enums are schema-wide, not per-table)
ENUM_TABLE_SENTINEL = "__enums__"


class SchemaDiffer:
    """Compute structural differences between two ``DatabaseSchema`` objects."""

    @staticmethod
    def diff(old: DatabaseSchema, new: DatabaseSchema) -> list[SchemaChange]:
        """Return a sorted list of changes from *old* to *new*.

        Uses ``schema_hash()`` as a fast-path: if hashes match the schemas
        are structurally identical and an empty list is returned immediately.

        Note: ``schema_hash()`` is more sensitive than the structural identity
        used for FK/index comparison.  For example, changing ``on_delete`` on a
        FK changes the hash but ``_fk_identity()`` treats it as the same FK, so
        the diff will report zero FK changes.  This is intentional — the differ
        reports structural additions/removals, not metadata mutations.
        """
        if old.schema_hash() == new.schema_hash():
            return []

        changes: list[SchemaChange] = []

        old_tables = {t.name: t for t in old.tables}
        new_tables = {t.name: t for t in new.tables}

        _diff_tables(old_tables, new_tables, changes)
        _diff_columns(old_tables, new_tables, changes)
        _diff_foreign_keys(old_tables, new_tables, changes)
        _diff_indexes(old_tables, new_tables, changes)
        _diff_enums(old.enums, new.enums, changes)

        changes.sort(key=_change_sort_key)
        return changes


# ── Table-level diff ──────────────────────────────────────────────────────


def _diff_tables(
    old_tables: dict[str, TableSchema],
    new_tables: dict[str, TableSchema],
    changes: list[SchemaChange],
) -> None:
    for name in sorted(new_tables.keys() - old_tables.keys()):
        table = new_tables[name]
        changes.append(
            SchemaChange(
                change_type=SchemaChangeType.TABLE_ADDED,
                table_name=name,
                detail={"table": table.model_dump()},
            )
        )
    for name in sorted(old_tables.keys() - new_tables.keys()):
        table = old_tables[name]
        changes.append(
            SchemaChange(
                change_type=SchemaChangeType.TABLE_REMOVED,
                table_name=name,
                detail={"table": table.model_dump()},
            )
        )


# ── Column-level diff ────────────────────────────────────────────────────


def _diff_columns(
    old_tables: dict[str, TableSchema],
    new_tables: dict[str, TableSchema],
    changes: list[SchemaChange],
) -> None:
    common = sorted(old_tables.keys() & new_tables.keys())
    for table_name in common:
        old_cols = {c.name: c for c in old_tables[table_name].columns}
        new_cols = {c.name: c for c in new_tables[table_name].columns}

        for col_name in sorted(new_cols.keys() - old_cols.keys()):
            col = new_cols[col_name]
            changes.append(
                SchemaChange(
                    change_type=SchemaChangeType.COLUMN_ADDED,
                    table_name=table_name,
                    column_name=col_name,
                    detail={"column": col.model_dump()},
                )
            )

        for col_name in sorted(old_cols.keys() - new_cols.keys()):
            col = old_cols[col_name]
            changes.append(
                SchemaChange(
                    change_type=SchemaChangeType.COLUMN_REMOVED,
                    table_name=table_name,
                    column_name=col_name,
                    detail={"column": col.model_dump()},
                )
            )

        for col_name in sorted(old_cols.keys() & new_cols.keys()):
            _diff_column_properties(
                table_name, col_name, old_cols[col_name], new_cols[col_name], changes
            )


def _diff_column_properties(
    table_name: str,
    col_name: str,
    old_col: ColumnSchema,
    new_col: ColumnSchema,
    changes: list[SchemaChange],
) -> None:
    if old_col.data_type != new_col.data_type:
        changes.append(
            SchemaChange(
                change_type=SchemaChangeType.COLUMN_TYPE_CHANGED,
                table_name=table_name,
                column_name=col_name,
                old_value=old_col.data_type.value,
                new_value=new_col.data_type.value,
                detail={
                    "old_type": old_col.data_type.value,
                    "new_type": new_col.data_type.value,
                },
            )
        )

    if old_col.nullable != new_col.nullable:
        changes.append(
            SchemaChange(
                change_type=SchemaChangeType.COLUMN_NULLABILITY_CHANGED,
                table_name=table_name,
                column_name=col_name,
                old_value=str(old_col.nullable),
                new_value=str(new_col.nullable),
                detail={
                    "old_nullable": old_col.nullable,
                    "new_nullable": new_col.nullable,
                },
            )
        )

    if old_col.default != new_col.default:
        changes.append(
            SchemaChange(
                change_type=SchemaChangeType.COLUMN_DEFAULT_CHANGED,
                table_name=table_name,
                column_name=col_name,
                old_value=old_col.default,
                new_value=new_col.default,
                detail={
                    "old_default": old_col.default,
                    "new_default": new_col.default,
                },
            )
        )


# ── FK-level diff ────────────────────────────────────────────────────────


def _fk_identity(fk: ForeignKeySchema) -> _FkIdentity:
    return (frozenset(fk.columns), fk.ref_table, frozenset(fk.ref_columns))


def _diff_foreign_keys(
    old_tables: dict[str, TableSchema],
    new_tables: dict[str, TableSchema],
    changes: list[SchemaChange],
) -> None:
    common = sorted(old_tables.keys() & new_tables.keys())
    for table_name in common:
        old_fks = {_fk_identity(fk): fk for fk in old_tables[table_name].foreign_keys}
        new_fks = {_fk_identity(fk): fk for fk in new_tables[table_name].foreign_keys}

        for key in sorted(new_fks.keys() - old_fks.keys(), key=str):
            fk = new_fks[key]
            changes.append(
                SchemaChange(
                    change_type=SchemaChangeType.FOREIGN_KEY_ADDED,
                    table_name=table_name,
                    detail={"fk": fk.model_dump()},
                )
            )

        for key in sorted(old_fks.keys() - new_fks.keys(), key=str):
            fk = old_fks[key]
            changes.append(
                SchemaChange(
                    change_type=SchemaChangeType.FOREIGN_KEY_REMOVED,
                    table_name=table_name,
                    detail={"fk": fk.model_dump()},
                )
            )


# ── Index-level diff ──────────────────────────────────────────────────────


def _idx_identity(idx: IndexSchema) -> _IdxIdentity:
    return (frozenset(idx.columns), idx.unique)


def _diff_indexes(
    old_tables: dict[str, TableSchema],
    new_tables: dict[str, TableSchema],
    changes: list[SchemaChange],
) -> None:
    common = sorted(old_tables.keys() & new_tables.keys())
    for table_name in common:
        old_idxs = {_idx_identity(i): i for i in old_tables[table_name].indexes}
        new_idxs = {_idx_identity(i): i for i in new_tables[table_name].indexes}

        for key in sorted(new_idxs.keys() - old_idxs.keys(), key=str):
            idx = new_idxs[key]
            changes.append(
                SchemaChange(
                    change_type=SchemaChangeType.INDEX_ADDED,
                    table_name=table_name,
                    detail={"index": idx.model_dump()},
                )
            )

        for key in sorted(old_idxs.keys() - new_idxs.keys(), key=str):
            idx = old_idxs[key]
            changes.append(
                SchemaChange(
                    change_type=SchemaChangeType.INDEX_REMOVED,
                    table_name=table_name,
                    detail={"index": idx.model_dump()},
                )
            )


# ── Enum-level diff ───────────────────────────────────────────────────────


def _diff_enums(
    old_enums: dict[str, list[str]],
    new_enums: dict[str, list[str]],
    changes: list[SchemaChange],
) -> None:
    for name in sorted(new_enums.keys() - old_enums.keys()):
        changes.append(
            SchemaChange(
                change_type=SchemaChangeType.ENUM_CHANGED,
                table_name=ENUM_TABLE_SENTINEL,
                column_name=name,
                detail={"old_values": None, "new_values": list(new_enums[name])},
            )
        )
    for name in sorted(old_enums.keys() - new_enums.keys()):
        changes.append(
            SchemaChange(
                change_type=SchemaChangeType.ENUM_CHANGED,
                table_name=ENUM_TABLE_SENTINEL,
                column_name=name,
                detail={"old_values": list(old_enums[name]), "new_values": None},
            )
        )
    for name in sorted(old_enums.keys() & new_enums.keys()):
        if set(old_enums[name]) != set(new_enums[name]):
            changes.append(
                SchemaChange(
                    change_type=SchemaChangeType.ENUM_CHANGED,
                    table_name=ENUM_TABLE_SENTINEL,
                    column_name=name,
                    detail={
                        "old_values": list(old_enums[name]),
                        "new_values": list(new_enums[name]),
                    },
                )
            )


# ── Sorting ───────────────────────────────────────────────────────────────

_CHANGE_TYPE_ORDER: dict[SchemaChangeType, int] = {
    SchemaChangeType.TABLE_ADDED: 0,
    SchemaChangeType.TABLE_REMOVED: 1,
    SchemaChangeType.COLUMN_ADDED: 2,
    SchemaChangeType.COLUMN_REMOVED: 3,
    SchemaChangeType.COLUMN_TYPE_CHANGED: 4,
    SchemaChangeType.COLUMN_NULLABILITY_CHANGED: 5,
    SchemaChangeType.COLUMN_DEFAULT_CHANGED: 6,
    SchemaChangeType.FOREIGN_KEY_ADDED: 7,
    SchemaChangeType.FOREIGN_KEY_REMOVED: 8,
    SchemaChangeType.INDEX_ADDED: 9,
    SchemaChangeType.INDEX_REMOVED: 10,
    SchemaChangeType.ENUM_CHANGED: 11,
}

if set(_CHANGE_TYPE_ORDER) != set(SchemaChangeType):
    msg = "_CHANGE_TYPE_ORDER must cover every SchemaChangeType variant"
    raise RuntimeError(msg)


def _change_sort_key(change: SchemaChange) -> tuple[int, str, str]:
    return (
        _CHANGE_TYPE_ORDER[change.change_type],
        change.table_name,
        change.column_name or "",
    )
