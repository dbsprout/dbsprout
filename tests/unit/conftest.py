"""Shared test helpers for graph-related unit tests.

These helpers build minimal DatabaseSchema/TableSchema/ColumnSchema
objects for testing FK graph construction, cycle detection, and cycle
breaking without needing a real database.
"""

from __future__ import annotations

from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    TableSchema,
)


def _col(name: str, *, nullable: bool = True) -> ColumnSchema:
    """Minimal column for test schemas."""
    return ColumnSchema(name=name, data_type=ColumnType.INTEGER, nullable=nullable)


def _table(
    name: str,
    fks: list[tuple[str, str, bool]] | None = None,
    self_ref: str | None = None,
    fk_name: str | None = None,
) -> TableSchema:
    """Build a table with optional FKs.

    fks: list of (column, ref_table, nullable) tuples.
    self_ref: column name for self-referencing FK.
    fk_name: explicit FK name (for self_ref).
    """
    columns = [_col("id", nullable=False)]
    foreign_keys: list[ForeignKeySchema] = []
    if fks:
        for col_name, ref_table, nullable in fks:
            columns.append(_col(col_name, nullable=nullable))
            foreign_keys.append(
                ForeignKeySchema(
                    columns=[col_name],
                    ref_table=ref_table,
                    ref_columns=["id"],
                )
            )
    if self_ref:
        columns.append(_col(self_ref))
        foreign_keys.append(
            ForeignKeySchema(
                name=fk_name,
                columns=[self_ref],
                ref_table=name,
                ref_columns=["id"],
            )
        )
    return TableSchema(
        name=name,
        columns=columns,
        primary_key=["id"],
        foreign_keys=foreign_keys,
    )


def _schema(*tables: TableSchema) -> DatabaseSchema:
    """Build a DatabaseSchema from tables."""
    return DatabaseSchema(tables=list(tables))
