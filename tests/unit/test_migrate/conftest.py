"""Shared test fixtures for migrate module tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    TableSchema,
)

if TYPE_CHECKING:
    from dbsprout.migrate.models import SchemaChange, SchemaChangeType


@pytest.fixture
def minimal_schema() -> DatabaseSchema:
    """Single-table schema: users(id INTEGER PK)."""
    col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)
    table = TableSchema(name="users", columns=[col], primary_key=["id"])
    return DatabaseSchema(tables=[table], dialect="sqlite")


@pytest.fixture
def two_table_schema() -> DatabaseSchema:
    """Two-table schema: users + orders with FK."""
    id_col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)
    user_id_col = ColumnSchema(name="user_id", data_type=ColumnType.INTEGER, nullable=False)
    users = TableSchema(name="users", columns=[id_col], primary_key=["id"])
    orders = TableSchema(
        name="orders",
        columns=[id_col, user_id_col],
        primary_key=["id"],
        foreign_keys=[ForeignKeySchema(columns=["user_id"], ref_table="users", ref_columns=["id"])],
    )
    return DatabaseSchema(tables=[users, orders], dialect="sqlite")


def _find_change(
    changes: list[SchemaChange],
    change_type: SchemaChangeType,
    table_name: str,
    column_name: str | None = None,
) -> SchemaChange | None:
    """Helper to find a specific change in the list."""
    for c in changes:
        if (
            c.change_type == change_type
            and c.table_name == table_name
            and (column_name is None or c.column_name == column_name)
        ):
            return c
    return None
