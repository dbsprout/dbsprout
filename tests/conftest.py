"""Shared test fixtures for DBSprout.

Factory functions return frozen Pydantic models reusable across test modules.
"""

from __future__ import annotations

import pytest

from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    TableSchema,
)


def _col(name: str, dtype: ColumnType = ColumnType.INTEGER, **kw: object) -> ColumnSchema:
    return ColumnSchema(name=name, data_type=dtype, **kw)  # type: ignore[arg-type]


def _fk(cols: list[str], ref: str, ref_cols: list[str], **kw: object) -> ForeignKeySchema:
    return ForeignKeySchema(columns=cols, ref_table=ref, ref_columns=ref_cols, **kw)  # type: ignore[arg-type]


@pytest.fixture
def simple_schema() -> DatabaseSchema:
    """Two tables (users, posts) with one FK — the minimal relational schema."""
    users = TableSchema(
        name="users",
        columns=[
            _col("id", primary_key=True),
            _col("email", ColumnType.VARCHAR, nullable=False, unique=True),
        ],
        primary_key=["id"],
    )
    posts = TableSchema(
        name="posts",
        columns=[
            _col("id", primary_key=True),
            _col("user_id", nullable=False),
            _col("title", ColumnType.VARCHAR, nullable=False),
        ],
        primary_key=["id"],
        foreign_keys=[_fk(["user_id"], "users", ["id"], on_delete="CASCADE")],
    )
    return DatabaseSchema(tables=[users, posts])


@pytest.fixture
def junction_table_schema() -> DatabaseSchema:
    """Users, roles, and a classic M:N user_roles junction table."""
    users = TableSchema(
        name="users",
        columns=[_col("id", primary_key=True)],
        primary_key=["id"],
    )
    roles = TableSchema(
        name="roles",
        columns=[_col("id", primary_key=True)],
        primary_key=["id"],
    )
    user_roles = TableSchema(
        name="user_roles",
        columns=[_col("user_id"), _col("role_id")],
        primary_key=["user_id", "role_id"],
        foreign_keys=[
            _fk(["user_id"], "users", ["id"]),
            _fk(["role_id"], "roles", ["id"]),
        ],
    )
    return DatabaseSchema(tables=[users, roles, user_roles])


@pytest.fixture
def all_types_schema() -> DatabaseSchema:
    """One table with one column per ColumnType value."""
    columns = [_col(ct.value, ct) for ct in ColumnType]
    table = TableSchema(name="all_types", columns=columns)
    return DatabaseSchema(tables=[table])
