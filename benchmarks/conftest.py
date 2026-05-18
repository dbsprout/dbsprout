"""Shared fixtures for the performance benchmark suite."""

from __future__ import annotations

import tracemalloc
from typing import TYPE_CHECKING, Any

import pytest

from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    TableSchema,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _ecommerce_table() -> TableSchema:
    """A realistic 10-column e-commerce-like table (no FK, no autoincrement)."""
    return TableSchema(
        name="orders",
        columns=[
            ColumnSchema(name="id", data_type=ColumnType.INTEGER),
            ColumnSchema(name="customer_email", data_type=ColumnType.VARCHAR),
            ColumnSchema(name="full_name", data_type=ColumnType.VARCHAR),
            ColumnSchema(name="amount", data_type=ColumnType.DECIMAL),
            ColumnSchema(name="quantity", data_type=ColumnType.INTEGER),
            ColumnSchema(name="is_paid", data_type=ColumnType.BOOLEAN),
            ColumnSchema(name="created_at", data_type=ColumnType.DATETIME),
            ColumnSchema(name="ship_date", data_type=ColumnType.DATE),
            ColumnSchema(name="notes", data_type=ColumnType.TEXT),
            ColumnSchema(name="tracking_uuid", data_type=ColumnType.UUID),
        ],
    )


@pytest.fixture
def ecommerce_table() -> TableSchema:
    return _ecommerce_table()


@pytest.fixture
def ecommerce_schema(ecommerce_table: TableSchema) -> DatabaseSchema:
    return DatabaseSchema(tables=[ecommerce_table])


@pytest.fixture
def fk_child_table() -> TableSchema:
    """A child table with a single-column FK to a parent ``users`` table."""
    return TableSchema(
        name="orders",
        columns=[
            ColumnSchema(name="id", data_type=ColumnType.INTEGER),
            ColumnSchema(name="user_id", data_type=ColumnType.INTEGER),
        ],
        foreign_keys=[
            ForeignKeySchema(columns=["user_id"], ref_table="users", ref_columns=["id"]),
        ],
    )


@pytest.fixture
def measure_peak_mb() -> Callable[[Callable[[], Any]], float]:
    """Return a helper that runs *fn* and reports peak allocation in MB."""

    def _measure(fn: Callable[[], Any]) -> float:
        tracemalloc.start()
        try:
            fn()
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        return peak / (1024 * 1024)

    return _measure
