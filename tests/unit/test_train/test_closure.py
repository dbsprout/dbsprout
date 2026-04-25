"""Unit tests for the FK closure pass."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import polars as pl

from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    TableSchema,
)
from dbsprout.train.closure import close_fk_graph

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pytest


class _StubEngine:
    """Returns canned parent rows for a given (table, pk_values) request."""

    def __init__(self, parent_rows: dict[str, dict[Any, dict[str, Any]]]) -> None:
        self._rows = parent_rows
        self.calls: list[tuple[str, frozenset[Any]]] = []

    def fetch_by_pk(
        self,
        table: TableSchema,
        pk_column: str,
        values: Iterable[Any],
    ) -> pl.DataFrame:
        vals = frozenset(values)
        self.calls.append((table.name, vals))
        rows = [self._rows[table.name][v] for v in vals if v in self._rows[table.name]]
        if not rows:
            return pl.DataFrame(schema={pk_column: pl.Int64})
        return pl.DataFrame(rows)


def _schema() -> DatabaseSchema:
    users = TableSchema(
        name="users",
        columns=[
            ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False, primary_key=True)
        ],
        primary_key=["id"],
        foreign_keys=[],
    )
    orders = TableSchema(
        name="orders",
        columns=[
            ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False, primary_key=True),
            ColumnSchema(name="user_id", data_type=ColumnType.INTEGER, nullable=True),
        ],
        primary_key=["id"],
        foreign_keys=[
            ForeignKeySchema(columns=["user_id"], ref_table="users", ref_columns=["id"]),
        ],
    )
    return DatabaseSchema(dialect="sqlite", tables=[users, orders])


def test_closure_adds_missing_parent() -> None:
    engine = _StubEngine({"users": {1: {"id": 1}, 2: {"id": 2}}})
    samples = {
        "users": pl.DataFrame({"id": [1]}),
        "orders": pl.DataFrame({"id": [10, 11], "user_id": [1, 2]}),
    }
    report = close_fk_graph(samples, _schema(), engine, max_iterations=4)
    assert sorted(samples["users"]["id"].to_list()) == [1, 2]
    assert report.additions == {"users": 1}


def test_closure_skips_null_fk_values() -> None:
    engine = _StubEngine({"users": {1: {"id": 1}}})
    samples = {
        "users": pl.DataFrame({"id": [1]}),
        "orders": pl.DataFrame({"id": [10, 11], "user_id": [1, None]}),
    }
    report = close_fk_graph(samples, _schema(), engine, max_iterations=4)
    assert report.additions == {}


def test_closure_warns_on_no_pk(caplog: pytest.LogCaptureFixture) -> None:
    schema = DatabaseSchema(
        dialect="sqlite",
        tables=[
            TableSchema(
                name="users",
                columns=[ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=True)],
                primary_key=[],  # no PK
                foreign_keys=[],
            ),
            TableSchema(
                name="orders",
                columns=[
                    ColumnSchema(
                        name="id",
                        data_type=ColumnType.INTEGER,
                        nullable=False,
                        primary_key=True,
                    ),
                    ColumnSchema(name="user_id", data_type=ColumnType.INTEGER, nullable=True),
                ],
                primary_key=["id"],
                foreign_keys=[
                    ForeignKeySchema(columns=["user_id"], ref_table="users", ref_columns=["id"]),
                ],
            ),
        ],
    )
    engine = _StubEngine({})
    samples = {
        "users": pl.DataFrame({"id": [1]}),
        "orders": pl.DataFrame({"id": [10], "user_id": [99]}),
    }
    with caplog.at_level(logging.WARNING, logger="dbsprout.train.closure"):
        close_fk_graph(samples, schema, engine, max_iterations=4)
    assert any("no primary key" in r.message for r in caplog.records)


def test_closure_terminates_on_cycle(caplog: pytest.LogCaptureFixture) -> None:
    """A 2-table mutual FK cycle must terminate via max_iterations."""
    a = TableSchema(
        name="a",
        columns=[
            ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False, primary_key=True),
            ColumnSchema(name="b_id", data_type=ColumnType.INTEGER, nullable=True),
        ],
        primary_key=["id"],
        foreign_keys=[
            ForeignKeySchema(columns=["b_id"], ref_table="b", ref_columns=["id"]),
        ],
    )
    b = TableSchema(
        name="b",
        columns=[
            ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False, primary_key=True),
            ColumnSchema(name="a_id", data_type=ColumnType.INTEGER, nullable=True),
        ],
        primary_key=["id"],
        foreign_keys=[
            ForeignKeySchema(columns=["a_id"], ref_table="a", ref_columns=["id"]),
        ],
    )
    schema = DatabaseSchema(dialect="sqlite", tables=[a, b])
    # Engine returns chained refs forever.
    engine = _StubEngine(
        {
            "a": {i: {"id": i, "b_id": i + 1} for i in range(100)},
            "b": {i: {"id": i, "a_id": i + 1} for i in range(100)},
        }
    )
    samples = {
        "a": pl.DataFrame({"id": [1], "b_id": [2]}),
        "b": pl.DataFrame({"id": [1], "a_id": [2]}),
    }
    with caplog.at_level(logging.WARNING, logger="dbsprout.train.closure"):
        report = close_fk_graph(samples, schema, engine, max_iterations=3)
    assert report.iterations == 3
    assert any("closure terminated" in r.message for r in caplog.records)


def test_closure_skips_composite_fk_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Composite (multi-column) FKs are not yet supported by closure — warn + skip."""
    parent = TableSchema(
        name="composite_pk",
        columns=[
            ColumnSchema(name="a", data_type=ColumnType.INTEGER, nullable=False, primary_key=True),
            ColumnSchema(name="b", data_type=ColumnType.INTEGER, nullable=False, primary_key=True),
        ],
        primary_key=["a", "b"],
        foreign_keys=[],
    )
    child = TableSchema(
        name="child",
        columns=[
            ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False, primary_key=True),
            ColumnSchema(name="ref_a", data_type=ColumnType.INTEGER, nullable=True),
            ColumnSchema(name="ref_b", data_type=ColumnType.INTEGER, nullable=True),
        ],
        primary_key=["id"],
        foreign_keys=[
            ForeignKeySchema(
                columns=["ref_a", "ref_b"],
                ref_table="composite_pk",
                ref_columns=["a", "b"],
            ),
        ],
    )
    schema = DatabaseSchema(dialect="sqlite", tables=[parent, child])
    engine = _StubEngine({})
    samples = {
        "composite_pk": pl.DataFrame({"a": [1], "b": [1]}),
        "child": pl.DataFrame({"id": [10], "ref_a": [99], "ref_b": [99]}),
    }
    with caplog.at_level(logging.WARNING, logger="dbsprout.train.closure"):
        report = close_fk_graph(samples, schema, engine, max_iterations=4)
    assert any("composite FK" in r.message for r in caplog.records)
    assert engine.calls == []
    assert report.additions == {}


def test_closure_logs_unresolved_for_empty_parent(caplog: pytest.LogCaptureFixture) -> None:
    engine = _StubEngine({"users": {}})  # parent table empty
    samples = {
        "users": pl.DataFrame(schema={"id": pl.Int64}),
        "orders": pl.DataFrame({"id": [10], "user_id": [99]}),
    }
    with caplog.at_level(logging.WARNING, logger="dbsprout.train.closure"):
        report = close_fk_graph(samples, _schema(), engine, max_iterations=4)
    assert report.unresolved_per_table.get("orders", 0) == 1
    assert any("empty but child" in r.message for r in caplog.records)
