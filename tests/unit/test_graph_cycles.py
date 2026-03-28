"""Tests for dbsprout.schema.graph — cycle detection via Tarjan SCC."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dbsprout.schema.graph import detect_cycles
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    TableSchema,
)


def _col(name: str, *, nullable: bool = True) -> ColumnSchema:
    return ColumnSchema(name=name, data_type=ColumnType.INTEGER, nullable=nullable)


def _table(
    name: str,
    fks: list[tuple[str, str, bool]] | None = None,
    self_ref: str | None = None,
) -> TableSchema:
    """Build a table with optional FKs.

    fks: list of (column, ref_table, nullable) tuples.
    self_ref: column name for self-referencing FK.
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
    return DatabaseSchema(tables=list(tables))


# ── Acyclic ──────────────────────────────────────────────────────────────


class TestAcyclic:
    def test_no_cycles_returns_empty(self) -> None:
        result = detect_cycles(
            _schema(
                _table("users"),
                _table("orders", fks=[("user_id", "users", True)]),
            )
        )
        assert result == []

    def test_empty_schema(self) -> None:
        assert detect_cycles(_schema()) == []


# ── Basic cycles ─────────────────────────────────────────────────────────


class TestTwoTableCycle:
    def test_detects_cycle(self) -> None:
        result = detect_cycles(
            _schema(
                _table("a", fks=[("b_id", "b", True)]),
                _table("b", fks=[("a_id", "a", True)]),
            )
        )
        assert len(result) == 1
        assert result[0].tables == frozenset({"a", "b"})

    def test_edges_contain_both_fks(self) -> None:
        result = detect_cycles(
            _schema(
                _table("a", fks=[("b_id", "b", True)]),
                _table("b", fks=[("a_id", "a", True)]),
            )
        )
        assert len(result[0].edges) == 2


class TestThreeTableCycle:
    def test_detects_cycle(self) -> None:
        result = detect_cycles(
            _schema(
                _table("a", fks=[("c_id", "c", True)]),
                _table("b", fks=[("a_id", "a", True)]),
                _table("c", fks=[("b_id", "b", True)]),
            )
        )
        assert len(result) == 1
        assert result[0].tables == frozenset({"a", "b", "c"})
        assert len(result[0].edges) == 3


# ── Multiple cycles ──────────────────────────────────────────────────────


class TestMultipleIndependentCycles:
    def test_two_separate_cycles(self) -> None:
        result = detect_cycles(
            _schema(
                _table("a", fks=[("b_id", "b", True)]),
                _table("b", fks=[("a_id", "a", True)]),
                _table("c", fks=[("d_id", "d", True)]),
                _table("d", fks=[("c_id", "c", True)]),
            )
        )
        assert len(result) == 2
        tables_sets = {frozenset(ci.tables) for ci in result}
        assert frozenset({"a", "b"}) in tables_sets
        assert frozenset({"c", "d"}) in tables_sets


# ── Nullable candidates ──────────────────────────────────────────────────


class TestNullableCandidates:
    def test_nullable_fk_is_candidate(self) -> None:
        result = detect_cycles(
            _schema(
                _table("a", fks=[("b_id", "b", True)]),
                _table("b", fks=[("a_id", "a", False)]),
            )
        )
        assert len(result) == 1
        candidates = result[0].candidate_breaks
        assert len(candidates) == 1
        assert candidates[0].source_table == "a"
        assert candidates[0].foreign_key.columns == ["b_id"]

    def test_no_nullable_fk_empty_candidates(self) -> None:
        result = detect_cycles(
            _schema(
                _table("a", fks=[("b_id", "b", False)]),
                _table("b", fks=[("a_id", "a", False)]),
            )
        )
        assert len(result) == 1
        assert result[0].candidate_breaks == ()

    def test_both_nullable_both_candidates(self) -> None:
        result = detect_cycles(
            _schema(
                _table("a", fks=[("b_id", "b", True)]),
                _table("b", fks=[("a_id", "a", True)]),
            )
        )
        assert len(result) == 1
        assert len(result[0].candidate_breaks) == 2


# ── Self-ref not reported ────────────────────────────────────────────────


class TestSelfRefNotCycle:
    def test_self_ref_only_not_cycle(self) -> None:
        result = detect_cycles(_schema(_table("employees", self_ref="manager_id")))
        assert result == []

    def test_self_ref_with_acyclic_fks(self) -> None:
        result = detect_cycles(
            _schema(
                _table("departments"),
                _table(
                    "employees",
                    fks=[("dept_id", "departments", True)],
                    self_ref="manager_id",
                ),
            )
        )
        assert result == []


# ── Overlapping cycles (single SCC) ─────────────────────────────────────


class TestOverlappingCycles:
    def test_single_scc_all_edges(self) -> None:
        """A→B→C→A and A→B→D→A share the A→B edge. Single SCC={A,B,C,D}."""
        a = TableSchema(
            name="a",
            columns=[_col("id", nullable=False), _col("c_id"), _col("d_id")],
            primary_key=["id"],
            foreign_keys=[
                ForeignKeySchema(columns=["c_id"], ref_table="c", ref_columns=["id"]),
                ForeignKeySchema(columns=["d_id"], ref_table="d", ref_columns=["id"]),
            ],
        )
        result = detect_cycles(
            _schema(
                a,
                _table("b", fks=[("a_id", "a", True)]),
                _table("c", fks=[("b_id", "b", True)]),
                _table("d", fks=[("b_id", "b", True)]),
            )
        )
        assert len(result) == 1
        assert result[0].tables == frozenset({"a", "b", "c", "d"})
        # A→C, A→D, B→A, C→B, D→B = 5 edges
        assert len(result[0].edges) == 5


# ── Model properties ─────────────────────────────────────────────────────


class TestCycleInfoProperties:
    def test_frozen(self) -> None:
        result = detect_cycles(
            _schema(
                _table("a", fks=[("b_id", "b", True)]),
                _table("b", fks=[("a_id", "a", True)]),
            )
        )
        with pytest.raises(ValidationError):
            result[0].tables = frozenset({"x"})  # type: ignore[misc]

    def test_deterministic_ordering(self) -> None:
        schema = _schema(
            _table("x", fks=[("y_id", "y", True)]),
            _table("y", fks=[("x_id", "x", True)]),
            _table("a", fks=[("b_id", "b", True)]),
            _table("b", fks=[("a_id", "a", True)]),
        )
        r1 = detect_cycles(schema)
        r2 = detect_cycles(schema)
        assert len(r1) == 2
        # Sorted by table names: {a,b} before {x,y}
        assert r1[0].tables == frozenset({"a", "b"})
        assert r1[1].tables == frozenset({"x", "y"})
        assert r1[0].tables == r2[0].tables
        assert r1[1].tables == r2[1].tables
