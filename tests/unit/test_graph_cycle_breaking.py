"""Tests for dbsprout.schema.graph — cycle breaking via resolve_cycles."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dbsprout.schema.graph import UnresolvableCycleError, resolve_cycles
from dbsprout.schema.models import ForeignKeySchema, TableSchema
from tests.unit.conftest import _col, _schema, _table

# ── Acyclic passthrough ──────────────────────────────────────────────────


class TestAcyclicPassthrough:
    def test_no_cycles_empty_deferred(self) -> None:
        result = resolve_cycles(
            _schema(
                _table("users"),
                _table("orders", fks=[("user_id", "users", True)]),
            )
        )
        assert result.deferred_fks == ()
        assert result.graph.insertion_order == (("users",), ("orders",))

    def test_empty_schema(self) -> None:
        result = resolve_cycles(_schema())
        assert result.deferred_fks == ()
        assert result.graph.tables == ()


# ── Simple cycle breaking ────────────────────────────────────────────────


class TestSimpleCycleBreak:
    def test_two_table_cycle(self) -> None:
        result = resolve_cycles(
            _schema(
                _table("a", fks=[("b_id", "b", True)]),
                _table("b", fks=[("a_id", "a", True)]),
            )
        )
        assert len(result.deferred_fks) == 1
        # Graph should now be acyclic
        assert len(result.graph.insertion_order) >= 1
        # All tables present
        all_tables = {t for batch in result.graph.insertion_order for t in batch}
        assert all_tables == {"a", "b"}

    def test_three_table_cycle(self) -> None:
        result = resolve_cycles(
            _schema(
                _table("a", fks=[("c_id", "c", True)]),
                _table("b", fks=[("a_id", "a", True)]),
                _table("c", fks=[("b_id", "b", True)]),
            )
        )
        assert len(result.deferred_fks) == 1
        all_tables = {t for batch in result.graph.insertion_order for t in batch}
        assert all_tables == {"a", "b", "c"}

    def test_deferred_fk_has_reason(self) -> None:
        result = resolve_cycles(
            _schema(
                _table("a", fks=[("b_id", "b", True)]),
                _table("b", fks=[("a_id", "a", True)]),
            )
        )
        assert result.deferred_fks[0].reason == "cycle_break"


# ── Multiple cycles ──────────────────────────────────────────────────────


class TestMultipleCycles:
    def test_two_independent_cycles(self) -> None:
        result = resolve_cycles(
            _schema(
                _table("a", fks=[("b_id", "b", True)]),
                _table("b", fks=[("a_id", "a", True)]),
                _table("c", fks=[("d_id", "d", True)]),
                _table("d", fks=[("c_id", "c", True)]),
            )
        )
        assert len(result.deferred_fks) == 2
        all_tables = {t for batch in result.graph.insertion_order for t in batch}
        assert all_tables == {"a", "b", "c", "d"}


# ── Unresolvable cycle ───────────────────────────────────────────────────


class TestUnresolvableCycle:
    def test_no_nullable_raises(self) -> None:
        with pytest.raises(UnresolvableCycleError):
            resolve_cycles(
                _schema(
                    _table("a", fks=[("b_id", "b", False)]),
                    _table("b", fks=[("a_id", "a", False)]),
                )
            )

    def test_error_message_contains_tables(self) -> None:
        with pytest.raises(UnresolvableCycleError, match="a") as exc_info:
            resolve_cycles(
                _schema(
                    _table("a", fks=[("b_id", "b", False)]),
                    _table("b", fks=[("a_id", "a", False)]),
                )
            )
        assert "b" in str(exc_info.value)
        assert "nullable" in str(exc_info.value).lower()

    def test_error_has_cycle_info(self) -> None:
        with pytest.raises(UnresolvableCycleError) as exc_info:
            resolve_cycles(
                _schema(
                    _table("a", fks=[("b_id", "b", False)]),
                    _table("b", fks=[("a_id", "a", False)]),
                )
            )
        assert exc_info.value.cycle_info.tables == frozenset({"a", "b"})

    def test_one_resolvable_one_not(self) -> None:
        """Two cycles: A↔B (nullable), C↔D (non-nullable). Should raise for C↔D."""
        with pytest.raises(UnresolvableCycleError) as exc_info:
            resolve_cycles(
                _schema(
                    _table("a", fks=[("b_id", "b", True)]),
                    _table("b", fks=[("a_id", "a", True)]),
                    _table("c", fks=[("d_id", "d", False)]),
                    _table("d", fks=[("c_id", "c", False)]),
                )
            )
        assert exc_info.value.cycle_info.tables == frozenset({"c", "d"})


# ── Edge selection heuristic ─────────────────────────────────────────────


class TestEdgeSelectionHeuristic:
    def test_prefer_table_with_most_outgoing(self) -> None:
        """In A→B→C→A, if A has 2 outgoing edges within SCC (A→B, A→C),
        prefer to defer one of A's edges."""
        a = TableSchema(
            name="a",
            columns=[
                _col("id", nullable=False),
                _col("b_id"),
                _col("c_id"),
            ],
            primary_key=["id"],
            foreign_keys=[
                ForeignKeySchema(columns=["b_id"], ref_table="b", ref_columns=["id"]),
                ForeignKeySchema(columns=["c_id"], ref_table="c", ref_columns=["id"]),
            ],
        )
        result = resolve_cycles(
            _schema(
                a,
                _table("b", fks=[("a_id", "a", True)]),
                _table("c", fks=[("b_id", "b", True)]),
            )
        )
        assert len(result.deferred_fks) >= 1
        # A has most outgoing edges (2), so one of A's edges should be deferred
        assert result.deferred_fks[0].source_table == "a"


# ── Self-ref not affected ────────────────────────────────────────────────


class TestSelfRefNotAffected:
    def test_self_ref_preserved_after_cycle_break(self) -> None:
        result = resolve_cycles(
            _schema(
                _table("employees", fks=[("dept_id", "departments", True)], self_ref="manager_id"),
                _table("departments", fks=[("head_id", "employees", True)]),
            )
        )
        assert "employees" in result.graph.self_referencing
        assert len(result.deferred_fks) == 1


# ── Multiple FKs same target ────────────────────────────────────────────


class TestMultipleFKsSameTarget:
    def test_partial_defer_keeps_dependency(self) -> None:
        """A has 2 FKs to B (one nullable, one not). B has nullable FK to A.
        Deferring B→A breaks the cycle. A still depends on B via both FKs."""
        a = TableSchema(
            name="a",
            columns=[
                _col("id", nullable=False),
                _col("b_id_nullable"),
                _col("b_id_required", nullable=False),
            ],
            primary_key=["id"],
            foreign_keys=[
                ForeignKeySchema(columns=["b_id_nullable"], ref_table="b", ref_columns=["id"]),
                ForeignKeySchema(columns=["b_id_required"], ref_table="b", ref_columns=["id"]),
            ],
        )
        result = resolve_cycles(_schema(a, _table("b", fks=[("a_id", "a", True)])))
        # Cycle requires deferring both A→B(nullable) and B→A to fully break,
        # or just B→A. Either way, A still depends on B via non-nullable FK.
        assert len(result.deferred_fks) >= 1
        # A still depends on B (non-deferred FK remains)
        assert "b" in result.graph.dependencies["a"]


# ── Resolved graph is frozen ─────────────────────────────────────────────


class TestResolvedGraphFrozen:
    def test_immutable(self) -> None:
        result = resolve_cycles(
            _schema(
                _table("users"),
                _table("orders", fks=[("user_id", "users", True)]),
            )
        )
        with pytest.raises(ValidationError):
            result.graph = None  # type: ignore[assignment]
