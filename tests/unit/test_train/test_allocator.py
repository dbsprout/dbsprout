"""Pure unit tests for the stratified-sample allocator."""

from __future__ import annotations

import pytest

from dbsprout.train.allocator import allocate_budget


def test_proportional_allocation_two_tables() -> None:
    allocs = {
        a.table: a.target
        for a in allocate_budget(
            row_counts={"big": 900, "small": 100},
            budget=100,
            min_per_table=0,
            max_per_table=100,
        )
    }
    assert allocs == {"big": 90, "small": 10}


def test_floor_clamp_redistributes_residual() -> None:
    allocs = {
        a.table: a.target
        for a in allocate_budget(
            row_counts={"big": 9990, "tiny": 10},
            budget=100,
            min_per_table=10,
            max_per_table=100,
        )
    }
    assert allocs["tiny"] == 10
    assert allocs["big"] == 90


def test_ceiling_clamp() -> None:
    allocs = {
        a.table: a.target
        for a in allocate_budget(
            row_counts={"a": 1, "b": 1},
            budget=1000,
            min_per_table=0,
            max_per_table=100,
        )
    }
    assert allocs == {"a": 1, "b": 1}


def test_empty_tables_skipped() -> None:
    allocs = {
        a.table: a.target
        for a in allocate_budget(
            row_counts={"present": 100, "empty": 0},
            budget=50,
            min_per_table=0,
            max_per_table=100,
        )
    }
    assert "empty" not in allocs
    assert allocs["present"] == 50


def test_single_table_gets_min_of_budget_and_ceiling() -> None:
    allocs = {
        a.table: a.target
        for a in allocate_budget(
            row_counts={"only": 5000},
            budget=100,
            min_per_table=0,
            max_per_table=100,
        )
    }
    assert allocs == {"only": 100}


def test_budget_larger_than_row_count_caps_at_row_count() -> None:
    allocs = {
        a.table: a.target
        for a in allocate_budget(
            row_counts={"a": 7},
            budget=1000,
            min_per_table=0,
            max_per_table=10000,
        )
    }
    assert allocs == {"a": 7}


def test_clamp_flags_set_correctly() -> None:
    out = {
        a.table: a
        for a in allocate_budget(
            row_counts={"big": 9990, "tiny": 10},
            budget=100,
            min_per_table=10,
            max_per_table=100,
        )
    }
    assert out["tiny"].floor_clamped is True
    assert out["tiny"].ceiling_clamped is False


def test_residual_within_n_tables_of_budget() -> None:
    """AC-2: after clamping, sum is within +/- n_tables of requested budget."""
    row_counts = {f"t{i}": 100 for i in range(7)}
    allocs = list(
        allocate_budget(row_counts=row_counts, budget=33, min_per_table=0, max_per_table=100)
    )
    assert abs(sum(a.target for a in allocs) - 33) <= len(allocs)


def test_zero_budget_returns_zero_targets() -> None:
    allocs = list(
        allocate_budget(
            row_counts={"a": 100},
            budget=0,
            min_per_table=0,
            max_per_table=10,
        )
    )
    assert allocs[0].target == 0


def test_min_floor_overshoots_budget_by_design() -> None:
    """``min_per_table`` is a hard floor: total may legitimately exceed budget.

    Regression test documenting the intentional behavior described in the
    docstring: when ``min_per_table * n > budget``, the floor wins.
    """
    allocs = list(
        allocate_budget(
            row_counts={"a": 100, "b": 100, "c": 100},
            budget=10,
            min_per_table=10,
            max_per_table=100,
        )
    )
    assert sum(a.target for a in allocs) == 30
    assert all(a.target == 10 for a in allocs)


def test_min_per_table_capped_at_row_count() -> None:
    """Row count caps the floor: a 5-row table never overshoots into target=10.

    Neither the user-configured floor nor ceiling binds — the row count itself
    is the binding cap, so both clamp flags must be False.
    """
    allocs = {
        a.table: a
        for a in allocate_budget(
            row_counts={"a": 5},
            budget=100,
            min_per_table=10,
            max_per_table=100,
        )
    }
    assert allocs["a"].target == 5
    assert allocs["a"].floor_clamped is False
    assert allocs["a"].ceiling_clamped is False


def test_min_greater_than_max_raises() -> None:
    with pytest.raises(ValueError, match="must be <= max_per_table"):
        allocate_budget(
            row_counts={"a": 100},
            budget=10,
            min_per_table=20,
            max_per_table=10,
        )
