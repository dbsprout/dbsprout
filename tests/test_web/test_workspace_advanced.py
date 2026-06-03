"""Tests for ``Workspace.update_table_advanced`` (P2b-2).

The advanced-pack edit persists ``TableSpec.correlations`` (``CorrelationRule``)
and ``TableSpec.derived`` (``DerivedColumn``) on the workspace spec. Both
``DataSpec`` and ``TableSpec`` are ``frozen=True`` so the helper must swap them
out via ``model_copy(update=...)`` — never mutate in place. Either list may be
left unchanged by passing ``None`` (partial update).

These tests lock the helper contract: identity changes on success, no in-place
mutation of the original models, ``None`` leaves the other list intact, and the
right exceptions when the spec / table is missing (the router maps those to
409 / 404).
"""

from __future__ import annotations

import pytest

from dbsprout.spec.models import (
    CorrelationRule,
    DataSpec,
    DerivedColumn,
    GeneratorConfig,
    TableSpec,
)
from dbsprout.web.workspace import Workspace


def _seed_spec(workspace: Workspace) -> DataSpec:
    """Seed a 2-table spec; return the original so identity checks are easy."""
    spec = DataSpec(
        tables=[
            TableSpec(
                table_name="users",
                row_count=100,
                columns={
                    "id": GeneratorConfig(provider="builtin.sequence", unique=True),
                    "city": GeneratorConfig(provider="mimesis.city"),
                    "state": GeneratorConfig(provider="mimesis.state"),
                },
            ),
            TableSpec(
                table_name="orders",
                row_count=200,
                columns={
                    "id": GeneratorConfig(provider="builtin.sequence", unique=True),
                    "user_id": GeneratorConfig(provider="numpy.integer"),
                    "qty": GeneratorConfig(provider="numpy.integer"),
                    "price": GeneratorConfig(provider="numpy.float"),
                },
            ),
        ],
        schema_hash="deadbeef",
    )
    workspace.set_spec(spec)
    return spec


def test_update_advanced_sets_correlations() -> None:
    ws = Workspace()
    _seed_spec(ws)
    rules = [CorrelationRule(columns=["city", "state"], strategy="lookup")]

    returned = ws.update_table_advanced("users", correlations=rules)

    assert returned.correlations == rules
    stored = ws.get_spec()
    assert stored is not None
    users = stored.get_table_spec("users")
    assert users is not None
    assert users.correlations == rules


def test_update_advanced_sets_derived() -> None:
    ws = Workspace()
    _seed_spec(ws)
    derived = [
        DerivedColumn(column="price", expression="qty * 9.99", depends_on=["qty"]),
    ]

    returned = ws.update_table_advanced("orders", derived=derived)

    assert returned.derived == derived
    stored = ws.get_spec()
    assert stored is not None
    orders = stored.get_table_spec("orders")
    assert orders is not None
    assert orders.derived == derived


def test_update_advanced_sets_both() -> None:
    ws = Workspace()
    _seed_spec(ws)
    rules = [CorrelationRule(columns=["qty", "price"])]
    derived = [DerivedColumn(column="price", expression="qty * 2", depends_on=["qty"])]

    returned = ws.update_table_advanced("orders", correlations=rules, derived=derived)

    assert returned.correlations == rules
    assert returned.derived == derived


def test_update_advanced_replaces_spec_immutably() -> None:
    ws = Workspace()
    original = _seed_spec(ws)
    ws.update_table_advanced(
        "users",
        correlations=[CorrelationRule(columns=["city", "state"])],
    )

    after = ws.get_spec()
    assert after is not None
    # Identity change — frozen ``DataSpec`` is replaced, not mutated.
    assert after is not original
    # The original spec object remains untouched (frozen invariants hold).
    original_users = original.get_table_spec("users")
    assert original_users is not None
    assert original_users.correlations == []


def test_update_advanced_none_leaves_other_list_intact() -> None:
    ws = Workspace()
    _seed_spec(ws)
    # First set derived, then update only correlations — derived must survive.
    derived = [DerivedColumn(column="price", expression="qty * 2", depends_on=["qty"])]
    ws.update_table_advanced("orders", derived=derived)

    ws.update_table_advanced(
        "orders",
        correlations=[CorrelationRule(columns=["qty", "price"])],
    )

    orders = ws.get_spec().get_table_spec("orders")  # type: ignore[union-attr]
    assert orders is not None
    assert orders.derived == derived
    assert orders.correlations == [CorrelationRule(columns=["qty", "price"])]


def test_update_advanced_does_not_touch_sibling_tables() -> None:
    ws = Workspace()
    _seed_spec(ws)
    ws.update_table_advanced(
        "users",
        correlations=[CorrelationRule(columns=["city", "state"])],
    )
    after = ws.get_spec()
    assert after is not None
    orders = after.get_table_spec("orders")
    assert orders is not None
    assert orders.correlations == []
    assert orders.row_count == 200


def test_update_advanced_preserves_columns_and_row_count() -> None:
    ws = Workspace()
    _seed_spec(ws)
    ws.update_table_advanced(
        "users",
        correlations=[CorrelationRule(columns=["city", "state"])],
    )
    users = ws.get_spec().get_table_spec("users")  # type: ignore[union-attr]
    assert users is not None
    assert users.row_count == 100
    assert set(users.columns) == {"id", "city", "state"}


def test_update_advanced_preserves_top_level_metadata() -> None:
    ws = Workspace()
    spec = DataSpec(
        tables=[
            TableSpec(
                table_name="t",
                columns={"id": GeneratorConfig(provider="builtin.sequence", unique=True)},
            ),
        ],
        schema_hash="cafef00d",
        model_used="heuristic_fallback",
        global_seed=1234,
    )
    ws.set_spec(spec)
    ws.update_table_advanced("t", correlations=[])
    after = ws.get_spec()
    assert after is not None
    assert after.schema_hash == "cafef00d"
    assert after.model_used == "heuristic_fallback"
    assert after.global_seed == 1234


def test_update_advanced_no_spec_raises_valueerror() -> None:
    ws = Workspace()
    with pytest.raises(ValueError, match="no spec"):
        ws.update_table_advanced("users", correlations=[])


def test_update_advanced_unknown_table_raises_keyerror() -> None:
    ws = Workspace()
    _seed_spec(ws)
    with pytest.raises(KeyError, match="ghosts"):
        ws.update_table_advanced("ghosts", correlations=[])
