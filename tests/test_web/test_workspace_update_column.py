"""Tests for ``Workspace.update_column`` (S-119).

The web spec-edit endpoint mutates the active workspace spec **immutably** —
both ``DataSpec`` and ``TableSpec`` are ``frozen=True`` so the helper must
swap them out via ``model_copy(update=...)``. Mutation containment lives in
the workspace; the router only validates input and calls this helper.

These tests lock the helper contract: identity changes on success, no
in-place mutation of the original models, and clean ``LookupError`` raises
when the spec / table / column is missing (the router maps those to 404).
"""

from __future__ import annotations

import pytest

from dbsprout.spec.models import DataSpec, GeneratorConfig, TableSpec
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
                    "email": GeneratorConfig(provider="mimesis.email"),
                },
            ),
            TableSpec(
                table_name="orders",
                row_count=200,
                columns={
                    "id": GeneratorConfig(provider="builtin.sequence", unique=True),
                    "user_id": GeneratorConfig(provider="numpy.integer"),
                },
            ),
        ],
        schema_hash="deadbeef",
    )
    workspace.set_spec(spec)
    return spec


def test_update_column_happy_path_returns_new_config() -> None:
    ws = Workspace()
    _seed_spec(ws)
    new_cfg = GeneratorConfig(provider="mimesis.first_name", nullable_rate=0.2)

    returned = ws.update_column("users", "email", new_cfg)

    assert returned == new_cfg
    stored = ws.get_spec()
    assert stored is not None
    users = stored.get_table_spec("users")
    assert users is not None
    assert users.columns["email"] == new_cfg


def test_update_column_replaces_spec_immutably() -> None:
    ws = Workspace()
    original = _seed_spec(ws)
    new_cfg = GeneratorConfig(provider="mimesis.first_name")

    ws.update_column("users", "email", new_cfg)

    after = ws.get_spec()
    assert after is not None
    # Identity change — frozen ``DataSpec`` is replaced, not mutated.
    assert after is not original
    # The original spec object remains untouched (frozen invariants hold).
    original_users = original.get_table_spec("users")
    assert original_users is not None
    assert original_users.columns["email"] == GeneratorConfig(provider="mimesis.email")


def test_update_column_does_not_touch_sibling_columns() -> None:
    ws = Workspace()
    _seed_spec(ws)
    new_cfg = GeneratorConfig(provider="mimesis.first_name")

    ws.update_column("users", "email", new_cfg)
    after = ws.get_spec()
    assert after is not None
    users = after.get_table_spec("users")
    assert users is not None
    # Sibling column on the same table is untouched.
    assert users.columns["id"] == GeneratorConfig(provider="builtin.sequence", unique=True)


def test_update_column_does_not_touch_sibling_tables() -> None:
    ws = Workspace()
    _seed_spec(ws)
    new_cfg = GeneratorConfig(provider="mimesis.first_name")

    ws.update_column("users", "email", new_cfg)
    after = ws.get_spec()
    assert after is not None
    orders = after.get_table_spec("orders")
    assert orders is not None
    assert orders.columns["user_id"] == GeneratorConfig(provider="numpy.integer")
    assert orders.row_count == 200


def test_update_column_no_spec_raises_lookuperror() -> None:
    ws = Workspace()
    with pytest.raises(LookupError, match="no spec"):
        ws.update_column("users", "email", GeneratorConfig(provider="mimesis.email"))


def test_update_column_unknown_table_raises_lookuperror() -> None:
    ws = Workspace()
    _seed_spec(ws)
    with pytest.raises(LookupError, match="ghosts"):
        ws.update_column("ghosts", "id", GeneratorConfig(provider="mimesis.email"))


def test_update_column_unknown_column_raises_lookuperror() -> None:
    ws = Workspace()
    _seed_spec(ws)
    with pytest.raises(LookupError, match="ghost_col"):
        ws.update_column("users", "ghost_col", GeneratorConfig(provider="mimesis.email"))


def test_update_column_preserves_top_level_metadata() -> None:
    """``schema_hash``, ``model_used``, ``global_seed`` survive the swap."""
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
    ws.update_column(
        "t",
        "id",
        GeneratorConfig(provider="numpy.integer", unique=True),
    )
    after = ws.get_spec()
    assert after is not None
    assert after.schema_hash == "cafef00d"
    assert after.model_used == "heuristic_fallback"
    assert after.global_seed == 1234
