"""Workspace.update_table_row_count immutability tests (S-121).

The helper performs an immutable swap on the in-memory ``DataSpec``: it
replaces the matching ``TableSpec`` via ``model_copy(update=...)`` and
rebuilds the ``DataSpec.tables`` list with the same ordering, then stores a
fresh ``DataSpec`` on the workspace. The tests below pin that contract:

* the returned ``row_count`` is the new value,
* the workspace ``DataSpec`` object identity changes (proving immutability),
* sibling tables' ``row_count`` is preserved,
* ``ValueError`` is raised when no spec is loaded (matches ``update_spec``),
* ``KeyError`` is raised when the table is absent from the spec.
"""

from __future__ import annotations

import pytest

from dbsprout.spec.models import DataSpec, GeneratorConfig, TableSpec
from dbsprout.web.workspace import Workspace


def _two_table_spec() -> DataSpec:
    return DataSpec(
        tables=[
            TableSpec(
                table_name="users",
                row_count=100,
                columns={"id": GeneratorConfig(provider="seq", method="int")},
            ),
            TableSpec(
                table_name="orders",
                row_count=500,
                columns={"id": GeneratorConfig(provider="seq", method="int")},
            ),
        ],
    )


def test_update_table_row_count_returns_new_value() -> None:
    ws = Workspace()
    ws.set_spec(_two_table_spec())

    new_value = ws.update_table_row_count("users", 250)

    assert new_value == 250


def test_update_table_row_count_replaces_spec_object() -> None:
    ws = Workspace()
    ws.set_spec(_two_table_spec())
    original_spec = ws.get_spec()

    ws.update_table_row_count("users", 250)

    updated_spec = ws.get_spec()
    assert updated_spec is not None
    assert updated_spec is not original_spec, "spec must be a fresh object (immutability)"
    users = updated_spec.get_table_spec("users")
    assert users is not None
    assert users.row_count == 250


def test_update_table_row_count_preserves_other_tables() -> None:
    ws = Workspace()
    ws.set_spec(_two_table_spec())

    ws.update_table_row_count("users", 250)

    spec = ws.get_spec()
    assert spec is not None
    orders = spec.get_table_spec("orders")
    assert orders is not None
    assert orders.row_count == 500, "sibling table's row_count must be untouched"


def test_update_table_row_count_preserves_table_order() -> None:
    ws = Workspace()
    ws.set_spec(_two_table_spec())

    ws.update_table_row_count("orders", 999)

    spec = ws.get_spec()
    assert spec is not None
    assert [t.table_name for t in spec.tables] == ["users", "orders"]


def test_update_table_row_count_no_spec_raises_value_error() -> None:
    ws = Workspace()

    with pytest.raises(ValueError, match="no spec loaded"):
        ws.update_table_row_count("users", 100)


def test_update_table_row_count_unknown_table_raises_key_error() -> None:
    ws = Workspace()
    ws.set_spec(_two_table_spec())

    with pytest.raises(KeyError):
        ws.update_table_row_count("ghosts", 100)


def test_update_table_row_count_does_not_mutate_original_table_spec() -> None:
    """Frozen ``TableSpec`` instances must never be mutated in place."""
    ws = Workspace()
    ws.set_spec(_two_table_spec())
    original_users = ws.get_spec().tables[0]  # type: ignore[union-attr]
    assert original_users.row_count == 100

    ws.update_table_row_count("users", 250)

    # The original TableSpec object still has the old row_count.
    assert original_users.row_count == 100
