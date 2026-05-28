"""Tests for :mod:`dbsprout.spec.constraints` (S-119).

The web spec-edit endpoint must reject changes that would break referential
integrity invariants: a PK column or any column referenced by a foreign key
must remain ``unique=True`` (PK / FK-target columns *are* the join columns
the generator relies on). The constraint guard is a pure function on the
schema + new config — no workspace state, no Pydantic coupling beyond the
already-validated ``GeneratorConfig``.

These tests lock the rule set the router consumes.
"""

from __future__ import annotations

from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    TableSchema,
)
from dbsprout.spec.constraints import check_column_update
from dbsprout.spec.models import GeneratorConfig


def _schema_pk_and_fk() -> DatabaseSchema:
    users = TableSchema(
        name="users",
        columns=[
            ColumnSchema(name="id", data_type=ColumnType.INTEGER, primary_key=True),
            ColumnSchema(name="email", data_type=ColumnType.VARCHAR, max_length=255),
        ],
        primary_key=["id"],
    )
    orders = TableSchema(
        name="orders",
        columns=[
            ColumnSchema(name="id", data_type=ColumnType.INTEGER, primary_key=True),
            ColumnSchema(name="user_id", data_type=ColumnType.INTEGER),
        ],
        primary_key=["id"],
        foreign_keys=[
            ForeignKeySchema(
                columns=("user_id",),
                ref_table="users",
                ref_columns=("id",),
            ),
        ],
    )
    return DatabaseSchema(tables=[users, orders])


def test_pk_keeping_unique_is_allowed() -> None:
    schema = _schema_pk_and_fk()
    cfg = GeneratorConfig(provider="builtin.sequence", unique=True)
    assert check_column_update(schema, "users", "id", cfg) is None


def test_pk_dropping_unique_is_rejected_mentions_primary_key() -> None:
    schema = _schema_pk_and_fk()
    cfg = GeneratorConfig(provider="mimesis.full_name", unique=False)
    reason = check_column_update(schema, "users", "id", cfg)
    assert reason is not None
    assert "primary key" in reason.lower()


def test_fk_target_column_dropping_unique_is_rejected_mentions_foreign_key() -> None:
    """``users.id`` is a PK *and* the target of orders.user_id → uniqueness must stay."""
    schema = _schema_pk_and_fk()
    cfg = GeneratorConfig(provider="mimesis.full_name", unique=False)
    reason = check_column_update(schema, "users", "id", cfg)
    assert reason is not None
    # The reason must surface at least one of the violated invariants;
    # for a PK *and* FK-target column the message names the primary key
    # because the PK guard fires first (deterministic ordering).
    assert "primary key" in reason.lower() or "foreign key" in reason.lower()


def test_fk_target_non_pk_column_dropping_unique_is_rejected() -> None:
    """A column referenced by an FK but not itself a PK still needs uniqueness."""
    parent = TableSchema(
        name="parent",
        columns=[
            ColumnSchema(name="id", data_type=ColumnType.INTEGER, primary_key=True),
            ColumnSchema(name="external_key", data_type=ColumnType.VARCHAR, unique=True),
        ],
        primary_key=["id"],
    )
    child = TableSchema(
        name="child",
        columns=[
            ColumnSchema(name="id", data_type=ColumnType.INTEGER, primary_key=True),
            ColumnSchema(name="ext", data_type=ColumnType.VARCHAR),
        ],
        primary_key=["id"],
        foreign_keys=[
            ForeignKeySchema(
                columns=("ext",),
                ref_table="parent",
                ref_columns=("external_key",),
            ),
        ],
    )
    schema = DatabaseSchema(tables=[parent, child])
    cfg = GeneratorConfig(provider="mimesis.word", unique=False)
    reason = check_column_update(schema, "parent", "external_key", cfg)
    assert reason is not None
    assert "foreign key" in reason.lower()


def test_self_referential_fk_target_dropping_unique_is_rejected() -> None:
    employees = TableSchema(
        name="employees",
        columns=[
            ColumnSchema(name="id", data_type=ColumnType.INTEGER, primary_key=True),
            ColumnSchema(name="manager_id", data_type=ColumnType.INTEGER),
        ],
        primary_key=["id"],
        foreign_keys=[
            ForeignKeySchema(
                columns=("manager_id",),
                ref_table="employees",
                ref_columns=("id",),
            ),
        ],
    )
    schema = DatabaseSchema(tables=[employees])
    cfg = GeneratorConfig(provider="numpy.integer", unique=False)
    reason = check_column_update(schema, "employees", "id", cfg)
    assert reason is not None


def test_non_pk_non_target_column_change_is_allowed() -> None:
    """Plain data columns can be tuned freely."""
    schema = _schema_pk_and_fk()
    cfg = GeneratorConfig(provider="mimesis.email", unique=False, nullable_rate=0.1)
    assert check_column_update(schema, "users", "email", cfg) is None


def test_pk_provider_swap_keeping_unique_is_allowed() -> None:
    """Swapping provider/method on a PK is fine as long as ``unique`` stays."""
    schema = _schema_pk_and_fk()
    cfg = GeneratorConfig(provider="numpy.integer", method="sequence", unique=True)
    assert check_column_update(schema, "users", "id", cfg) is None


def test_missing_table_returns_reason() -> None:
    schema = _schema_pk_and_fk()
    cfg = GeneratorConfig(provider="mimesis.email")
    reason = check_column_update(schema, "ghosts", "id", cfg)
    assert reason is not None
    assert "ghosts" in reason


def test_missing_column_returns_reason() -> None:
    schema = _schema_pk_and_fk()
    cfg = GeneratorConfig(provider="mimesis.email")
    reason = check_column_update(schema, "users", "ghost_col", cfg)
    assert reason is not None
    assert "ghost_col" in reason


def test_fk_source_column_change_is_allowed() -> None:
    """The FK *source* column (orders.user_id) is sampled from parent PKs at
    generation time — its own ``unique`` flag doesn't affect referential
    integrity, so it stays freely editable.
    """
    schema = _schema_pk_and_fk()
    cfg = GeneratorConfig(provider="numpy.integer", unique=False)
    assert check_column_update(schema, "orders", "user_id", cfg) is None
