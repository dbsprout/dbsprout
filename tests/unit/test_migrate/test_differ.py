"""Tests for dbsprout.migrate.differ — schema diff algorithm."""

from __future__ import annotations

import pytest

from dbsprout.migrate.differ import ENUM_TABLE_SENTINEL, SchemaDiffer
from dbsprout.migrate.models import SchemaChange, SchemaChangeType
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    IndexSchema,
    TableSchema,
)

# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def minimal_schema() -> DatabaseSchema:
    """Single-table schema: users(id INTEGER PK)."""
    col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)
    table = TableSchema(name="users", columns=[col], primary_key=["id"])
    return DatabaseSchema(tables=[table], dialect="sqlite")


@pytest.fixture
def two_table_schema() -> DatabaseSchema:
    """Two-table schema: users + orders with FK."""
    id_col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)
    user_id_col = ColumnSchema(name="user_id", data_type=ColumnType.INTEGER, nullable=False)
    users = TableSchema(name="users", columns=[id_col], primary_key=["id"])
    orders = TableSchema(
        name="orders",
        columns=[id_col, user_id_col],
        primary_key=["id"],
        foreign_keys=[ForeignKeySchema(columns=["user_id"], ref_table="users", ref_columns=["id"])],
    )
    return DatabaseSchema(tables=[users, orders], dialect="sqlite")


def _find_change(
    changes: list[SchemaChange],
    change_type: SchemaChangeType,
    table_name: str,
    column_name: str | None = None,
) -> SchemaChange | None:
    """Helper to find a specific change in the list."""
    for c in changes:
        if (
            c.change_type == change_type
            and c.table_name == table_name
            and (column_name is None or c.column_name == column_name)
        ):
            return c
    return None


# ── Identical / empty schema tests (AC-2) ────────────────────────────────


class TestIdenticalSchemas:
    def test_same_object_returns_empty_list(self, minimal_schema: DatabaseSchema) -> None:
        assert SchemaDiffer.diff(minimal_schema, minimal_schema) == []

    def test_equal_objects_returns_empty_list(self) -> None:
        col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)
        table = TableSchema(name="users", columns=[col], primary_key=["id"])
        schema_a = DatabaseSchema(tables=[table], dialect="sqlite")
        schema_b = DatabaseSchema(tables=[table], dialect="sqlite")
        assert SchemaDiffer.diff(schema_a, schema_b) == []

    def test_both_empty_returns_empty_list(self) -> None:
        empty = DatabaseSchema(tables=[])
        assert SchemaDiffer.diff(empty, empty) == []


class TestEmptySchemas:
    def test_empty_to_populated(self, minimal_schema: DatabaseSchema) -> None:
        empty = DatabaseSchema(tables=[])
        changes = SchemaDiffer.diff(empty, minimal_schema)
        assert len(changes) == 1
        assert changes[0].change_type == SchemaChangeType.TABLE_ADDED
        assert changes[0].table_name == "users"

    def test_populated_to_empty(self, minimal_schema: DatabaseSchema) -> None:
        empty = DatabaseSchema(tables=[])
        changes = SchemaDiffer.diff(minimal_schema, empty)
        assert len(changes) == 1
        assert changes[0].change_type == SchemaChangeType.TABLE_REMOVED
        assert changes[0].table_name == "users"


# ── Table-level changes (AC-3, AC-4) ─────────────────────────────────────


class TestTableChanges:
    def test_table_added(self, minimal_schema: DatabaseSchema) -> None:
        col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)
        products = TableSchema(name="products", columns=[col], primary_key=["id"])
        new_schema = minimal_schema.model_copy(
            update={"tables": [*minimal_schema.tables, products]}
        )
        changes = SchemaDiffer.diff(minimal_schema, new_schema)
        added = _find_change(changes, SchemaChangeType.TABLE_ADDED, "products")
        assert added is not None
        assert added.detail is not None
        assert "table" in added.detail

    def test_table_removed(self, two_table_schema: DatabaseSchema) -> None:
        new_schema = two_table_schema.model_copy(
            update={"tables": [two_table_schema.tables[0]]}  # keep only users
        )
        changes = SchemaDiffer.diff(two_table_schema, new_schema)
        removed = _find_change(changes, SchemaChangeType.TABLE_REMOVED, "orders")
        assert removed is not None
        assert removed.detail is not None
        assert "table" in removed.detail

    def test_multiple_tables_added_and_removed(self) -> None:
        col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)
        old = DatabaseSchema(
            tables=[
                TableSchema(name="alpha", columns=[col], primary_key=["id"]),
                TableSchema(name="beta", columns=[col], primary_key=["id"]),
            ]
        )
        new = DatabaseSchema(
            tables=[
                TableSchema(name="beta", columns=[col], primary_key=["id"]),
                TableSchema(name="gamma", columns=[col], primary_key=["id"]),
                TableSchema(name="delta", columns=[col], primary_key=["id"]),
            ]
        )
        changes = SchemaDiffer.diff(old, new)
        added = [c for c in changes if c.change_type == SchemaChangeType.TABLE_ADDED]
        removed = [c for c in changes if c.change_type == SchemaChangeType.TABLE_REMOVED]
        assert len(added) == 2
        assert {c.table_name for c in added} == {"gamma", "delta"}
        assert len(removed) == 1
        assert removed[0].table_name == "alpha"


# ── Column-level changes (AC-5, AC-6, AC-7, AC-8, AC-9, AC-17) ──────────


class TestColumnChanges:
    def test_column_added(self, minimal_schema: DatabaseSchema) -> None:
        old_table = minimal_schema.tables[0]
        email_col = ColumnSchema(name="email", data_type=ColumnType.VARCHAR)
        new_table = old_table.model_copy(update={"columns": [*old_table.columns, email_col]})
        new_schema = minimal_schema.model_copy(update={"tables": [new_table]})
        changes = SchemaDiffer.diff(minimal_schema, new_schema)
        added = _find_change(changes, SchemaChangeType.COLUMN_ADDED, "users", "email")
        assert added is not None
        assert added.detail is not None
        assert "column" in added.detail

    def test_column_removed(self) -> None:
        id_col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)
        email_col = ColumnSchema(name="email", data_type=ColumnType.VARCHAR)
        old = DatabaseSchema(
            tables=[TableSchema(name="users", columns=[id_col, email_col], primary_key=["id"])]
        )
        new = DatabaseSchema(
            tables=[TableSchema(name="users", columns=[id_col], primary_key=["id"])]
        )
        changes = SchemaDiffer.diff(old, new)
        removed = _find_change(changes, SchemaChangeType.COLUMN_REMOVED, "users", "email")
        assert removed is not None
        assert removed.detail is not None
        assert "column" in removed.detail

    def test_column_type_changed(self) -> None:
        id_col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)
        old_email = ColumnSchema(name="email", data_type=ColumnType.VARCHAR)
        new_email = ColumnSchema(name="email", data_type=ColumnType.TEXT)
        old = DatabaseSchema(
            tables=[TableSchema(name="users", columns=[id_col, old_email], primary_key=["id"])]
        )
        new = DatabaseSchema(
            tables=[TableSchema(name="users", columns=[id_col, new_email], primary_key=["id"])]
        )
        changes = SchemaDiffer.diff(old, new)
        changed = _find_change(changes, SchemaChangeType.COLUMN_TYPE_CHANGED, "users", "email")
        assert changed is not None
        assert changed.old_value == "varchar"
        assert changed.new_value == "text"

    def test_column_nullability_changed(self) -> None:
        id_col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)
        old_email = ColumnSchema(name="email", data_type=ColumnType.VARCHAR, nullable=True)
        new_email = ColumnSchema(name="email", data_type=ColumnType.VARCHAR, nullable=False)
        old = DatabaseSchema(
            tables=[TableSchema(name="users", columns=[id_col, old_email], primary_key=["id"])]
        )
        new = DatabaseSchema(
            tables=[TableSchema(name="users", columns=[id_col, new_email], primary_key=["id"])]
        )
        changes = SchemaDiffer.diff(old, new)
        changed = _find_change(
            changes, SchemaChangeType.COLUMN_NULLABILITY_CHANGED, "users", "email"
        )
        assert changed is not None
        assert changed.detail is not None
        assert changed.detail["old_nullable"] is True
        assert changed.detail["new_nullable"] is False

    def test_column_default_changed(self) -> None:
        id_col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)
        old_status = ColumnSchema(name="status", data_type=ColumnType.VARCHAR, default=None)
        new_status = ColumnSchema(name="status", data_type=ColumnType.VARCHAR, default="'active'")
        old = DatabaseSchema(
            tables=[TableSchema(name="users", columns=[id_col, old_status], primary_key=["id"])]
        )
        new = DatabaseSchema(
            tables=[TableSchema(name="users", columns=[id_col, new_status], primary_key=["id"])]
        )
        changes = SchemaDiffer.diff(old, new)
        changed = _find_change(changes, SchemaChangeType.COLUMN_DEFAULT_CHANGED, "users", "status")
        assert changed is not None
        assert changed.detail is not None
        assert changed.detail["old_default"] is None
        assert changed.detail["new_default"] == "'active'"

    def test_renamed_column_is_remove_plus_add(self) -> None:
        """AC-17: renamed column → COLUMN_REMOVED + COLUMN_ADDED."""
        id_col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)
        old = DatabaseSchema(
            tables=[
                TableSchema(
                    name="users",
                    columns=[
                        id_col,
                        ColumnSchema(name="email", data_type=ColumnType.VARCHAR),
                    ],
                    primary_key=["id"],
                )
            ]
        )
        new = DatabaseSchema(
            tables=[
                TableSchema(
                    name="users",
                    columns=[
                        id_col,
                        ColumnSchema(name="email_address", data_type=ColumnType.VARCHAR),
                    ],
                    primary_key=["id"],
                )
            ]
        )
        changes = SchemaDiffer.diff(old, new)
        removed = _find_change(changes, SchemaChangeType.COLUMN_REMOVED, "users", "email")
        added = _find_change(changes, SchemaChangeType.COLUMN_ADDED, "users", "email_address")
        assert removed is not None
        assert added is not None


# ── FK changes (AC-10, AC-11, AC-19) ─────────────────────────────────────


class TestForeignKeyChanges:
    def test_foreign_key_added(self) -> None:
        id_col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)
        uid_col = ColumnSchema(name="user_id", data_type=ColumnType.INTEGER, nullable=False)
        users = TableSchema(name="users", columns=[id_col], primary_key=["id"])
        old_orders = TableSchema(name="orders", columns=[id_col, uid_col], primary_key=["id"])
        new_orders = TableSchema(
            name="orders",
            columns=[id_col, uid_col],
            primary_key=["id"],
            foreign_keys=[
                ForeignKeySchema(columns=["user_id"], ref_table="users", ref_columns=["id"])
            ],
        )
        old = DatabaseSchema(tables=[users, old_orders])
        new = DatabaseSchema(tables=[users, new_orders])
        changes = SchemaDiffer.diff(old, new)
        added = _find_change(changes, SchemaChangeType.FOREIGN_KEY_ADDED, "orders")
        assert added is not None
        assert added.detail is not None
        assert "fk" in added.detail

    def test_foreign_key_removed(self) -> None:
        id_col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)
        uid_col = ColumnSchema(name="user_id", data_type=ColumnType.INTEGER, nullable=False)
        users = TableSchema(name="users", columns=[id_col], primary_key=["id"])
        old_orders = TableSchema(
            name="orders",
            columns=[id_col, uid_col],
            primary_key=["id"],
            foreign_keys=[
                ForeignKeySchema(columns=["user_id"], ref_table="users", ref_columns=["id"])
            ],
        )
        new_orders = TableSchema(name="orders", columns=[id_col, uid_col], primary_key=["id"])
        old = DatabaseSchema(tables=[users, old_orders])
        new = DatabaseSchema(tables=[users, new_orders])
        changes = SchemaDiffer.diff(old, new)
        removed = _find_change(changes, SchemaChangeType.FOREIGN_KEY_REMOVED, "orders")
        assert removed is not None
        assert removed.detail is not None
        assert "fk" in removed.detail

    def test_self_referencing_fk_added(self) -> None:
        """AC-19: self-referencing FK detected correctly."""
        id_col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)
        parent_col = ColumnSchema(name="parent_id", data_type=ColumnType.INTEGER, nullable=True)
        old = DatabaseSchema(
            tables=[
                TableSchema(
                    name="categories",
                    columns=[id_col, parent_col],
                    primary_key=["id"],
                )
            ]
        )
        new = DatabaseSchema(
            tables=[
                TableSchema(
                    name="categories",
                    columns=[id_col, parent_col],
                    primary_key=["id"],
                    foreign_keys=[
                        ForeignKeySchema(
                            columns=["parent_id"],
                            ref_table="categories",
                            ref_columns=["id"],
                        )
                    ],
                )
            ]
        )
        changes = SchemaDiffer.diff(old, new)
        added = _find_change(changes, SchemaChangeType.FOREIGN_KEY_ADDED, "categories")
        assert added is not None

    def test_fk_on_delete_change_not_detected(self) -> None:
        """FK metadata (on_delete, deferrable) change is not tracked; structural identity only."""
        id_col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)
        uid_col = ColumnSchema(name="user_id", data_type=ColumnType.INTEGER, nullable=False)
        users = TableSchema(name="users", columns=[id_col], primary_key=["id"])
        old_orders = TableSchema(
            name="orders",
            columns=[id_col, uid_col],
            primary_key=["id"],
            foreign_keys=[
                ForeignKeySchema(
                    columns=["user_id"],
                    ref_table="users",
                    ref_columns=["id"],
                    on_delete="CASCADE",
                )
            ],
        )
        new_orders = TableSchema(
            name="orders",
            columns=[id_col, uid_col],
            primary_key=["id"],
            foreign_keys=[
                ForeignKeySchema(
                    columns=["user_id"],
                    ref_table="users",
                    ref_columns=["id"],
                    on_delete="SET NULL",
                )
            ],
        )
        old = DatabaseSchema(tables=[users, old_orders])
        new = DatabaseSchema(tables=[users, new_orders])
        changes = SchemaDiffer.diff(old, new)
        fk_changes = [
            c
            for c in changes
            if c.change_type
            in (SchemaChangeType.FOREIGN_KEY_ADDED, SchemaChangeType.FOREIGN_KEY_REMOVED)
        ]
        assert len(fk_changes) == 0

    def test_composite_fk(self) -> None:
        """AC-19: multi-column FK identity based on column set."""
        id_col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)
        oid_col = ColumnSchema(name="order_id", data_type=ColumnType.INTEGER, nullable=False)
        pid_col = ColumnSchema(name="product_id", data_type=ColumnType.INTEGER, nullable=False)
        old = DatabaseSchema(
            tables=[
                TableSchema(
                    name="line_items",
                    columns=[id_col, oid_col, pid_col],
                    primary_key=["id"],
                )
            ]
        )
        new = DatabaseSchema(
            tables=[
                TableSchema(
                    name="line_items",
                    columns=[id_col, oid_col, pid_col],
                    primary_key=["id"],
                    foreign_keys=[
                        ForeignKeySchema(
                            columns=["order_id", "product_id"],
                            ref_table="order_products",
                            ref_columns=["order_id", "product_id"],
                        )
                    ],
                )
            ]
        )
        changes = SchemaDiffer.diff(old, new)
        added = _find_change(changes, SchemaChangeType.FOREIGN_KEY_ADDED, "line_items")
        assert added is not None


# ── Index changes (AC-12, AC-13) ──────────────────────────────────────────


class TestIndexChanges:
    def test_index_added(self) -> None:
        id_col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)
        email_col = ColumnSchema(name="email", data_type=ColumnType.VARCHAR)
        old = DatabaseSchema(
            tables=[TableSchema(name="users", columns=[id_col, email_col], primary_key=["id"])]
        )
        new = DatabaseSchema(
            tables=[
                TableSchema(
                    name="users",
                    columns=[id_col, email_col],
                    primary_key=["id"],
                    indexes=[IndexSchema(columns=["email"], unique=True)],
                )
            ]
        )
        changes = SchemaDiffer.diff(old, new)
        added = _find_change(changes, SchemaChangeType.INDEX_ADDED, "users")
        assert added is not None
        assert added.detail is not None
        assert "index" in added.detail

    def test_index_uniqueness_change_is_remove_plus_add(self) -> None:
        """Changing unique flag → INDEX_REMOVED + INDEX_ADDED."""
        id_col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)
        email_col = ColumnSchema(name="email", data_type=ColumnType.VARCHAR)
        old = DatabaseSchema(
            tables=[
                TableSchema(
                    name="users",
                    columns=[id_col, email_col],
                    primary_key=["id"],
                    indexes=[IndexSchema(columns=["email"], unique=False)],
                )
            ]
        )
        new = DatabaseSchema(
            tables=[
                TableSchema(
                    name="users",
                    columns=[id_col, email_col],
                    primary_key=["id"],
                    indexes=[IndexSchema(columns=["email"], unique=True)],
                )
            ]
        )
        changes = SchemaDiffer.diff(old, new)
        added = _find_change(changes, SchemaChangeType.INDEX_ADDED, "users")
        removed = _find_change(changes, SchemaChangeType.INDEX_REMOVED, "users")
        assert added is not None
        assert removed is not None

    def test_index_removed(self) -> None:
        id_col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)
        email_col = ColumnSchema(name="email", data_type=ColumnType.VARCHAR)
        old = DatabaseSchema(
            tables=[
                TableSchema(
                    name="users",
                    columns=[id_col, email_col],
                    primary_key=["id"],
                    indexes=[IndexSchema(columns=["email"], unique=True)],
                )
            ]
        )
        new = DatabaseSchema(
            tables=[TableSchema(name="users", columns=[id_col, email_col], primary_key=["id"])]
        )
        changes = SchemaDiffer.diff(old, new)
        removed = _find_change(changes, SchemaChangeType.INDEX_REMOVED, "users")
        assert removed is not None
        assert removed.detail is not None
        assert "index" in removed.detail


# ── Enum changes (AC-20) ─────────────────────────────────────────────────


class TestEnumChanges:
    def test_enum_added(self) -> None:
        old = DatabaseSchema(tables=[])
        new = DatabaseSchema(tables=[], enums={"status": ["active", "inactive"]})
        changes = SchemaDiffer.diff(old, new)
        changed = _find_change(changes, SchemaChangeType.ENUM_CHANGED, ENUM_TABLE_SENTINEL)
        assert changed is not None
        assert changed.detail is not None
        assert changed.detail["old_values"] is None
        assert set(changed.detail["new_values"]) == {"active", "inactive"}

    def test_enum_removed(self) -> None:
        old = DatabaseSchema(tables=[], enums={"status": ["active", "inactive"]})
        new = DatabaseSchema(tables=[])
        changes = SchemaDiffer.diff(old, new)
        changed = _find_change(changes, SchemaChangeType.ENUM_CHANGED, ENUM_TABLE_SENTINEL)
        assert changed is not None
        assert changed.detail is not None
        assert set(changed.detail["old_values"]) == {"active", "inactive"}
        assert changed.detail["new_values"] is None

    def test_enum_values_changed(self) -> None:
        old = DatabaseSchema(tables=[], enums={"status": ["active", "inactive"]})
        new = DatabaseSchema(tables=[], enums={"status": ["active", "inactive", "pending"]})
        changes = SchemaDiffer.diff(old, new)
        changed = _find_change(changes, SchemaChangeType.ENUM_CHANGED, ENUM_TABLE_SENTINEL)
        assert changed is not None
        assert changed.detail is not None
        assert set(changed.detail["old_values"]) == {"active", "inactive"}
        assert set(changed.detail["new_values"]) == {"active", "inactive", "pending"}

    def test_enum_values_reordered_no_change(self) -> None:
        old = DatabaseSchema(tables=[], enums={"status": ["active", "inactive"]})
        new = DatabaseSchema(tables=[], enums={"status": ["inactive", "active"]})
        changes = SchemaDiffer.diff(old, new)
        enum_changes = [c for c in changes if c.change_type == SchemaChangeType.ENUM_CHANGED]
        assert len(enum_changes) == 0


# ── Edge cases and ordering (AC-17, AC-18, AC-19) ────────────────────────


class TestEdgeCases:
    def test_ordering_independence(self) -> None:
        """AC-18: same content in different order → no changes."""
        id_col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)
        name_col = ColumnSchema(name="name", data_type=ColumnType.VARCHAR)
        users = TableSchema(name="users", columns=[id_col, name_col], primary_key=["id"])
        orders = TableSchema(name="orders", columns=[id_col], primary_key=["id"])
        schema_a = DatabaseSchema(tables=[users, orders])
        schema_b = DatabaseSchema(tables=[orders, users])
        assert SchemaDiffer.diff(schema_a, schema_b) == []

    def test_ordering_independence_with_changes(self) -> None:
        """AC-18: shuffled order + one column added → same single change."""
        id_col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)
        name_col = ColumnSchema(name="name", data_type=ColumnType.VARCHAR)
        email_col = ColumnSchema(name="email", data_type=ColumnType.VARCHAR)
        users_old = TableSchema(name="users", columns=[id_col, name_col], primary_key=["id"])
        users_new = TableSchema(
            name="users", columns=[id_col, name_col, email_col], primary_key=["id"]
        )
        orders = TableSchema(name="orders", columns=[id_col], primary_key=["id"])
        old = DatabaseSchema(tables=[users_old, orders])
        new = DatabaseSchema(tables=[orders, users_new])  # reversed order
        changes = SchemaDiffer.diff(old, new)
        assert len(changes) == 1
        assert changes[0].change_type == SchemaChangeType.COLUMN_ADDED
        assert changes[0].table_name == "users"
        assert changes[0].column_name == "email"

    def test_multiple_simultaneous_changes(self) -> None:
        """Multiple change types in one diff."""
        id_col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)
        email_col = ColumnSchema(name="email", data_type=ColumnType.VARCHAR)
        old = DatabaseSchema(
            tables=[
                TableSchema(name="users", columns=[id_col, email_col], primary_key=["id"]),
                TableSchema(name="orders", columns=[id_col], primary_key=["id"]),
            ],
            enums={"status": ["a", "b"]},
        )
        new = DatabaseSchema(
            tables=[
                TableSchema(name="users", columns=[id_col], primary_key=["id"]),  # email removed
                TableSchema(name="products", columns=[id_col], primary_key=["id"]),  # new table
            ],
            enums={"status": ["a", "b", "c"]},  # enum changed
        )
        changes = SchemaDiffer.diff(old, new)
        types = {c.change_type for c in changes}
        assert SchemaChangeType.TABLE_ADDED in types  # products
        assert SchemaChangeType.TABLE_REMOVED in types  # orders
        assert SchemaChangeType.COLUMN_REMOVED in types  # email
        assert SchemaChangeType.ENUM_CHANGED in types

    def test_changes_are_deterministically_ordered(self) -> None:
        """Changes should be sorted by table name for predictability."""
        id_col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)
        old = DatabaseSchema(tables=[])
        new = DatabaseSchema(
            tables=[
                TableSchema(name="zebra", columns=[id_col], primary_key=["id"]),
                TableSchema(name="alpha", columns=[id_col], primary_key=["id"]),
                TableSchema(name="middle", columns=[id_col], primary_key=["id"]),
            ]
        )
        changes = SchemaDiffer.diff(old, new)
        table_names = [c.table_name for c in changes]
        assert table_names == sorted(table_names)

    def test_cross_type_sort_ordering(self) -> None:
        """Change types are ordered: table > column > FK > index > enum."""
        id_col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)
        email_col = ColumnSchema(name="email", data_type=ColumnType.VARCHAR)
        old = DatabaseSchema(
            tables=[
                TableSchema(
                    name="users",
                    columns=[id_col, email_col],
                    primary_key=["id"],
                    indexes=[IndexSchema(columns=["email"], unique=True)],
                ),
            ],
            enums={"role": ["admin", "user"]},
        )
        new = DatabaseSchema(
            tables=[
                TableSchema(name="users", columns=[id_col], primary_key=["id"]),
                TableSchema(name="products", columns=[id_col], primary_key=["id"]),
            ],
            enums={"role": ["admin", "user", "guest"]},
        )
        changes = SchemaDiffer.diff(old, new)
        change_types = [c.change_type for c in changes]
        assert change_types.index(SchemaChangeType.TABLE_ADDED) < change_types.index(
            SchemaChangeType.COLUMN_REMOVED
        )
        assert change_types.index(SchemaChangeType.COLUMN_REMOVED) < change_types.index(
            SchemaChangeType.INDEX_REMOVED
        )
        assert change_types.index(SchemaChangeType.INDEX_REMOVED) < change_types.index(
            SchemaChangeType.ENUM_CHANGED
        )

    def test_composite_primary_key_with_column_added(self) -> None:
        id_col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)
        tid_col = ColumnSchema(name="tenant_id", data_type=ColumnType.INTEGER, nullable=False)
        old = DatabaseSchema(
            tables=[
                TableSchema(
                    name="items",
                    columns=[id_col, tid_col],
                    primary_key=["id", "tenant_id"],
                )
            ]
        )
        new_col = ColumnSchema(name="name", data_type=ColumnType.VARCHAR)
        new = DatabaseSchema(
            tables=[
                TableSchema(
                    name="items",
                    columns=[id_col, tid_col, new_col],
                    primary_key=["id", "tenant_id"],
                )
            ]
        )
        changes = SchemaDiffer.diff(old, new)
        assert len(changes) == 1
        assert changes[0].change_type == SchemaChangeType.COLUMN_ADDED
        assert changes[0].column_name == "name"
