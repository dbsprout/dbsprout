"""Tests for dbsprout.migrate.updater — incremental seed updater."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from dbsprout.config.models import DBSproutConfig
from dbsprout.migrate.models import SchemaChange, SchemaChangeType
from dbsprout.migrate.update_models import UpdateAction, UpdatePlan, UpdateResult
from dbsprout.migrate.updater import IncrementalUpdater
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    TableSchema,
)


@pytest.fixture
def updater(minimal_schema: DatabaseSchema) -> IncrementalUpdater:
    """IncrementalUpdater with a minimal schema, default config, and fixed seed."""
    config = DBSproutConfig()
    return IncrementalUpdater(schema=minimal_schema, config=config, seed=42)


# ── Constructor ─────────────────────────────────────────────────────


class TestInit:
    """Verify constructor stores schema, config, and seed."""

    def test_init_stores_schema_config_seed(self, minimal_schema: DatabaseSchema) -> None:
        config = DBSproutConfig()
        updater = IncrementalUpdater(schema=minimal_schema, config=config, seed=99)

        assert updater._schema is minimal_schema
        assert updater._config is config
        assert updater._seed == 99


# ── plan() — empty input ────────────────────────────────────────────


class TestPlanEmpty:
    """plan([]) returns an empty UpdatePlan."""

    def test_plan_empty_changes(self, updater: IncrementalUpdater) -> None:
        result = updater.plan([], {})

        assert isinstance(result, UpdatePlan)
        assert result.actions == []


# ── plan() — column-level changes ───────────────────────────────────


class TestPlanColumnChanges:
    """Each column-level SchemaChangeType maps to the correct UpdateAction."""

    def test_plan_column_added(self, updater: IncrementalUpdater) -> None:
        change = SchemaChange(
            change_type=SchemaChangeType.COLUMN_ADDED,
            table_name="users",
            column_name="email",
            new_value="varchar",
            detail={"column": {"name": "email", "data_type": "varchar", "nullable": True}},
        )
        plan = updater.plan([change], {})

        assert len(plan.actions) == 1
        assert plan.actions[0].action == UpdateAction.GENERATE_COLUMN
        assert plan.actions[0].change is change

    def test_plan_column_removed(self, updater: IncrementalUpdater) -> None:
        change = SchemaChange(
            change_type=SchemaChangeType.COLUMN_REMOVED,
            table_name="users",
            column_name="email",
            old_value="varchar",
        )
        plan = updater.plan([change], {})

        assert len(plan.actions) == 1
        assert plan.actions[0].action == UpdateAction.DROP_COLUMN

    def test_plan_column_type_changed(self, updater: IncrementalUpdater) -> None:
        change = SchemaChange(
            change_type=SchemaChangeType.COLUMN_TYPE_CHANGED,
            table_name="users",
            column_name="age",
            old_value="integer",
            new_value="bigint",
            detail={"old_type": "integer", "new_type": "bigint"},
        )
        plan = updater.plan([change], {})

        assert len(plan.actions) == 1
        assert plan.actions[0].action == UpdateAction.REGENERATE_COLUMN

    def test_plan_nullability_to_not_null(self, updater: IncrementalUpdater) -> None:
        change = SchemaChange(
            change_type=SchemaChangeType.COLUMN_NULLABILITY_CHANGED,
            table_name="users",
            column_name="email",
            old_value="True",
            new_value="False",
            detail={"old_nullable": True, "new_nullable": False},
        )
        plan = updater.plan([change], {})

        assert len(plan.actions) == 1
        assert plan.actions[0].action == UpdateAction.REGENERATE_COLUMN

    def test_plan_nullability_to_nullable(self, updater: IncrementalUpdater) -> None:
        change = SchemaChange(
            change_type=SchemaChangeType.COLUMN_NULLABILITY_CHANGED,
            table_name="users",
            column_name="email",
            old_value="False",
            new_value="True",
            detail={"old_nullable": False, "new_nullable": True},
        )
        plan = updater.plan([change], {})

        assert len(plan.actions) == 1
        assert plan.actions[0].action == UpdateAction.NO_ACTION

    def test_plan_default_changed(self, updater: IncrementalUpdater) -> None:
        change = SchemaChange(
            change_type=SchemaChangeType.COLUMN_DEFAULT_CHANGED,
            table_name="users",
            column_name="status",
            old_value="active",
            new_value="pending",
            detail={"old_default": "active", "new_default": "pending"},
        )
        plan = updater.plan([change], {})

        assert len(plan.actions) == 1
        assert plan.actions[0].action == UpdateAction.NO_ACTION


# ── plan() — table-level changes ────────────────────────────────────


class TestPlanTableChanges:
    """Table add / remove maps to GENERATE_TABLE / DROP_TABLE."""

    def test_plan_table_added(self, updater: IncrementalUpdater) -> None:
        change = SchemaChange(
            change_type=SchemaChangeType.TABLE_ADDED,
            table_name="products",
            detail={"table": {"name": "products"}},
        )
        plan = updater.plan([change], {})

        assert len(plan.actions) == 1
        assert plan.actions[0].action == UpdateAction.GENERATE_TABLE

    def test_plan_table_removed(self, updater: IncrementalUpdater) -> None:
        change = SchemaChange(
            change_type=SchemaChangeType.TABLE_REMOVED,
            table_name="products",
        )
        plan = updater.plan([change], {})

        assert len(plan.actions) == 1
        assert plan.actions[0].action == UpdateAction.DROP_TABLE


# ── plan() — foreign key changes ────────────────────────────────────


class TestPlanForeignKeyChanges:
    """FK add → VALIDATE_FK; FK remove → NO_ACTION."""

    def test_plan_fk_added(self, updater: IncrementalUpdater) -> None:
        change = SchemaChange(
            change_type=SchemaChangeType.FOREIGN_KEY_ADDED,
            table_name="orders",
            detail={
                "fk": {
                    "columns": ["user_id"],
                    "ref_table": "users",
                    "ref_columns": ["id"],
                }
            },
        )
        plan = updater.plan([change], {})

        assert len(plan.actions) == 1
        assert plan.actions[0].action == UpdateAction.VALIDATE_FK

    def test_plan_fk_removed(self, updater: IncrementalUpdater) -> None:
        change = SchemaChange(
            change_type=SchemaChangeType.FOREIGN_KEY_REMOVED,
            table_name="orders",
            detail={
                "fk": {
                    "columns": ["user_id"],
                    "ref_table": "users",
                    "ref_columns": ["id"],
                }
            },
        )
        plan = updater.plan([change], {})

        assert len(plan.actions) == 1
        assert plan.actions[0].action == UpdateAction.NO_ACTION


# ── plan() — index changes ──────────────────────────────────────────


class TestPlanIndexChanges:
    """Unique index add → DEDUPLICATE; non-unique / remove → NO_ACTION."""

    def test_plan_index_added_unique(self, updater: IncrementalUpdater) -> None:
        change = SchemaChange(
            change_type=SchemaChangeType.INDEX_ADDED,
            table_name="users",
            detail={"index": {"columns": ["email"], "unique": True}},
        )
        plan = updater.plan([change], {})

        assert len(plan.actions) == 1
        assert plan.actions[0].action == UpdateAction.DEDUPLICATE

    def test_plan_index_added_non_unique(self, updater: IncrementalUpdater) -> None:
        change = SchemaChange(
            change_type=SchemaChangeType.INDEX_ADDED,
            table_name="users",
            detail={"index": {"columns": ["email"], "unique": False}},
        )
        plan = updater.plan([change], {})

        assert len(plan.actions) == 1
        assert plan.actions[0].action == UpdateAction.NO_ACTION

    def test_plan_index_removed(self, updater: IncrementalUpdater) -> None:
        change = SchemaChange(
            change_type=SchemaChangeType.INDEX_REMOVED,
            table_name="users",
            detail={"index": {"columns": ["email"], "unique": True}},
        )
        plan = updater.plan([change], {})

        assert len(plan.actions) == 1
        assert plan.actions[0].action == UpdateAction.NO_ACTION


# ── plan() — enum changes ───────────────────────────────────────────


class TestPlanEnumChanges:
    """ENUM_CHANGED → REPLACE_ENUM."""

    def test_plan_enum_changed(self, updater: IncrementalUpdater) -> None:
        change = SchemaChange(
            change_type=SchemaChangeType.ENUM_CHANGED,
            table_name="users",
            column_name="role",
            detail={
                "old_values": ["admin", "user"],
                "new_values": ["admin", "user", "moderator"],
            },
        )
        plan = updater.plan([change], {})

        assert len(plan.actions) == 1
        assert plan.actions[0].action == UpdateAction.REPLACE_ENUM


# ── plan() — description quality ─────────────────────────────────────


class TestPlanDescriptions:
    """Every PlannedAction must carry a non-empty human-readable description."""

    _ALL_CHANGES: ClassVar[list[tuple[SchemaChangeType, dict[str, Any] | None]]] = [
        (SchemaChangeType.COLUMN_ADDED, {"column": {"name": "x", "data_type": "text"}}),
        (SchemaChangeType.COLUMN_REMOVED, None),
        (SchemaChangeType.TABLE_ADDED, {"table": {"name": "t"}}),
        (SchemaChangeType.TABLE_REMOVED, None),
        (
            SchemaChangeType.COLUMN_TYPE_CHANGED,
            {"old_type": "int", "new_type": "bigint"},
        ),
        (
            SchemaChangeType.COLUMN_NULLABILITY_CHANGED,
            {"old_nullable": True, "new_nullable": False},
        ),
        (
            SchemaChangeType.COLUMN_NULLABILITY_CHANGED,
            {"old_nullable": False, "new_nullable": True},
        ),
        (
            SchemaChangeType.COLUMN_DEFAULT_CHANGED,
            {"old_default": "a", "new_default": "b"},
        ),
        (
            SchemaChangeType.FOREIGN_KEY_ADDED,
            {"fk": {"columns": ["c"], "ref_table": "t", "ref_columns": ["id"]}},
        ),
        (
            SchemaChangeType.FOREIGN_KEY_REMOVED,
            {"fk": {"columns": ["c"], "ref_table": "t", "ref_columns": ["id"]}},
        ),
        (SchemaChangeType.INDEX_ADDED, {"index": {"columns": ["c"], "unique": True}}),
        (SchemaChangeType.INDEX_ADDED, {"index": {"columns": ["c"], "unique": False}}),
        (SchemaChangeType.INDEX_REMOVED, {"index": {"columns": ["c"], "unique": False}}),
        (SchemaChangeType.ENUM_CHANGED, {"old_values": ["a"], "new_values": ["a", "b"]}),
    ]

    @pytest.mark.parametrize(
        ("change_type", "detail"),
        _ALL_CHANGES,
        ids=[f"{ct.value}{'_' + str(i) if i else ''}" for i, (ct, _) in enumerate(_ALL_CHANGES)],
    )
    def test_plan_description_is_human_readable(
        self,
        updater: IncrementalUpdater,
        change_type: SchemaChangeType,
        detail: dict[str, Any] | None,
    ) -> None:
        change = SchemaChange(
            change_type=change_type,
            table_name="users",
            column_name="col",
            detail=detail,
        )
        plan = updater.plan([change], {})

        assert len(plan.actions) == 1
        desc = plan.actions[0].description
        assert isinstance(desc, str)
        assert len(desc) > 0


# ── apply() — DROP_COLUMN / DROP_TABLE (Task 4) ───────────────────


class TestApplyDropColumn:
    """_apply_drop_column removes the column key from every row."""

    def test_drop_column_removes_key(self, updater: IncrementalUpdater) -> None:
        data = {"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}
        change = SchemaChange(
            change_type=SchemaChangeType.COLUMN_REMOVED,
            table_name="users",
            column_name="name",
        )
        result = updater.update([change], data)
        assert all("name" not in row for row in result.tables_data["users"])
        assert all("id" in row for row in result.tables_data["users"])

    def test_drop_column_empty_table(self, updater: IncrementalUpdater) -> None:
        data: dict[str, list[dict[str, Any]]] = {"users": []}
        change = SchemaChange(
            change_type=SchemaChangeType.COLUMN_REMOVED,
            table_name="users",
            column_name="name",
        )
        result = updater.update([change], data)
        assert result.tables_data["users"] == []

    def test_drop_column_preserves_other_tables(self, updater: IncrementalUpdater) -> None:
        data = {
            "users": [{"id": 1, "name": "A"}],
            "posts": [{"id": 1, "title": "T"}],
        }
        change = SchemaChange(
            change_type=SchemaChangeType.COLUMN_REMOVED,
            table_name="users",
            column_name="name",
        )
        result = updater.update([change], data)
        assert result.tables_data["posts"] == [{"id": 1, "title": "T"}]


class TestApplyDropTable:
    """_apply_drop_table removes the table from data and tracks stats."""

    def test_drop_table_removes_key(self, updater: IncrementalUpdater) -> None:
        data = {"users": [{"id": 1}], "posts": [{"id": 1}]}
        change = SchemaChange(change_type=SchemaChangeType.TABLE_REMOVED, table_name="posts")
        result = updater.update([change], data)
        assert "posts" not in result.tables_data
        assert "users" in result.tables_data

    def test_drop_table_stats(self, updater: IncrementalUpdater) -> None:
        data = {"users": [{"id": 1}, {"id": 2}]}
        change = SchemaChange(change_type=SchemaChangeType.TABLE_REMOVED, table_name="users")
        result = updater.update([change], data)
        assert "users" in result.tables_removed
        assert result.rows_removed == 2


# ── apply() — GENERATE_COLUMN (Task 5) ────────────────────────────


class TestApplyGenerateColumn:
    """_apply_generate_column adds column values to existing rows."""

    def test_column_added_nullable_sets_none(self, updater: IncrementalUpdater) -> None:
        data = {"users": [{"id": 1}, {"id": 2}, {"id": 3}]}
        change = SchemaChange(
            change_type=SchemaChangeType.COLUMN_ADDED,
            table_name="users",
            column_name="email",
            detail={
                "column": {
                    "name": "email",
                    "data_type": "varchar",
                    "nullable": True,
                }
            },
        )
        result = updater.update([change], data)
        assert all(row["email"] is None for row in result.tables_data["users"])
        assert len(result.tables_data["users"]) == 3

    def test_column_added_not_null_generates_values(self, updater: IncrementalUpdater) -> None:
        data = {"users": [{"id": 1}, {"id": 2}, {"id": 3}]}
        change = SchemaChange(
            change_type=SchemaChangeType.COLUMN_ADDED,
            table_name="users",
            column_name="age",
            detail={
                "column": {
                    "name": "age",
                    "data_type": "integer",
                    "nullable": False,
                }
            },
        )
        result = updater.update([change], data)
        for row in result.tables_data["users"]:
            assert "age" in row
            assert row["age"] is not None
            assert isinstance(row["age"], int)

    def test_column_added_not_null_varchar_generates_strings(
        self, updater: IncrementalUpdater
    ) -> None:
        data = {"users": [{"id": 1}, {"id": 2}]}
        change = SchemaChange(
            change_type=SchemaChangeType.COLUMN_ADDED,
            table_name="users",
            column_name="bio",
            detail={
                "column": {
                    "name": "bio",
                    "data_type": "varchar",
                    "nullable": False,
                    "max_length": 50,
                }
            },
        )
        result = updater.update([change], data)
        for row in result.tables_data["users"]:
            assert isinstance(row["bio"], str)

    def test_deterministic_generation(self, minimal_schema: DatabaseSchema) -> None:
        config = DBSproutConfig()
        u1 = IncrementalUpdater(schema=minimal_schema, config=config, seed=42)
        u2 = IncrementalUpdater(schema=minimal_schema, config=config, seed=42)
        data = {"users": [{"id": 1}, {"id": 2}]}
        change = SchemaChange(
            change_type=SchemaChangeType.COLUMN_ADDED,
            table_name="users",
            column_name="score",
            detail={
                "column": {
                    "name": "score",
                    "data_type": "integer",
                    "nullable": False,
                }
            },
        )
        r1 = u1.update([change], data)
        r2 = u2.update([change], data)
        assert r1.tables_data == r2.tables_data

    def test_nullability_to_not_null_fills_nones(self) -> None:
        """COLUMN_NULLABILITY_CHANGED (→NOT NULL) fills only None values."""
        email_col = ColumnSchema(
            name="email",
            data_type=ColumnType.VARCHAR,
            nullable=False,
            max_length=50,
        )
        id_col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)
        table = TableSchema(name="users", columns=[id_col, email_col], primary_key=["id"])
        schema = DatabaseSchema(tables=[table], dialect="sqlite")
        updater = IncrementalUpdater(schema=schema, config=DBSproutConfig(), seed=42)

        data = {
            "users": [
                {"id": 1, "email": "existing@test.com"},
                {"id": 2, "email": None},
                {"id": 3, "email": "another@test.com"},
            ]
        }
        change = SchemaChange(
            change_type=SchemaChangeType.COLUMN_NULLABILITY_CHANGED,
            table_name="users",
            column_name="email",
            detail={"old_nullable": True, "new_nullable": False},
        )
        result = updater.update([change], data)
        # Existing values preserved
        assert result.tables_data["users"][0]["email"] == "existing@test.com"
        assert result.tables_data["users"][2]["email"] == "another@test.com"
        # None value filled
        assert result.tables_data["users"][1]["email"] is not None


# ── apply() — REGENERATE_COLUMN (Task 6) ──────────────────────────


class TestApplyRegenerateColumn:
    """_apply_regenerate_column regenerates values for type changes."""

    def test_type_changed_regenerates_all(self, updater: IncrementalUpdater) -> None:
        data = {"users": [{"id": 1, "xdata": 25}, {"id": 2, "xdata": 30}]}
        change = SchemaChange(
            change_type=SchemaChangeType.COLUMN_TYPE_CHANGED,
            table_name="users",
            column_name="xdata",
            old_value="integer",
            new_value="varchar",
            detail={"old_type": "integer", "new_type": "varchar"},
        )
        result = updater.update([change], data)
        for row in result.tables_data["users"]:
            assert isinstance(row["xdata"], str)
        assert len(result.tables_data["users"]) == 2


# ── apply() — GENERATE_TABLE (Task 7) ─────────────────────────────


class TestApplyGenerateTable:
    """_apply_generate_table generates rows for a brand-new table."""

    def test_table_added_generates_rows(self, updater: IncrementalUpdater) -> None:
        id_col = ColumnSchema(
            name="id",
            data_type=ColumnType.INTEGER,
            nullable=False,
            autoincrement=True,
        )
        name_col = ColumnSchema(
            name="name",
            data_type=ColumnType.VARCHAR,
            nullable=False,
            max_length=50,
        )
        change = SchemaChange(
            change_type=SchemaChangeType.TABLE_ADDED,
            table_name="products",
            detail={
                "table": TableSchema(
                    name="products",
                    columns=[id_col, name_col],
                    primary_key=["id"],
                ).model_dump()
            },
        )
        result = updater.update([change], {})
        assert "products" in result.tables_data
        assert len(result.tables_data["products"]) == 100  # default_rows
        assert "products" in result.tables_added

    def test_table_added_with_fk_references_parent(self) -> None:
        id_col = ColumnSchema(
            name="id",
            data_type=ColumnType.INTEGER,
            nullable=False,
            autoincrement=True,
        )
        user_id_col = ColumnSchema(name="user_id", data_type=ColumnType.INTEGER, nullable=False)
        users_table = TableSchema(name="users", columns=[id_col], primary_key=["id"])
        orders_table = TableSchema(
            name="orders",
            columns=[id_col, user_id_col],
            primary_key=["id"],
            foreign_keys=[
                ForeignKeySchema(columns=["user_id"], ref_table="users", ref_columns=["id"])
            ],
        )
        schema = DatabaseSchema(tables=[users_table, orders_table], dialect="sqlite")
        updater = IncrementalUpdater(schema=schema, config=DBSproutConfig(), seed=42)

        existing: dict[str, list[dict[str, Any]]] = {"users": [{"id": 1}, {"id": 2}, {"id": 3}]}
        change = SchemaChange(
            change_type=SchemaChangeType.TABLE_ADDED,
            table_name="orders",
            detail={"table": orders_table.model_dump()},
        )
        result = updater.update([change], existing)
        assert "orders" in result.tables_data
        parent_ids = {1, 2, 3}
        for row in result.tables_data["orders"]:
            assert row["user_id"] in parent_ids

    def test_table_added_respects_config_rows(self) -> None:
        id_col = ColumnSchema(
            name="id",
            data_type=ColumnType.INTEGER,
            nullable=False,
            autoincrement=True,
        )
        table = TableSchema(name="products", columns=[id_col], primary_key=["id"])
        schema = DatabaseSchema(tables=[table], dialect="sqlite")
        config = DBSproutConfig(tables={"products": {"rows": 10}})
        updater = IncrementalUpdater(schema=schema, config=config, seed=42)

        change = SchemaChange(
            change_type=SchemaChangeType.TABLE_ADDED,
            table_name="products",
            detail={"table": table.model_dump()},
        )
        result = updater.update([change], {})
        assert len(result.tables_data["products"]) == 10


# ── apply() — VALIDATE_FK (Task 8) ────────────────────────────────


class TestApplyValidateFk:
    """_apply_validate_fk replaces invalid FK values with valid parent refs."""

    def test_all_valid_refs_unchanged(self) -> None:
        id_col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)
        user_id_col = ColumnSchema(name="user_id", data_type=ColumnType.INTEGER, nullable=False)
        users_table = TableSchema(name="users", columns=[id_col], primary_key=["id"])
        orders_table = TableSchema(
            name="orders",
            columns=[id_col, user_id_col],
            primary_key=["id"],
            foreign_keys=[
                ForeignKeySchema(
                    columns=["user_id"],
                    ref_table="users",
                    ref_columns=["id"],
                )
            ],
        )
        schema = DatabaseSchema(tables=[users_table, orders_table], dialect="sqlite")
        updater = IncrementalUpdater(schema=schema, config=DBSproutConfig(), seed=42)

        data = {
            "users": [{"id": 1}, {"id": 2}],
            "orders": [{"id": 1, "user_id": 1}, {"id": 2, "user_id": 2}],
        }
        change = SchemaChange(
            change_type=SchemaChangeType.FOREIGN_KEY_ADDED,
            table_name="orders",
            detail={
                "fk": {
                    "columns": ["user_id"],
                    "ref_table": "users",
                    "ref_columns": ["id"],
                }
            },
        )
        result = updater.update([change], data)
        assert result.tables_data["orders"][0]["user_id"] == 1
        assert result.tables_data["orders"][1]["user_id"] == 2

    def test_invalid_refs_replaced(self) -> None:
        id_col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)
        user_id_col = ColumnSchema(name="user_id", data_type=ColumnType.INTEGER, nullable=False)
        users_table = TableSchema(name="users", columns=[id_col], primary_key=["id"])
        orders_table = TableSchema(
            name="orders",
            columns=[id_col, user_id_col],
            primary_key=["id"],
            foreign_keys=[
                ForeignKeySchema(
                    columns=["user_id"],
                    ref_table="users",
                    ref_columns=["id"],
                )
            ],
        )
        schema = DatabaseSchema(tables=[users_table, orders_table], dialect="sqlite")
        updater = IncrementalUpdater(schema=schema, config=DBSproutConfig(), seed=42)

        data = {
            "users": [{"id": 1}, {"id": 2}],
            "orders": [{"id": 1, "user_id": 999}, {"id": 2, "user_id": 1}],
        }
        change = SchemaChange(
            change_type=SchemaChangeType.FOREIGN_KEY_ADDED,
            table_name="orders",
            detail={
                "fk": {
                    "columns": ["user_id"],
                    "ref_table": "users",
                    "ref_columns": ["id"],
                }
            },
        )
        result = updater.update([change], data)
        assert result.tables_data["orders"][0]["user_id"] in {1, 2}
        assert result.tables_data["orders"][1]["user_id"] == 1


# ── apply() — DEDUPLICATE / REPLACE_ENUM (Task 9) ─────────────────


class TestApplyDeduplicate:
    """_apply_deduplicate removes duplicate values from unique columns."""

    def test_deduplicate_removes_duplicates(self) -> None:
        email_col = ColumnSchema(
            name="email",
            data_type=ColumnType.VARCHAR,
            nullable=False,
            unique=True,
            max_length=50,
        )
        id_col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)
        table = TableSchema(name="users", columns=[id_col, email_col], primary_key=["id"])
        schema = DatabaseSchema(tables=[table], dialect="sqlite")
        updater = IncrementalUpdater(schema=schema, config=DBSproutConfig(), seed=42)

        data = {
            "users": [
                {"id": 1, "email": "same@test.com"},
                {"id": 2, "email": "same@test.com"},
                {"id": 3, "email": "unique@test.com"},
            ]
        }
        change = SchemaChange(
            change_type=SchemaChangeType.INDEX_ADDED,
            table_name="users",
            detail={"index": {"columns": ["email"], "unique": True}},
        )
        result = updater.update([change], data)
        emails = [row["email"] for row in result.tables_data["users"]]
        assert len(set(emails)) == len(emails)

    def test_deduplicate_no_dupes_unchanged(self) -> None:
        email_col = ColumnSchema(
            name="email",
            data_type=ColumnType.VARCHAR,
            nullable=False,
            unique=True,
            max_length=50,
        )
        id_col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)
        table = TableSchema(name="users", columns=[id_col, email_col], primary_key=["id"])
        schema = DatabaseSchema(tables=[table], dialect="sqlite")
        updater = IncrementalUpdater(schema=schema, config=DBSproutConfig(), seed=42)

        data = {
            "users": [
                {"id": 1, "email": "a@test.com"},
                {"id": 2, "email": "b@test.com"},
            ]
        }
        change = SchemaChange(
            change_type=SchemaChangeType.INDEX_ADDED,
            table_name="users",
            detail={"index": {"columns": ["email"], "unique": True}},
        )
        result = updater.update([change], data)
        assert result.tables_data["users"][0]["email"] == "a@test.com"
        assert result.tables_data["users"][1]["email"] == "b@test.com"


class TestApplyReplaceEnum:
    """_apply_replace_enum replaces invalid enum values with new ones."""

    def test_replace_invalid_enum_values(self) -> None:
        role_col = ColumnSchema(
            name="role",
            data_type=ColumnType.ENUM,
            nullable=False,
            enum_values=["admin", "editor"],
        )
        id_col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)
        table = TableSchema(name="users", columns=[id_col, role_col], primary_key=["id"])
        schema = DatabaseSchema(tables=[table], dialect="sqlite")
        updater = IncrementalUpdater(schema=schema, config=DBSproutConfig(), seed=42)

        data = {
            "users": [
                {"id": 1, "role": "admin"},
                {"id": 2, "role": "old_role"},
                {"id": 3, "role": "editor"},
            ]
        }
        change = SchemaChange(
            change_type=SchemaChangeType.ENUM_CHANGED,
            table_name="users",
            column_name="role",
            detail={
                "old_values": ["admin", "user", "old_role"],
                "new_values": ["admin", "editor"],
            },
        )
        result = updater.update([change], data)
        assert result.tables_data["users"][0]["role"] == "admin"
        assert result.tables_data["users"][1]["role"] in {"admin", "editor"}
        assert result.tables_data["users"][2]["role"] == "editor"

    def test_replace_enum_all_valid_unchanged(self) -> None:
        role_col = ColumnSchema(
            name="role",
            data_type=ColumnType.ENUM,
            nullable=False,
            enum_values=["admin", "user"],
        )
        id_col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)
        table = TableSchema(name="users", columns=[id_col, role_col], primary_key=["id"])
        schema = DatabaseSchema(tables=[table], dialect="sqlite")
        updater = IncrementalUpdater(schema=schema, config=DBSproutConfig(), seed=42)

        data = {
            "users": [
                {"id": 1, "role": "admin"},
                {"id": 2, "role": "user"},
            ]
        }
        change = SchemaChange(
            change_type=SchemaChangeType.ENUM_CHANGED,
            table_name="users",
            column_name="role",
            detail={
                "old_values": ["admin", "user"],
                "new_values": ["admin", "user", "editor"],
            },
        )
        result = updater.update([change], data)
        assert result.tables_data["users"][0]["role"] == "admin"
        assert result.tables_data["users"][1]["role"] == "user"


# ── apply() — NO_ACTION / orchestration (Task 10) ─────────────────


class TestApplyNoAction:
    """NO_ACTION changes pass through data without modification."""

    def test_nullability_to_nullable_unchanged(self, updater: IncrementalUpdater) -> None:
        data = {"users": [{"id": 1, "email": "a@b.com"}]}
        change = SchemaChange(
            change_type=SchemaChangeType.COLUMN_NULLABILITY_CHANGED,
            table_name="users",
            column_name="email",
            detail={"old_nullable": False, "new_nullable": True},
        )
        result = updater.update([change], data)
        assert result.tables_data == data

    def test_default_changed_unchanged(self, updater: IncrementalUpdater) -> None:
        data = {"users": [{"id": 1, "status": "active"}]}
        change = SchemaChange(
            change_type=SchemaChangeType.COLUMN_DEFAULT_CHANGED,
            table_name="users",
            column_name="status",
            detail={"old_default": "active", "new_default": "pending"},
        )
        result = updater.update([change], data)
        assert result.tables_data["users"][0]["status"] == "active"

    def test_fk_removed_unchanged(self, updater: IncrementalUpdater) -> None:
        data = {"orders": [{"id": 1, "user_id": 5}]}
        change = SchemaChange(
            change_type=SchemaChangeType.FOREIGN_KEY_REMOVED,
            table_name="orders",
            detail={
                "fk": {
                    "columns": ["user_id"],
                    "ref_table": "users",
                    "ref_columns": ["id"],
                }
            },
        )
        result = updater.update([change], data)
        assert result.tables_data["orders"][0]["user_id"] == 5

    def test_index_non_unique_unchanged(self, updater: IncrementalUpdater) -> None:
        data = {"users": [{"id": 1, "email": "a@b.com"}]}
        change = SchemaChange(
            change_type=SchemaChangeType.INDEX_ADDED,
            table_name="users",
            detail={"index": {"columns": ["email"], "unique": False}},
        )
        result = updater.update([change], data)
        assert result.tables_data == data

    def test_index_removed_unchanged(self, updater: IncrementalUpdater) -> None:
        data = {"users": [{"id": 1, "email": "a@b.com"}]}
        change = SchemaChange(
            change_type=SchemaChangeType.INDEX_REMOVED,
            table_name="users",
            detail={"index": {"columns": ["email"], "unique": True}},
        )
        result = updater.update([change], data)
        assert result.tables_data == data


class TestUpdateConvenience:
    """update() composes plan() + apply()."""

    def test_update_calls_plan_then_apply(self, updater: IncrementalUpdater) -> None:
        data = {"users": [{"id": 1, "name": "A"}]}
        change = SchemaChange(
            change_type=SchemaChangeType.COLUMN_REMOVED,
            table_name="users",
            column_name="name",
        )
        result = updater.update([change], data)
        assert isinstance(result, UpdateResult)
        assert "name" not in result.tables_data["users"][0]


class TestMultipleChanges:
    """Multiple changes in a single update call."""

    def test_multiple_changes_same_table(self, updater: IncrementalUpdater) -> None:
        data = {"users": [{"id": 1, "name": "A", "old_col": "x"}]}
        changes = [
            SchemaChange(
                change_type=SchemaChangeType.COLUMN_REMOVED,
                table_name="users",
                column_name="old_col",
            ),
            SchemaChange(
                change_type=SchemaChangeType.COLUMN_ADDED,
                table_name="users",
                column_name="new_col",
                detail={
                    "column": {
                        "name": "new_col",
                        "data_type": "varchar",
                        "nullable": True,
                    }
                },
            ),
        ]
        result = updater.update(changes, data)
        assert "old_col" not in result.tables_data["users"][0]
        assert "new_col" in result.tables_data["users"][0]

    def test_generate_table_runs_before_drop_table(self) -> None:
        """GENERATE_TABLE runs at order 0, DROP_TABLE at order 3.

        When a table is both removed and re-added (e.g. a schema
        replacement), the generate happens first but the drop runs
        after — so no duplicate table name remains.  This verifies
        the ordering: GENERATE_TABLE before DROP_TABLE.
        """
        id_col = ColumnSchema(
            name="id",
            data_type=ColumnType.INTEGER,
            nullable=False,
            autoincrement=True,
        )
        name_col = ColumnSchema(
            name="name",
            data_type=ColumnType.VARCHAR,
            nullable=False,
            max_length=50,
        )
        new_table = TableSchema(name="items", columns=[id_col, name_col], primary_key=["id"])
        schema = DatabaseSchema(tables=[new_table], dialect="sqlite")
        updater = IncrementalUpdater(schema=schema, config=DBSproutConfig(), seed=42)

        data: dict[str, list[dict[str, Any]]] = {"old_table": [{"id": 1, "old_field": "x"}]}
        changes = [
            SchemaChange(
                change_type=SchemaChangeType.TABLE_REMOVED,
                table_name="old_table",
            ),
            SchemaChange(
                change_type=SchemaChangeType.TABLE_ADDED,
                table_name="items",
                detail={"table": new_table.model_dump()},
            ),
        ]
        result = updater.update(changes, data)
        # New table was generated
        assert "items" in result.tables_data
        assert len(result.tables_data["items"]) == 100
        # Old table was dropped
        assert "old_table" not in result.tables_data

    def test_drop_table_runs_after_validate_fk(self) -> None:
        """DROP_TABLE must run last so parent data is available for FK validation."""
        id_col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)
        parent_id_col = ColumnSchema(name="parent_id", data_type=ColumnType.INTEGER, nullable=False)
        parent = TableSchema(name="parent", columns=[id_col], primary_key=["id"])
        child = TableSchema(
            name="child",
            columns=[id_col, parent_id_col],
            primary_key=["id"],
            foreign_keys=[
                ForeignKeySchema(
                    columns=["parent_id"],
                    ref_table="parent",
                    ref_columns=["id"],
                )
            ],
        )
        schema = DatabaseSchema(tables=[parent, child], dialect="sqlite")
        updater = IncrementalUpdater(schema=schema, config=DBSproutConfig(), seed=42)

        data = {
            "parent": [{"id": 1}, {"id": 2}],
            "child": [{"id": 1, "parent_id": 999}],
        }
        changes = [
            SchemaChange(
                change_type=SchemaChangeType.TABLE_REMOVED,
                table_name="parent",
            ),
            SchemaChange(
                change_type=SchemaChangeType.FOREIGN_KEY_ADDED,
                table_name="child",
                detail={
                    "fk": {
                        "columns": ["parent_id"],
                        "ref_table": "parent",
                        "ref_columns": ["id"],
                    }
                },
            ),
        ]
        result = updater.update(changes, data)
        assert "parent" not in result.tables_data


class TestTableAddedTopologicalOrder:
    """AC-28: Multiple TABLE_ADDED with FK dependency uses topological order."""

    def test_two_new_tables_with_fk_dependency(self) -> None:
        """Parent table must be generated before child even if child sorts first."""
        id_col = ColumnSchema(
            name="id",
            data_type=ColumnType.INTEGER,
            nullable=False,
            autoincrement=True,
        )
        parent_id_col = ColumnSchema(
            name="alpha_id",
            data_type=ColumnType.INTEGER,
            nullable=False,
        )
        # "alpha" depends on "beta" via FK — alpha sorts first alphabetically
        beta_table = TableSchema(name="beta", columns=[id_col], primary_key=["id"])
        alpha_table = TableSchema(
            name="alpha",
            columns=[id_col, parent_id_col],
            primary_key=["id"],
            foreign_keys=[
                ForeignKeySchema(
                    columns=["alpha_id"],
                    ref_table="beta",
                    ref_columns=["id"],
                ),
            ],
        )
        schema = DatabaseSchema(tables=[beta_table, alpha_table], dialect="sqlite")
        config = DBSproutConfig(tables={"alpha": {"rows": 5}, "beta": {"rows": 5}})
        updater = IncrementalUpdater(schema=schema, config=config, seed=42)

        changes = [
            SchemaChange(
                change_type=SchemaChangeType.TABLE_ADDED,
                table_name="alpha",
                detail={"table": alpha_table.model_dump()},
            ),
            SchemaChange(
                change_type=SchemaChangeType.TABLE_ADDED,
                table_name="beta",
                detail={"table": beta_table.model_dump()},
            ),
        ]
        result = updater.update(changes, {})
        assert "alpha" in result.tables_data
        assert "beta" in result.tables_data
        # FK integrity: every alpha.alpha_id must be in beta.id
        beta_ids = {row["id"] for row in result.tables_data["beta"]}
        for row in result.tables_data["alpha"]:
            assert row["alpha_id"] in beta_ids


class TestTableRemovedThenAddedSameName:
    """AC-29: TABLE_REMOVED + TABLE_ADDED same name → drop then recreate."""

    def test_same_name_drop_and_recreate(self) -> None:
        id_col = ColumnSchema(
            name="id",
            data_type=ColumnType.INTEGER,
            nullable=False,
            autoincrement=True,
        )
        name_col = ColumnSchema(
            name="name",
            data_type=ColumnType.VARCHAR,
            nullable=False,
            max_length=50,
        )
        new_table = TableSchema(
            name="products",
            columns=[id_col, name_col],
            primary_key=["id"],
        )
        schema = DatabaseSchema(tables=[new_table], dialect="sqlite")
        updater = IncrementalUpdater(schema=schema, config=DBSproutConfig(), seed=42)

        data: dict[str, list[dict[str, Any]]] = {
            "products": [{"id": 1, "old_field": "x"}],
        }
        changes = [
            SchemaChange(
                change_type=SchemaChangeType.TABLE_REMOVED,
                table_name="products",
            ),
            SchemaChange(
                change_type=SchemaChangeType.TABLE_ADDED,
                table_name="products",
                detail={"table": new_table.model_dump()},
            ),
        ]
        result = updater.update(changes, data)
        # Table should exist with new structure (no old_field)
        assert "products" in result.tables_data
        assert len(result.tables_data["products"]) == 100
        assert "old_field" not in result.tables_data["products"][0]
        assert "name" in result.tables_data["products"][0]


class TestApplyEmpty:
    """Edge cases for apply with no changes."""

    def test_empty_changes_returns_data_unmodified(self, updater: IncrementalUpdater) -> None:
        data = {"users": [{"id": 1, "name": "A"}]}
        result = updater.update([], data)
        assert result.tables_data == data
        assert result.rows_modified == 0

    def test_apply_does_not_mutate_input(self, updater: IncrementalUpdater) -> None:
        original_data = {"users": [{"id": 1, "name": "A"}]}
        change = SchemaChange(
            change_type=SchemaChangeType.COLUMN_REMOVED,
            table_name="users",
            column_name="name",
        )
        result = updater.update([change], original_data)
        assert "name" in original_data["users"][0]
        assert "name" not in result.tables_data["users"][0]


# ── Integrity validation & stats accuracy (Task 11) ─────────────


class TestIntegrityAndStats:
    """Verify data integrity after updates and accuracy of UpdateResult stats."""

    def test_fk_integrity_after_validate(self) -> None:
        """AC-22: After FOREIGN_KEY_ADDED, every FK value exists in parent PKs."""
        id_col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)
        user_id_col = ColumnSchema(name="user_id", data_type=ColumnType.INTEGER, nullable=False)
        users_table = TableSchema(name="users", columns=[id_col], primary_key=["id"])
        orders_table = TableSchema(
            name="orders",
            columns=[id_col, user_id_col],
            primary_key=["id"],
            foreign_keys=[
                ForeignKeySchema(
                    columns=["user_id"],
                    ref_table="users",
                    ref_columns=["id"],
                )
            ],
        )
        schema = DatabaseSchema(tables=[users_table, orders_table], dialect="sqlite")
        updater = IncrementalUpdater(schema=schema, config=DBSproutConfig(), seed=42)

        parent_ids = {10, 20, 30}
        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": pk} for pk in parent_ids],
            "orders": [
                {"id": 1, "user_id": 999},
                {"id": 2, "user_id": -1},
                {"id": 3, "user_id": 10},
                {"id": 4, "user_id": 0},
            ],
        }
        change = SchemaChange(
            change_type=SchemaChangeType.FOREIGN_KEY_ADDED,
            table_name="orders",
            detail={
                "fk": {
                    "columns": ["user_id"],
                    "ref_table": "users",
                    "ref_columns": ["id"],
                }
            },
        )
        result = updater.update([change], data)

        for row in result.tables_data["orders"]:
            assert row["user_id"] in parent_ids, (
                f"FK value {row['user_id']} not in parent PKs {parent_ids}"
            )

    def test_uniqueness_after_deduplicate(self) -> None:
        """AC-23: After INDEX_ADDED (unique), no duplicate values."""
        email_col = ColumnSchema(
            name="email",
            data_type=ColumnType.VARCHAR,
            nullable=False,
            unique=True,
            max_length=50,
        )
        id_col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)
        table = TableSchema(name="users", columns=[id_col, email_col], primary_key=["id"])
        schema = DatabaseSchema(tables=[table], dialect="sqlite")
        updater = IncrementalUpdater(schema=schema, config=DBSproutConfig(), seed=42)

        data: dict[str, list[dict[str, Any]]] = {
            "users": [
                {"id": 1, "email": "dup@test.com"},
                {"id": 2, "email": "dup@test.com"},
                {"id": 3, "email": "dup@test.com"},
                {"id": 4, "email": "unique@test.com"},
            ]
        }
        change = SchemaChange(
            change_type=SchemaChangeType.INDEX_ADDED,
            table_name="users",
            detail={"index": {"columns": ["email"], "unique": True}},
        )
        result = updater.update([change], data)
        emails = [row["email"] for row in result.tables_data["users"]]
        assert len(set(emails)) == len(emails), (
            f"Duplicate emails found after deduplicate: {emails}"
        )

    def test_not_null_after_generate_column(self) -> None:
        """AC-24: After COLUMN_ADDED (NOT NULL), no None values."""
        id_col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)
        table = TableSchema(name="users", columns=[id_col], primary_key=["id"])
        schema = DatabaseSchema(tables=[table], dialect="sqlite")
        updater = IncrementalUpdater(schema=schema, config=DBSproutConfig(), seed=42)

        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}, {"id": 5}]
        }
        change = SchemaChange(
            change_type=SchemaChangeType.COLUMN_ADDED,
            table_name="users",
            column_name="status",
            detail={
                "column": {
                    "name": "status",
                    "data_type": "varchar",
                    "nullable": False,
                    "max_length": 30,
                }
            },
        )
        result = updater.update([change], data)
        for row in result.tables_data["users"]:
            assert "status" in row, "Column 'status' missing from row"
            assert row["status"] is not None, (
                f"None value found in NOT NULL column 'status': row={row}"
            )

    def test_stats_rows_modified(self) -> None:
        """AC-17: rows_modified counts unique rows touched."""
        id_col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)
        table = TableSchema(name="users", columns=[id_col], primary_key=["id"])
        schema = DatabaseSchema(tables=[table], dialect="sqlite")
        updater = IncrementalUpdater(schema=schema, config=DBSproutConfig(), seed=42)

        data: dict[str, list[dict[str, Any]]] = {
            "users": [
                {"id": 1, "old_col": "a"},
                {"id": 2, "old_col": "b"},
                {"id": 3, "old_col": "c"},
            ]
        }
        change = SchemaChange(
            change_type=SchemaChangeType.COLUMN_REMOVED,
            table_name="users",
            column_name="old_col",
        )
        result = updater.update([change], data)
        assert result.rows_modified == 3

    def test_stats_rows_added(self) -> None:
        """AC-17: rows_added counts rows in newly generated tables."""
        id_col = ColumnSchema(
            name="id",
            data_type=ColumnType.INTEGER,
            nullable=False,
            autoincrement=True,
        )
        name_col = ColumnSchema(
            name="name",
            data_type=ColumnType.VARCHAR,
            nullable=False,
            max_length=50,
        )
        table = TableSchema(name="items", columns=[id_col, name_col], primary_key=["id"])
        schema = DatabaseSchema(tables=[table], dialect="sqlite")
        config = DBSproutConfig(tables={"items": {"rows": 25}})
        updater = IncrementalUpdater(schema=schema, config=config, seed=42)

        change = SchemaChange(
            change_type=SchemaChangeType.TABLE_ADDED,
            table_name="items",
            detail={"table": table.model_dump()},
        )
        result = updater.update([change], {})
        assert result.rows_added == 25
        assert len(result.tables_data["items"]) == 25

    def test_stats_tables_added_removed(self) -> None:
        """AC-17: tables_added/removed lists are populated correctly."""
        id_col = ColumnSchema(
            name="id",
            data_type=ColumnType.INTEGER,
            nullable=False,
            autoincrement=True,
        )
        new_table = TableSchema(name="products", columns=[id_col], primary_key=["id"])
        schema = DatabaseSchema(tables=[new_table], dialect="sqlite")
        updater = IncrementalUpdater(schema=schema, config=DBSproutConfig(), seed=42)

        data: dict[str, list[dict[str, Any]]] = {
            "legacy": [{"id": 1}, {"id": 2}],
        }
        changes = [
            SchemaChange(
                change_type=SchemaChangeType.TABLE_ADDED,
                table_name="products",
                detail={"table": new_table.model_dump()},
            ),
            SchemaChange(
                change_type=SchemaChangeType.TABLE_REMOVED,
                table_name="legacy",
            ),
        ]
        result = updater.update(changes, data)

        assert "products" in result.tables_added
        assert "legacy" in result.tables_removed
        assert "products" in result.tables_data
        assert "legacy" not in result.tables_data
        assert result.rows_removed == 2


# ── Edge case guard branches ──────────────────────────────────────


class TestEdgeCaseGuards:
    """Cover defensive guard branches for missing tables/columns/data."""

    def test_drop_column_missing_table(self, updater: IncrementalUpdater) -> None:
        change = SchemaChange(
            change_type=SchemaChangeType.COLUMN_REMOVED,
            table_name="nonexistent",
            column_name="col",
        )
        result = updater.update([change], {})
        assert result.tables_data == {}

    def test_generate_column_missing_table(self, updater: IncrementalUpdater) -> None:
        change = SchemaChange(
            change_type=SchemaChangeType.COLUMN_ADDED,
            table_name="nonexistent",
            column_name="col",
            detail={"column": {"name": "col", "data_type": "integer", "nullable": False}},
        )
        result = updater.update([change], {})
        assert result.tables_data == {}

    def test_generate_column_empty_table(self, updater: IncrementalUpdater) -> None:
        change = SchemaChange(
            change_type=SchemaChangeType.COLUMN_ADDED,
            table_name="users",
            column_name="col",
            detail={"column": {"name": "col", "data_type": "integer", "nullable": False}},
        )
        result = updater.update([change], {"users": []})
        assert result.tables_data["users"] == []

    def test_regenerate_column_missing_table(self, updater: IncrementalUpdater) -> None:
        change = SchemaChange(
            change_type=SchemaChangeType.COLUMN_TYPE_CHANGED,
            table_name="nonexistent",
            column_name="col",
            detail={"old_type": "integer", "new_type": "varchar"},
        )
        result = updater.update([change], {})
        assert result.tables_data == {}

    def test_regenerate_column_empty_table(self, updater: IncrementalUpdater) -> None:
        change = SchemaChange(
            change_type=SchemaChangeType.COLUMN_TYPE_CHANGED,
            table_name="users",
            column_name="col",
            detail={"old_type": "integer", "new_type": "varchar"},
        )
        result = updater.update([change], {"users": []})
        assert result.tables_data["users"] == []

    def test_validate_fk_missing_table(self, updater: IncrementalUpdater) -> None:
        change = SchemaChange(
            change_type=SchemaChangeType.FOREIGN_KEY_ADDED,
            table_name="nonexistent",
            detail={"fk": {"columns": ["id"], "ref_table": "users", "ref_columns": ["id"]}},
        )
        result = updater.update([change], {})
        assert result.tables_data == {}

    def test_validate_fk_missing_parent(self, updater: IncrementalUpdater) -> None:
        change = SchemaChange(
            change_type=SchemaChangeType.FOREIGN_KEY_ADDED,
            table_name="orders",
            detail={
                "fk": {"columns": ["user_id"], "ref_table": "nonexistent", "ref_columns": ["id"]}
            },
        )
        result = updater.update([change], {"orders": [{"id": 1, "user_id": 5}]})
        assert result.tables_data["orders"][0]["user_id"] == 5

    def test_validate_fk_empty_parent(self, updater: IncrementalUpdater) -> None:
        change = SchemaChange(
            change_type=SchemaChangeType.FOREIGN_KEY_ADDED,
            table_name="orders",
            detail={"fk": {"columns": ["user_id"], "ref_table": "users", "ref_columns": ["id"]}},
        )
        result = updater.update(
            [change],
            {"users": [], "orders": [{"id": 1, "user_id": 5}]},
        )
        assert result.tables_data["orders"][0]["user_id"] == 5

    def test_validate_fk_composite(self) -> None:
        id_col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)
        a_col = ColumnSchema(name="a", data_type=ColumnType.INTEGER, nullable=False)
        b_col = ColumnSchema(name="b", data_type=ColumnType.INTEGER, nullable=False)
        parent = TableSchema(name="parent", columns=[a_col, b_col], primary_key=["a", "b"])
        child = TableSchema(
            name="child",
            columns=[id_col, a_col, b_col],
            primary_key=["id"],
            foreign_keys=[
                ForeignKeySchema(columns=["a", "b"], ref_table="parent", ref_columns=["a", "b"]),
            ],
        )
        schema = DatabaseSchema(tables=[parent, child], dialect="sqlite")
        updater = IncrementalUpdater(schema=schema, config=DBSproutConfig(), seed=42)

        data = {
            "parent": [{"a": 1, "b": 10}, {"a": 2, "b": 20}],
            "child": [{"id": 1, "a": 999, "b": 888}],
        }
        change = SchemaChange(
            change_type=SchemaChangeType.FOREIGN_KEY_ADDED,
            table_name="child",
            detail={
                "fk": {"columns": ["a", "b"], "ref_table": "parent", "ref_columns": ["a", "b"]},
            },
        )
        result = updater.update([change], data)
        row = result.tables_data["child"][0]
        assert (row["a"], row["b"]) in {(1, 10), (2, 20)}

    def test_deduplicate_missing_table(self, updater: IncrementalUpdater) -> None:
        change = SchemaChange(
            change_type=SchemaChangeType.INDEX_ADDED,
            table_name="nonexistent",
            detail={"index": {"columns": ["email"], "unique": True}},
        )
        result = updater.update([change], {})
        assert result.tables_data == {}

    def test_deduplicate_empty_columns(self, updater: IncrementalUpdater) -> None:
        change = SchemaChange(
            change_type=SchemaChangeType.INDEX_ADDED,
            table_name="users",
            detail={"index": {"columns": [], "unique": True}},
        )
        result = updater.update([change], {"users": [{"id": 1}]})
        assert result.tables_data["users"] == [{"id": 1}]

    def test_deduplicate_empty_table(self, updater: IncrementalUpdater) -> None:
        change = SchemaChange(
            change_type=SchemaChangeType.INDEX_ADDED,
            table_name="users",
            detail={"index": {"columns": ["email"], "unique": True}},
        )
        result = updater.update([change], {"users": []})
        assert result.tables_data["users"] == []

    def test_replace_enum_missing_table(self, updater: IncrementalUpdater) -> None:
        change = SchemaChange(
            change_type=SchemaChangeType.ENUM_CHANGED,
            table_name="nonexistent",
            column_name="role",
            detail={"old_values": ["a"], "new_values": ["b"]},
        )
        result = updater.update([change], {})
        assert result.tables_data == {}

    def test_replace_enum_empty_new_values(self, updater: IncrementalUpdater) -> None:
        change = SchemaChange(
            change_type=SchemaChangeType.ENUM_CHANGED,
            table_name="users",
            column_name="role",
            detail={"old_values": ["a"], "new_values": []},
        )
        result = updater.update([change], {"users": [{"id": 1, "role": "a"}]})
        assert result.tables_data["users"][0]["role"] == "a"

    def test_fill_null_no_nulls_present(self) -> None:
        email_col = ColumnSchema(
            name="email",
            data_type=ColumnType.VARCHAR,
            nullable=False,
            max_length=50,
        )
        id_col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)
        table = TableSchema(name="users", columns=[id_col, email_col], primary_key=["id"])
        schema = DatabaseSchema(tables=[table], dialect="sqlite")
        updater = IncrementalUpdater(schema=schema, config=DBSproutConfig(), seed=42)

        data = {"users": [{"id": 1, "email": "a@b.com"}, {"id": 2, "email": "c@d.com"}]}
        change = SchemaChange(
            change_type=SchemaChangeType.COLUMN_NULLABILITY_CHANGED,
            table_name="users",
            column_name="email",
            detail={"old_nullable": True, "new_nullable": False},
        )
        result = updater.update([change], data)
        assert result.tables_data["users"][0]["email"] == "a@b.com"
        assert result.tables_data["users"][1]["email"] == "c@d.com"

    def test_fill_null_table_not_in_schema(self, updater: IncrementalUpdater) -> None:
        data = {"other": [{"id": 1, "col": None}]}
        change = SchemaChange(
            change_type=SchemaChangeType.COLUMN_NULLABILITY_CHANGED,
            table_name="other",
            column_name="col",
            detail={"old_nullable": True, "new_nullable": False},
        )
        result = updater.update([change], data)
        assert result.tables_data["other"][0]["col"] is None

    def test_fill_null_column_not_in_schema(self) -> None:
        id_col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)
        table = TableSchema(name="users", columns=[id_col], primary_key=["id"])
        schema = DatabaseSchema(tables=[table], dialect="sqlite")
        updater = IncrementalUpdater(schema=schema, config=DBSproutConfig(), seed=42)

        data = {"users": [{"id": 1, "mystery": None}]}
        change = SchemaChange(
            change_type=SchemaChangeType.COLUMN_NULLABILITY_CHANGED,
            table_name="users",
            column_name="mystery",
            detail={"old_nullable": True, "new_nullable": False},
        )
        result = updater.update([change], data)
        assert result.tables_data["users"][0]["mystery"] is None

    def test_generate_column_values_unknown_type(
        self,
        updater: IncrementalUpdater,
    ) -> None:
        data = {"users": [{"id": 1}]}
        change = SchemaChange(
            change_type=SchemaChangeType.COLUMN_ADDED,
            table_name="users",
            column_name="weird",
            detail={
                "column": {
                    "name": "weird",
                    "data_type": "nonexistent_type",
                    "nullable": False,
                },
            },
        )
        result = updater.update([change], data)
        assert result.tables_data["users"][0]["weird"] is not None
