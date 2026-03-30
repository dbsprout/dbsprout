"""Tests for dbsprout.generate.orchestrator — generation pipeline coordinator."""

from __future__ import annotations

from dbsprout.config.models import DBSproutConfig, TableOverride
from dbsprout.generate.orchestrator import orchestrate
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    TableSchema,
)


def _col(  # noqa: PLR0913
    name: str,
    *,
    nullable: bool = True,
    pk: bool = False,
    unique: bool = False,
    autoincrement: bool = False,
    data_type: ColumnType = ColumnType.INTEGER,
) -> ColumnSchema:
    return ColumnSchema(
        name=name,
        data_type=data_type,
        nullable=nullable,
        primary_key=pk,
        unique=unique,
        autoincrement=autoincrement,
    )


def _users_orders_schema() -> DatabaseSchema:
    """Two tables: users (parent) → orders (child with FK)."""
    return DatabaseSchema(
        tables=[
            TableSchema(
                name="users",
                columns=[
                    _col("id", nullable=False, pk=True, autoincrement=True),
                    _col(
                        "email",
                        nullable=False,
                        unique=True,
                        data_type=ColumnType.VARCHAR,
                    ),
                ],
                primary_key=["id"],
            ),
            TableSchema(
                name="orders",
                columns=[
                    _col("id", nullable=False, pk=True, autoincrement=True),
                    _col("user_id", nullable=False),
                    _col("amount", nullable=False, data_type=ColumnType.FLOAT),
                ],
                primary_key=["id"],
                foreign_keys=[
                    ForeignKeySchema(
                        columns=["user_id"],
                        ref_table="users",
                        ref_columns=["id"],
                    ),
                ],
            ),
        ],
    )


class TestTopologicalOrder:
    def test_parent_generated_before_child(self) -> None:
        """Users must be generated before orders (FK dependency)."""
        schema = _users_orders_schema()
        config = DBSproutConfig()

        result = orchestrate(schema, config, seed=42, default_rows=5)

        assert "users" in result.tables_data
        assert "orders" in result.tables_data
        # users generated first — orders FK should reference valid user IDs
        user_ids = {r["id"] for r in result.tables_data["users"]}
        for order in result.tables_data["orders"]:
            assert order["user_id"] in user_ids


class TestFKIntegrity:
    def test_fk_values_reference_parent_pks(self) -> None:
        """Every FK value must exist in the parent table's PKs."""
        schema = _users_orders_schema()
        config = DBSproutConfig()

        result = orchestrate(schema, config, seed=42, default_rows=10)

        user_ids = {r["id"] for r in result.tables_data["users"]}
        for order in result.tables_data["orders"]:
            assert order["user_id"] in user_ids


class TestConstraints:
    def test_unique_constraints_enforced(self) -> None:
        """UNIQUE columns must have no duplicates."""
        schema = _users_orders_schema()
        config = DBSproutConfig()

        result = orchestrate(schema, config, seed=42, default_rows=20)

        emails = [r["email"] for r in result.tables_data["users"]]
        assert len(emails) == len(set(emails))

    def test_autoincrement_pk_sequential(self) -> None:
        """Auto-increment PKs must be sequential 1, 2, 3, ..."""
        schema = _users_orders_schema()
        config = DBSproutConfig()

        result = orchestrate(schema, config, seed=42, default_rows=5)

        user_ids = [r["id"] for r in result.tables_data["users"]]
        assert user_ids == [1, 2, 3, 4, 5]


class TestPerTableConfig:
    def test_per_table_row_override(self) -> None:
        """Per-table config overrides default row count."""
        schema = _users_orders_schema()
        config = DBSproutConfig(
            tables={
                "users": TableOverride(rows=3),
                "orders": TableOverride(rows=7),
            },
        )

        result = orchestrate(schema, config, seed=42, default_rows=100)

        assert len(result.tables_data["users"]) == 3
        assert len(result.tables_data["orders"]) == 7

    def test_excluded_table_skipped(self) -> None:
        """Excluded tables must not appear in output."""
        schema = _users_orders_schema()
        config = DBSproutConfig(
            tables={
                "orders": TableOverride(exclude=True),
            },
        )

        result = orchestrate(schema, config, seed=42, default_rows=5)

        assert "users" in result.tables_data
        assert "orders" not in result.tables_data


class TestDeterministic:
    def test_same_seed_same_structure(self) -> None:
        """Same seed must produce same row counts and PK sequences."""
        schema = _users_orders_schema()
        config = DBSproutConfig()

        r1 = orchestrate(schema, config, seed=99, default_rows=10)
        r2 = orchestrate(schema, config, seed=99, default_rows=10)

        # Same table set and row counts
        assert set(r1.tables_data.keys()) == set(r2.tables_data.keys())
        for table in r1.tables_data:
            assert len(r1.tables_data[table]) == len(r2.tables_data[table])

        # Same PK sequences (autoincrement is deterministic)
        assert [r["id"] for r in r1.tables_data["users"]] == [
            r["id"] for r in r2.tables_data["users"]
        ]
        # Same FK references (FK sampling with same seed is deterministic)
        assert [r["user_id"] for r in r1.tables_data["orders"]] == [
            r["user_id"] for r in r2.tables_data["orders"]
        ]


class TestEmptySchema:
    def test_empty_schema_returns_empty(self) -> None:
        """Schema with no tables must return empty result."""
        schema = DatabaseSchema(tables=[])
        config = DBSproutConfig()

        result = orchestrate(schema, config, seed=42, default_rows=10)

        assert result.tables_data == {}
        assert result.total_rows == 0
        assert result.total_tables == 0


class TestSpecEngine:
    def test_engine_spec_produces_data(self) -> None:
        """engine='spec' uses SpecDrivenEngine and produces valid data."""
        schema = _users_orders_schema()
        config = DBSproutConfig()

        result = orchestrate(schema, config, seed=42, default_rows=5, engine="spec")

        assert "users" in result.tables_data
        assert "orders" in result.tables_data
        assert len(result.tables_data["users"]) == 5


class TestGenerateResult:
    def test_result_stats(self) -> None:
        """GenerateResult must report correct stats."""
        schema = _users_orders_schema()
        config = DBSproutConfig()

        result = orchestrate(schema, config, seed=42, default_rows=5)

        assert result.total_tables == 2
        assert result.total_rows == 10  # 5 users + 5 orders
        assert isinstance(result.tables_data, dict)
