"""Tests for dbsprout.generate.engines.spec_driven — spec-driven generation engine."""

from __future__ import annotations

import uuid
from typing import Any

from dbsprout.generate.engines.spec_driven import SpecDrivenEngine
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    ForeignKeySchema,
    TableSchema,
)
from dbsprout.spec.models import GeneratorConfig, TableSpec


def _col(
    name: str,
    *,
    nullable: bool = True,
    pk: bool = False,
    autoincrement: bool = False,
    data_type: ColumnType = ColumnType.INTEGER,
) -> ColumnSchema:
    return ColumnSchema(
        name=name,
        data_type=data_type,
        nullable=nullable,
        primary_key=pk,
        autoincrement=autoincrement,
    )


class TestMimesisDispatch:
    def test_dispatches_mimesis_email(self) -> None:
        """mimesis.Person.email produces email-like strings."""
        table = TableSchema(
            name="users",
            columns=[_col("email", data_type=ColumnType.VARCHAR)],
            primary_key=[],
        )
        spec = TableSpec(
            table_name="users",
            columns={"email": GeneratorConfig(provider="mimesis.Person.email")},
        )
        engine = SpecDrivenEngine(seed=42)
        rows = engine.generate_table(table, spec, 5)

        assert len(rows) == 5
        for row in rows:
            assert isinstance(row["email"], str)
            assert "@" in row["email"]

    def test_dispatches_mimesis_full_name(self) -> None:
        """mimesis.Person.full_name produces non-empty strings."""
        table = TableSchema(
            name="users",
            columns=[_col("name", data_type=ColumnType.VARCHAR)],
            primary_key=[],
        )
        spec = TableSpec(
            table_name="users",
            columns={"name": GeneratorConfig(provider="mimesis.Person.full_name")},
        )
        engine = SpecDrivenEngine(seed=42)
        rows = engine.generate_table(table, spec, 5)

        assert all(len(row["name"]) > 0 for row in rows)


class TestNumpyDistributions:
    def test_numpy_uniform(self) -> None:
        """numpy.uniform with min/max produces values in range."""
        table = TableSchema(
            name="items",
            columns=[_col("price", data_type=ColumnType.FLOAT)],
            primary_key=[],
        )
        spec = TableSpec(
            table_name="items",
            columns={
                "price": GeneratorConfig(
                    provider="numpy.uniform",
                    distribution="uniform",
                    min_value=10.0,
                    max_value=100.0,
                ),
            },
        )
        engine = SpecDrivenEngine(seed=42)
        rows = engine.generate_table(table, spec, 50)

        for row in rows:
            assert 10.0 <= row["price"] <= 100.0

    def test_numpy_normal(self) -> None:
        """numpy.normal with mean/std produces reasonable values."""
        table = TableSchema(
            name="scores",
            columns=[_col("score", data_type=ColumnType.FLOAT)],
            primary_key=[],
        )
        spec = TableSpec(
            table_name="scores",
            columns={
                "score": GeneratorConfig(
                    provider="numpy.normal",
                    distribution="normal",
                    distribution_params={"mean": 50.0, "std": 10.0},
                    min_value=0.0,
                    max_value=100.0,
                ),
            },
        )
        engine = SpecDrivenEngine(seed=42)
        rows = engine.generate_table(table, spec, 100)

        values = [row["score"] for row in rows]
        assert all(0.0 <= v <= 100.0 for v in values)
        # Mean should be roughly near 50
        avg = sum(values) / len(values)
        assert 30.0 < avg < 70.0


class TestEnumValues:
    def test_enum_values_respected(self) -> None:
        """Only enum values appear in output."""
        table = TableSchema(
            name="tasks",
            columns=[_col("status", data_type=ColumnType.VARCHAR)],
            primary_key=[],
        )
        spec = TableSpec(
            table_name="tasks",
            columns={
                "status": GeneratorConfig(
                    provider="builtin.default",
                    enum_values=["active", "inactive", "pending"],
                ),
            },
        )
        engine = SpecDrivenEngine(seed=42)
        rows = engine.generate_table(table, spec, 50)

        valid = {"active", "inactive", "pending"}
        for row in rows:
            assert row["status"] in valid


class TestNullableRate:
    def test_nullable_rate_applied(self) -> None:
        """~10% of values are None when nullable_rate=0.1."""
        table = TableSchema(
            name="items",
            columns=[_col("notes", data_type=ColumnType.TEXT, nullable=True)],
            primary_key=[],
        )
        spec = TableSpec(
            table_name="items",
            columns={
                "notes": GeneratorConfig(
                    provider="mimesis.Text.word",
                    nullable_rate=0.5,
                ),
            },
        )
        engine = SpecDrivenEngine(seed=42)
        rows = engine.generate_table(table, spec, 200)

        null_count = sum(1 for r in rows if r["notes"] is None)
        # With 0.5 rate and 200 rows, expect roughly 100 nulls (allow wide margin)
        assert 50 < null_count < 150


class TestFKAndAutoincrement:
    def test_fk_and_autoincrement_none(self) -> None:
        """FK columns and autoincrement PKs are set to None."""
        table = TableSchema(
            name="orders",
            columns=[
                _col("id", nullable=False, pk=True, autoincrement=True),
                _col("user_id"),
            ],
            primary_key=["id"],
            foreign_keys=[
                ForeignKeySchema(
                    columns=["user_id"],
                    ref_table="users",
                    ref_columns=["id"],
                ),
            ],
        )
        spec = TableSpec(
            table_name="orders",
            columns={
                "id": GeneratorConfig(provider="builtin.autoincrement"),
                "user_id": GeneratorConfig(provider="builtin.default"),
            },
        )
        engine = SpecDrivenEngine(seed=42)
        rows = engine.generate_table(table, spec, 5)

        for row in rows:
            assert row["id"] is None
            assert row["user_id"] is None


class TestUnknownProviderFallback:
    def test_unknown_provider_produces_values(self) -> None:
        """Unknown provider falls back to heuristic-style generation."""
        table = TableSchema(
            name="misc",
            columns=[_col("data", data_type=ColumnType.VARCHAR)],
            primary_key=[],
        )
        spec = TableSpec(
            table_name="misc",
            columns={
                "data": GeneratorConfig(provider="nonexistent.Unknown.method"),
            },
        )
        engine = SpecDrivenEngine(seed=42)
        rows = engine.generate_table(table, spec, 5)

        # Should produce something (not crash)
        assert len(rows) == 5


class TestBuiltinUUID:
    def test_uuid4_generation(self) -> None:
        """builtin.uuid4 produces valid UUID strings."""
        table = TableSchema(
            name="tokens",
            columns=[_col("token", data_type=ColumnType.UUID)],
            primary_key=[],
        )
        spec = TableSpec(
            table_name="tokens",
            columns={
                "token": GeneratorConfig(provider="builtin.uuid4"),
            },
        )
        engine = SpecDrivenEngine(seed=42)
        rows = engine.generate_table(table, spec, 5)

        for row in rows:
            parsed = uuid.UUID(row["token"])
            assert isinstance(parsed, uuid.UUID)

    def test_uuid4_deterministic(self) -> None:
        """Same seed produces identical UUIDs."""
        table = TableSchema(
            name="tokens",
            columns=[_col("token", data_type=ColumnType.UUID)],
            primary_key=[],
        )
        spec = TableSpec(
            table_name="tokens",
            columns={"token": GeneratorConfig(provider="builtin.uuid4")},
        )
        e1 = SpecDrivenEngine(seed=99)
        e2 = SpecDrivenEngine(seed=99)

        rows1 = e1.generate_table(table, spec, 5)
        rows2 = e2.generate_table(table, spec, 5)

        assert [r["token"] for r in rows1] == [r["token"] for r in rows2]


class TestDerivedAndCorrelationWarnings:
    def test_derived_columns_logged_warning(self, caplog: Any) -> None:
        """Derived columns log a warning and are skipped."""
        import logging  # noqa: PLC0415

        from dbsprout.spec.models import DerivedColumn  # noqa: PLC0415

        table = TableSchema(
            name="users",
            columns=[_col("name", data_type=ColumnType.VARCHAR)],
            primary_key=[],
        )
        spec = TableSpec(
            table_name="users",
            columns={"name": GeneratorConfig(provider="mimesis.Text.word")},
            derived=[
                DerivedColumn(column="full", expression="x", depends_on=["name"]),
            ],
        )
        engine = SpecDrivenEngine(seed=42)
        with caplog.at_level(logging.WARNING):
            rows = engine.generate_table(table, spec, 3)

        assert len(rows) == 3
        assert "derived columns" in caplog.text.lower()

    def test_correlations_logged_warning(self, caplog: Any) -> None:
        """Correlation rules log a warning and are skipped."""
        import logging  # noqa: PLC0415

        from dbsprout.spec.models import CorrelationRule  # noqa: PLC0415

        table = TableSchema(
            name="users",
            columns=[_col("city", data_type=ColumnType.VARCHAR)],
            primary_key=[],
        )
        spec = TableSpec(
            table_name="users",
            columns={"city": GeneratorConfig(provider="mimesis.Address.city")},
            correlations=[CorrelationRule(columns=["city", "state"])],
        )
        engine = SpecDrivenEngine(seed=42)
        with caplog.at_level(logging.WARNING):
            rows = engine.generate_table(table, spec, 3)

        assert len(rows) == 3
        assert "correlation" in caplog.text.lower()


class TestDeterministic:
    def test_same_seed_same_rows(self) -> None:
        """Same seed + same spec = identical rows."""
        table = TableSchema(
            name="items",
            columns=[_col("val", data_type=ColumnType.FLOAT)],
            primary_key=[],
        )
        spec = TableSpec(
            table_name="items",
            columns={
                "val": GeneratorConfig(
                    provider="numpy.uniform",
                    min_value=0.0,
                    max_value=100.0,
                ),
            },
        )
        e1 = SpecDrivenEngine(seed=99)
        e2 = SpecDrivenEngine(seed=99)

        rows1 = e1.generate_table(table, spec, 10)
        rows2 = e2.generate_table(table, spec, 10)

        assert rows1 == rows2
