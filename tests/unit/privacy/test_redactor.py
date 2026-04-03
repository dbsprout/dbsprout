"""Tests for dbsprout.privacy.redactor — schema redaction for redacted tier."""

from __future__ import annotations

from dbsprout.privacy.redactor import redact_schema
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    TableSchema,
)


def _make_schema() -> DatabaseSchema:
    """Build a small test schema with 2 tables, FK, and comments."""
    return DatabaseSchema(
        tables=[
            TableSchema(
                name="users",
                columns=[
                    ColumnSchema(
                        name="id",
                        data_type=ColumnType.INTEGER,
                        primary_key=True,
                        nullable=False,
                    ),
                    ColumnSchema(
                        name="email",
                        data_type=ColumnType.VARCHAR,
                        max_length=255,
                        comment="User email address",
                    ),
                ],
                primary_key=["id"],
                comment="Main user table",
            ),
            TableSchema(
                name="orders",
                columns=[
                    ColumnSchema(
                        name="id",
                        data_type=ColumnType.INTEGER,
                        primary_key=True,
                        nullable=False,
                    ),
                    ColumnSchema(
                        name="user_id",
                        data_type=ColumnType.INTEGER,
                        nullable=False,
                    ),
                    ColumnSchema(
                        name="total",
                        data_type=ColumnType.DECIMAL,
                        precision=10,
                        scale=2,
                    ),
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
        dialect="postgresql",
        source="test",
    )


class TestRedactSchemaColumnNames:
    """Column names are hashed."""

    def test_column_names_are_hashed(self) -> None:
        schema = _make_schema()
        redacted = redact_schema(schema)
        col_names = [c.name for t in redacted.tables for c in t.columns]
        assert all(n.startswith("col_") for n in col_names)
        assert "email" not in col_names
        assert "user_id" not in col_names

    def test_primary_key_names_are_hashed(self) -> None:
        schema = _make_schema()
        redacted = redact_schema(schema)
        for table in redacted.tables:
            for pk_col in table.primary_key:
                assert pk_col.startswith("col_")


class TestRedactSchemaTableNames:
    """Table names are hashed."""

    def test_table_names_are_hashed(self) -> None:
        schema = _make_schema()
        redacted = redact_schema(schema)
        table_names = [t.name for t in redacted.tables]
        assert all(n.startswith("tbl_") for n in table_names)
        assert "users" not in table_names
        assert "orders" not in table_names


class TestRedactSchemaComments:
    """Comments are stripped."""

    def test_table_comments_stripped(self) -> None:
        schema = _make_schema()
        redacted = redact_schema(schema)
        for table in redacted.tables:
            assert table.comment is None

    def test_column_comments_stripped(self) -> None:
        schema = _make_schema()
        redacted = redact_schema(schema)
        for table in redacted.tables:
            for col in table.columns:
                assert col.comment is None


class TestRedactSchemaPreservesStructure:
    """Types, nullability, PK, FK structure preserved."""

    def test_preserves_column_types(self) -> None:
        schema = _make_schema()
        redacted = redact_schema(schema)
        original_types = [c.data_type for t in schema.tables for c in t.columns]
        redacted_types = [c.data_type for t in redacted.tables for c in t.columns]
        assert original_types == redacted_types

    def test_preserves_nullable(self) -> None:
        schema = _make_schema()
        redacted = redact_schema(schema)
        original_nullable = [c.nullable for t in schema.tables for c in t.columns]
        redacted_nullable = [c.nullable for t in redacted.tables for c in t.columns]
        assert original_nullable == redacted_nullable

    def test_preserves_max_length(self) -> None:
        schema = _make_schema()
        redacted = redact_schema(schema)
        original_ml = [c.max_length for t in schema.tables for c in t.columns]
        redacted_ml = [c.max_length for t in redacted.tables for c in t.columns]
        assert original_ml == redacted_ml

    def test_preserves_precision_scale(self) -> None:
        schema = _make_schema()
        redacted = redact_schema(schema)
        total_col_orig = schema.tables[1].columns[2]
        total_col_redacted = redacted.tables[1].columns[2]
        assert total_col_redacted.precision == total_col_orig.precision
        assert total_col_redacted.scale == total_col_orig.scale


class TestRedactSchemaFkConsistency:
    """FK references use consistently hashed names."""

    def test_fk_ref_table_is_hashed(self) -> None:
        schema = _make_schema()
        redacted = redact_schema(schema)
        fk = redacted.tables[1].foreign_keys[0]
        assert fk.ref_table.startswith("tbl_")
        assert fk.ref_table == redacted.tables[0].name

    def test_fk_columns_are_hashed(self) -> None:
        schema = _make_schema()
        redacted = redact_schema(schema)
        fk = redacted.tables[1].foreign_keys[0]
        assert all(c.startswith("col_") for c in fk.columns)
        assert all(c.startswith("col_") for c in fk.ref_columns)


class TestRedactSchemaImmutability:
    """Redaction returns new object, original untouched."""

    def test_returns_new_object(self) -> None:
        schema = _make_schema()
        redacted = redact_schema(schema)
        assert redacted is not schema

    def test_original_not_mutated(self) -> None:
        schema = _make_schema()
        redact_schema(schema)
        assert schema.tables[0].name == "users"
        assert schema.tables[0].columns[1].name == "email"

    def test_deterministic(self) -> None:
        schema = _make_schema()
        r1 = redact_schema(schema)
        r2 = redact_schema(schema)
        assert r1.tables[0].name == r2.tables[0].name
        assert r1.tables[0].columns[0].name == r2.tables[0].columns[0].name
