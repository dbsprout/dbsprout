"""Tests for dbsprout.privacy.redactor — schema redaction for redacted tier."""

from __future__ import annotations

from dbsprout.privacy.redactor import RedactionMap, de_redact_spec, redact_schema
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    IndexSchema,
    TableSchema,
)
from dbsprout.spec.models import DataSpec, GeneratorConfig, TableSpec

_FIXED_SALT = b"test-salt-16byte"


def _make_schema() -> DatabaseSchema:
    """Build a small test schema with 2 tables, FK, comments, and metadata."""
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
                        default="'nobody@example.com'",
                    ),
                    ColumnSchema(
                        name="status",
                        data_type=ColumnType.ENUM,
                        enum_values=["active", "suspended", "banned"],
                        check_constraint="status IN ('active','suspended','banned')",
                    ),
                ],
                primary_key=["id"],
                comment="Main user table",
                indexes=[
                    IndexSchema(name="idx_users_email", columns=["email"], unique=True),
                ],
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
                        name="fk_orders_user_id",
                        columns=["user_id"],
                        ref_table="users",
                        ref_columns=["id"],
                    ),
                ],
            ),
        ],
        enums={"user_status": ["active", "suspended"]},
        dialect="postgresql",
        source="test",
    )


class TestRedactSchemaReturnsTyple:
    """redact_schema returns (DatabaseSchema, RedactionMap)."""

    def test_returns_tuple(self) -> None:
        schema = _make_schema()
        result = redact_schema(schema, salt=_FIXED_SALT)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_first_element_is_database_schema(self) -> None:
        schema = _make_schema()
        redacted, _ = redact_schema(schema, salt=_FIXED_SALT)
        assert isinstance(redacted, DatabaseSchema)

    def test_second_element_is_redaction_map(self) -> None:
        schema = _make_schema()
        _, rmap = redact_schema(schema, salt=_FIXED_SALT)
        assert isinstance(rmap, RedactionMap)


class TestRedactionMap:
    """RedactionMap stores mappings and salt."""

    def test_stores_salt(self) -> None:
        _, rmap = redact_schema(_make_schema(), salt=_FIXED_SALT)
        assert rmap.salt == _FIXED_SALT

    def test_stores_table_map(self) -> None:
        _, rmap = redact_schema(_make_schema(), salt=_FIXED_SALT)
        assert "users" in rmap.table_map
        assert "orders" in rmap.table_map
        assert rmap.table_map["users"].startswith("tbl_")

    def test_stores_column_maps(self) -> None:
        _, rmap = redact_schema(_make_schema(), salt=_FIXED_SALT)
        assert "users" in rmap.column_maps
        assert "email" in rmap.column_maps["users"]

    def test_map_entries_match_redacted_names(self) -> None:
        redacted, rmap = redact_schema(_make_schema(), salt=_FIXED_SALT)
        assert redacted.tables[0].name == rmap.table_map["users"]
        assert redacted.tables[0].columns[1].name == rmap.column_maps["users"]["email"]


class TestSaltedHashing:
    """HMAC-SHA256 salted hashing."""

    def test_different_salts_produce_different_hashes(self) -> None:
        schema = _make_schema()
        r1, _ = redact_schema(schema, salt=b"salt-a-0000000000")
        r2, _ = redact_schema(schema, salt=b"salt-b-0000000000")
        assert r1.tables[0].name != r2.tables[0].name

    def test_same_salt_is_deterministic(self) -> None:
        schema = _make_schema()
        r1, _ = redact_schema(schema, salt=_FIXED_SALT)
        r2, _ = redact_schema(schema, salt=_FIXED_SALT)
        assert r1.tables[0].name == r2.tables[0].name

    def test_random_salt_generated_when_none(self) -> None:
        schema = _make_schema()
        _, rmap = redact_schema(schema)
        assert len(rmap.salt) == 16


class TestRedactSchemaColumnNames:
    """Column names are hashed."""

    def test_column_names_are_hashed(self) -> None:
        redacted, _ = redact_schema(_make_schema(), salt=_FIXED_SALT)
        col_names = [c.name for t in redacted.tables for c in t.columns]
        assert all(n.startswith("col_") for n in col_names)
        assert "email" not in col_names

    def test_primary_key_names_are_hashed(self) -> None:
        redacted, _ = redact_schema(_make_schema(), salt=_FIXED_SALT)
        for table in redacted.tables:
            for pk_col in table.primary_key:
                assert pk_col.startswith("col_")


class TestRedactSchemaTableNames:
    """Table names are hashed."""

    def test_table_names_are_hashed(self) -> None:
        redacted, _ = redact_schema(_make_schema(), salt=_FIXED_SALT)
        table_names = [t.name for t in redacted.tables]
        assert all(n.startswith("tbl_") for n in table_names)
        assert "users" not in table_names


class TestRedactSchemaComments:
    """Comments are stripped."""

    def test_table_comments_stripped(self) -> None:
        redacted, _ = redact_schema(_make_schema(), salt=_FIXED_SALT)
        for table in redacted.tables:
            assert table.comment is None

    def test_column_comments_stripped(self) -> None:
        redacted, _ = redact_schema(_make_schema(), salt=_FIXED_SALT)
        for table in redacted.tables:
            for col in table.columns:
                assert col.comment is None


class TestRedactSchemaEnhanced:
    """Enhanced redaction: enum_values, defaults, constraints, names."""

    def test_enum_values_stripped(self) -> None:
        redacted, _ = redact_schema(_make_schema(), salt=_FIXED_SALT)
        status_col = redacted.tables[0].columns[2]
        assert status_col.enum_values is None

    def test_default_values_stripped(self) -> None:
        redacted, _ = redact_schema(_make_schema(), salt=_FIXED_SALT)
        email_col = redacted.tables[0].columns[1]
        assert email_col.default is None

    def test_check_constraint_stripped(self) -> None:
        redacted, _ = redact_schema(_make_schema(), salt=_FIXED_SALT)
        status_col = redacted.tables[0].columns[2]
        assert status_col.check_constraint is None

    def test_fk_names_redacted(self) -> None:
        redacted, _ = redact_schema(_make_schema(), salt=_FIXED_SALT)
        fk = redacted.tables[1].foreign_keys[0]
        assert fk.name is not None
        assert fk.name.startswith("fk_")
        assert "orders" not in fk.name

    def test_index_names_redacted(self) -> None:
        redacted, _ = redact_schema(_make_schema(), salt=_FIXED_SALT)
        idx = redacted.tables[0].indexes[0]
        assert idx.name is not None
        assert idx.name.startswith("idx_")
        assert "users" not in idx.name

    def test_schema_enums_redacted(self) -> None:
        redacted, _ = redact_schema(_make_schema(), salt=_FIXED_SALT)
        assert "user_status" not in redacted.enums
        assert len(redacted.enums) == 1
        key = next(iter(redacted.enums.keys()))
        assert key.startswith("enum_")


class TestRedactSchemaPreservesStructure:
    """Types, nullability, PK, FK structure preserved."""

    def test_preserves_column_types(self) -> None:
        schema = _make_schema()
        redacted, _ = redact_schema(schema, salt=_FIXED_SALT)
        original_types = [c.data_type for t in schema.tables for c in t.columns]
        redacted_types = [c.data_type for t in redacted.tables for c in t.columns]
        assert original_types == redacted_types

    def test_preserves_nullable(self) -> None:
        schema = _make_schema()
        redacted, _ = redact_schema(schema, salt=_FIXED_SALT)
        original_nullable = [c.nullable for t in schema.tables for c in t.columns]
        redacted_nullable = [c.nullable for t in redacted.tables for c in t.columns]
        assert original_nullable == redacted_nullable

    def test_preserves_precision_scale(self) -> None:
        schema = _make_schema()
        redacted, _ = redact_schema(schema, salt=_FIXED_SALT)
        total_col_orig = schema.tables[1].columns[2]
        total_col_redacted = redacted.tables[1].columns[2]
        assert total_col_redacted.precision == total_col_orig.precision
        assert total_col_redacted.scale == total_col_orig.scale


class TestRedactSchemaFkConsistency:
    """FK references use consistently hashed names."""

    def test_fk_ref_table_is_hashed(self) -> None:
        redacted, _ = redact_schema(_make_schema(), salt=_FIXED_SALT)
        fk = redacted.tables[1].foreign_keys[0]
        assert fk.ref_table.startswith("tbl_")
        assert fk.ref_table == redacted.tables[0].name

    def test_fk_columns_are_hashed(self) -> None:
        redacted, _ = redact_schema(_make_schema(), salt=_FIXED_SALT)
        fk = redacted.tables[1].foreign_keys[0]
        assert all(c.startswith("col_") for c in fk.columns)
        assert all(c.startswith("col_") for c in fk.ref_columns)


class TestRedactSchemaImmutability:
    """Redaction returns new object, original untouched."""

    def test_returns_new_object(self) -> None:
        schema = _make_schema()
        redacted, _ = redact_schema(schema, salt=_FIXED_SALT)
        assert redacted is not schema

    def test_original_not_mutated(self) -> None:
        schema = _make_schema()
        redact_schema(schema, salt=_FIXED_SALT)
        assert schema.tables[0].name == "users"
        assert schema.tables[0].columns[1].name == "email"


class TestDeRedactSpec:
    """de_redact_spec reverses hashed names in DataSpec."""

    def test_reverses_table_names(self) -> None:
        schema = _make_schema()
        _, rmap = redact_schema(schema, salt=_FIXED_SALT)

        hashed_table = rmap.table_map["users"]
        hashed_col = rmap.column_maps["users"]["email"]
        spec = DataSpec(
            tables=[
                TableSpec(
                    table_name=hashed_table,
                    columns={
                        hashed_col: GeneratorConfig(provider="mimesis.Person.email"),
                    },
                ),
            ],
            schema_hash="test",
            model_used="test",
        )

        result = de_redact_spec(spec, rmap)
        assert result.tables[0].table_name == "users"

    def test_reverses_column_names(self) -> None:
        schema = _make_schema()
        _, rmap = redact_schema(schema, salt=_FIXED_SALT)

        hashed_table = rmap.table_map["users"]
        hashed_col = rmap.column_maps["users"]["email"]
        spec = DataSpec(
            tables=[
                TableSpec(
                    table_name=hashed_table,
                    columns={
                        hashed_col: GeneratorConfig(provider="mimesis.Person.email"),
                    },
                ),
            ],
            schema_hash="test",
            model_used="test",
        )

        result = de_redact_spec(spec, rmap)
        assert "email" in result.tables[0].columns

    def test_preserves_generator_config(self) -> None:
        schema = _make_schema()
        _, rmap = redact_schema(schema, salt=_FIXED_SALT)

        hashed_table = rmap.table_map["users"]
        hashed_col = rmap.column_maps["users"]["id"]
        spec = DataSpec(
            tables=[
                TableSpec(
                    table_name=hashed_table,
                    columns={
                        hashed_col: GeneratorConfig(provider="builtin.autoincrement"),
                    },
                ),
            ],
            schema_hash="test",
            model_used="test",
        )

        result = de_redact_spec(spec, rmap)
        assert result.tables[0].columns["id"].provider == "builtin.autoincrement"

    def test_returns_new_object(self) -> None:
        schema = _make_schema()
        _, rmap = redact_schema(schema, salt=_FIXED_SALT)

        hashed_table = rmap.table_map["users"]
        spec = DataSpec(
            tables=[
                TableSpec(table_name=hashed_table, columns={}),
            ],
            schema_hash="test",
            model_used="test",
        )

        result = de_redact_spec(spec, rmap)
        assert result is not spec
