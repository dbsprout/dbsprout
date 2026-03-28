"""Tests for dbsprout.schema.dialect — PostgreSQL type normalization."""

from __future__ import annotations

import sqlalchemy.types as sa_types
from sqlalchemy.dialects import postgresql as pg

from dbsprout.schema.dialect import normalize_type
from dbsprout.schema.models import ColumnType


class TestPgTypesAlreadyHandled:
    """PG types that inherit from generic SA types — should work with no changes."""

    def test_jsonb(self) -> None:
        col_type, meta = normalize_type(pg.JSONB(), "postgresql", "jsonb")
        assert col_type is ColumnType.JSON
        assert meta == {}

    def test_json(self) -> None:
        col_type, meta = normalize_type(pg.JSON(), "postgresql", "json")
        assert col_type is ColumnType.JSON
        assert meta == {}

    def test_bytea(self) -> None:
        col_type, meta = normalize_type(pg.BYTEA(), "postgresql", "bytea")
        assert col_type is ColumnType.BINARY
        assert meta == {}

    def test_timestamp_with_timezone(self) -> None:
        col_type, meta = normalize_type(
            pg.TIMESTAMP(timezone=True), "postgresql", "timestamp with time zone"
        )
        assert col_type is ColumnType.TIMESTAMP
        assert meta == {}

    def test_timestamp_without_timezone(self) -> None:
        col_type, meta = normalize_type(
            pg.TIMESTAMP(timezone=False), "postgresql", "timestamp without time zone"
        )
        assert col_type is ColumnType.TIMESTAMP
        assert meta == {}

    def test_pg_enum(self) -> None:
        col_type, meta = normalize_type(
            pg.ENUM("active", "inactive", name="status_enum"),
            "postgresql",
            "status_enum",
        )
        assert col_type is ColumnType.ENUM
        assert meta == {"enum_values": ["active", "inactive"]}

    def test_pg_uuid(self) -> None:
        """PG UUID triggers raw_type override."""
        col_type, meta = normalize_type(pg.UUID(), "postgresql", "UUID")
        assert col_type is ColumnType.UUID
        assert meta == {}


class TestPgTypesNewEntries:
    """Types that need new entries in _SIMPLE_TYPE_MAP."""

    def test_array_of_integer(self) -> None:
        col_type, meta = normalize_type(sa_types.ARRAY(sa_types.Integer), "postgresql", "INTEGER[]")
        assert col_type is ColumnType.ARRAY
        assert meta == {}

    def test_array_of_text(self) -> None:
        col_type, meta = normalize_type(sa_types.ARRAY(sa_types.Text), "postgresql", "TEXT[]")
        assert col_type is ColumnType.ARRAY
        assert meta == {}

    def test_generic_uuid(self) -> None:
        """sa_types.Uuid (generic) should also map to UUID."""
        col_type, meta = normalize_type(sa_types.Uuid(), "postgresql", "UUID")
        assert col_type is ColumnType.UUID
        assert meta == {}


class TestPgOnlyTypes:
    """PG-specific types that need a dialect-gated _PG_TYPE_MAP."""

    def test_inet(self) -> None:
        col_type, meta = normalize_type(pg.INET(), "postgresql", "inet")
        assert col_type is ColumnType.VARCHAR
        assert meta == {}

    def test_cidr(self) -> None:
        col_type, meta = normalize_type(pg.CIDR(), "postgresql", "cidr")
        assert col_type is ColumnType.VARCHAR
        assert meta == {}

    def test_money(self) -> None:
        col_type, meta = normalize_type(pg.MONEY(), "postgresql", "money")
        assert col_type is ColumnType.DECIMAL
        assert meta == {}

    def test_interval(self) -> None:
        col_type, meta = normalize_type(pg.INTERVAL(), "postgresql", "interval")
        assert col_type is ColumnType.VARCHAR
        assert meta == {}

    def test_tsvector(self) -> None:
        col_type, meta = normalize_type(pg.TSVECTOR(), "postgresql", "tsvector")
        assert col_type is ColumnType.TEXT
        assert meta == {}

    def test_tsquery(self) -> None:
        col_type, meta = normalize_type(pg.TSQUERY(), "postgresql", "tsquery")
        assert col_type is ColumnType.TEXT
        assert meta == {}

    def test_pg_types_not_matched_on_other_dialect(self) -> None:
        """PG-only types should fall through to UNKNOWN on non-PG dialects."""
        col_type, _meta = normalize_type(pg.INET(), "sqlite", "inet")
        assert col_type is ColumnType.UNKNOWN
