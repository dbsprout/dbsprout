"""Tests for dbsprout.schema.dialect — MSSQL type normalization."""

from __future__ import annotations

from sqlalchemy.dialects import mssql

from dbsprout.schema.dialect import normalize_type
from dbsprout.schema.models import ColumnType


class TestMssqlTypesAlreadyHandled:
    """MSSQL types that inherit from generic SA types — no map entries needed."""

    def test_nvarchar(self) -> None:
        col_type, meta = normalize_type(mssql.NVARCHAR(length=255), "mssql", "nvarchar(255)")
        assert col_type is ColumnType.VARCHAR
        assert meta == {"max_length": 255}

    def test_nchar(self) -> None:
        col_type, meta = normalize_type(mssql.NCHAR(length=10), "mssql", "nchar(10)")
        assert col_type is ColumnType.VARCHAR
        assert meta == {"max_length": 10}

    def test_ntext(self) -> None:
        col_type, meta = normalize_type(mssql.NTEXT(), "mssql", "ntext")
        assert col_type is ColumnType.TEXT
        assert meta == {}

    def test_smalldatetime(self) -> None:
        col_type, meta = normalize_type(mssql.SMALLDATETIME(), "mssql", "smalldatetime")
        assert col_type is ColumnType.DATETIME
        assert meta == {}

    def test_bit(self) -> None:
        col_type, meta = normalize_type(mssql.BIT(), "mssql", "bit")
        assert col_type is ColumnType.BOOLEAN
        assert meta == {}

    def test_image(self) -> None:
        col_type, meta = normalize_type(mssql.IMAGE(), "mssql", "image")
        assert col_type is ColumnType.BINARY
        assert meta == {}

    def test_xml(self) -> None:
        col_type, meta = normalize_type(mssql.XML(), "mssql", "xml")
        assert col_type is ColumnType.TEXT
        assert meta == {}

    def test_uniqueidentifier(self) -> None:
        """UNIQUEIDENTIFIER inherits from sa_types.Uuid — generic dispatch."""
        col_type, meta = normalize_type(mssql.UNIQUEIDENTIFIER(), "mssql", "uniqueidentifier")
        assert col_type is ColumnType.UUID
        assert meta == {}

    def test_varbinary(self) -> None:
        col_type, meta = normalize_type(mssql.VARBINARY(), "mssql", "varbinary")
        assert col_type is ColumnType.BINARY
        assert meta == {}

    def test_sql_variant(self) -> None:
        col_type, meta = normalize_type(mssql.SQL_VARIANT(), "mssql", "sql_variant")
        assert col_type is ColumnType.UNKNOWN
        assert meta == {}


class TestMssqlTypeMapEntries:
    """MSSQL types that need explicit _MSSQL_TYPE_MAP entries."""

    def test_datetime2_maps_to_timestamp(self) -> None:
        col_type, meta = normalize_type(mssql.DATETIME2(), "mssql", "datetime2")
        assert col_type is ColumnType.TIMESTAMP
        assert meta == {}

    def test_datetimeoffset_maps_to_timestamp(self) -> None:
        col_type, meta = normalize_type(mssql.DATETIMEOFFSET(), "mssql", "datetimeoffset")
        assert col_type is ColumnType.TIMESTAMP
        assert meta == {}

    def test_money_maps_to_decimal(self) -> None:
        col_type, meta = normalize_type(mssql.MONEY(), "mssql", "money")
        assert col_type is ColumnType.DECIMAL
        assert meta == {}

    def test_smallmoney_maps_to_decimal(self) -> None:
        col_type, meta = normalize_type(mssql.SMALLMONEY(), "mssql", "smallmoney")
        assert col_type is ColumnType.DECIMAL
        assert meta == {}


class TestMssqlTypesNotMatchedOnOtherDialect:
    """MSSQL-only map entries should fall through on non-MSSQL dialects."""

    def test_money_falls_to_unknown_on_sqlite(self) -> None:
        """MONEY inherits from TypeEngine directly, not Numeric, so falls to UNKNOWN."""
        col_type, _meta = normalize_type(mssql.MONEY(), "sqlite", "money")
        assert col_type is ColumnType.UNKNOWN

    def test_datetime2_falls_to_datetime_on_sqlite(self) -> None:
        """Without MSSQL map, DATETIME2 falls through to DATETIME via DateTime."""
        col_type, _meta = normalize_type(mssql.DATETIME2(), "sqlite", "datetime2")
        assert col_type is ColumnType.DATETIME
