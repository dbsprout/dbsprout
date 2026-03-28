"""Tests for dbsprout.schema.dialect — type normalization."""

from __future__ import annotations

from sqlalchemy import types as sa_types

from dbsprout.schema.dialect import normalize_type
from dbsprout.schema.models import ColumnType


class TestNormalizeTypeIntegers:
    """Integer family: INTEGER, BIGINT, SMALLINT."""

    def test_integer(self) -> None:
        col_type, meta = normalize_type(sa_types.Integer(), "sqlite", "INTEGER")
        assert col_type is ColumnType.INTEGER
        assert meta == {}

    def test_bigint(self) -> None:
        col_type, meta = normalize_type(sa_types.BigInteger(), "sqlite", "BIGINT")
        assert col_type is ColumnType.BIGINT
        assert meta == {}

    def test_smallint(self) -> None:
        col_type, meta = normalize_type(sa_types.SmallInteger(), "sqlite", "SMALLINT")
        assert col_type is ColumnType.SMALLINT
        assert meta == {}


class TestNormalizeTypeStrings:
    """String family: VARCHAR, TEXT."""

    def test_varchar_with_length(self) -> None:
        col_type, meta = normalize_type(sa_types.VARCHAR(255), "sqlite", "VARCHAR(255)")
        assert col_type is ColumnType.VARCHAR
        assert meta == {"max_length": 255}

    def test_varchar_no_length(self) -> None:
        col_type, meta = normalize_type(sa_types.VARCHAR(), "sqlite", "VARCHAR")
        assert col_type is ColumnType.VARCHAR
        assert meta == {}

    def test_text(self) -> None:
        col_type, meta = normalize_type(sa_types.Text(), "sqlite", "TEXT")
        assert col_type is ColumnType.TEXT
        assert meta == {}

    def test_string_with_length(self) -> None:
        col_type, meta = normalize_type(sa_types.String(100), "sqlite", "VARCHAR(100)")
        assert col_type is ColumnType.VARCHAR
        assert meta == {"max_length": 100}


class TestNormalizeTypeNumerics:
    """Numeric family: FLOAT, DECIMAL."""

    def test_float(self) -> None:
        col_type, meta = normalize_type(sa_types.Float(), "sqlite", "FLOAT")
        assert col_type is ColumnType.FLOAT
        assert meta == {}

    def test_real(self) -> None:
        col_type, meta = normalize_type(sa_types.REAL(), "sqlite", "REAL")
        assert col_type is ColumnType.FLOAT
        assert meta == {}

    def test_double(self) -> None:
        col_type, meta = normalize_type(sa_types.DOUBLE(), "sqlite", "DOUBLE")
        assert col_type is ColumnType.FLOAT
        assert meta == {}

    def test_decimal_with_precision_and_scale(self) -> None:
        col_type, meta = normalize_type(sa_types.DECIMAL(10, 2), "sqlite", "DECIMAL(10,2)")
        assert col_type is ColumnType.DECIMAL
        assert meta == {"precision": 10, "scale": 2}

    def test_numeric_with_precision_only(self) -> None:
        col_type, meta = normalize_type(sa_types.Numeric(8), "sqlite", "NUMERIC(8)")
        assert col_type is ColumnType.DECIMAL
        assert meta == {"precision": 8}

    def test_numeric_no_precision(self) -> None:
        col_type, meta = normalize_type(sa_types.Numeric(), "sqlite", "NUMERIC")
        assert col_type is ColumnType.DECIMAL
        assert meta == {}


class TestNormalizeTypeBoolean:
    def test_boolean(self) -> None:
        col_type, meta = normalize_type(sa_types.Boolean(), "sqlite", "BOOLEAN")
        assert col_type is ColumnType.BOOLEAN
        assert meta == {}


class TestNormalizeTypeDatetime:
    """Date/time family."""

    def test_date(self) -> None:
        col_type, meta = normalize_type(sa_types.Date(), "sqlite", "DATE")
        assert col_type is ColumnType.DATE
        assert meta == {}

    def test_datetime(self) -> None:
        col_type, meta = normalize_type(sa_types.DateTime(), "sqlite", "DATETIME")
        assert col_type is ColumnType.DATETIME
        assert meta == {}

    def test_timestamp(self) -> None:
        col_type, meta = normalize_type(sa_types.TIMESTAMP(), "sqlite", "TIMESTAMP")
        assert col_type is ColumnType.TIMESTAMP
        assert meta == {}

    def test_time(self) -> None:
        col_type, meta = normalize_type(sa_types.Time(), "sqlite", "TIME")
        assert col_type is ColumnType.TIME
        assert meta == {}


class TestNormalizeTypeBinary:
    def test_blob(self) -> None:
        col_type, meta = normalize_type(sa_types.LargeBinary(), "sqlite", "BLOB")
        assert col_type is ColumnType.BINARY
        assert meta == {}


class TestNormalizeTypeJson:
    def test_json(self) -> None:
        col_type, meta = normalize_type(sa_types.JSON(), "sqlite", "JSON")
        assert col_type is ColumnType.JSON
        assert meta == {}


class TestNormalizeTypeUuid:
    """UUID detected via raw_type string override."""

    def test_uuid_from_raw_type(self) -> None:
        col_type, meta = normalize_type(sa_types.VARCHAR(36), "sqlite", "UUID")
        assert col_type is ColumnType.UUID
        assert meta == {}

    def test_uuid_case_insensitive(self) -> None:
        col_type, meta = normalize_type(sa_types.VARCHAR(36), "sqlite", "uuid")
        assert col_type is ColumnType.UUID
        assert meta == {}


class TestNormalizeTypeUnknown:
    def test_null_type(self) -> None:
        col_type, meta = normalize_type(sa_types.NullType(), "sqlite", "")
        assert col_type is ColumnType.UNKNOWN
        assert meta == {}


class TestNormalizeTypeEnum:
    def test_enum_type(self) -> None:
        col_type, meta = normalize_type(
            sa_types.Enum("active", "inactive", "deleted"),
            "sqlite",
            "VARCHAR(8)",
        )
        assert col_type is ColumnType.ENUM
        assert meta == {"enum_values": ["active", "deleted", "inactive"]}
