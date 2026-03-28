"""Tests for dbsprout.schema.dialect — MySQL type normalization."""

from __future__ import annotations

from sqlalchemy.dialects import mysql

from dbsprout.schema.dialect import normalize_type
from dbsprout.schema.models import ColumnType


class TestMysqlTypesAlreadyHandled:
    """MySQL types that inherit from generic SA types — no code changes needed."""

    def test_mediumint(self) -> None:
        col_type, meta = normalize_type(mysql.MEDIUMINT(), "mysql", "mediumint")
        assert col_type is ColumnType.INTEGER
        assert meta == {}

    def test_double(self) -> None:
        col_type, meta = normalize_type(mysql.DOUBLE(), "mysql", "double")
        assert col_type is ColumnType.FLOAT
        assert meta == {}

    def test_enum(self) -> None:
        col_type, meta = normalize_type(
            mysql.ENUM("active", "inactive"), "mysql", "enum('active','inactive')"
        )
        assert col_type is ColumnType.ENUM
        assert meta == {"enum_values": ["active", "inactive"]}

    def test_json(self) -> None:
        col_type, meta = normalize_type(mysql.JSON(), "mysql", "json")
        assert col_type is ColumnType.JSON
        assert meta == {}


class TestMysqlTypeMapEntries:
    """MySQL types that need explicit _MYSQL_TYPE_MAP entries."""

    def test_mediumtext(self) -> None:
        col_type, meta = normalize_type(mysql.MEDIUMTEXT(), "mysql", "mediumtext")
        assert col_type is ColumnType.TEXT
        assert meta == {}

    def test_longtext(self) -> None:
        col_type, meta = normalize_type(mysql.LONGTEXT(), "mysql", "longtext")
        assert col_type is ColumnType.TEXT
        assert meta == {}

    def test_tinytext(self) -> None:
        col_type, meta = normalize_type(mysql.TINYTEXT(), "mysql", "tinytext")
        assert col_type is ColumnType.TEXT
        assert meta == {}

    def test_tinyblob(self) -> None:
        col_type, meta = normalize_type(mysql.TINYBLOB(), "mysql", "tinyblob")
        assert col_type is ColumnType.BINARY
        assert meta == {}

    def test_mediumblob(self) -> None:
        col_type, meta = normalize_type(mysql.MEDIUMBLOB(), "mysql", "mediumblob")
        assert col_type is ColumnType.BINARY
        assert meta == {}

    def test_longblob(self) -> None:
        col_type, meta = normalize_type(mysql.LONGBLOB(), "mysql", "longblob")
        assert col_type is ColumnType.BINARY
        assert meta == {}

    def test_year(self) -> None:
        col_type, meta = normalize_type(mysql.YEAR(), "mysql", "year")
        assert col_type is ColumnType.INTEGER
        assert meta == {}

    def test_bit(self) -> None:
        col_type, meta = normalize_type(mysql.BIT(), "mysql", "bit")
        assert col_type is ColumnType.INTEGER
        assert meta == {}


class TestMysqlTinyintBoolean:
    """TINYINT(1) → BOOLEAN, TINYINT() → SMALLINT."""

    def test_tinyint_display_width_1_is_boolean(self) -> None:
        col_type, meta = normalize_type(mysql.TINYINT(display_width=1), "mysql", "tinyint(1)")
        assert col_type is ColumnType.BOOLEAN
        assert meta == {}

    def test_tinyint_no_display_width_is_smallint(self) -> None:
        col_type, meta = normalize_type(mysql.TINYINT(), "mysql", "tinyint")
        assert col_type is ColumnType.SMALLINT
        assert meta == {}

    def test_tinyint_display_width_3_is_smallint(self) -> None:
        col_type, meta = normalize_type(mysql.TINYINT(display_width=3), "mysql", "tinyint(3)")
        assert col_type is ColumnType.SMALLINT
        assert meta == {}

    def test_tinyint_not_matched_on_other_dialect(self) -> None:
        """TINYINT on non-MySQL dialect should fall to INTEGER (generic Integer)."""
        col_type, _meta = normalize_type(mysql.TINYINT(), "sqlite", "tinyint")
        assert col_type is ColumnType.INTEGER


class TestMysqlSetType:
    """MySQL SET type → VARCHAR with enum_values."""

    def test_set_with_values(self) -> None:
        col_type, meta = normalize_type(
            mysql.SET("read", "write", "admin"), "mysql", "set('read','write','admin')"
        )
        assert col_type is ColumnType.ENUM
        assert meta == {"enum_values": ["admin", "read", "write"]}

    def test_set_not_matched_on_other_dialect(self) -> None:
        """SET on non-MySQL dialect should fall to VARCHAR (generic String)."""
        col_type, _meta = normalize_type(mysql.SET("a", "b"), "sqlite", "set('a','b')")
        assert col_type is ColumnType.VARCHAR


class TestMysqlTypesNotMatchedOnOtherDialect:
    def test_mysql_types_fall_to_unknown_on_sqlite(self) -> None:
        """MySQL-only types should fall through on non-MySQL dialects."""
        col_type, _meta = normalize_type(mysql.YEAR(), "sqlite", "year")
        assert col_type is ColumnType.UNKNOWN
