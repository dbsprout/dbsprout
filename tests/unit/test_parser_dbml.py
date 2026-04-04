"""Tests for dbsprout.schema.parsers.dbml — DBML file parser."""

from __future__ import annotations

import pytest

from dbsprout.schema.models import ColumnType
from dbsprout.schema.parsers.dbml import can_parse_dbml, parse_dbml

_SAMPLE_DBML = """
Table users {
  id integer [pk, increment]
  email varchar [not null, unique]
  name varchar
  status user_status
  Note: 'Users table'
}

Table orders {
  id integer [pk, increment]
  user_id integer [not null, ref: > users.id]
  total decimal
  created_at timestamp
}

Enum user_status {
  active
  inactive
  pending
}
"""


class TestParseTablesAndColumns:
    def test_tables_extracted(self) -> None:
        """All tables are extracted."""
        schema = parse_dbml(_SAMPLE_DBML)
        names = schema.table_names()
        assert "users" in names
        assert "orders" in names

    def test_columns_extracted(self) -> None:
        """Columns are extracted with correct names."""
        schema = parse_dbml(_SAMPLE_DBML)
        users = schema.get_table("users")
        assert users is not None
        col_names = [c.name for c in users.columns]
        assert "id" in col_names
        assert "email" in col_names
        assert "name" in col_names


class TestColumnTypeNormalization:
    def test_integer_type(self) -> None:
        """integer → ColumnType.INTEGER."""
        schema = parse_dbml(_SAMPLE_DBML)
        users = schema.get_table("users")
        assert users is not None
        id_col = users.get_column("id")
        assert id_col is not None
        assert id_col.data_type == ColumnType.INTEGER

    def test_varchar_type(self) -> None:
        """varchar → ColumnType.VARCHAR."""
        schema = parse_dbml(_SAMPLE_DBML)
        users = schema.get_table("users")
        assert users is not None
        email_col = users.get_column("email")
        assert email_col is not None
        assert email_col.data_type == ColumnType.VARCHAR

    def test_timestamp_type(self) -> None:
        """timestamp → ColumnType.TIMESTAMP."""
        schema = parse_dbml(_SAMPLE_DBML)
        orders = schema.get_table("orders")
        assert orders is not None
        ts_col = orders.get_column("created_at")
        assert ts_col is not None
        assert ts_col.data_type == ColumnType.TIMESTAMP


class TestPrimaryKey:
    def test_pk_detected(self) -> None:
        """PK columns identified."""
        schema = parse_dbml(_SAMPLE_DBML)
        users = schema.get_table("users")
        assert users is not None
        assert users.primary_key == ["id"]
        id_col = users.get_column("id")
        assert id_col is not None
        assert id_col.primary_key is True


class TestForeignKeys:
    def test_fk_from_ref(self) -> None:
        """DBML ref → ForeignKeySchema."""
        schema = parse_dbml(_SAMPLE_DBML)
        orders = schema.get_table("orders")
        assert orders is not None
        assert len(orders.foreign_keys) == 1
        fk = orders.foreign_keys[0]
        assert fk.columns == ["user_id"]
        assert fk.ref_table == "users"
        assert fk.ref_columns == ["id"]


class TestConstraints:
    def test_not_null(self) -> None:
        """NOT NULL constraint extracted."""
        schema = parse_dbml(_SAMPLE_DBML)
        users = schema.get_table("users")
        assert users is not None
        email = users.get_column("email")
        assert email is not None
        assert email.nullable is False

    def test_unique(self) -> None:
        """UNIQUE constraint extracted."""
        schema = parse_dbml(_SAMPLE_DBML)
        users = schema.get_table("users")
        assert users is not None
        email = users.get_column("email")
        assert email is not None
        assert email.unique is True

    def test_autoincrement(self) -> None:
        """increment → autoincrement."""
        schema = parse_dbml(_SAMPLE_DBML)
        users = schema.get_table("users")
        assert users is not None
        id_col = users.get_column("id")
        assert id_col is not None
        assert id_col.autoincrement is True


class TestEnumExtraction:
    def test_enums_extracted(self) -> None:
        """DBML enums → schema enums."""
        schema = parse_dbml(_SAMPLE_DBML)
        assert "user_status" in schema.enums
        assert "active" in schema.enums["user_status"]
        assert "inactive" in schema.enums["user_status"]
        assert "pending" in schema.enums["user_status"]


class TestCanParse:
    def test_dbml_extension(self) -> None:
        """.dbml extension detected."""
        assert can_parse_dbml("schema.dbml") is True

    def test_non_dbml_extension(self) -> None:
        """Non-.dbml extension rejected."""
        assert can_parse_dbml("schema.sql") is False

    def test_dbml_keyword(self) -> None:
        """DBML keyword in content detected."""
        assert can_parse_dbml("Table users {") is True


class TestMalformedDBML:
    def test_malformed_raises_error(self) -> None:
        """Bad DBML raises ValueError."""
        with pytest.raises(ValueError, match="DBML"):
            parse_dbml("{{{{ invalid dbml syntax !!!!")


class TestManyToManyRef:
    def test_many_to_many_skipped(self) -> None:
        """<> ref logged as warning and skipped."""
        dbml = """
Table students {
  id integer [pk]
}
Table courses {
  id integer [pk]
}
Ref: students.id <> courses.id
"""
        schema = parse_dbml(dbml)
        students = schema.get_table("students")
        courses = schema.get_table("courses")
        assert students is not None
        assert courses is not None
        # No FKs created for many-to-many
        assert len(students.foreign_keys) == 0
        assert len(courses.foreign_keys) == 0


class TestReverseRef:
    def test_reverse_ref_less_than(self) -> None:
        """< ref creates FK on the right side."""
        dbml = """
Table users {
  id integer [pk]
}
Table orders {
  id integer [pk]
  user_id integer
}
Ref: users.id < orders.user_id
"""
        schema = parse_dbml(dbml)
        orders = schema.get_table("orders")
        assert orders is not None
        assert len(orders.foreign_keys) == 1
        assert orders.foreign_keys[0].ref_table == "users"


class TestSourceFile:
    def test_source_file_preserved(self) -> None:
        """source_file is set on the schema."""
        schema = parse_dbml(_SAMPLE_DBML, source_file="test.dbml")
        assert schema.source_file == "test.dbml"
