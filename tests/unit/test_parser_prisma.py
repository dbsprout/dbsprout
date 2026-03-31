"""Tests for dbsprout.schema.parsers.prisma — Prisma schema parser."""

from __future__ import annotations

import pytest

from dbsprout.schema.models import ColumnType
from dbsprout.schema.parsers.prisma import can_parse_prisma, parse_prisma

_SAMPLE_PRISMA = """
model User {
  id        Int      @id @default(autoincrement())
  email     String   @unique
  name      String?
  status    Status
  orders    Order[]
  createdAt DateTime @default(now())
}

model Order {
  id     Int   @id @default(autoincrement())
  userId Int
  total  Float
  user   User  @relation(fields: [userId], references: [id])
}

enum Status {
  ACTIVE
  INACTIVE
  PENDING
}
"""


class TestParseModels:
    def test_tables_extracted(self) -> None:
        schema = parse_prisma(_SAMPLE_PRISMA)
        names = schema.table_names()
        assert "user" in names
        assert "order" in names

    def test_columns_extracted(self) -> None:
        schema = parse_prisma(_SAMPLE_PRISMA)
        user = schema.get_table("user")
        assert user is not None
        col_names = [c.name for c in user.columns]
        assert "id" in col_names
        assert "email" in col_names
        assert "name" in col_names
        assert "createdAt" in col_names


class TestScalarTypes:
    def test_int_type(self) -> None:
        schema = parse_prisma(_SAMPLE_PRISMA)
        user = schema.get_table("user")
        assert user is not None
        id_col = user.get_column("id")
        assert id_col is not None
        assert id_col.data_type == ColumnType.INTEGER

    def test_string_type(self) -> None:
        schema = parse_prisma(_SAMPLE_PRISMA)
        user = schema.get_table("user")
        assert user is not None
        email = user.get_column("email")
        assert email is not None
        assert email.data_type == ColumnType.VARCHAR

    def test_float_type(self) -> None:
        schema = parse_prisma(_SAMPLE_PRISMA)
        order = schema.get_table("order")
        assert order is not None
        total = order.get_column("total")
        assert total is not None
        assert total.data_type == ColumnType.FLOAT

    def test_datetime_type(self) -> None:
        schema = parse_prisma(_SAMPLE_PRISMA)
        user = schema.get_table("user")
        assert user is not None
        ts = user.get_column("createdAt")
        assert ts is not None
        assert ts.data_type == ColumnType.TIMESTAMP


class TestAttributes:
    def test_id_attribute(self) -> None:
        """@id → primary_key + autoincrement."""
        schema = parse_prisma(_SAMPLE_PRISMA)
        user = schema.get_table("user")
        assert user is not None
        assert user.primary_key == ["id"]
        id_col = user.get_column("id")
        assert id_col is not None
        assert id_col.primary_key is True
        assert id_col.autoincrement is True

    def test_unique_attribute(self) -> None:
        """@unique → unique=True."""
        schema = parse_prisma(_SAMPLE_PRISMA)
        user = schema.get_table("user")
        assert user is not None
        email = user.get_column("email")
        assert email is not None
        assert email.unique is True


class TestOptional:
    def test_optional_field(self) -> None:
        """Type? → nullable=True."""
        schema = parse_prisma(_SAMPLE_PRISMA)
        user = schema.get_table("user")
        assert user is not None
        name = user.get_column("name")
        assert name is not None
        assert name.nullable is True

    def test_required_field(self) -> None:
        """Type (no ?) → nullable=False."""
        schema = parse_prisma(_SAMPLE_PRISMA)
        user = schema.get_table("user")
        assert user is not None
        email = user.get_column("email")
        assert email is not None
        assert email.nullable is False


class TestRelations:
    def test_relation_to_fk(self) -> None:
        """@relation(fields, references) → ForeignKeySchema."""
        schema = parse_prisma(_SAMPLE_PRISMA)
        order = schema.get_table("order")
        assert order is not None
        assert len(order.foreign_keys) == 1
        fk = order.foreign_keys[0]
        assert fk.columns == ["userId"]
        assert fk.ref_table == "user"
        assert fk.ref_columns == ["id"]

    def test_relation_field_not_column(self) -> None:
        """Relation fields (user User) are NOT columns."""
        schema = parse_prisma(_SAMPLE_PRISMA)
        order = schema.get_table("order")
        assert order is not None
        col_names = [c.name for c in order.columns]
        assert "user" not in col_names

    def test_array_relation_not_column(self) -> None:
        """Array relation (orders Order[]) is NOT a column."""
        schema = parse_prisma(_SAMPLE_PRISMA)
        user = schema.get_table("user")
        assert user is not None
        col_names = [c.name for c in user.columns]
        assert "orders" not in col_names


class TestEnums:
    def test_enum_extraction(self) -> None:
        schema = parse_prisma(_SAMPLE_PRISMA)
        assert "Status" in schema.enums
        assert "ACTIVE" in schema.enums["Status"]
        assert "INACTIVE" in schema.enums["Status"]

    def test_enum_field_type(self) -> None:
        """Field with enum type gets ENUM data_type."""
        schema = parse_prisma(_SAMPLE_PRISMA)
        user = schema.get_table("user")
        assert user is not None
        status = user.get_column("status")
        assert status is not None
        assert status.data_type == ColumnType.ENUM
        assert status.enum_values is not None
        assert "ACTIVE" in status.enum_values


class TestCanParse:
    def test_prisma_extension(self) -> None:
        assert can_parse_prisma("schema.prisma") is True

    def test_non_prisma(self) -> None:
        assert can_parse_prisma("schema.sql") is False

    def test_model_keyword(self) -> None:
        assert can_parse_prisma("model User {") is True


class TestMalformed:
    def test_no_models_raises(self) -> None:
        with pytest.raises(ValueError, match="model"):
            parse_prisma("datasource db { provider = 'postgresql' }")


class TestSourceFile:
    def test_source_file_preserved(self) -> None:
        schema = parse_prisma(_SAMPLE_PRISMA, source_file="schema.prisma")
        assert schema.source_file == "schema.prisma"
