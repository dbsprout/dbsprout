"""Tests for dbsprout.schema.parsers.plantuml — PlantUML ERD parser."""

from __future__ import annotations

import pytest

from dbsprout.schema.models import ColumnType
from dbsprout.schema.parsers.plantuml import can_parse_plantuml, parse_plantuml

_SAMPLE_PUML = """
@startuml
entity "users" as users {
  *id : integer <<PK>>
  --
  *email : varchar
  name : varchar
}
entity "orders" as orders {
  *id : integer <<PK>>
  --
  *user_id : integer <<FK>>
  total : decimal
  created_at : timestamp
}
users ||--o{ orders
@enduml
"""


class TestParseEntities:
    def test_tables_extracted(self) -> None:
        schema = parse_plantuml(_SAMPLE_PUML)
        names = schema.table_names()
        assert "users" in names
        assert "orders" in names

    def test_columns_extracted(self) -> None:
        schema = parse_plantuml(_SAMPLE_PUML)
        users = schema.get_table("users")
        assert users is not None
        col_names = [c.name for c in users.columns]
        assert "id" in col_names
        assert "email" in col_names
        assert "name" in col_names


class TestStereotypes:
    def test_pk_stereotype(self) -> None:
        """<<PK>> → primary_key."""
        schema = parse_plantuml(_SAMPLE_PUML)
        users = schema.get_table("users")
        assert users is not None
        assert users.primary_key == ["id"]
        id_col = users.get_column("id")
        assert id_col is not None
        assert id_col.primary_key is True

    def test_not_null_asterisk(self) -> None:
        """* prefix → nullable=False."""
        schema = parse_plantuml(_SAMPLE_PUML)
        users = schema.get_table("users")
        assert users is not None
        email = users.get_column("email")
        assert email is not None
        assert email.nullable is False

    def test_nullable_without_asterisk(self) -> None:
        """No * → nullable=True."""
        schema = parse_plantuml(_SAMPLE_PUML)
        users = schema.get_table("users")
        assert users is not None
        name = users.get_column("name")
        assert name is not None
        assert name.nullable is True


class TestTypeNormalization:
    def test_integer_type(self) -> None:
        schema = parse_plantuml(_SAMPLE_PUML)
        users = schema.get_table("users")
        assert users is not None
        id_col = users.get_column("id")
        assert id_col is not None
        assert id_col.data_type == ColumnType.INTEGER

    def test_varchar_type(self) -> None:
        schema = parse_plantuml(_SAMPLE_PUML)
        users = schema.get_table("users")
        assert users is not None
        email = users.get_column("email")
        assert email is not None
        assert email.data_type == ColumnType.VARCHAR

    def test_timestamp_type(self) -> None:
        schema = parse_plantuml(_SAMPLE_PUML)
        orders = schema.get_table("orders")
        assert orders is not None
        ts = orders.get_column("created_at")
        assert ts is not None
        assert ts.data_type == ColumnType.TIMESTAMP


class TestForeignKeys:
    def test_fk_from_relationship(self) -> None:
        schema = parse_plantuml(_SAMPLE_PUML)
        orders = schema.get_table("orders")
        assert orders is not None
        assert len(orders.foreign_keys) >= 1
        fk = orders.foreign_keys[0]
        assert fk.columns == ["user_id"]
        assert fk.ref_table == "users"


class TestCanParse:
    def test_puml_extension(self) -> None:
        assert can_parse_plantuml("schema.puml") is True

    def test_plantuml_extension(self) -> None:
        assert can_parse_plantuml("schema.plantuml") is True

    def test_pu_extension(self) -> None:
        assert can_parse_plantuml("schema.pu") is True

    def test_startuml_keyword(self) -> None:
        assert can_parse_plantuml("@startuml\nentity") is True

    def test_non_plantuml(self) -> None:
        assert can_parse_plantuml("schema.sql") is False


class TestNonEntity:
    def test_no_entity_raises(self) -> None:
        with pytest.raises(ValueError, match="entity"):
            parse_plantuml("@startuml\nclass Foo\n@enduml")


class TestReversedRelationship:
    def test_reversed_cardinality(self) -> None:
        """Left-side many creates FK on left."""
        puml = """
@startuml
entity "orders" as orders {
  *id : integer <<PK>>
  *user_id : integer <<FK>>
}
entity "users" as users {
  *id : integer <<PK>>
}
orders }o--|| users
@enduml
"""
        schema = parse_plantuml(puml)
        orders = schema.get_table("orders")
        assert orders is not None
        assert len(orders.foreign_keys) == 1
        assert orders.foreign_keys[0].ref_table == "users"


class TestSourceFile:
    def test_source_file_preserved(self) -> None:
        schema = parse_plantuml(_SAMPLE_PUML, source_file="test.puml")
        assert schema.source_file == "test.puml"
