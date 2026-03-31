"""Tests for dbsprout.schema.parsers.mermaid — Mermaid ERD parser."""

from __future__ import annotations

import pytest

from dbsprout.schema.models import ColumnType
from dbsprout.schema.parsers.mermaid import can_parse_mermaid, parse_mermaid

_SAMPLE_MERMAID = """
erDiagram
    USERS {
        int id PK
        string email
        string name
    }
    ORDERS {
        int id PK
        int user_id FK
        decimal total
        timestamp created_at
    }
    USERS ||--o{ ORDERS : "has"
"""


class TestParseEntities:
    def test_tables_extracted(self) -> None:
        """Entity blocks produce tables."""
        schema = parse_mermaid(_SAMPLE_MERMAID)
        names = schema.table_names()
        assert "users" in names
        assert "orders" in names

    def test_columns_extracted(self) -> None:
        """Columns extracted from entity blocks."""
        schema = parse_mermaid(_SAMPLE_MERMAID)
        users = schema.get_table("users")
        assert users is not None
        col_names = [c.name for c in users.columns]
        assert "id" in col_names
        assert "email" in col_names
        assert "name" in col_names


class TestTypeNormalization:
    def test_int_type(self) -> None:
        """int → INTEGER."""
        schema = parse_mermaid(_SAMPLE_MERMAID)
        users = schema.get_table("users")
        assert users is not None
        id_col = users.get_column("id")
        assert id_col is not None
        assert id_col.data_type == ColumnType.INTEGER

    def test_string_type(self) -> None:
        """string → VARCHAR."""
        schema = parse_mermaid(_SAMPLE_MERMAID)
        users = schema.get_table("users")
        assert users is not None
        email = users.get_column("email")
        assert email is not None
        assert email.data_type == ColumnType.VARCHAR

    def test_decimal_type(self) -> None:
        """decimal → DECIMAL."""
        schema = parse_mermaid(_SAMPLE_MERMAID)
        orders = schema.get_table("orders")
        assert orders is not None
        total = orders.get_column("total")
        assert total is not None
        assert total.data_type == ColumnType.DECIMAL

    def test_timestamp_type(self) -> None:
        """timestamp → TIMESTAMP."""
        schema = parse_mermaid(_SAMPLE_MERMAID)
        orders = schema.get_table("orders")
        assert orders is not None
        ts = orders.get_column("created_at")
        assert ts is not None
        assert ts.data_type == ColumnType.TIMESTAMP


class TestPrimaryKey:
    def test_pk_marker(self) -> None:
        """PK marker → primary_key."""
        schema = parse_mermaid(_SAMPLE_MERMAID)
        users = schema.get_table("users")
        assert users is not None
        assert users.primary_key == ["id"]
        id_col = users.get_column("id")
        assert id_col is not None
        assert id_col.primary_key is True


class TestForeignKeys:
    def test_fk_from_relationship(self) -> None:
        """Relationship line + FK marker → ForeignKeySchema."""
        schema = parse_mermaid(_SAMPLE_MERMAID)
        orders = schema.get_table("orders")
        assert orders is not None
        assert len(orders.foreign_keys) >= 1
        fk = orders.foreign_keys[0]
        assert fk.columns == ["user_id"]
        assert fk.ref_table == "users"


class TestCanParse:
    def test_mermaid_extension(self) -> None:
        """.mermaid extension detected."""
        assert can_parse_mermaid("diagram.mermaid") is True

    def test_mmd_extension(self) -> None:
        """.mmd extension detected."""
        assert can_parse_mermaid("schema.mmd") is True

    def test_erdiagram_keyword(self) -> None:
        """erDiagram keyword in content detected."""
        assert can_parse_mermaid("erDiagram\n  USERS {") is True

    def test_non_mermaid(self) -> None:
        """Non-mermaid content rejected."""
        assert can_parse_mermaid("schema.sql") is False


class TestNonERDiagram:
    def test_non_erd_raises_error(self) -> None:
        """Non-ERD mermaid content raises ValueError."""
        with pytest.raises(ValueError, match="erDiagram"):
            parse_mermaid("graph TD\n    A --> B")


class TestSourceFile:
    def test_source_file_preserved(self) -> None:
        """source_file set on schema."""
        schema = parse_mermaid(_SAMPLE_MERMAID, source_file="test.mermaid")
        assert schema.source_file == "test.mermaid"


class TestOneToOneRelationship:
    def test_one_to_one_creates_fk(self) -> None:
        """||--|| creates FK on right side by convention."""
        mermaid = """
erDiagram
    USERS {
        int id PK
    }
    PROFILES {
        int id PK
        int user_id FK
    }
    USERS ||--|| PROFILES : "has"
"""
        schema = parse_mermaid(mermaid)
        profiles = schema.get_table("profiles")
        assert profiles is not None
        assert len(profiles.foreign_keys) == 1
        assert profiles.foreign_keys[0].ref_table == "users"


class TestNoFKColumn:
    def test_missing_fk_column_skipped(self) -> None:
        """Relationship with no matching FK column is skipped."""
        mermaid = """
erDiagram
    USERS {
        int id PK
    }
    ORDERS {
        int id PK
        decimal total
    }
    USERS ||--o{ ORDERS : "has"
"""
        schema = parse_mermaid(mermaid)
        orders = schema.get_table("orders")
        assert orders is not None
        # No user_id column → no FK created
        assert len(orders.foreign_keys) == 0


class TestMarkdownExtraction:
    def test_erdiagram_from_markdown_fence(self) -> None:
        """Extract erDiagram from markdown code fence."""
        md = """
# Schema

```mermaid
erDiagram
    PRODUCTS {
        int id PK
        string name
    }
```
"""
        schema = parse_mermaid(md)
        assert schema.get_table("products") is not None
