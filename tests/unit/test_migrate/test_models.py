"""Tests for dbsprout.migrate.models — schema change data models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dbsprout.migrate.models import SchemaChange, SchemaChangeType

# ── SchemaChangeType enum tests (AC-14) ──────────────────────────────────


_EXPECTED_VARIANTS = frozenset(
    {
        "TABLE_ADDED",
        "TABLE_REMOVED",
        "COLUMN_ADDED",
        "COLUMN_REMOVED",
        "COLUMN_TYPE_CHANGED",
        "COLUMN_NULLABILITY_CHANGED",
        "COLUMN_DEFAULT_CHANGED",
        "FOREIGN_KEY_ADDED",
        "FOREIGN_KEY_REMOVED",
        "INDEX_ADDED",
        "INDEX_REMOVED",
        "ENUM_CHANGED",
    }
)


class TestSchemaChangeType:
    def test_has_all_variants(self) -> None:
        actual = {m.name for m in SchemaChangeType}
        expected = _EXPECTED_VARIANTS
        assert actual == expected

    def test_values_are_lowercase_snake_case(self) -> None:
        for member in SchemaChangeType:
            assert member.value == member.name.lower()

    def test_is_str_enum(self) -> None:
        assert isinstance(SchemaChangeType.TABLE_ADDED, str)
        assert SchemaChangeType.TABLE_ADDED == "table_added"


# ── SchemaChange model tests (AC-15) ─────────────────────────────────────


class TestSchemaChange:
    def test_required_fields(self) -> None:
        change = SchemaChange(
            change_type=SchemaChangeType.TABLE_ADDED,
            table_name="users",
        )
        assert change.change_type == SchemaChangeType.TABLE_ADDED
        assert change.table_name == "users"

    def test_optional_fields_default_none(self) -> None:
        change = SchemaChange(
            change_type=SchemaChangeType.TABLE_ADDED,
            table_name="users",
        )
        assert change.column_name is None
        assert change.old_value is None
        assert change.new_value is None
        assert change.detail is None

    def test_all_fields_populated(self) -> None:
        change = SchemaChange(
            change_type=SchemaChangeType.COLUMN_TYPE_CHANGED,
            table_name="users",
            column_name="email",
            old_value="varchar",
            new_value="text",
            detail={"old_type": "varchar", "new_type": "text"},
        )
        assert change.column_name == "email"
        assert change.old_value == "varchar"
        assert change.new_value == "text"
        assert change.detail == {"old_type": "varchar", "new_type": "text"}

    def test_frozen_model(self) -> None:
        change = SchemaChange(
            change_type=SchemaChangeType.TABLE_ADDED,
            table_name="users",
        )
        with pytest.raises(ValidationError):
            change.table_name = "orders"  # type: ignore[misc]

    def test_missing_required_raises(self) -> None:
        with pytest.raises(ValidationError):
            SchemaChange(table_name="users")  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            SchemaChange(change_type=SchemaChangeType.TABLE_ADDED)  # type: ignore[call-arg]
