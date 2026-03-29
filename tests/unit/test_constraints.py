"""Tests for dbsprout.generate.constraints — UNIQUE + NOT NULL enforcement."""

from __future__ import annotations

import pytest

from dbsprout.generate.constraints import ConstraintError, enforce_constraints
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    ForeignKeySchema,
    IndexSchema,
    TableSchema,
)


def _col(  # noqa: PLR0913
    name: str,
    *,
    nullable: bool = True,
    pk: bool = False,
    unique: bool = False,
    autoincrement: bool = False,
    data_type: ColumnType = ColumnType.INTEGER,
) -> ColumnSchema:
    return ColumnSchema(
        name=name,
        data_type=data_type,
        nullable=nullable,
        primary_key=pk,
        unique=unique,
        autoincrement=autoincrement,
    )


class TestSingleColumnUnique:
    def test_no_duplicates_after_enforcement(self) -> None:
        """UNIQUE column must have no duplicate values."""
        table = TableSchema(
            name="users",
            columns=[
                _col("id", nullable=False, pk=True, autoincrement=True),
                _col("email", nullable=False, unique=True, data_type=ColumnType.VARCHAR),
            ],
            primary_key=["id"],
        )
        # Deliberately create rows with duplicate values in email
        rows = [
            {"id": None, "email": "alice@example.com"},
            {"id": None, "email": "alice@example.com"},  # duplicate
            {"id": None, "email": "bob@example.com"},
            {"id": None, "email": "bob@example.com"},  # duplicate
            {"id": None, "email": "carol@example.com"},
        ]

        result = enforce_constraints(table, rows, seed=42)

        emails = [r["email"] for r in result]
        assert len(emails) == len(set(emails)), f"Duplicates found: {emails}"


class TestCompositeUnique:
    def test_composite_unique_index_enforced(self) -> None:
        """Composite UNIQUE index must have no duplicate tuples."""
        table = TableSchema(
            name="enrollments",
            columns=[
                _col("id", nullable=False, pk=True, autoincrement=True),
                _col("student_id"),
                _col("course_id"),
            ],
            primary_key=["id"],
            indexes=[
                IndexSchema(
                    name="uq_enrollment",
                    columns=["student_id", "course_id"],
                    unique=True,
                ),
            ],
        )
        rows = [
            {"id": None, "student_id": 1, "course_id": 101},
            {"id": None, "student_id": 1, "course_id": 101},  # duplicate tuple
            {"id": None, "student_id": 2, "course_id": 102},
            {"id": None, "student_id": 2, "course_id": 102},  # duplicate tuple
        ]

        result = enforce_constraints(table, rows, seed=42)

        tuples = [(r["student_id"], r["course_id"]) for r in result]
        assert len(tuples) == len(set(tuples)), f"Duplicate tuples found: {tuples}"


class TestNotNull:
    def test_no_none_values_in_non_nullable_columns(self) -> None:
        """Non-nullable columns must have no None values after enforcement."""
        table = TableSchema(
            name="products",
            columns=[
                _col("id", nullable=False, pk=True, autoincrement=True),
                _col("name", nullable=False, data_type=ColumnType.VARCHAR),
                _col("price", nullable=False, data_type=ColumnType.FLOAT),
                _col("description", nullable=True, data_type=ColumnType.TEXT),
            ],
            primary_key=["id"],
        )
        rows = [
            {"id": None, "name": None, "price": None, "description": None},
            {"id": None, "name": "Widget", "price": 9.99, "description": "A widget"},
            {"id": None, "name": None, "price": 5.00, "description": None},
        ]

        result = enforce_constraints(table, rows, seed=42)

        for row in result:
            assert row["name"] is not None
            assert row["price"] is not None
            # description is nullable, so None is allowed
        # Nullable column can remain None
        assert any(r["description"] is None for r in result)


class TestAutoincrementPK:
    def test_autoincrement_pk_sequential(self) -> None:
        """Auto-increment PK must be assigned sequential 1, 2, 3, ..."""
        table = TableSchema(
            name="users",
            columns=[
                _col("id", nullable=False, pk=True, autoincrement=True),
                _col("name", data_type=ColumnType.VARCHAR),
            ],
            primary_key=["id"],
        )
        rows = [
            {"id": None, "name": "Alice"},
            {"id": None, "name": "Bob"},
            {"id": None, "name": "Carol"},
        ]

        result = enforce_constraints(table, rows, seed=42)

        assert [r["id"] for r in result] == [1, 2, 3]


class TestConstraintError:
    def test_impossible_unique_raises_error(self) -> None:
        """Boolean UNIQUE with >2 rows must raise ConstraintError."""
        table = TableSchema(
            name="flags",
            columns=[
                _col("id", nullable=False, pk=True, autoincrement=True),
                _col("active", nullable=False, unique=True, data_type=ColumnType.BOOLEAN),
            ],
            primary_key=["id"],
        )
        rows = [
            {"id": None, "active": True},
            {"id": None, "active": False},
            {"id": None, "active": True},  # 3rd row — impossible for boolean UNIQUE
        ]

        with pytest.raises(ConstraintError, match="active"):
            enforce_constraints(table, rows, seed=42)


class TestFKColumnsSkipped:
    def test_fk_columns_not_touched(self) -> None:
        """FK columns must not be regenerated by constraint enforcement."""
        table = TableSchema(
            name="orders",
            columns=[
                _col("id", nullable=False, pk=True, autoincrement=True),
                _col("user_id", nullable=False),
            ],
            primary_key=["id"],
            foreign_keys=[
                ForeignKeySchema(
                    columns=["user_id"],
                    ref_table="users",
                    ref_columns=["id"],
                ),
            ],
        )
        rows = [
            {"id": None, "user_id": 10},
            {"id": None, "user_id": 20},
            {"id": None, "user_id": 10},  # same FK value is fine (not UNIQUE)
        ]

        result = enforce_constraints(table, rows, seed=42)

        # FK values must be preserved exactly
        assert [r["user_id"] for r in result] == [10, 20, 10]


class TestDeterministic:
    def test_same_seed_same_results(self) -> None:
        """Same seed must produce identical constraint resolution."""
        table = TableSchema(
            name="items",
            columns=[
                _col("id", nullable=False, pk=True, autoincrement=True),
                _col("code", nullable=False, unique=True, data_type=ColumnType.VARCHAR),
            ],
            primary_key=["id"],
        )
        rows = [
            {"id": None, "code": "AAA"},
            {"id": None, "code": "AAA"},  # duplicate
            {"id": None, "code": "BBB"},
        ]

        result1 = enforce_constraints(table, [dict(r) for r in rows], seed=77)
        result2 = enforce_constraints(table, [dict(r) for r in rows], seed=77)

        for r1, r2 in zip(result1, result2, strict=True):
            assert r1 == r2


class TestReturnsNewList:
    def test_input_not_mutated(self) -> None:
        """enforce_constraints must return a new list without mutating input."""
        table = TableSchema(
            name="items",
            columns=[
                _col("id", nullable=False, pk=True, autoincrement=True),
                _col("value"),
            ],
            primary_key=["id"],
        )
        original_rows = [
            {"id": None, "value": 42},
            {"id": None, "value": 99},
        ]
        # Save copies to verify no mutation
        saved = [dict(r) for r in original_rows]

        result = enforce_constraints(table, original_rows, seed=42)

        # Result is a new list
        assert result is not original_rows
        # Original rows are unchanged
        for orig, saved_row in zip(original_rows, saved, strict=True):
            assert orig == saved_row


class TestCompositePKUnique:
    def test_composite_pk_deduplication(self) -> None:
        """Composite PK (multi-column) is implicitly UNIQUE and must be deduped."""
        table = TableSchema(
            name="course_sections",
            columns=[
                _col("course_id", nullable=False, pk=True),
                _col("section_id", nullable=False, pk=True),
            ],
            primary_key=["course_id", "section_id"],
        )
        rows = [
            {"course_id": 1, "section_id": 100},
            {"course_id": 1, "section_id": 100},  # duplicate PK tuple
            {"course_id": 2, "section_id": 200},
        ]

        result = enforce_constraints(table, rows, seed=42)

        tuples = [(r["course_id"], r["section_id"]) for r in result]
        assert len(tuples) == len(set(tuples))


class TestCompositeConstraintError:
    def test_impossible_composite_unique_raises(self) -> None:
        """Composite UNIQUE that cannot be satisfied must raise ConstraintError."""
        table = TableSchema(
            name="flags",
            columns=[
                _col("id", nullable=False, pk=True, autoincrement=True),
                _col("a", nullable=False, data_type=ColumnType.BOOLEAN),
                _col("b", nullable=False, data_type=ColumnType.BOOLEAN),
            ],
            primary_key=["id"],
            indexes=[
                IndexSchema(
                    name="uq_flags",
                    columns=["a", "b"],
                    unique=True,
                ),
            ],
        )
        # 5 rows but only 4 possible (True,True), (True,False), (False,True), (False,False)
        rows = [
            {"id": None, "a": True, "b": True},
            {"id": None, "a": True, "b": False},
            {"id": None, "a": False, "b": True},
            {"id": None, "a": False, "b": False},
            {"id": None, "a": True, "b": True},  # 5th — impossible
        ]

        with pytest.raises(ConstraintError, match="UNIQUE"):
            enforce_constraints(table, rows, seed=42)


class TestNonUniqueIndexSkipped:
    def test_non_unique_index_not_enforced(self) -> None:
        """Non-unique indexes should not trigger deduplication."""
        table = TableSchema(
            name="orders",
            columns=[
                _col("id", nullable=False, pk=True, autoincrement=True),
                _col("status"),
            ],
            primary_key=["id"],
            indexes=[
                IndexSchema(
                    name="idx_status",
                    columns=["status"],
                    unique=False,
                ),
            ],
        )
        rows = [
            {"id": None, "status": 1},
            {"id": None, "status": 1},  # duplicate is fine for non-unique index
        ]

        result = enforce_constraints(table, rows, seed=42)

        assert result[0]["status"] == 1
        assert result[1]["status"] == 1


class TestDecimalAndFallbackRegen:
    def test_not_null_decimal_regenerated(self) -> None:
        """DECIMAL NOT NULL column with None must get a regenerated value."""
        table = TableSchema(
            name="prices",
            columns=[
                _col("id", nullable=False, pk=True, autoincrement=True),
                ColumnSchema(
                    name="amount",
                    data_type=ColumnType.DECIMAL,
                    nullable=False,
                    precision=10,
                    scale=2,
                ),
            ],
            primary_key=["id"],
        )
        rows = [{"id": None, "amount": None}]

        result = enforce_constraints(table, rows, seed=42)

        assert result[0]["amount"] is not None
        assert isinstance(result[0]["amount"], float)

    def test_not_null_unknown_type_gets_fallback(self) -> None:
        """Unknown column types should get an integer fallback for NOT NULL."""
        table = TableSchema(
            name="misc",
            columns=[
                _col("id", nullable=False, pk=True, autoincrement=True),
                _col("data", nullable=False, data_type=ColumnType.JSON),
            ],
            primary_key=["id"],
        )
        rows = [{"id": None, "data": None}]

        result = enforce_constraints(table, rows, seed=42)

        assert result[0]["data"] is not None

    def test_not_null_enum_uses_enum_values(self) -> None:
        """ENUM NOT NULL column should regenerate from enum_values."""
        table = TableSchema(
            name="tasks",
            columns=[
                _col("id", nullable=False, pk=True, autoincrement=True),
                ColumnSchema(
                    name="status",
                    data_type=ColumnType.ENUM,
                    nullable=False,
                    enum_values=["active", "inactive", "pending"],
                ),
            ],
            primary_key=["id"],
        )
        rows = [{"id": None, "status": None}]

        result = enforce_constraints(table, rows, seed=42)

        assert result[0]["status"] in {"active", "inactive", "pending"}

    def test_not_null_uuid_gets_uuid_string(self) -> None:
        """UUID NOT NULL column should regenerate as a UUID string."""
        table = TableSchema(
            name="tokens",
            columns=[
                _col("id", nullable=False, pk=True, autoincrement=True),
                _col("token", nullable=False, data_type=ColumnType.UUID),
            ],
            primary_key=["id"],
        )
        rows = [{"id": None, "token": None}]

        result = enforce_constraints(table, rows, seed=42)

        token = result[0]["token"]
        assert token is not None
        assert isinstance(token, str)
        assert len(token) == 36  # UUID format: 8-4-4-4-12
