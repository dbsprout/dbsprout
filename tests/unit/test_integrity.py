"""Tests for dbsprout.quality.integrity — integrity validation."""

from __future__ import annotations

from typing import Any

from dbsprout.quality.integrity import validate_integrity
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
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


def _parent_child_schema() -> DatabaseSchema:
    return DatabaseSchema(
        tables=[
            TableSchema(
                name="users",
                columns=[
                    _col("id", nullable=False, pk=True, autoincrement=True),
                    _col("email", nullable=False, unique=True, data_type=ColumnType.VARCHAR),
                ],
                primary_key=["id"],
            ),
            TableSchema(
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
            ),
        ],
    )


# ── FK satisfaction tests ───────────────────────────────────────────


class TestFKSatisfaction:
    def test_fk_passes(self) -> None:
        """All FK values exist in parent → passes."""
        schema = _parent_child_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": 1, "email": "a@b.com"}, {"id": 2, "email": "c@d.com"}],
            "orders": [{"id": 1, "user_id": 1}, {"id": 2, "user_id": 2}],
        }

        report = validate_integrity(data, schema)

        fk_checks = [c for c in report.checks if c.check == "fk_satisfaction"]
        assert len(fk_checks) == 1
        assert fk_checks[0].passed

    def test_fk_fails(self) -> None:
        """FK value references nonexistent parent PK → fails."""
        schema = _parent_child_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": 1, "email": "a@b.com"}],
            "orders": [{"id": 1, "user_id": 999}],  # 999 not in users
        }

        report = validate_integrity(data, schema)

        fk_checks = [c for c in report.checks if c.check == "fk_satisfaction"]
        assert not fk_checks[0].passed
        assert not report.passed

    def test_fk_nullable_none_skipped(self) -> None:
        """None FK values are not violations (nullable FK)."""
        schema = DatabaseSchema(
            tables=[
                TableSchema(
                    name="users",
                    columns=[_col("id", nullable=False, pk=True)],
                    primary_key=["id"],
                ),
                TableSchema(
                    name="orders",
                    columns=[
                        _col("id", nullable=False, pk=True),
                        _col("user_id", nullable=True),  # nullable FK
                    ],
                    primary_key=["id"],
                    foreign_keys=[
                        ForeignKeySchema(
                            columns=["user_id"],
                            ref_table="users",
                            ref_columns=["id"],
                        ),
                    ],
                ),
            ],
        )
        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": 1}],
            "orders": [{"id": 1, "user_id": None}],  # None is OK
        }

        report = validate_integrity(data, schema)

        fk_checks = [c for c in report.checks if c.check == "fk_satisfaction"]
        assert fk_checks[0].passed


# ── PK uniqueness tests ────────────────────────────────────────────


class TestPKUniqueness:
    def test_pk_passes(self) -> None:
        """All PKs unique → passes."""
        schema = _parent_child_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": 1, "email": "a"}, {"id": 2, "email": "b"}],
            "orders": [{"id": 1, "user_id": 1}, {"id": 2, "user_id": 1}],
        }

        report = validate_integrity(data, schema)

        pk_checks = [c for c in report.checks if c.check == "pk_uniqueness"]
        assert all(c.passed for c in pk_checks)

    def test_pk_fails(self) -> None:
        """Duplicate PK → fails."""
        schema = _parent_child_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": 1, "email": "a"}, {"id": 1, "email": "b"}],  # dup PK
            "orders": [],
        }

        report = validate_integrity(data, schema)

        pk_checks = [c for c in report.checks if c.check == "pk_uniqueness"]
        users_pk = [c for c in pk_checks if c.table == "users"]
        assert not users_pk[0].passed


# ── UNIQUE constraint tests ─────────────────────────────────────────


class TestUnique:
    def test_unique_passes(self) -> None:
        """No duplicates in UNIQUE column → passes."""
        schema = _parent_child_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": 1, "email": "a@b.com"}, {"id": 2, "email": "c@d.com"}],
            "orders": [],
        }

        report = validate_integrity(data, schema)

        unique_checks = [c for c in report.checks if c.check == "unique"]
        assert all(c.passed for c in unique_checks)

    def test_unique_fails(self) -> None:
        """Duplicate in UNIQUE column → fails."""
        schema = _parent_child_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": 1, "email": "same"}, {"id": 2, "email": "same"}],
            "orders": [],
        }

        report = validate_integrity(data, schema)

        unique_checks = [c for c in report.checks if c.check == "unique"]
        email_check = [c for c in unique_checks if c.column == "email"]
        assert not email_check[0].passed

    def test_unique_null_excluded(self) -> None:
        """None values excluded from uniqueness check (SQL semantics)."""
        schema = DatabaseSchema(
            tables=[
                TableSchema(
                    name="items",
                    columns=[
                        _col("id", nullable=False, pk=True),
                        _col("code", nullable=True, unique=True, data_type=ColumnType.VARCHAR),
                    ],
                    primary_key=["id"],
                ),
            ],
        )
        data: dict[str, list[dict[str, Any]]] = {
            "items": [
                {"id": 1, "code": None},
                {"id": 2, "code": None},  # Two NULLs OK per SQL standard
                {"id": 3, "code": "ABC"},
            ],
        }

        report = validate_integrity(data, schema)

        unique_checks = [c for c in report.checks if c.check == "unique"]
        assert all(c.passed for c in unique_checks)


# ── NOT NULL tests ──────────────────────────────────────────────────


class TestNotNull:
    def test_not_null_passes(self) -> None:
        """No None in non-nullable columns → passes."""
        schema = _parent_child_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": 1, "email": "a@b.com"}],
            "orders": [{"id": 1, "user_id": 1}],
        }

        report = validate_integrity(data, schema)

        nn_checks = [c for c in report.checks if c.check == "not_null"]
        assert all(c.passed for c in nn_checks)

    def test_not_null_fails(self) -> None:
        """None in non-nullable column → fails."""
        schema = _parent_child_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": 1, "email": None}],  # email is NOT NULL
            "orders": [],
        }

        report = validate_integrity(data, schema)

        nn_checks = [c for c in report.checks if c.check == "not_null"]
        email_nn = [c for c in nn_checks if c.column == "email"]
        assert not email_nn[0].passed


# ── Overall report tests ────────────────────────────────────────────


class TestOverall:
    def test_all_pass(self) -> None:
        """All checks pass → report.passed = True."""
        schema = _parent_child_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": 1, "email": "a@b.com"}, {"id": 2, "email": "c@d.com"}],
            "orders": [{"id": 1, "user_id": 1}],
        }

        report = validate_integrity(data, schema)

        assert report.passed
        assert all(c.passed for c in report.checks)

    def test_any_fail(self) -> None:
        """Any check fails → report.passed = False."""
        schema = _parent_child_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": 1, "email": "a"}, {"id": 1, "email": "b"}],  # dup PK
            "orders": [],
        }

        report = validate_integrity(data, schema)

        assert not report.passed


class TestCompositeFKSatisfaction:
    def test_composite_fk_passes(self) -> None:
        """Composite FK tuples all exist in parent → passes."""
        schema = DatabaseSchema(
            tables=[
                TableSchema(
                    name="course_sections",
                    columns=[
                        _col("course_id", nullable=False, pk=True),
                        _col("section_id", nullable=False, pk=True),
                    ],
                    primary_key=["course_id", "section_id"],
                ),
                TableSchema(
                    name="enrollments",
                    columns=[
                        _col("id", nullable=False, pk=True),
                        _col("course_id"),
                        _col("section_id"),
                    ],
                    primary_key=["id"],
                    foreign_keys=[
                        ForeignKeySchema(
                            columns=["course_id", "section_id"],
                            ref_table="course_sections",
                            ref_columns=["course_id", "section_id"],
                        ),
                    ],
                ),
            ],
        )
        data: dict[str, list[dict[str, Any]]] = {
            "course_sections": [
                {"course_id": 1, "section_id": 100},
                {"course_id": 2, "section_id": 200},
            ],
            "enrollments": [
                {"id": 1, "course_id": 1, "section_id": 100},
                {"id": 2, "course_id": 2, "section_id": 200},
            ],
        }

        report = validate_integrity(data, schema)

        fk_checks = [c for c in report.checks if c.check == "fk_satisfaction"]
        assert fk_checks[0].passed

    def test_composite_fk_fails(self) -> None:
        """Composite FK tuple not in parent → fails."""
        schema = DatabaseSchema(
            tables=[
                TableSchema(
                    name="course_sections",
                    columns=[
                        _col("course_id", nullable=False, pk=True),
                        _col("section_id", nullable=False, pk=True),
                    ],
                    primary_key=["course_id", "section_id"],
                ),
                TableSchema(
                    name="enrollments",
                    columns=[
                        _col("id", nullable=False, pk=True),
                        _col("course_id"),
                        _col("section_id"),
                    ],
                    primary_key=["id"],
                    foreign_keys=[
                        ForeignKeySchema(
                            columns=["course_id", "section_id"],
                            ref_table="course_sections",
                            ref_columns=["course_id", "section_id"],
                        ),
                    ],
                ),
            ],
        )
        data: dict[str, list[dict[str, Any]]] = {
            "course_sections": [{"course_id": 1, "section_id": 100}],
            "enrollments": [{"id": 1, "course_id": 9, "section_id": 999}],
        }

        report = validate_integrity(data, schema)

        fk_checks = [c for c in report.checks if c.check == "fk_satisfaction"]
        assert not fk_checks[0].passed

    def test_composite_fk_partial_null_skipped(self) -> None:
        """Composite FK with any NULL column → skipped."""
        schema = DatabaseSchema(
            tables=[
                TableSchema(
                    name="parent",
                    columns=[
                        _col("a", nullable=False, pk=True),
                        _col("b", nullable=False, pk=True),
                    ],
                    primary_key=["a", "b"],
                ),
                TableSchema(
                    name="child",
                    columns=[
                        _col("id", nullable=False, pk=True),
                        _col("a"),
                        _col("b"),
                    ],
                    primary_key=["id"],
                    foreign_keys=[
                        ForeignKeySchema(
                            columns=["a", "b"],
                            ref_table="parent",
                            ref_columns=["a", "b"],
                        ),
                    ],
                ),
            ],
        )
        data: dict[str, list[dict[str, Any]]] = {
            "parent": [{"a": 1, "b": 2}],
            "child": [{"id": 1, "a": None, "b": 2}],  # partial null
        }

        report = validate_integrity(data, schema)

        fk_checks = [c for c in report.checks if c.check == "fk_satisfaction"]
        assert fk_checks[0].passed


class TestCompositePKUniqueness:
    def test_composite_pk_unique(self) -> None:
        """Composite PK with unique tuples → passes."""
        schema = DatabaseSchema(
            tables=[
                TableSchema(
                    name="t",
                    columns=[
                        _col("a", nullable=False, pk=True),
                        _col("b", nullable=False, pk=True),
                    ],
                    primary_key=["a", "b"],
                ),
            ],
        )
        data: dict[str, list[dict[str, Any]]] = {
            "t": [{"a": 1, "b": 1}, {"a": 1, "b": 2}],
        }

        report = validate_integrity(data, schema)

        pk_checks = [c for c in report.checks if c.check == "pk_uniqueness"]
        assert pk_checks[0].passed


class TestCompositeUniqueIndex:
    def test_composite_unique_index(self) -> None:
        """Composite UNIQUE index with duplicate tuples → fails."""
        from dbsprout.schema.models import IndexSchema  # noqa: PLC0415

        schema = DatabaseSchema(
            tables=[
                TableSchema(
                    name="enrollments",
                    columns=[
                        _col("id", nullable=False, pk=True),
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
                ),
            ],
        )
        data: dict[str, list[dict[str, Any]]] = {
            "enrollments": [
                {"id": 1, "student_id": 1, "course_id": 101},
                {"id": 2, "student_id": 1, "course_id": 101},  # dup tuple
            ],
        }

        report = validate_integrity(data, schema)

        unique_checks = [
            c for c in report.checks if c.check == "unique" and "student_id" in c.column
        ]
        assert not unique_checks[0].passed


class TestNonNullableFKWithNull:
    def test_non_nullable_fk_with_null_fails(self) -> None:
        """Non-nullable FK column with None must fail NOT NULL check."""
        schema = _parent_child_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": 1, "email": "a@b.com"}],
            "orders": [{"id": 1, "user_id": None}],  # non-nullable FK with None
        }

        report = validate_integrity(data, schema)

        nn_checks = [c for c in report.checks if c.check == "not_null" and c.column == "user_id"]
        assert len(nn_checks) == 1
        assert not nn_checks[0].passed


class TestEdgeCases:
    def test_parent_missing_from_data(self) -> None:
        """FK check skips when parent table not in tables_data."""
        schema = _parent_child_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "orders": [{"id": 1, "user_id": 1}],
            # "users" key missing entirely
        }

        report = validate_integrity(data, schema)

        # FK check should not appear (parent missing)
        fk_checks = [c for c in report.checks if c.check == "fk_satisfaction"]
        assert len(fk_checks) == 0

    def test_table_with_no_pk(self) -> None:
        """Table with no PK produces no PK uniqueness check."""
        schema = DatabaseSchema(
            tables=[
                TableSchema(
                    name="logs",
                    columns=[
                        _col("message", data_type=ColumnType.VARCHAR),
                    ],
                    primary_key=[],
                ),
            ],
        )
        data: dict[str, list[dict[str, Any]]] = {
            "logs": [{"message": "hello"}, {"message": "hello"}],
        }

        report = validate_integrity(data, schema)

        pk_checks = [c for c in report.checks if c.check == "pk_uniqueness"]
        assert len(pk_checks) == 0


class TestEmptySchema:
    def test_empty_schema_passes(self) -> None:
        """No tables → passes with empty checks."""
        schema = DatabaseSchema(tables=[])
        data: dict[str, list[dict[str, Any]]] = {}

        report = validate_integrity(data, schema)

        assert report.passed
        assert report.checks == []
