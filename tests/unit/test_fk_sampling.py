"""Tests for dbsprout.generate.fk_sampling — FK value sampling from parent PKs."""

from __future__ import annotations

from dbsprout.generate.fk_sampling import sample_fk_values
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    ForeignKeySchema,
    TableSchema,
)


def _col(name: str, *, nullable: bool = True, pk: bool = False) -> ColumnSchema:
    return ColumnSchema(
        name=name,
        data_type=ColumnType.INTEGER,
        nullable=nullable,
        primary_key=pk,
    )


class TestSingleColumnFK:
    def test_all_fk_values_exist_in_parent_pks(self) -> None:
        """Every FK value must reference a valid parent PK."""
        child = TableSchema(
            name="employees",
            columns=[
                _col("id", nullable=False, pk=True),
                _col("dept_id"),
            ],
            primary_key=["id"],
            foreign_keys=[
                ForeignKeySchema(
                    columns=["dept_id"],
                    ref_table="departments",
                    ref_columns=["id"],
                )
            ],
        )
        parent_rows = [{"id": 10}, {"id": 20}, {"id": 30}]
        parent_data = {"departments": parent_rows}
        rows = [{"id": i, "dept_id": None} for i in range(1, 51)]

        result = sample_fk_values(child, parent_data, rows, seed=42)

        assert result is rows  # mutates in-place, returns same list
        parent_pks = {r["id"] for r in parent_rows}
        for row in result:
            assert row["dept_id"] in parent_pks


class TestCompositeFK:
    def test_composite_fk_tuples_match_parent_pk_tuples(self) -> None:
        """Multi-column FK must sample complete PK tuples from parent."""
        child = TableSchema(
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
                )
            ],
        )
        parent_rows = [
            {"course_id": 1, "section_id": 100},
            {"course_id": 2, "section_id": 200},
            {"course_id": 3, "section_id": 300},
        ]
        parent_data = {"course_sections": parent_rows}
        rows = [{"id": i, "course_id": None, "section_id": None} for i in range(1, 21)]

        result = sample_fk_values(child, parent_data, rows, seed=42)

        parent_tuples = {(r["course_id"], r["section_id"]) for r in parent_rows}
        for row in result:
            assert (row["course_id"], row["section_id"]) in parent_tuples


class TestSelfReferencingFK:
    def test_first_20_pct_none_rest_reference_earlier_rows(self) -> None:
        """Self-referencing: first 20% get None, rest reference earlier PKs."""
        table = TableSchema(
            name="categories",
            columns=[
                _col("id", nullable=False, pk=True),
                _col("parent_id"),
            ],
            primary_key=["id"],
            foreign_keys=[
                ForeignKeySchema(
                    columns=["parent_id"],
                    ref_table="categories",
                    ref_columns=["id"],
                )
            ],
        )
        rows = [{"id": i, "parent_id": None} for i in range(1, 51)]
        parent_data: dict[str, list[dict[str, object]]] = {}

        result = sample_fk_values(table, parent_data, rows, seed=42)

        null_count = len(rows) // 5  # 20% = 10
        # First 20% must be None
        for row in result[:null_count]:
            assert row["parent_id"] is None

        # Remaining must reference an earlier row's PK
        pk_values = [row["id"] for row in result]
        for i in range(null_count, len(result)):
            fk_val = result[i]["parent_id"]
            assert fk_val is not None
            # Must reference a PK from rows 0..i-1
            assert fk_val in pk_values[:i]


class TestDeterministic:
    def test_same_seed_same_fk_values(self) -> None:
        """Same seed must produce identical FK assignments."""
        table = TableSchema(
            name="orders",
            columns=[
                _col("id", nullable=False, pk=True),
                _col("user_id"),
            ],
            primary_key=["id"],
            foreign_keys=[
                ForeignKeySchema(
                    columns=["user_id"],
                    ref_table="users",
                    ref_columns=["id"],
                )
            ],
        )
        parent_rows = [{"id": i} for i in range(1, 11)]
        parent_data = {"users": parent_rows}

        rows1 = [{"id": i, "user_id": None} for i in range(1, 31)]
        rows2 = [{"id": i, "user_id": None} for i in range(1, 31)]

        sample_fk_values(table, parent_data, rows1, seed=99)
        sample_fk_values(table, parent_data, rows2, seed=99)

        for r1, r2 in zip(rows1, rows2, strict=True):
            assert r1["user_id"] == r2["user_id"]

    def test_different_seed_different_fk_values(self) -> None:
        """Different seeds should produce different FK assignments."""
        table = TableSchema(
            name="orders",
            columns=[
                _col("id", nullable=False, pk=True),
                _col("user_id"),
            ],
            primary_key=["id"],
            foreign_keys=[
                ForeignKeySchema(
                    columns=["user_id"],
                    ref_table="users",
                    ref_columns=["id"],
                )
            ],
        )
        parent_rows = [{"id": i} for i in range(1, 101)]
        parent_data = {"users": parent_rows}

        rows1 = [{"id": i, "user_id": None} for i in range(1, 51)]
        rows2 = [{"id": i, "user_id": None} for i in range(1, 51)]

        sample_fk_values(table, parent_data, rows1, seed=42)
        sample_fk_values(table, parent_data, rows2, seed=99)

        # At least some values should differ
        diffs = sum(
            1 for r1, r2 in zip(rows1, rows2, strict=True) if r1["user_id"] != r2["user_id"]
        )
        assert diffs > 0


class TestEmptyParent:
    def test_empty_parent_returns_none(self) -> None:
        """If parent has 0 rows, all FK values must be None."""
        table = TableSchema(
            name="orders",
            columns=[
                _col("id", nullable=False, pk=True),
                _col("user_id"),
            ],
            primary_key=["id"],
            foreign_keys=[
                ForeignKeySchema(
                    columns=["user_id"],
                    ref_table="users",
                    ref_columns=["id"],
                )
            ],
        )
        parent_data: dict[str, list[dict[str, object]]] = {"users": []}
        rows = [{"id": i, "user_id": None} for i in range(1, 11)]

        result = sample_fk_values(table, parent_data, rows, seed=42)

        for row in result:
            assert row["user_id"] is None


class TestMultipleFKs:
    def test_table_with_two_fks_to_different_parents(self) -> None:
        """A table with FKs to two different parents — both must be valid."""
        table = TableSchema(
            name="orders",
            columns=[
                _col("id", nullable=False, pk=True),
                _col("user_id"),
                _col("product_id"),
            ],
            primary_key=["id"],
            foreign_keys=[
                ForeignKeySchema(
                    columns=["user_id"],
                    ref_table="users",
                    ref_columns=["id"],
                ),
                ForeignKeySchema(
                    columns=["product_id"],
                    ref_table="products",
                    ref_columns=["id"],
                ),
            ],
        )
        users = [{"id": i} for i in range(1, 6)]
        products = [{"id": i * 10} for i in range(1, 4)]
        parent_data = {"users": users, "products": products}
        rows = [{"id": i, "user_id": None, "product_id": None} for i in range(1, 21)]

        result = sample_fk_values(table, parent_data, rows, seed=42)

        user_pks = {r["id"] for r in users}
        product_pks = {r["id"] for r in products}
        for row in result:
            assert row["user_id"] in user_pks
            assert row["product_id"] in product_pks


class TestDeferredFKSkipped:
    def test_fk_to_missing_parent_stays_none(self) -> None:
        """Deferred FK (parent not in parent_data) → FK columns stay None."""
        table = TableSchema(
            name="orders",
            columns=[
                _col("id", nullable=False, pk=True),
                _col("user_id"),
            ],
            primary_key=["id"],
            foreign_keys=[
                ForeignKeySchema(
                    columns=["user_id"],
                    ref_table="users",
                    ref_columns=["id"],
                )
            ],
        )
        parent_data: dict[str, list[dict[str, object]]] = {}  # no "users" key
        rows = [{"id": i, "user_id": None} for i in range(1, 11)]

        result = sample_fk_values(table, parent_data, rows, seed=42)

        for row in result:
            assert row["user_id"] is None


class TestEdgeCases:
    def test_self_ref_single_row_gets_none(self) -> None:
        """Single row in self-referencing table must get None."""
        table = TableSchema(
            name="categories",
            columns=[
                _col("id", nullable=False, pk=True),
                _col("parent_id"),
            ],
            primary_key=["id"],
            foreign_keys=[
                ForeignKeySchema(
                    columns=["parent_id"],
                    ref_table="categories",
                    ref_columns=["id"],
                )
            ],
        )
        rows: list[dict[str, object]] = [{"id": 1, "parent_id": None}]

        result = sample_fk_values(table, {}, rows, seed=42)

        assert result[0]["parent_id"] is None

    def test_empty_rows_returns_empty(self) -> None:
        """Empty rows list should return empty list without errors."""
        table = TableSchema(
            name="orders",
            columns=[
                _col("id", nullable=False, pk=True),
                _col("user_id"),
            ],
            primary_key=["id"],
            foreign_keys=[
                ForeignKeySchema(
                    columns=["user_id"],
                    ref_table="users",
                    ref_columns=["id"],
                )
            ],
        )
        parent_data = {"users": [{"id": 1}]}
        rows: list[dict[str, object]] = []

        result = sample_fk_values(table, parent_data, rows, seed=42)

        assert result == []
