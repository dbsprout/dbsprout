"""Tests for dbsprout.schema.graph — FK dependency graph + topological sort."""

from __future__ import annotations

import time
from graphlib import CycleError

import pytest
from pydantic import ValidationError

from dbsprout.schema.graph import FKGraph
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    TableSchema,
)


def _col(name: str) -> ColumnSchema:
    """Minimal column for test schemas."""
    return ColumnSchema(name=name, data_type=ColumnType.INTEGER)


def _table(
    name: str,
    fks: list[tuple[str, str]] | None = None,
    self_ref: str | None = None,
    fk_name: str | None = None,
) -> TableSchema:
    """Build a table with optional FKs.

    fks: list of (column, ref_table) pairs for foreign keys
    self_ref: column name for self-referencing FK
    fk_name: explicit FK name (for self_ref or single FK)
    """
    columns = [_col("id")]
    foreign_keys: list[ForeignKeySchema] = []
    if fks:
        for col_name, ref_table in fks:
            columns.append(_col(col_name))
            foreign_keys.append(
                ForeignKeySchema(
                    columns=[col_name],
                    ref_table=ref_table,
                    ref_columns=["id"],
                )
            )
    if self_ref:
        columns.append(_col(self_ref))
        foreign_keys.append(
            ForeignKeySchema(
                name=fk_name,
                columns=[self_ref],
                ref_table=name,
                ref_columns=["id"],
            )
        )
    return TableSchema(
        name=name,
        columns=columns,
        primary_key=["id"],
        foreign_keys=foreign_keys,
    )


def _schema(*tables: TableSchema) -> DatabaseSchema:
    """Build a DatabaseSchema from tables."""
    return DatabaseSchema(tables=list(tables))


# ── Task 1: Core graph construction tests ────────────────────────────────


class TestEmptySchema:
    def test_empty_returns_empty(self) -> None:
        graph = FKGraph.from_schema(_schema())
        assert graph.insertion_order == ()
        assert graph.tables == ()

    def test_empty_dependencies(self) -> None:
        graph = FKGraph.from_schema(_schema())
        assert graph.dependencies == {}


class TestSingleTable:
    def test_single_table_no_fks(self) -> None:
        graph = FKGraph.from_schema(_schema(_table("users")))
        assert graph.insertion_order == (("users",),)

    def test_single_table_in_tables(self) -> None:
        graph = FKGraph.from_schema(_schema(_table("users")))
        assert graph.tables == ("users",)


class TestTwoIndependentTables:
    def test_both_in_same_batch(self) -> None:
        graph = FKGraph.from_schema(_schema(_table("users"), _table("products")))
        assert len(graph.insertion_order) == 1
        assert set(graph.insertion_order[0]) == {"users", "products"}


class TestSimpleFK:
    def test_parent_before_child(self) -> None:
        graph = FKGraph.from_schema(
            _schema(
                _table("users"),
                _table("orders", fks=[("user_id", "users")]),
            )
        )
        assert graph.insertion_order == (("users",), ("orders",))

    def test_dependencies(self) -> None:
        graph = FKGraph.from_schema(
            _schema(
                _table("users"),
                _table("orders", fks=[("user_id", "users")]),
            )
        )
        assert graph.dependencies == {
            "users": frozenset(),
            "orders": frozenset({"users"}),
        }


class TestLinearChain:
    def test_four_batches(self) -> None:
        graph = FKGraph.from_schema(
            _schema(
                _table("d"),
                _table("c", fks=[("d_id", "d")]),
                _table("b", fks=[("c_id", "c")]),
                _table("a", fks=[("b_id", "b")]),
            )
        )
        assert graph.insertion_order == (("d",), ("c",), ("b",), ("a",))


class TestDiamond:
    def test_three_batches(self) -> None:
        graph = FKGraph.from_schema(
            _schema(
                _table("a"),
                _table("b", fks=[("a_id", "a")]),
                _table("c", fks=[("a_id", "a")]),
                _table("d", fks=[("b_id", "b"), ("c_id", "c")]),
            )
        )
        assert len(graph.insertion_order) == 3
        assert graph.insertion_order[0] == ("a",)
        assert set(graph.insertion_order[1]) == {"b", "c"}
        assert graph.insertion_order[2] == ("d",)


# ── Task 3: Self-ref + edge case tests ──────────────────────────────────


class TestSelfReferencingFK:
    def test_self_ref_not_in_dependencies(self) -> None:
        graph = FKGraph.from_schema(
            _schema(_table("employees", self_ref="manager_id", fk_name="fk_mgr"))
        )
        assert graph.dependencies["employees"] == frozenset()
        assert graph.insertion_order == (("employees",),)

    def test_self_ref_captured(self) -> None:
        graph = FKGraph.from_schema(
            _schema(_table("employees", self_ref="manager_id", fk_name="fk_mgr"))
        )
        assert "employees" in graph.self_referencing
        fks = graph.self_referencing["employees"]
        assert len(fks) == 1
        assert fks[0].columns == ["manager_id"]
        assert fks[0].ref_table == "employees"

    def test_self_ref_unnamed_fk(self) -> None:
        graph = FKGraph.from_schema(_schema(_table("categories", self_ref="parent_id")))
        assert "categories" in graph.self_referencing
        fks = graph.self_referencing["categories"]
        assert fks[0].name is None

    def test_self_ref_with_other_fks(self) -> None:
        """Table with both self-ref and external FKs."""
        t = TableSchema(
            name="employees",
            columns=[_col("id"), _col("dept_id"), _col("manager_id")],
            primary_key=["id"],
            foreign_keys=[
                ForeignKeySchema(columns=["dept_id"], ref_table="departments", ref_columns=["id"]),
                ForeignKeySchema(columns=["manager_id"], ref_table="employees", ref_columns=["id"]),
            ],
        )
        graph = FKGraph.from_schema(_schema(_table("departments"), t))
        assert graph.dependencies["employees"] == frozenset({"departments"})
        assert "employees" in graph.self_referencing


class TestMultipleFKsToSameParent:
    def test_single_edge(self) -> None:
        """Two FKs to same parent → single dependency edge."""
        t = TableSchema(
            name="orders",
            columns=[_col("id"), _col("billing_addr_id"), _col("shipping_addr_id")],
            primary_key=["id"],
            foreign_keys=[
                ForeignKeySchema(
                    columns=["billing_addr_id"], ref_table="addresses", ref_columns=["id"]
                ),
                ForeignKeySchema(
                    columns=["shipping_addr_id"], ref_table="addresses", ref_columns=["id"]
                ),
            ],
        )
        graph = FKGraph.from_schema(_schema(_table("addresses"), t))
        assert graph.dependencies["orders"] == frozenset({"addresses"})
        assert graph.insertion_order == (("addresses",), ("orders",))


class TestJunctionTable:
    def test_junction_after_parents(self) -> None:
        graph = FKGraph.from_schema(
            _schema(
                _table("users"),
                _table("roles"),
                _table("user_roles", fks=[("user_id", "users"), ("role_id", "roles")]),
            )
        )
        assert len(graph.insertion_order) == 2
        assert set(graph.insertion_order[0]) == {"roles", "users"}
        assert graph.insertion_order[1] == ("user_roles",)


class TestCompositeFk:
    def test_composite_fk_single_edge(self) -> None:
        """Multi-column FK creates a single dependency edge."""
        parent = TableSchema(
            name="order_lines",
            columns=[_col("order_id"), _col("line_no")],
            primary_key=["order_id", "line_no"],
        )
        child = TableSchema(
            name="line_items",
            columns=[_col("id"), _col("order_id"), _col("line_no")],
            primary_key=["id"],
            foreign_keys=[
                ForeignKeySchema(
                    columns=["order_id", "line_no"],
                    ref_table="order_lines",
                    ref_columns=["order_id", "line_no"],
                )
            ],
        )
        graph = FKGraph.from_schema(_schema(parent, child))
        assert graph.dependencies["line_items"] == frozenset({"order_lines"})
        assert graph.insertion_order == (("order_lines",), ("line_items",))


# ── Task 4: External refs + cycle + properties ──────────────────────────


class TestExternalRefs:
    def test_external_ref_filtered_from_deps(self) -> None:
        """FK to table not in schema → filtered from dependencies."""
        t = TableSchema(
            name="orders",
            columns=[_col("id"), _col("user_id")],
            primary_key=["id"],
            foreign_keys=[
                ForeignKeySchema(
                    columns=["user_id"],
                    ref_table="external_users",
                    ref_columns=["id"],
                )
            ],
        )
        graph = FKGraph.from_schema(_schema(t))
        assert graph.dependencies["orders"] == frozenset()
        assert "external_users" not in graph.tables

    def test_external_ref_exposed(self) -> None:
        t = TableSchema(
            name="orders",
            columns=[_col("id"), _col("user_id")],
            primary_key=["id"],
            foreign_keys=[
                ForeignKeySchema(
                    columns=["user_id"],
                    ref_table="ext_schema.users",
                    ref_columns=["id"],
                )
            ],
        )
        graph = FKGraph.from_schema(_schema(t))
        assert "orders" in graph.external_refs
        assert "ext_schema.users" in graph.external_refs["orders"]

    def test_external_ref_not_in_insertion_order(self) -> None:
        t = TableSchema(
            name="orders",
            columns=[_col("id"), _col("user_id")],
            primary_key=["id"],
            foreign_keys=[
                ForeignKeySchema(
                    columns=["user_id"],
                    ref_table="external_users",
                    ref_columns=["id"],
                )
            ],
        )
        graph = FKGraph.from_schema(_schema(t))
        all_tables = [t for batch in graph.insertion_order for t in batch]
        assert "external_users" not in all_tables
        assert "orders" in all_tables


class TestCycleDetection:
    def test_cycle_raises(self) -> None:
        with pytest.raises(CycleError):
            FKGraph.from_schema(
                _schema(
                    _table("a", fks=[("b_id", "b")]),
                    _table("b", fks=[("a_id", "a")]),
                )
            )

    def test_three_table_cycle_raises(self) -> None:
        with pytest.raises(CycleError):
            FKGraph.from_schema(
                _schema(
                    _table("a", fks=[("c_id", "c")]),
                    _table("b", fks=[("a_id", "a")]),
                    _table("c", fks=[("b_id", "b")]),
                )
            )


class TestDependents:
    def test_dependents_returns_children(self) -> None:
        graph = FKGraph.from_schema(
            _schema(
                _table("users"),
                _table("orders", fks=[("user_id", "users")]),
                _table("reviews", fks=[("user_id", "users")]),
            )
        )
        assert graph.dependents("users") == frozenset({"orders", "reviews"})

    def test_dependents_leaf_returns_empty(self) -> None:
        graph = FKGraph.from_schema(
            _schema(
                _table("users"),
                _table("orders", fks=[("user_id", "users")]),
            )
        )
        assert graph.dependents("orders") == frozenset()

    def test_dependents_unknown_table_raises(self) -> None:
        graph = FKGraph.from_schema(_schema(_table("users")))
        with pytest.raises(KeyError, match="nonexistent"):
            graph.dependents("nonexistent")


class TestImmutability:
    def test_frozen_model(self) -> None:
        graph = FKGraph.from_schema(_schema(_table("users")))
        with pytest.raises(ValidationError):
            graph.tables = ("modified",)  # type: ignore[misc]


class TestDeterministicOrdering:
    def test_same_schema_same_order(self) -> None:
        schema = _schema(
            _table("c"),
            _table("b", fks=[("c_id", "c")]),
            _table("a", fks=[("c_id", "c")]),
        )
        order1 = FKGraph.from_schema(schema).insertion_order
        order2 = FKGraph.from_schema(schema).insertion_order
        assert order1 == order2
        # a and b should be sorted within their batch
        assert order1[1] == ("a", "b")


# ── Task 5: Realistic schema + performance ───────────────────────────────


class TestRealisticSchema:
    """Tech spec example: categories, countries, roles, users, products, orders,
    order_items, reviews."""

    def test_tech_spec_batches(self) -> None:
        schema = _schema(
            _table("categories", self_ref="parent_id"),
            _table("countries"),
            _table("roles"),
            _table("users", fks=[("role_id", "roles"), ("country_id", "countries")]),
            _table("products", fks=[("category_id", "categories")]),
            _table("orders", fks=[("user_id", "users")]),
            _table(
                "order_items",
                fks=[("order_id", "orders"), ("product_id", "products")],
            ),
            _table(
                "reviews",
                fks=[("user_id", "users"), ("product_id", "products")],
            ),
        )
        graph = FKGraph.from_schema(schema)

        # Should be 4 batches
        assert len(graph.insertion_order) == 4

        # Batch 0: tables with no FK deps
        assert set(graph.insertion_order[0]) == {"categories", "countries", "roles"}

        # Batch 1: depend only on batch 0
        assert set(graph.insertion_order[1]) == {"products", "users"}

        # Batch 2: depend on batch 0+1
        assert set(graph.insertion_order[2]) == {"orders", "reviews"}

        # Batch 3: depends on batch 2
        assert graph.insertion_order[3] == ("order_items",)

    def test_self_refs_captured(self) -> None:
        schema = _schema(
            _table("categories", self_ref="parent_id"),
            _table("products", fks=[("category_id", "categories")]),
        )
        graph = FKGraph.from_schema(schema)
        assert "categories" in graph.self_referencing
        assert "products" not in graph.self_referencing


class TestPerformance:
    def test_200_table_chain_under_100ms(self) -> None:
        tables = [_table(f"t_{i:03d}") for i in range(200)]
        # Add FK chain: t_001 -> t_000, t_002 -> t_001, ...
        chained: list[TableSchema] = [tables[0]]
        for i in range(1, 200):
            chained.append(_table(f"t_{i:03d}", fks=[(f"t_{i - 1:03d}_id", f"t_{i - 1:03d}")]))
        schema = _schema(*chained)

        start = time.perf_counter()
        graph = FKGraph.from_schema(schema)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.5, f"Took {elapsed:.3f}s (>500ms)"
        assert len(graph.tables) == 200
        assert len(graph.insertion_order) == 200  # linear chain = 200 batches

    def test_200_table_star_under_100ms(self) -> None:
        """200 tables all depending on one root — 2 batches, tests parallel grouping."""
        root = _table("root")
        children = [_table(f"child_{i:03d}", fks=[("root_id", "root")]) for i in range(199)]
        schema = _schema(root, *children)

        start = time.perf_counter()
        graph = FKGraph.from_schema(schema)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.5, f"Took {elapsed:.3f}s (>500ms)"
        assert len(graph.insertion_order) == 2
        assert graph.insertion_order[0] == ("root",)
        assert len(graph.insertion_order[1]) == 199
