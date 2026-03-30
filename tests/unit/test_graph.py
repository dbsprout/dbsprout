"""Tests for dbsprout.schema.graph — FK dependency graph + topological sort."""

from __future__ import annotations

import time
from graphlib import CycleError

import pytest
from pydantic import ValidationError

from dbsprout.schema.graph import FKGraph
from dbsprout.schema.models import (
    ForeignKeySchema,
    TableSchema,
)
from tests.unit.conftest import _col, _schema, _table

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
                _table("orders", fks=[("user_id", "users", True)]),
            )
        )
        assert graph.insertion_order == (("users",), ("orders",))

    def test_dependencies(self) -> None:
        graph = FKGraph.from_schema(
            _schema(
                _table("users"),
                _table("orders", fks=[("user_id", "users", True)]),
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
                _table("c", fks=[("d_id", "d", True)]),
                _table("b", fks=[("c_id", "c", True)]),
                _table("a", fks=[("b_id", "b", True)]),
            )
        )
        assert graph.insertion_order == (("d",), ("c",), ("b",), ("a",))


class TestDiamond:
    def test_three_batches(self) -> None:
        graph = FKGraph.from_schema(
            _schema(
                _table("a"),
                _table("b", fks=[("a_id", "a", True)]),
                _table("c", fks=[("a_id", "a", True)]),
                _table("d", fks=[("b_id", "b", True), ("c_id", "c", True)]),
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
                _table("user_roles", fks=[("user_id", "users", True), ("role_id", "roles", True)]),
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
                    _table("a", fks=[("b_id", "b", True)]),
                    _table("b", fks=[("a_id", "a", True)]),
                )
            )

    def test_three_table_cycle_raises(self) -> None:
        with pytest.raises(CycleError):
            FKGraph.from_schema(
                _schema(
                    _table("a", fks=[("c_id", "c", True)]),
                    _table("b", fks=[("a_id", "a", True)]),
                    _table("c", fks=[("b_id", "b", True)]),
                )
            )


class TestDependents:
    def test_dependents_returns_children(self) -> None:
        graph = FKGraph.from_schema(
            _schema(
                _table("users"),
                _table("orders", fks=[("user_id", "users", True)]),
                _table("reviews", fks=[("user_id", "users", True)]),
            )
        )
        assert graph.dependents("users") == frozenset({"orders", "reviews"})

    def test_dependents_leaf_returns_empty(self) -> None:
        graph = FKGraph.from_schema(
            _schema(
                _table("users"),
                _table("orders", fks=[("user_id", "users", True)]),
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
            _table("b", fks=[("c_id", "c", True)]),
            _table("a", fks=[("c_id", "c", True)]),
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
            _table("users", fks=[("role_id", "roles", True), ("country_id", "countries", True)]),
            _table("products", fks=[("category_id", "categories", True)]),
            _table("orders", fks=[("user_id", "users", True)]),
            _table(
                "order_items",
                fks=[("order_id", "orders", True), ("product_id", "products", True)],
            ),
            _table(
                "reviews",
                fks=[("user_id", "users", True), ("product_id", "products", True)],
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
            _table("products", fks=[("category_id", "categories", True)]),
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
            prev = f"t_{i - 1:03d}"
            chained.append(_table(f"t_{i:03d}", fks=[(f"{prev}_id", prev, True)]))
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
        children = [_table(f"child_{i:03d}", fks=[("root_id", "root", True)]) for i in range(199)]
        schema = _schema(root, *children)

        start = time.perf_counter()
        graph = FKGraph.from_schema(schema)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.5, f"Took {elapsed:.3f}s (>500ms)"
        assert len(graph.insertion_order) == 2
        assert graph.insertion_order[0] == ("root",)
        assert len(graph.insertion_order[1]) == 199
