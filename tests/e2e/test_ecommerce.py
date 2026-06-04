"""E2E: e-commerce schema (self-ref categories, composite order_items)."""

from __future__ import annotations

from pathlib import Path

import pytest

from dbsprout.schema.models import ColumnType
from tests.e2e._pipeline import (
    assert_full_integrity,
    assert_programmatic_integrity,
    run_pipeline,
)

FIXTURE = Path(__file__).parent.parent / "fixtures" / "schemas" / "ecommerce.sql"


@pytest.mark.integration
def test_ecommerce_full_integrity(tmp_path: Path) -> None:
    result = run_pipeline(FIXTURE, tmp_path, rows=50, seed=42)
    assert_full_integrity(result)
    assert_programmatic_integrity(result)
    assert len(result.schema.tables) >= 10
    for rows in result.seed_data.values():
        assert len(rows) == 50


@pytest.mark.integration
def test_ecommerce_self_ref_and_composite(tmp_path: Path) -> None:
    result = run_pipeline(FIXTURE, tmp_path, rows=50, seed=42)
    schema = result.schema

    categories = schema.get_table("categories")
    assert categories is not None
    self_refs = [fk for fk in categories.foreign_keys if fk.ref_table == "categories"]
    assert self_refs, "categories must have a self-referencing FK"

    order_items = schema.get_table("order_items")
    assert order_items is not None
    assert len(order_items.primary_key) == 2, order_items.primary_key
    assert len(order_items.foreign_keys) >= 2

    users = schema.get_table("users")
    assert users is not None
    status = next(c for c in users.columns if c.name == "status")
    assert status.data_type is ColumnType.ENUM
    assert status.enum_values


@pytest.mark.integration
def test_ecommerce_deterministic(tmp_path: Path) -> None:
    # Same seed → byte-identical seed data (reproducibility). Composite-PK
    # junction tables (inventory, order_items) are now FK-aware deduped (P5-6),
    # so any seed validates clean — see test_ecommerce_inventory_unique_across_seeds.
    a = run_pipeline(FIXTURE, tmp_path / "a", rows=50, seed=42)
    b = run_pipeline(FIXTURE, tmp_path / "b", rows=50, seed=42)
    assert a.seed_data == b.seed_data


@pytest.mark.integration
def test_ecommerce_inventory_unique_across_seeds(tmp_path: Path) -> None:
    """P5-6: the all-FK composite-PK junction tables (inventory, order_items)
    must have NO pk_uniqueness violation regardless of seed (the bug surfaced
    on most seeds via FK-sampling collisions; seed=42 happened to pass)."""
    junction_tables = {"inventory", "order_items"}
    for seed in (0, 1, 7, 11, 17, 23):
        result = run_pipeline(FIXTURE, tmp_path / str(seed), rows=50, seed=seed)
        checks = result.report["integrity"]["checks"]
        pk_failures = [
            c
            for c in checks
            if c["check"] == "pk_uniqueness" and c["table"] in junction_tables and not c["passed"]
        ]
        assert not pk_failures, f"seed={seed}: pk_uniqueness failures {pk_failures}"
        # Whole-report integrity must also hold for these seeds.
        assert result.report["integrity"]["passed"] is True, (
            f"seed={seed}: {result.report['integrity']}"
        )
