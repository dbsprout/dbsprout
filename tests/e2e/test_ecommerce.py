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
    # Use the suite's standard rows=50/seed=42: composite-PK junction
    # tables (inventory, order_items) can collide at very low row counts
    # (pre-existing generator behavior — see follow-up). 50 rows is the
    # AC's documented count and produces 100% integrity.
    a = run_pipeline(FIXTURE, tmp_path / "a", rows=50, seed=42)
    b = run_pipeline(FIXTURE, tmp_path / "b", rows=50, seed=42)
    assert a.seed_data == b.seed_data
