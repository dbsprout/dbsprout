"""End-to-end SQLite extraction test."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from dbsprout.train.extractor import SampleExtractor
from dbsprout.train.manifest import read_manifest
from dbsprout.train.models import ExtractorConfig

if TYPE_CHECKING:
    from pathlib import Path


def test_full_extract_sqlite(sqlite_db: str, tmp_path: Path) -> None:
    out = tmp_path / "run"
    cfg = ExtractorConfig(
        sample_rows=50,
        output_dir=out,
        seed=7,
        max_per_table=30,
        quiet=True,
    )
    SampleExtractor().extract(source=sqlite_db, config=cfg)

    assert (out / "samples" / "users.parquet").exists()
    assert (out / "samples" / "orders.parquet").exists()
    assert (out / "samples" / "order_items.parquet").exists()
    assert (out / "manifest.json").exists()

    manifest = read_manifest(out / "manifest.json")
    assert manifest.dialect == "sqlite"
    assert manifest.seed == 7
    assert manifest.schema_hash != ""
    assert manifest.requested_budget == 50

    items = pl.read_parquet(out / "samples" / "order_items.parquet")
    orders = pl.read_parquet(out / "samples" / "orders.parquet")
    items_order_ids = set(items["order_id"].to_list())
    orders_ids = set(orders["id"].to_list())
    assert items_order_ids.issubset(orders_ids)


def test_fk_closure_actually_adds_rows(sqlite_db: str, tmp_path: Path) -> None:
    """Force closure to fire by giving children a much larger budget than parents."""
    out = tmp_path / "run"
    # `min_per_table=1` keeps users tiny so most order_items reference unsampled users.
    cfg = ExtractorConfig(
        sample_rows=200,
        output_dir=out,
        seed=11,
        min_per_table=1,
        max_per_table=120,
        quiet=True,
    )
    result = SampleExtractor().extract(source=sqlite_db, config=cfg)

    closure_added = sum(r.fk_closure_added for r in result.tables)
    breakdown = [(r.table, r.target, r.sampled, r.fk_closure_added) for r in result.tables]
    assert closure_added > 0, (
        f"FK closure added no rows in a scenario engineered to require it; breakdown={breakdown}"
    )

    # And FK satisfaction must hold for every child→parent edge in the sampled set.
    items = pl.read_parquet(out / "samples" / "order_items.parquet")
    orders = pl.read_parquet(out / "samples" / "orders.parquet")
    products = pl.read_parquet(out / "samples" / "products.parquet")
    users = pl.read_parquet(out / "samples" / "users.parquet")
    assert set(items["order_id"].to_list()).issubset(set(orders["id"].to_list()))
    assert set(items["product_id"].to_list()).issubset(set(products["id"].to_list()))
    assert set(orders["user_id"].to_list()).issubset(set(users["id"].to_list()))


def test_determinism_same_seed(sqlite_db: str, tmp_path: Path) -> None:
    """Two runs with the same seed produce byte-identical Parquet files."""
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    cfg_a = ExtractorConfig(
        sample_rows=20,
        output_dir=out_a,
        seed=99,
        max_per_table=10,
        quiet=True,
    )
    cfg_b = ExtractorConfig(
        sample_rows=20,
        output_dir=out_b,
        seed=99,
        max_per_table=10,
        quiet=True,
    )
    SampleExtractor().extract(source=sqlite_db, config=cfg_a)
    SampleExtractor().extract(source=sqlite_db, config=cfg_b)

    for name in ("users.parquet", "orders.parquet", "order_items.parquet"):
        a = (out_a / "samples" / name).read_bytes()
        b = (out_b / "samples" / name).read_bytes()
        assert a == b, f"non-deterministic output for {name}"
