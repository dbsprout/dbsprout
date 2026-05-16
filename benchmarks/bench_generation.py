"""Generation-path performance benchmarks (S-074)."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Any

import pytest

from dbsprout.generate.engines.heuristic import HeuristicEngine
from dbsprout.generate.fk_sampling import sample_fk_values
from dbsprout.generate.vectorized import generate_vectorized
from dbsprout.schema.introspect import introspect
from dbsprout.spec.heuristics import map_columns

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_benchmark.fixture import BenchmarkFixture

    from dbsprout.schema.models import DatabaseSchema, TableSchema

pytestmark = pytest.mark.benchmark


def test_bench_heuristic_generation(
    benchmark: BenchmarkFixture,
    ecommerce_table: TableSchema,
    ecommerce_schema: DatabaseSchema,
) -> None:
    """Heuristic engine: 10-col table, 1000 rows per round."""
    mappings = map_columns(ecommerce_schema)["orders"]
    engine = HeuristicEngine()
    rows = benchmark(engine.generate_table, ecommerce_table, mappings, 1000)
    assert len(rows) == 1000


def test_bench_numpy_vectorized(benchmark: BenchmarkFixture) -> None:
    """NumPy fast path: generate 1,000,000 integers."""
    values = benchmark(
        generate_vectorized,
        "random_int",
        1_000_000,
        seed=42,
        params={"min": 0, "max": 1_000_000},
    )
    assert values is not None
    assert len(values) == 1_000_000


def test_bench_fk_sampling(
    benchmark: BenchmarkFixture,
    fk_child_table: TableSchema,
) -> None:
    """FK sampling: 10k child rows referencing 10k parent rows."""
    parent_data: dict[str, list[dict[str, Any]]] = {"users": [{"id": i} for i in range(10_000)]}

    def _run() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = [{"id": i, "user_id": None} for i in range(10_000)]
        return sample_fk_values(fk_child_table, parent_data, rows, seed=42)

    result = benchmark(_run)
    assert len(result) == 10_000
    assert result[0]["user_id"] is not None


def test_bench_introspection(benchmark: BenchmarkFixture, tmp_path: Path) -> None:
    """Schema introspection of a 5-table SQLite database."""
    db_path = tmp_path / "bench.db"
    conn = sqlite3.connect(db_path)
    for i in range(5):
        conn.execute(f"CREATE TABLE t{i} (id INTEGER PRIMARY KEY, name TEXT, val REAL)")
    conn.commit()
    conn.close()
    url = f"sqlite:///{db_path}"
    schema = benchmark(introspect, url)
    assert len(schema.tables) == 5
