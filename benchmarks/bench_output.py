"""Output-writer and memory performance benchmarks (S-074)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dbsprout.generate.engines.heuristic import HeuristicEngine
from dbsprout.output.sql_writer import SQLWriter
from dbsprout.spec.heuristics import map_columns

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Any

    from pytest_benchmark.fixture import BenchmarkFixture

    from dbsprout.schema.models import DatabaseSchema, TableSchema

pytestmark = pytest.mark.benchmark


def test_bench_sql_output(
    benchmark: BenchmarkFixture,
    ecommerce_table: TableSchema,
    ecommerce_schema: DatabaseSchema,
    tmp_path: Path,
) -> None:
    """SQL INSERT writer: 10k rows for a 10-column table."""
    mappings = map_columns(ecommerce_schema)["orders"]
    rows = HeuristicEngine().generate_table(ecommerce_table, mappings, 10_000)
    writer = SQLWriter()

    def _run() -> list[Path]:
        return writer.write(
            {"orders": rows},
            ecommerce_schema,
            ["orders"],
            tmp_path,
            dialect="postgresql",
            batch_size=1000,
        )

    written = benchmark(_run)
    assert len(written) == 1


def test_bench_generation_memory(
    measure_peak_mb: Callable[[Callable[[], Any]], float],
    ecommerce_table: TableSchema,
    ecommerce_schema: DatabaseSchema,
) -> None:
    """Peak memory for a 100k-row heuristic generation pass stays bounded."""
    mappings = map_columns(ecommerce_schema)["orders"]
    engine = HeuristicEngine()
    peak_mb = measure_peak_mb(lambda: engine.generate_table(ecommerce_table, mappings, 100_000))
    # Generous ceiling — guards against pathological regressions, not a tight SLA.
    assert peak_mb < 1024, f"peak {peak_mb:.1f} MB exceeded 1024 MB"
