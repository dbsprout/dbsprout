"""Generation orchestrator — coordinates the full generate pipeline.

Wires together: FK graph → topological order → heuristic mapping →
engine generation → FK sampling → constraint enforcement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from graphlib import CycleError
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dbsprout.config.models import DBSproutConfig
    from dbsprout.schema.models import DatabaseSchema

from dbsprout.generate.constraints import enforce_constraints
from dbsprout.generate.engines.heuristic import HeuristicEngine
from dbsprout.generate.fk_sampling import sample_fk_values
from dbsprout.schema.graph import FKGraph, resolve_cycles
from dbsprout.spec.heuristics import map_columns


@dataclass(frozen=True)
class GenerateResult:
    """Result of a generation run."""

    tables_data: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    insertion_order: list[str] = field(default_factory=list)
    total_rows: int = 0
    total_tables: int = 0
    duration_seconds: float = 0.0


def orchestrate(
    schema: DatabaseSchema,
    config: DBSproutConfig,
    seed: int,
    default_rows: int,
    engine: str = "heuristic",
) -> GenerateResult:
    """Run the full generation pipeline.

    Returns a ``GenerateResult`` with generated data and stats.
    """
    start = time.monotonic()

    if not schema.tables:
        return GenerateResult(duration_seconds=time.monotonic() - start)

    # Build FK graph (handle cycles)
    insertion_order = _build_insertion_order(schema)

    # Map columns to generators (once for whole schema)
    all_mappings = map_columns(schema)

    # Select engine
    use_spec = engine == "spec"
    heuristic_engine = HeuristicEngine(seed=seed)
    spec_engine = None
    all_table_specs = None
    if use_spec:
        from dbsprout.generate.engines.spec_driven import SpecDrivenEngine  # noqa: PLC0415
        from dbsprout.spec.analyzer import heuristic_fallback  # noqa: PLC0415

        spec_engine = SpecDrivenEngine(seed=seed)
        dataspec = heuristic_fallback(schema)
        all_table_specs = {ts.table_name: ts for ts in dataspec.tables}

    parent_data: dict[str, list[dict[str, Any]]] = {}

    for table_name in insertion_order:
        if _is_excluded(table_name, config):
            continue

        table_schema = schema.get_table(table_name)
        if table_schema is None:
            continue

        num_rows = _get_row_count(table_name, config, default_rows)

        # Generate with selected engine
        if use_spec and spec_engine is not None and all_table_specs is not None:
            table_spec = all_table_specs.get(table_name)
            if table_spec is not None:
                rows = spec_engine.generate_table(table_schema, table_spec, num_rows)
            else:
                mappings = all_mappings.get(table_name, {})
                rows = heuristic_engine.generate_table(table_schema, mappings, num_rows)
        else:
            mappings = all_mappings.get(table_name, {})
            rows = heuristic_engine.generate_table(table_schema, mappings, num_rows)
        sample_fk_values(table_schema, parent_data, rows, seed)
        rows = enforce_constraints(table_schema, rows, seed)

        parent_data[table_name] = rows

    duration = time.monotonic() - start
    total_rows = sum(len(rows) for rows in parent_data.values())
    generated_order = [t for t in insertion_order if t in parent_data]

    return GenerateResult(
        tables_data=parent_data,
        insertion_order=generated_order,
        total_rows=total_rows,
        total_tables=len(parent_data),
        duration_seconds=round(duration, 3),
    )


def _build_insertion_order(schema: DatabaseSchema) -> list[str]:
    """Build a flat insertion order from the FK graph."""
    try:
        graph = FKGraph.from_schema(schema)
    except CycleError:
        resolved = resolve_cycles(schema)
        graph = resolved.graph

    return _flatten_batches(graph.insertion_order)


def _flatten_batches(order: tuple[tuple[str, ...], ...]) -> list[str]:
    """Flatten batched insertion order into a flat list."""
    result: list[str] = []
    for batch in order:
        result.extend(sorted(batch))
    return result


def _get_row_count(
    table_name: str,
    config: DBSproutConfig,
    default_rows: int,
) -> int:
    """Resolve row count: per-table override > default_rows."""
    override = config.tables.get(table_name)
    if override and override.rows is not None:
        return override.rows
    return default_rows


def _is_excluded(table_name: str, config: DBSproutConfig) -> bool:
    """Check if a table is excluded in config."""
    override = config.tables.get(table_name)
    return override.exclude if override else False
