"""Generation orchestrator — coordinates the full generate pipeline.

Wires together: FK graph → topological order → heuristic mapping →
engine generation → FK sampling → constraint enforcement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from graphlib import CycleError
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from pathlib import Path

    from dbsprout.config.models import DBSproutConfig
    from dbsprout.schema.models import DatabaseSchema

from dbsprout.generate.constraints import enforce_constraints
from dbsprout.generate.fk_sampling import sample_fk_values
from dbsprout.plugins.dispatch import resolve_engine
from dbsprout.schema.graph import FKGraph, resolve_cycles
from dbsprout.spec.heuristics import map_columns


@dataclass(frozen=True)
class GenerateResult:
    """Result of a generation run.

    ``table_timings`` is a tuple of ``(table_name, row_count,
    generation_ms)`` triples, one per generated table, captured by
    instrumenting the per-table generation loop. It is empty when no
    tables were generated. Consumed by the state-writer (S-080) to
    persist per-table telemetry.
    """

    tables_data: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    insertion_order: list[str] = field(default_factory=list)
    total_rows: int = 0
    total_tables: int = 0
    duration_seconds: float = 0.0
    table_timings: tuple[tuple[str, int, int], ...] = ()


def orchestrate(  # noqa: PLR0913
    schema: DatabaseSchema,
    config: DBSproutConfig,
    seed: int,
    default_rows: int,
    engine: str = "heuristic",
    reference_data: dict[str, list[dict[str, Any]]] | None = None,
    lora_path: Path | None = None,
) -> GenerateResult:
    """Run the full generation pipeline.

    ``reference_data`` is consumed only by the ``statistical`` engine
    (per-table sample rows used to fit the Gaussian copula). Returns a
    ``GenerateResult`` with generated data and stats.
    """
    start = time.monotonic()

    if not schema.tables:
        return GenerateResult(duration_seconds=time.monotonic() - start)

    # Build FK graph (handle cycles)
    insertion_order = _build_insertion_order(schema)

    # Map columns to generators (once for whole schema)
    all_mappings = map_columns(schema)

    selection = _select_engines(engine, schema, seed, lora_path)
    parent_data: dict[str, list[dict[str, Any]]] = {}
    timings: list[tuple[str, int, int]] = []

    for table_name in insertion_order:
        if _is_excluded(table_name, config):
            continue

        table_schema = schema.get_table(table_name)
        if table_schema is None:
            continue

        num_rows = _get_row_count(table_name, config, default_rows)
        mappings = all_mappings.get(table_name, {})
        table_start = time.perf_counter_ns()
        rows = _generate_rows(
            selection,
            table_schema,
            mappings,
            num_rows,
            (reference_data or {}).get(table_name, []),
        )
        sample_fk_values(table_schema, parent_data, rows, seed)
        rows = enforce_constraints(table_schema, rows, seed)
        table_ms = (time.perf_counter_ns() - table_start) // 1_000_000

        parent_data[table_name] = rows
        timings.append((table_name, len(rows), table_ms))

    duration = time.monotonic() - start
    total_rows = sum(len(rows) for rows in parent_data.values())
    generated_order = [t for t in insertion_order if t in parent_data]

    return GenerateResult(
        tables_data=parent_data,
        insertion_order=generated_order,
        total_rows=total_rows,
        total_tables=len(parent_data),
        duration_seconds=round(duration, 3),
        table_timings=tuple(timings),
    )


@dataclass(frozen=True)
class _EngineSelection:
    """Resolved engines for a generation run."""

    engine: str
    heuristic: Any
    spec: Any = None
    statistical: Any = None
    table_specs: dict[str, Any] = field(default_factory=dict)


def _select_engines(
    engine: str,
    schema: DatabaseSchema,
    seed: int,
    lora_path: Path | None = None,
) -> _EngineSelection:
    """Resolve the engine(s) needed for *engine*.

    Registry-first dispatch so third-party engines registered via
    ``[project.entry-points."dbsprout.generators"]`` win over the
    hard-wired fallback. The heuristic engine is always resolved — the
    ``spec`` and ``statistical`` engines fall back to it per-table.

    When *engine* is ``"spec"`` and a ``lora_path`` is supplied (S-067b),
    the DataSpec is produced by the real local LLM provider
    (:class:`~dbsprout.spec.providers.embedded.EmbeddedProvider` wrapped in
    :class:`~dbsprout.spec.analyzer.SpecAnalyzer`, which already retries and
    then falls back to ``heuristic_fallback`` on provider failure). Without
    a ``lora_path`` the original S-025 ``heuristic_fallback`` path is used
    unchanged.
    """
    heuristic = resolve_engine("heuristic", seed=seed)
    if engine == "spec":
        if lora_path is not None:
            from dbsprout.spec.analyzer import SpecAnalyzer  # noqa: PLC0415
            from dbsprout.spec.providers.embedded import (  # noqa: PLC0415
                EmbeddedProvider,
            )

            provider = EmbeddedProvider(lora_path=lora_path)
            dataspec = SpecAnalyzer(provider).analyze(schema)
        else:
            from dbsprout.spec.analyzer import heuristic_fallback  # noqa: PLC0415

            dataspec = heuristic_fallback(schema)
        return _EngineSelection(
            engine="spec",
            heuristic=heuristic,
            spec=resolve_engine("spec_driven", seed=seed),
            table_specs={ts.table_name: ts for ts in dataspec.tables},
        )
    if engine == "statistical":
        return _EngineSelection(
            engine="statistical",
            heuristic=heuristic,
            statistical=resolve_engine("statistical", seed=seed),
        )
    return _EngineSelection(engine="heuristic", heuristic=heuristic)


def _generate_rows(
    selection: _EngineSelection,
    table_schema: Any,
    mappings: dict[str, Any],
    num_rows: int,
    reference_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Generate one table's rows with the selected engine."""
    rows: Any
    if selection.engine == "spec" and selection.spec is not None:
        table_spec = selection.table_specs.get(table_schema.name)
        if table_spec is not None:
            rows = selection.spec.generate_table(table_schema, table_spec, num_rows)
            return cast("list[dict[str, Any]]", rows)
    elif selection.engine == "statistical" and selection.statistical is not None:
        rows = selection.statistical.generate_table(
            table_schema, reference_rows, mappings, num_rows
        )
        return cast("list[dict[str, Any]]", rows)
    rows = selection.heuristic.generate_table(table_schema, mappings, num_rows)
    return cast("list[dict[str, Any]]", rows)


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
