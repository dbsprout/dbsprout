"""SampleExtractor: introspect -> allocate -> fetch -> close FK -> write."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Final

import polars as pl
import sqlalchemy as sa
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from dbsprout import __version__ as dbs_version
from dbsprout.schema.introspect import introspect
from dbsprout.train.allocator import allocate_budget
from dbsprout.train.closure import ParentFetcher, close_fk_graph
from dbsprout.train.manifest import write_manifest
from dbsprout.train.models import (
    ClosureReport,
    ExtractorConfig,
    SampleAllocation,
    SampleManifest,
    SampleResult,
    TableExtractionResult,
)
from dbsprout.train.random_select import build_random_query

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from dbsprout.schema.models import DatabaseSchema, TableSchema

logger = logging.getLogger("dbsprout.train.extractor")

_MEMORY_WARN_THRESHOLD: Final[int] = 500_000
_CLOSURE_HARD_CAP: Final[int] = 16


def _row_counts(engine: sa.Engine, schema: DatabaseSchema) -> dict[str, int]:
    """Approximate row counts per table; dialect-specific fast paths.

    PostgreSQL uses ``pg_class.reltuples`` (O(1)); reltuples == 0 is treated as
    empty (skipped from allocation). MySQL uses ``information_schema.tables``.
    SQLite uses ``COUNT(*)`` (acceptable for small DBs).
    """
    dialect = engine.dialect.name
    counts: dict[str, int] = {}
    with engine.connect() as conn:
        for table in schema.tables:
            if dialect == "postgresql":
                row = conn.execute(
                    sa.text("SELECT reltuples::BIGINT FROM pg_class WHERE relname = :n"),
                    {"n": table.name},
                ).scalar()
                counts[table.name] = int(row) if row else 0
            elif dialect == "mysql":
                row = conn.execute(
                    sa.text(
                        "SELECT TABLE_ROWS FROM information_schema.tables "
                        "WHERE table_schema = DATABASE() AND table_name = :n"
                    ),
                    {"n": table.name},
                ).scalar()
                counts[table.name] = int(row) if row else 0
            else:
                row = conn.execute(
                    sa.text(f'SELECT COUNT(*) FROM "{table.name}"'),  # noqa: S608  # nosec B608
                ).scalar()
                counts[table.name] = int(row or 0)
    return counts


def _has_rowid(engine: sa.Engine, table_name: str) -> bool:
    """Probe whether a SQLite table has a rowid (False for ``WITHOUT ROWID``).

    Reads the table's CREATE statement from ``sqlite_master`` and looks for the
    ``WITHOUT ROWID`` clause. This is more precise than catching
    ``OperationalError`` from a probe SELECT (which would also swallow locked /
    permission-denied errors and silently fall back to unseeded sampling).
    """
    with engine.connect() as conn:
        ddl = conn.execute(
            sa.text("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = :n"),
            {"n": table_name},
        ).scalar()
    if ddl is None:
        return True
    return "WITHOUT ROWID" not in ddl.upper()


def _fetch_random(
    engine: sa.Engine,
    *,
    table: sa.Table,
    n: int,
    seed: int,
    row_count: int,
) -> pl.DataFrame:
    """Fetch ``n`` random rows from *table* using a dialect-aware seeded query.

    Uses :meth:`Engine.begin` (not ``connect()``) so that, on PostgreSQL, the
    ``setseed`` call binds to the same transaction as the subsequent ``SELECT``;
    a fresh connection per statement would otherwise reset the seed and break
    determinism.
    """
    dialect = engine.dialect.name
    has_rowid = _has_rowid(engine, table.name) if dialect == "sqlite" else True
    q = build_random_query(
        table,
        n,
        dialect=dialect,
        seed=seed,
        row_count=row_count,
        has_rowid=has_rowid,
    )
    if q.warning:
        logger.warning("train.random: %s", q.warning)
    # ``begin()`` (not ``connect()``) so PG ``setseed`` binds to the same
    # transaction as the SELECT — otherwise the seed is reset between statements.
    with engine.begin() as conn:
        for setup_sql, setup_params in q.setup:
            conn.execute(sa.text(setup_sql), setup_params)
        rows = conn.execute(sa.text(q.sql), q.params).mappings().all()
    return pl.DataFrame([dict(r) for r in rows]) if rows else pl.DataFrame()


# Safely below SQLite's 32_766 bind limit and PG/MySQL packet caps.
_PK_FETCH_BATCH: Final[int] = 5_000


def _fetch_by_pk(
    engine: sa.Engine,
    *,
    table: TableSchema,
    pk_column: str,
    values: Iterable[Any],
) -> pl.DataFrame:
    """Fetch parent rows by PK in chunks to stay below per-statement bind limits."""
    md = sa.MetaData()
    sa_table = sa.Table(table.name, md, autoload_with=engine)
    value_list = list(values)
    if not value_list:
        return pl.DataFrame(schema={pk_column: pl.Int64})
    frames: list[pl.DataFrame] = []
    with engine.connect() as conn:
        for start in range(0, len(value_list), _PK_FETCH_BATCH):
            chunk = value_list[start : start + _PK_FETCH_BATCH]
            sql = sa_table.select().where(sa_table.c[pk_column].in_(chunk))
            rows = conn.execute(sql).mappings().all()
            if rows:
                frames.append(pl.DataFrame([dict(r) for r in rows]))
    if not frames:
        return pl.DataFrame(schema={pk_column: pl.Int64})
    return pl.concat(frames, how="diagonal_relaxed") if len(frames) > 1 else frames[0]


class _EngineAdapter(ParentFetcher):
    """Adapter exposing ``_fetch_by_pk`` through the closure ``ParentFetcher`` protocol."""

    def __init__(self, engine: sa.Engine) -> None:
        self.engine = engine

    def fetch_by_pk(
        self,
        table: TableSchema,
        pk_column: str,
        values: Iterable[Any],
    ) -> pl.DataFrame:
        """Delegate to :func:`_fetch_by_pk` using this adapter's engine."""
        return _fetch_by_pk(self.engine, table=table, pk_column=pk_column, values=values)


class SampleExtractor:
    """Built-in ``live_db`` train extractor."""

    def extract(self, *, source: str, config: ExtractorConfig) -> SampleResult:
        """Extract a stratified sample from *source* into Parquet files.

        Pipeline: introspect schema → compute per-table allocations → fetch
        random rows → close FK graph → write per-table Parquet + manifest.
        Always disposes the SQLAlchemy engine on exit.
        """
        start = time.perf_counter()
        engine = sa.create_engine(source)
        try:
            schema = introspect(source)
            allocations, _ = self._allocate(engine, schema, config)
            samples = self._fetch_samples(engine, allocations, config)

            cap = config.fk_closure_max_iterations or min(_CLOSURE_HARD_CAP, len(schema.tables))
            report = close_fk_graph(samples, schema, _EngineAdapter(engine), max_iterations=cap)

            samples_dir = config.output_dir / "samples"
            samples_dir.mkdir(parents=True, exist_ok=True)
            table_results = self._write_results(samples, allocations, report, samples_dir)

            duration = time.perf_counter() - start
            manifest = SampleManifest(
                dbsprout_version=dbs_version,
                extracted_at=datetime.now(timezone.utc),
                dialect=engine.dialect.name,
                schema_hash=schema.schema_hash(),
                seed=config.seed,
                requested_budget=config.sample_rows,
                effective_budget=sum(r.sampled + r.fk_closure_added for r in table_results),
                tables=tuple(table_results),
                fk_closure_iterations=report.iterations,
                fk_unresolved_per_table=report.unresolved_per_table,
                fk_unresolved_total=sum(report.unresolved_per_table.values()),
                duration_seconds=duration,
            )
            manifest_path = config.output_dir / "manifest.json"
            write_manifest(manifest_path, manifest)

            return SampleResult(
                output_dir=config.output_dir,
                manifest_path=manifest_path,
                tables=tuple(table_results),
                duration_seconds=duration,
            )
        finally:
            engine.dispose()

    def _allocate(
        self,
        engine: sa.Engine,
        schema: DatabaseSchema,
        config: ExtractorConfig,
    ) -> tuple[list[SampleAllocation], int]:
        """Compute per-table sample allocations and emit memory-pressure warnings.

        Returns ``(allocations, max_per_table)`` where ``max_per_table`` is the
        effective ceiling actually applied (either user-supplied or derived).
        """
        row_counts = _row_counts(engine, schema)
        n_tables = max(1, sum(1 for c in row_counts.values() if c > 0))
        max_per_table = config.max_per_table or max(50, 10 * config.sample_rows // n_tables)
        allocations = allocate_budget(
            row_counts=row_counts,
            budget=config.sample_rows,
            min_per_table=config.min_per_table,
            max_per_table=max_per_table,
        )
        for a in allocations:
            if a.target > _MEMORY_WARN_THRESHOLD:
                logger.warning(
                    "train.allocator: table '%s' target is %d rows "
                    "(~%d MB est. in memory); consider lowering --max-per-table",
                    a.table,
                    a.target,
                    a.target * 5 // 1024,
                )
        return allocations, max_per_table

    def _fetch_samples(
        self,
        engine: sa.Engine,
        allocations: list[SampleAllocation],
        config: ExtractorConfig,
    ) -> dict[str, pl.DataFrame]:
        """Run the per-table random fetch loop with a Rich progress display."""
        samples: dict[str, pl.DataFrame] = {}
        md = sa.MetaData()
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            disable=config.quiet,
        ) as progress:
            overall = progress.add_task("Extracting", total=sum(a.target for a in allocations))
            for a in allocations:
                if a.target == 0:
                    continue
                sa_table = sa.Table(a.table, md, autoload_with=engine)
                samples[a.table] = _fetch_random(
                    engine,
                    table=sa_table,
                    n=a.target,
                    seed=config.seed,
                    row_count=a.row_count,
                )
                progress.update(overall, advance=a.target, description=f"Extracting {a.table}")
        return samples

    def _write_results(
        self,
        samples: dict[str, pl.DataFrame],
        allocations: list[SampleAllocation],
        report: ClosureReport,
        samples_dir: Path,
    ) -> list[TableExtractionResult]:
        """Write per-table Parquet files and build :class:`TableExtractionResult` records.

        Includes a defense-in-depth path-traversal guard: a table whose
        resolved Parquet path escapes ``samples_dir`` (only possible if upstream
        identifier validation has been bypassed) is skipped with a warning.
        """
        samples_dir_resolved = samples_dir.resolve()
        table_results: list[TableExtractionResult] = []
        for a in allocations:
            if a.target == 0 or a.table not in samples:
                continue
            parquet_path = samples_dir / f"{a.table}.parquet"
            try:
                parquet_path.resolve().relative_to(samples_dir_resolved)
            except ValueError:
                logger.warning(
                    "skipping table %r: resolved Parquet path escapes output directory",
                    a.table,
                )
                continue
            samples[a.table].write_parquet(parquet_path, compression="zstd", compression_level=3)
            added = report.additions.get(a.table, 0)
            table_results.append(
                TableExtractionResult(
                    table=a.table,
                    target=a.target,
                    sampled=max(0, len(samples[a.table]) - added),
                    fk_closure_added=added,
                    parquet_path=parquet_path,
                )
            )
        return table_results
