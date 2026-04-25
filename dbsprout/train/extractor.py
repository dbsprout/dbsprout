"""SampleExtractor: introspect -> allocate -> fetch -> close FK -> write."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import polars as pl
import sqlalchemy as sa

from dbsprout import __version__ as dbs_version
from dbsprout.schema.introspect import introspect
from dbsprout.train.allocator import allocate_budget
from dbsprout.train.closure import ParentFetcher, close_fk_graph
from dbsprout.train.manifest import write_manifest
from dbsprout.train.models import (
    ExtractorConfig,
    SampleManifest,
    SampleResult,
    TableExtractionResult,
)
from dbsprout.train.random_select import build_random_query

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dbsprout.schema.models import DatabaseSchema, TableSchema

logger = logging.getLogger("dbsprout.train.extractor")

_MEMORY_WARN_THRESHOLD = 500_000
_CLOSURE_HARD_CAP = 16


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
    """Probe whether a SQLite table has a rowid (False for WITHOUT ROWID)."""
    with engine.connect() as conn:
        try:
            conn.execute(sa.text(f'SELECT rowid FROM "{table_name}" LIMIT 0'))  # noqa: S608  # nosec B608
        except sa.exc.OperationalError:
            return False
    return True


def _fetch_random(
    engine: sa.Engine,
    *,
    table: sa.Table,
    n: int,
    seed: int,
    row_count: int,
) -> pl.DataFrame:
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
    with engine.begin() as conn:  # one transaction for setup + main
        for setup_sql, setup_params in q.setup:
            conn.execute(sa.text(setup_sql), setup_params)
        rows = conn.execute(sa.text(q.sql), q.params).mappings().all()
    return pl.DataFrame([dict(r) for r in rows]) if rows else pl.DataFrame()


def _fetch_by_pk(
    engine: sa.Engine,
    *,
    table: TableSchema,
    pk_column: str,
    values: Iterable[Any],
) -> pl.DataFrame:
    md = sa.MetaData()
    sa_table = sa.Table(table.name, md, autoload_with=engine)
    sql = sa_table.select().where(sa_table.c[pk_column].in_(list(values)))
    with engine.connect() as conn:
        rows = conn.execute(sql).mappings().all()
    return (
        pl.DataFrame([dict(r) for r in rows])
        if rows
        else pl.DataFrame(schema={pk_column: pl.Int64})
    )


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
        return _fetch_by_pk(self.engine, table=table, pk_column=pk_column, values=values)


class SampleExtractor:
    """Built-in ``live_db`` train extractor."""

    def extract(self, *, source: str, config: ExtractorConfig) -> SampleResult:
        start = time.perf_counter()
        engine = sa.create_engine(source)
        try:
            schema = introspect(source)
            row_counts = _row_counts(engine, schema)

            n_tables = max(1, len([c for c in row_counts.values() if c > 0]))
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

            samples: dict[str, pl.DataFrame] = {}
            md = sa.MetaData()
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

            cap = config.fk_closure_max_iterations or min(_CLOSURE_HARD_CAP, len(schema.tables))
            report = close_fk_graph(samples, schema, _EngineAdapter(engine), max_iterations=cap)

            samples_dir = config.output_dir / "samples"
            samples_dir.mkdir(parents=True, exist_ok=True)
            table_results: list[TableExtractionResult] = []
            for a in allocations:
                if a.target == 0 or a.table not in samples:
                    continue
                parquet_path = samples_dir / f"{a.table}.parquet"
                samples[a.table].write_parquet(
                    parquet_path, compression="zstd", compression_level=3
                )
                added = report.additions.get(a.table, 0)
                table_results.append(
                    TableExtractionResult(
                        table=a.table,
                        target=a.target,
                        sampled=len(samples[a.table]) - added,
                        fk_closure_added=added,
                        parquet_path=parquet_path,
                    )
                )

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
