"""SQLite state layer for DBSprout.

The CLI writes generation-run telemetry to ``.dbsprout/state.db`` using
stdlib :mod:`sqlite3` (zero extra dependencies). Visual surfaces (HTML
report, TUI, web dashboard) read from the same file -- zero coupling
between the CLI and GUI code.

Concurrency: WAL journal mode is enabled on every connection so a GUI can
read while the CLI writes. ``record_run`` is fully transactional: a partial
failure rolls back the whole run (no orphaned ``runs`` row).

Schema migration is intentionally lightweight -- ``PRAGMA user_version``
plus idempotent ``CREATE TABLE IF NOT EXISTS`` and additive
``ALTER TABLE ... ADD COLUMN`` guarded by ``PRAGMA table_info``. Old
databases open without data loss; newly added columns appear with their
default value.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from dbsprout.state.models import LLMCall, QualityResult, RunRecord, TableStats

if TYPE_CHECKING:
    from collections.abc import Iterator

#: Current state-DB schema version. Bump when adding columns/tables and add
#: the corresponding additive step to ``_EXPECTED_COLUMNS``.
SCHEMA_VERSION = 1

_CREATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    duration_ms INTEGER,
    engine TEXT NOT NULL,
    llm_provider TEXT,
    llm_model TEXT,
    total_rows INTEGER,
    total_tables INTEGER,
    seed INTEGER,
    config_json TEXT
);

CREATE TABLE IF NOT EXISTS table_stats (
    id INTEGER PRIMARY KEY,
    run_id INTEGER REFERENCES runs(id),
    table_name TEXT NOT NULL,
    row_count INTEGER,
    generation_ms INTEGER,
    rows_per_sec REAL,
    errors INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS quality_results (
    id INTEGER PRIMARY KEY,
    run_id INTEGER REFERENCES runs(id),
    metric_type TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    score REAL,
    passed BOOLEAN,
    details_json TEXT
);

CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY,
    run_id INTEGER REFERENCES runs(id),
    timestamp TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    tokens_sent INTEGER,
    tokens_received INTEGER,
    cost_usd REAL,
    privacy_tier TEXT,
    cached BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs(started_at);
CREATE INDEX IF NOT EXISTS idx_table_stats_run ON table_stats(run_id);
CREATE INDEX IF NOT EXISTS idx_quality_run ON quality_results(run_id);
CREATE INDEX IF NOT EXISTS idx_llm_calls_run ON llm_calls(run_id);
"""

# Columns that must exist on each table at ``SCHEMA_VERSION``. Used to
# additively migrate older databases (``ALTER TABLE ... ADD COLUMN``).
_EXPECTED_COLUMNS: dict[str, dict[str, str]] = {
    "runs": {
        "started_at": "TEXT",
        "completed_at": "TEXT",
        "duration_ms": "INTEGER",
        "engine": "TEXT",
        "llm_provider": "TEXT",
        "llm_model": "TEXT",
        "total_rows": "INTEGER",
        "total_tables": "INTEGER",
        "seed": "INTEGER",
        "config_json": "TEXT",
    },
}


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


class StateDB:
    """Persistent SQLite store for generation-run telemetry."""

    def __init__(self, db_path: Path | str = ".dbsprout/state.db") -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            self._ensure_schema(conn)

    # ── connection ──────────────────────────────────────────────────

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
        finally:
            conn.close()

    # ── schema / migration ──────────────────────────────────────────

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(_CREATE_SCHEMA)
        self._migrate(conn)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Additively bring an older database up to ``SCHEMA_VERSION``.

        Idempotent: only columns missing from a pre-existing table are
        added; existing rows and data are never dropped.
        """
        for table, expected in _EXPECTED_COLUMNS.items():
            existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
            for column, decl in expected.items():
                if column not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    # ── introspection helpers ───────────────────────────────────────

    def schema_version(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("PRAGMA user_version").fetchone()[0])

    def journal_mode(self) -> str:
        with self._connect() as conn:
            return str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()

    # ── writes ──────────────────────────────────────────────────────

    def record_run(self, run: RunRecord) -> int:
        """Persist a run plus all child records atomically.

        Returns the autoincrement ``runs.id``. Any failure (including a
        malformed child) rolls back the entire run.
        """
        with self._connect() as conn:
            try:
                run_id = self._insert_run(conn, run)
                self._insert_table_stats(conn, run_id, run.table_stats)
                self._insert_quality_results(conn, run_id, run.quality_results)
                self._insert_llm_calls(conn, run_id, run.llm_calls)
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
        return run_id

    @staticmethod
    def _insert_run(conn: sqlite3.Connection, run: RunRecord) -> int:
        cur = conn.execute(
            """
            INSERT INTO runs (
                started_at, completed_at, duration_ms, engine,
                llm_provider, llm_model, total_rows, total_tables,
                seed, config_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _iso(run.started_at),
                _iso(run.completed_at),
                run.duration_ms,
                run.engine,
                run.llm_provider,
                run.llm_model,
                run.total_rows,
                run.total_tables,
                run.seed,
                run.config_json,
            ),
        )
        return int(cur.lastrowid or 0)

    @staticmethod
    def _insert_table_stats(conn: sqlite3.Connection, run_id: int, stats: list[TableStats]) -> None:
        conn.executemany(
            """
            INSERT INTO table_stats (
                run_id, table_name, row_count, generation_ms,
                rows_per_sec, errors
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    s.table_name,
                    s.row_count,
                    s.generation_ms,
                    s.rows_per_sec,
                    s.errors,
                )
                for s in stats
            ],
        )

    @staticmethod
    def _insert_quality_results(
        conn: sqlite3.Connection, run_id: int, results: list[QualityResult]
    ) -> None:
        conn.executemany(
            """
            INSERT INTO quality_results (
                run_id, metric_type, metric_name, score, passed,
                details_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    r.metric_type,
                    r.metric_name,
                    r.score,
                    r.passed,
                    r.details_json,
                )
                for r in results
            ],
        )

    @staticmethod
    def _insert_llm_calls(conn: sqlite3.Connection, run_id: int, calls: list[LLMCall]) -> None:
        conn.executemany(
            """
            INSERT INTO llm_calls (
                run_id, timestamp, provider, model, tokens_sent,
                tokens_received, cost_usd, privacy_tier, cached
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    _iso(c.timestamp),
                    c.provider,
                    c.model,
                    c.tokens_sent,
                    c.tokens_received,
                    c.cost_usd,
                    c.privacy_tier,
                    c.cached,
                )
                for c in calls
            ],
        )

    # ── reads ───────────────────────────────────────────────────────

    def get_runs(self) -> list[RunRecord]:
        """Return all runs, newest first (by ``started_at``)."""
        with self._connect() as conn:
            run_rows = conn.execute(
                "SELECT * FROM runs ORDER BY started_at DESC, id DESC"
            ).fetchall()
            return [self._hydrate_run(conn, row) for row in run_rows]

    @staticmethod
    def _hydrate_run(conn: sqlite3.Connection, row: sqlite3.Row) -> RunRecord:
        run_id = row["id"]
        ts_rows = conn.execute(
            "SELECT * FROM table_stats WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()
        qr_rows = conn.execute(
            "SELECT * FROM quality_results WHERE run_id = ? ORDER BY id",
            (run_id,),
        ).fetchall()
        call_rows = conn.execute(
            "SELECT * FROM llm_calls WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()
        return RunRecord(
            id=run_id,
            started_at=datetime.fromisoformat(row["started_at"]),
            completed_at=(
                datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None
            ),
            duration_ms=row["duration_ms"],
            engine=row["engine"],
            llm_provider=row["llm_provider"],
            llm_model=row["llm_model"],
            total_rows=row["total_rows"] or 0,
            total_tables=row["total_tables"] or 0,
            seed=row["seed"],
            config_json=row["config_json"],
            table_stats=[
                TableStats(
                    id=r["id"],
                    run_id=r["run_id"],
                    table_name=r["table_name"],
                    row_count=r["row_count"] or 0,
                    generation_ms=r["generation_ms"] or 0,
                    rows_per_sec=r["rows_per_sec"] or 0.0,
                    errors=r["errors"] or 0,
                )
                for r in ts_rows
            ],
            quality_results=[
                QualityResult(
                    id=r["id"],
                    run_id=r["run_id"],
                    metric_type=r["metric_type"],
                    metric_name=r["metric_name"],
                    score=r["score"] or 0.0,
                    passed=bool(r["passed"]),
                    details_json=r["details_json"],
                )
                for r in qr_rows
            ],
            llm_calls=[
                LLMCall(
                    id=r["id"],
                    run_id=r["run_id"],
                    timestamp=datetime.fromisoformat(r["timestamp"]),
                    provider=r["provider"],
                    model=r["model"],
                    tokens_sent=r["tokens_sent"] or 0,
                    tokens_received=r["tokens_received"] or 0,
                    cost_usd=r["cost_usd"] or 0.0,
                    privacy_tier=r["privacy_tier"],
                    cached=bool(r["cached"]),
                )
                for r in call_rows
            ],
        )
