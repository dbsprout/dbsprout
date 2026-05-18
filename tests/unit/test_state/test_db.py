"""Tests for dbsprout.state.db — SQLite state layer."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import pytest

from dbsprout.state.db import SCHEMA_VERSION, StateDB
from dbsprout.state.models import LLMCall, QualityResult, RunRecord, TableStats

if TYPE_CHECKING:
    from pathlib import Path

_T0 = datetime(2026, 5, 16, 12, 0, 0, tzinfo=timezone.utc)


def _table_names(db_path: Path) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    finally:
        conn.close()
    return {r[0] for r in rows}


def _make_run(offset_minutes: int = 0, engine: str = "heuristic") -> RunRecord:
    start = _T0 + timedelta(minutes=offset_minutes)
    return RunRecord(
        started_at=start,
        completed_at=start + timedelta(seconds=5),
        duration_ms=5000,
        engine=engine,
        llm_provider="ollama",
        llm_model="qwen2.5",
        total_rows=150,
        total_tables=2,
        seed=42,
        config_json=f'{{"engine": "{engine}"}}',
        table_stats=[
            TableStats(table_name="users", row_count=100, generation_ms=10),
            TableStats(table_name="orders", row_count=50, errors=1),
        ],
        quality_results=[
            QualityResult(
                metric_type="integrity",
                metric_name="fk_valid",
                score=1.0,
                passed=True,
                details_json='{"violations": 0}',
            )
        ],
        llm_calls=[
            LLMCall(
                timestamp=start,
                provider="ollama",
                model="qwen2.5",
                tokens_sent=200,
                tokens_received=80,
                cost_usd=0.0,
                privacy_tier="local",
                cached=True,
            )
        ],
    )


# ── Task 2: schema creation + WAL ───────────────────────────────────


class TestSchema:
    def test_creates_db_file_and_tables(self, tmp_path: Path) -> None:
        db_path = tmp_path / "nested" / "state.db"
        StateDB(db_path)
        assert db_path.exists()
        assert _table_names(db_path) >= {
            "runs",
            "table_stats",
            "quality_results",
            "llm_calls",
        }

    def test_wal_mode_enabled(self, tmp_path: Path) -> None:
        db_path = tmp_path / "state.db"
        db = StateDB(db_path)
        assert db.journal_mode() == "wal"

    def test_user_version_set(self, tmp_path: Path) -> None:
        db_path = tmp_path / "state.db"
        db = StateDB(db_path)
        assert db.schema_version() == SCHEMA_VERSION

    def test_accepts_str_path(self, tmp_path: Path) -> None:
        db_path = tmp_path / "state.db"
        StateDB(str(db_path))
        assert db_path.exists()


# ── Task 3: record_run ──────────────────────────────────────────────


class TestRecordRun:
    def test_returns_run_id_and_persists_children(self, tmp_path: Path) -> None:
        db = StateDB(tmp_path / "state.db")
        run_id = db.record_run(_make_run())
        assert isinstance(run_id, int)
        assert run_id > 0

        runs = db.get_runs()
        assert len(runs) == 1
        rec = runs[0]
        assert rec.id == run_id
        assert rec.engine == "heuristic"
        assert len(rec.table_stats) == 2
        assert len(rec.quality_results) == 1
        assert len(rec.llm_calls) == 1
        assert all(ts.run_id == run_id for ts in rec.table_stats)
        assert all(qr.run_id == run_id for qr in rec.quality_results)
        assert all(c.run_id == run_id for c in rec.llm_calls)

    def test_roundtrip_field_fidelity(self, tmp_path: Path) -> None:
        db = StateDB(tmp_path / "state.db")
        db.record_run(_make_run())
        rec = db.get_runs()[0]
        assert rec.total_rows == 150
        assert rec.total_tables == 2
        assert rec.seed == 42
        assert rec.completed_at is not None
        assert rec.duration_ms == 5000
        ts = {t.table_name: t for t in rec.table_stats}
        assert ts["orders"].errors == 1
        assert rec.quality_results[0].passed is True
        assert rec.llm_calls[0].cached is True
        assert rec.llm_calls[0].timestamp == _T0

    def test_bad_child_rolls_back_run(self, tmp_path: Path) -> None:
        db = StateDB(tmp_path / "state.db")

        class Boom:
            def __getattr__(self, name: str) -> object:
                raise RuntimeError("boom")

        bad = _make_run().model_copy(update={"llm_calls": [Boom()]})
        with pytest.raises(RuntimeError, match="boom"):
            db.record_run(bad)  # type: ignore[arg-type]
        assert db.get_runs() == []


# ── Task 4: get_runs ordering ───────────────────────────────────────


class TestGetRuns:
    def test_empty(self, tmp_path: Path) -> None:
        db = StateDB(tmp_path / "state.db")
        assert db.get_runs() == []

    def test_ordered_newest_first(self, tmp_path: Path) -> None:
        db = StateDB(tmp_path / "state.db")
        db.record_run(_make_run(offset_minutes=0, engine="a"))
        db.record_run(_make_run(offset_minutes=10, engine="b"))
        db.record_run(_make_run(offset_minutes=5, engine="c"))
        runs = db.get_runs()
        assert [r.engine for r in runs] == ["b", "c", "a"]


# ── Task 5: persistence + graceful migration ────────────────────────


class TestPersistenceAndMigration:
    def test_survives_reopen(self, tmp_path: Path) -> None:
        db_path = tmp_path / "state.db"
        db1 = StateDB(db_path)
        rid = db1.record_run(_make_run())
        db2 = StateDB(db_path)
        runs = db2.get_runs()
        assert len(runs) == 1
        assert runs[0].id == rid

    def test_graceful_migration_adds_missing_column(self, tmp_path: Path) -> None:
        db_path = tmp_path / "state.db"
        # Simulate an "old" v0 DB: runs table missing the `seed` column,
        # user_version below current.
        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE runs (
                id INTEGER PRIMARY KEY,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                duration_ms INTEGER,
                engine TEXT NOT NULL,
                llm_provider TEXT,
                llm_model TEXT,
                total_rows INTEGER,
                total_tables INTEGER,
                config_json TEXT
            );
            CREATE TABLE table_stats (
                id INTEGER PRIMARY KEY,
                run_id INTEGER REFERENCES runs(id),
                table_name TEXT NOT NULL,
                row_count INTEGER,
                generation_ms INTEGER,
                rows_per_sec REAL,
                errors INTEGER DEFAULT 0
            );
            CREATE TABLE quality_results (
                id INTEGER PRIMARY KEY,
                run_id INTEGER REFERENCES runs(id),
                metric_type TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                score REAL,
                passed BOOLEAN,
                details_json TEXT
            );
            CREATE TABLE llm_calls (
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
            INSERT INTO runs (started_at, engine, total_rows, total_tables)
            VALUES ('2026-01-01T00:00:00+00:00', 'legacy', 7, 1);
            PRAGMA user_version = 0;
            """
        )
        conn.commit()
        conn.close()

        db = StateDB(db_path)
        assert db.schema_version() == SCHEMA_VERSION
        cols = {
            r[1] for r in sqlite3.connect(db_path).execute("PRAGMA table_info(runs)").fetchall()
        }
        assert "seed" in cols
        # Legacy row preserved, new column NULL → model default.
        runs = db.get_runs()
        assert len(runs) == 1
        assert runs[0].engine == "legacy"
        assert runs[0].total_rows == 7
        assert runs[0].seed is None
        # New writes still work post-migration.
        db.record_run(_make_run())
        assert len(db.get_runs()) == 2
