"""Targeted unit tests for extractor helper functions to lift branch coverage."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import polars as pl

from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    TableSchema,
)
from dbsprout.train.extractor import (
    SampleExtractor,
    _has_rowid,
    _row_counts,
)
from dbsprout.train.models import ExtractorConfig

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _single_table_schema() -> DatabaseSchema:
    return DatabaseSchema(
        dialect="sqlite",
        tables=[
            TableSchema(
                name="users",
                columns=[
                    ColumnSchema(
                        name="id", data_type=ColumnType.INTEGER, nullable=False, primary_key=True
                    )
                ],
                primary_key=["id"],
                foreign_keys=[],
            )
        ],
    )


def _stub_engine(dialect: str, scalar_value: object) -> MagicMock:
    engine = MagicMock()
    engine.dialect.name = dialect
    conn = MagicMock()
    conn.execute.return_value.scalar.return_value = scalar_value
    engine.connect.return_value.__enter__.return_value = conn
    return engine


def test_row_counts_postgresql_uses_reltuples() -> None:
    engine = _stub_engine("postgresql", 4242)
    counts = _row_counts(engine, _single_table_schema())
    assert counts == {"users": 4242}


def test_row_counts_postgresql_zero_reltuples_treated_as_empty() -> None:
    engine = _stub_engine("postgresql", 0)
    counts = _row_counts(engine, _single_table_schema())
    assert counts == {"users": 0}


def test_row_counts_mysql_uses_information_schema() -> None:
    engine = _stub_engine("mysql", 1234)
    counts = _row_counts(engine, _single_table_schema())
    assert counts == {"users": 1234}


def test_row_counts_mysql_null_treated_as_zero() -> None:
    engine = _stub_engine("mysql", None)
    counts = _row_counts(engine, _single_table_schema())
    assert counts == {"users": 0}


def test_has_rowid_returns_true_for_normal_table() -> None:
    engine = _stub_engine("sqlite", "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    assert _has_rowid(engine, "users") is True


def test_has_rowid_returns_false_for_without_rowid_table() -> None:
    engine = _stub_engine("sqlite", "CREATE TABLE users (id INTEGER PRIMARY KEY) WITHOUT ROWID")
    assert _has_rowid(engine, "users") is False


def test_has_rowid_returns_true_when_table_missing() -> None:
    """Defensive: missing DDL → assume rowid (standard SQLite default)."""
    engine = _stub_engine("sqlite", None)
    assert _has_rowid(engine, "users") is True


def test_extractor_warns_when_target_exceeds_memory_threshold(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Target above 500_000 rows triggers the memory-pressure warning."""
    schema = _single_table_schema()
    cfg = ExtractorConfig(
        db_url="sqlite:///:memory:",
        sample_rows=600_000,
        output_dir=tmp_path,
        seed=1,
        max_per_table=600_000,
        quiet=True,
    )

    def fake_table_factory(name: str, *_a: object, **_k: object) -> MagicMock:
        m = MagicMock()
        m.name = name
        return m

    with (
        patch("dbsprout.train.extractor.introspect", return_value=schema),
        patch(
            "dbsprout.train.extractor._row_counts",
            return_value={"users": 10_000_000},
        ),
        patch(
            "dbsprout.train.extractor._fetch_random",
            return_value=pl.DataFrame({"id": list(range(10))}),
        ),
        patch("dbsprout.train.extractor._fetch_by_pk", return_value=pl.DataFrame()),
        patch("dbsprout.train.extractor.sa.create_engine") as ce,
        patch("dbsprout.train.extractor.sa.Table", side_effect=fake_table_factory),
    ):
        ce.return_value.dialect.name = "sqlite"
        with caplog.at_level("WARNING", logger="dbsprout.train.extractor"):
            SampleExtractor().extract(source=cfg.db_url, config=cfg)

    assert any("memory" in r.message.lower() for r in caplog.records)
