"""Unit tests for the dialect-aware random query builder."""

from __future__ import annotations

import pytest
import sqlalchemy as sa

from dbsprout.train.random_select import build_random_query


def _table(name: str = "users") -> sa.Table:
    md = sa.MetaData()
    return sa.Table(name, md, sa.Column("id", sa.Integer, primary_key=True))


def test_postgresql_small_table_uses_order_by_random() -> None:
    q = build_random_query(_table(), n=100, dialect="postgresql", seed=42, row_count=1000)
    assert "ORDER BY random()" in q.sql
    assert ":n" in q.sql
    assert q.params["n"] == 100


def test_postgresql_small_table_returns_setseed_setup_statement() -> None:
    q = build_random_query(_table(), n=100, dialect="postgresql", seed=42, row_count=1000)
    assert len(q.setup) == 1
    setup_sql, setup_params = q.setup[0]
    assert "setseed" in setup_sql
    assert "s" in setup_params
    assert "setseed" not in q.sql
    assert "ORDER BY random()" in q.sql


def test_postgresql_large_table_uses_tablesample() -> None:
    q = build_random_query(_table(), n=100, dialect="postgresql", seed=42, row_count=2_000_000)
    assert "TABLESAMPLE BERNOULLI" in q.sql
    assert "REPEATABLE" in q.sql
    assert q.params["n"] == 100
    assert q.params["seed"] == 42


def test_mysql_uses_rand_with_seed() -> None:
    q = build_random_query(_table(), n=50, dialect="mysql", seed=7, row_count=1000)
    assert "RAND" in q.sql.upper()
    assert "LIMIT" in q.sql.upper()


def test_sqlite_uses_rowid_arithmetic_when_seed_set() -> None:
    q = build_random_query(_table(), n=50, dialect="sqlite", seed=11, row_count=1000)
    assert "rowid" in q.sql.lower()
    assert q.params["n"] == 50


def test_sqlite_seed_zero_does_not_degenerate_to_rowid_order() -> None:
    """Regression: seed=0 must not collapse the LCG multiplier to 1 (silent quality bug)."""
    q0 = build_random_query(_table(), n=50, dialect="sqlite", seed=0, row_count=1000)
    q1 = build_random_query(_table(), n=50, dialect="sqlite", seed=1, row_count=1000)
    # `a` must not be 1 at seed=0 — that would make the ORDER BY monotonic in rowid.
    assert q0.params["a"] != 1
    # And the coefficients at seed=0 vs seed=1 must differ — separate samples per seed.
    assert (q0.params["a"], q0.params["b"]) != (q1.params["a"], q1.params["b"])


def test_sqlite_without_rowid_falls_back_to_random() -> None:
    q = build_random_query(
        _table(), n=50, dialect="sqlite", seed=11, row_count=1000, has_rowid=False
    )
    assert "random()" in q.sql.lower()
    assert "rowid" not in q.sql.lower()
    assert q.warning is not None
    assert "WITHOUT ROWID" in q.warning


def test_unknown_dialect_raises() -> None:
    with pytest.raises(ValueError, match="unsupported dialect"):
        build_random_query(_table(), n=10, dialect="oracle", seed=0, row_count=10)
