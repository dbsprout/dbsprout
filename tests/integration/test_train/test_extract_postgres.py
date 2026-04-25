"""End-to-end PostgreSQL extraction test (skipped without PG_TEST_DSN)."""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING

import pytest
import sqlalchemy as sa

from dbsprout.train.extractor import SampleExtractor
from dbsprout.train.models import ExtractorConfig

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


PG_DSN = os.environ.get("PG_TEST_DSN")

pytestmark = pytest.mark.skipif(
    PG_DSN is None,
    reason="PG_TEST_DSN env var not set; skipping PostgreSQL integration test",
)


@pytest.fixture
def pg_tables() -> Iterator[tuple[str, str, str]]:
    """Yield (dsn, users_table, orders_table) with UUID-suffixed table names so
    parallel test runs against the same DSN don't collide.
    """
    assert PG_DSN is not None
    suffix = uuid.uuid4().hex[:8]
    users_t = f"s062_users_{suffix}"
    orders_t = f"s062_orders_{suffix}"
    engine = sa.create_engine(PG_DSN)
    with engine.begin() as conn:
        conn.execute(sa.text(f'CREATE TABLE "{users_t}" (id INT PRIMARY KEY, name TEXT)'))
        conn.execute(
            sa.text(
                f'CREATE TABLE "{orders_t}" '
                f'(id INT PRIMARY KEY, user_id INT REFERENCES "{users_t}"(id))'
            )
        )
        for i in range(1, 21):
            conn.execute(
                sa.text(f'INSERT INTO "{users_t}" VALUES (:i, :n)'),  # noqa: S608  # nosec B608
                {"i": i, "n": f"u{i}"},
            )
        for i in range(1, 51):
            conn.execute(
                sa.text(f'INSERT INTO "{orders_t}" VALUES (:i, :u)'),  # noqa: S608  # nosec B608
                {"i": i, "u": (i % 20) + 1},
            )
        conn.execute(sa.text(f'ANALYZE "{users_t}"'))
        conn.execute(sa.text(f'ANALYZE "{orders_t}"'))
    engine.dispose()
    yield PG_DSN, users_t, orders_t
    engine = sa.create_engine(PG_DSN)
    with engine.begin() as conn:
        conn.execute(sa.text(f'DROP TABLE IF EXISTS "{orders_t}"'))
        conn.execute(sa.text(f'DROP TABLE IF EXISTS "{users_t}"'))
    engine.dispose()


def test_full_extract_postgres(pg_tables: tuple[str, str, str], tmp_path: Path) -> None:
    dsn, users_t, orders_t = pg_tables
    out = tmp_path / "run"
    cfg = ExtractorConfig(
        sample_rows=20,
        output_dir=out,
        seed=3,
        max_per_table=15,
        quiet=True,
    )
    SampleExtractor().extract(source=dsn, config=cfg)
    assert (out / "samples" / f"{users_t}.parquet").exists()
    assert (out / "samples" / f"{orders_t}.parquet").exists()
