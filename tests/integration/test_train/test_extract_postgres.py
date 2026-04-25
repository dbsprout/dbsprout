"""End-to-end PostgreSQL extraction test (skipped without PG_TEST_DSN)."""

from __future__ import annotations

import os
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
def pg_url() -> Iterator[str]:
    assert PG_DSN is not None
    engine = sa.create_engine(PG_DSN)
    with engine.begin() as conn:
        conn.execute(sa.text("DROP TABLE IF EXISTS s062_orders"))
        conn.execute(sa.text("DROP TABLE IF EXISTS s062_users"))
        conn.execute(sa.text("CREATE TABLE s062_users (id INT PRIMARY KEY, name TEXT)"))
        conn.execute(
            sa.text(
                "CREATE TABLE s062_orders "
                "(id INT PRIMARY KEY, user_id INT REFERENCES s062_users(id))"
            )
        )
        for i in range(1, 21):
            conn.execute(
                sa.text("INSERT INTO s062_users VALUES (:i, :n)"),
                {"i": i, "n": f"u{i}"},
            )
        for i in range(1, 51):
            conn.execute(
                sa.text("INSERT INTO s062_orders VALUES (:i, :u)"),
                {"i": i, "u": (i % 20) + 1},
            )
        conn.execute(sa.text("ANALYZE s062_users"))
        conn.execute(sa.text("ANALYZE s062_orders"))
    engine.dispose()
    yield PG_DSN
    engine = sa.create_engine(PG_DSN)
    with engine.begin() as conn:
        conn.execute(sa.text("DROP TABLE s062_orders"))
        conn.execute(sa.text("DROP TABLE s062_users"))
    engine.dispose()


def test_full_extract_postgres(pg_url: str, tmp_path: Path) -> None:
    out = tmp_path / "run"
    cfg = ExtractorConfig(
        db_url=pg_url,
        sample_rows=20,
        output_dir=out,
        seed=3,
        max_per_table=15,
        quiet=True,
    )
    SampleExtractor().extract(source=pg_url, config=cfg)
    assert (out / "samples" / "s062_users.parquet").exists()
    assert (out / "samples" / "s062_orders.parquet").exists()
