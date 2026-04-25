"""End-to-end MySQL extraction test (skipped without MYSQL_TEST_DSN)."""

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


MYSQL_DSN = os.environ.get("MYSQL_TEST_DSN")

pytestmark = pytest.mark.skipif(
    MYSQL_DSN is None,
    reason="MYSQL_TEST_DSN env var not set; skipping MySQL integration test",
)


@pytest.fixture
def mysql_url() -> Iterator[str]:
    assert MYSQL_DSN is not None
    engine = sa.create_engine(MYSQL_DSN)
    with engine.begin() as conn:
        conn.execute(sa.text("DROP TABLE IF EXISTS s062_orders"))
        conn.execute(sa.text("DROP TABLE IF EXISTS s062_users"))
        conn.execute(sa.text("CREATE TABLE s062_users (id INT PRIMARY KEY, name VARCHAR(64))"))
        conn.execute(
            sa.text(
                "CREATE TABLE s062_orders (id INT PRIMARY KEY, user_id INT, "
                "FOREIGN KEY (user_id) REFERENCES s062_users(id))"
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
    engine.dispose()
    yield MYSQL_DSN
    engine = sa.create_engine(MYSQL_DSN)
    with engine.begin() as conn:
        conn.execute(sa.text("DROP TABLE s062_orders"))
        conn.execute(sa.text("DROP TABLE s062_users"))
    engine.dispose()


def test_full_extract_mysql(mysql_url: str, tmp_path: Path) -> None:
    out = tmp_path / "run"
    cfg = ExtractorConfig(
        db_url=mysql_url,
        sample_rows=20,
        output_dir=out,
        seed=3,
        max_per_table=15,
        quiet=True,
    )
    SampleExtractor().extract(source=mysql_url, config=cfg)
    assert (out / "samples" / "s062_users.parquet").exists()
