"""End-to-end MySQL extraction test (skipped without MYSQL_TEST_DSN)."""

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


MYSQL_DSN = os.environ.get("MYSQL_TEST_DSN")

pytestmark = pytest.mark.skipif(
    MYSQL_DSN is None,
    reason="MYSQL_TEST_DSN env var not set; skipping MySQL integration test",
)


@pytest.fixture
def mysql_tables() -> Iterator[tuple[str, str, str]]:
    """Yield (dsn, users_table, orders_table) with UUID-suffixed names so
    parallel runs against the same DSN don't collide.
    """
    assert MYSQL_DSN is not None
    suffix = uuid.uuid4().hex[:8]
    users_t = f"s062_users_{suffix}"
    orders_t = f"s062_orders_{suffix}"
    engine = sa.create_engine(MYSQL_DSN)
    with engine.begin() as conn:
        conn.execute(sa.text(f"CREATE TABLE `{users_t}` (id INT PRIMARY KEY, name VARCHAR(64))"))
        conn.execute(
            sa.text(
                f"CREATE TABLE `{orders_t}` (id INT PRIMARY KEY, user_id INT, "
                f"FOREIGN KEY (user_id) REFERENCES `{users_t}`(id))"
            )
        )
        for i in range(1, 21):
            conn.execute(
                sa.text(f"INSERT INTO `{users_t}` VALUES (:i, :n)"),  # noqa: S608  # nosec B608
                {"i": i, "n": f"u{i}"},
            )
        for i in range(1, 51):
            conn.execute(
                sa.text(f"INSERT INTO `{orders_t}` VALUES (:i, :u)"),  # noqa: S608  # nosec B608
                {"i": i, "u": (i % 20) + 1},
            )
    engine.dispose()
    yield MYSQL_DSN, users_t, orders_t
    engine = sa.create_engine(MYSQL_DSN)
    with engine.begin() as conn:
        conn.execute(sa.text(f"DROP TABLE IF EXISTS `{orders_t}`"))
        conn.execute(sa.text(f"DROP TABLE IF EXISTS `{users_t}`"))
    engine.dispose()


def test_full_extract_mysql(mysql_tables: tuple[str, str, str], tmp_path: Path) -> None:
    dsn, users_t, orders_t = mysql_tables
    out = tmp_path / "run"
    cfg = ExtractorConfig(
        db_url=dsn,
        sample_rows=20,
        output_dir=out,
        seed=3,
        max_per_table=15,
        quiet=True,
    )
    SampleExtractor().extract(source=dsn, config=cfg)
    assert (out / "samples" / f"{users_t}.parquet").exists()
    assert (out / "samples" / f"{orders_t}.parquet").exists()
