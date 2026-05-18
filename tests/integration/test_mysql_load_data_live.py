"""Integration tests for MySQL LOAD DATA direct insertion.

Requires Docker with MySQL and pymysql. Auto-skips when either unavailable.
"""

from __future__ import annotations

import glob
import random
import tempfile
import time
from typing import Any

import pytest

pytest.importorskip("pymysql", reason="pymysql not installed (pip install dbsprout[db])")

from dbsprout.output.mysql_load_data import MysqlLoadDataWriter, _parse_mysql_url
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    TableSchema,
)

from .conftest import create_mysql_tables, drop_mysql_tables


@pytest.mark.integration
class TestMysqlLoadDataLive:
    """AC: Integration test with Testcontainers MySQL."""

    def test_inserts_correct_row_counts(
        self,
        mysql_url: str,
        test_schema: DatabaseSchema,
        test_rows: dict[str, list[dict[str, Any]]],
    ) -> None:
        """100 users + 500 posts inserted, row counts match."""
        create_mysql_tables(mysql_url, test_schema)
        try:
            MysqlLoadDataWriter().write(test_rows, test_schema, ["users", "posts"], mysql_url)

            import pymysql  # noqa: PLC0415

            params = _parse_mysql_url(mysql_url)
            conn = pymysql.connect(**params)
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM `users`")
                    assert cur.fetchone()[0] == 100
                    cur.execute("SELECT COUNT(*) FROM `posts`")
                    assert cur.fetchone()[0] == 500
            finally:
                conn.close()
        finally:
            drop_mysql_tables(mysql_url, test_schema)

    def test_fk_integrity(
        self,
        mysql_url: str,
        test_schema: DatabaseSchema,
        test_rows: dict[str, list[dict[str, Any]]],
    ) -> None:
        """All FK references resolve via JOIN."""
        create_mysql_tables(mysql_url, test_schema)
        try:
            MysqlLoadDataWriter().write(test_rows, test_schema, ["users", "posts"], mysql_url)

            import pymysql  # noqa: PLC0415

            params = _parse_mysql_url(mysql_url)
            conn = pymysql.connect(**params)
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT COUNT(*) FROM `posts` p JOIN `users` u ON p.`user_id` = u.`id`"
                    )
                    assert cur.fetchone()[0] == 500
            finally:
                conn.close()
        finally:
            drop_mysql_tables(mysql_url, test_schema)

    def test_temp_files_cleaned(
        self,
        mysql_url: str,
        test_schema: DatabaseSchema,
        test_rows: dict[str, list[dict[str, Any]]],
    ) -> None:
        """No .tsv temp files remain after write completes."""
        tmp_dir = tempfile.gettempdir()
        before = set(glob.glob(f"{tmp_dir}/*.tsv"))

        create_mysql_tables(mysql_url, test_schema)
        try:
            MysqlLoadDataWriter().write(test_rows, test_schema, ["users", "posts"], mysql_url)

            after = set(glob.glob(f"{tmp_dir}/*.tsv"))
            new_files = after - before
            assert len(new_files) == 0, f"Leftover temp files: {new_files}"
        finally:
            drop_mysql_tables(mysql_url, test_schema)

    def test_returns_insert_result(
        self,
        mysql_url: str,
        test_schema: DatabaseSchema,
        test_rows: dict[str, list[dict[str, Any]]],
    ) -> None:
        """InsertResult has correct counts and positive duration."""
        create_mysql_tables(mysql_url, test_schema)
        try:
            result = MysqlLoadDataWriter().write(
                test_rows, test_schema, ["users", "posts"], mysql_url
            )
            assert result.tables_inserted == 2
            assert result.total_rows == 600
            assert result.duration_seconds > 0
        finally:
            drop_mysql_tables(mysql_url, test_schema)

    def test_fk_checks_reenabled(
        self,
        mysql_url: str,
        test_schema: DatabaseSchema,
        test_rows: dict[str, list[dict[str, Any]]],
    ) -> None:
        """After write, FK checks are re-enabled on the connection."""
        create_mysql_tables(mysql_url, test_schema)
        try:
            MysqlLoadDataWriter().write(test_rows, test_schema, ["users", "posts"], mysql_url)

            import pymysql  # noqa: PLC0415

            params = _parse_mysql_url(mysql_url)
            conn = pymysql.connect(**params)
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT @@foreign_key_checks")
                    assert cur.fetchone()[0] == 1
            finally:
                conn.close()
        finally:
            drop_mysql_tables(mysql_url, test_schema)


@pytest.mark.integration
class TestMysqlLoadDataUpsertLive:
    """S-094-F1: upsert via TEMPORARY staging — re-run updates, not duplicates."""

    def test_rerun_updates_not_duplicates(
        self,
        mysql_url: str,
        test_schema: DatabaseSchema,
    ) -> None:
        """Second run with upsert=True updates the row, count stays 1."""
        create_mysql_tables(mysql_url, test_schema)
        try:
            import pymysql  # noqa: PLC0415

            first = {"users": [{"id": 1, "email": "old@test.com"}], "posts": []}
            MysqlLoadDataWriter().write(
                first, test_schema, ["users", "posts"], mysql_url, upsert=True
            )
            second = {"users": [{"id": 1, "email": "new@test.com"}], "posts": []}
            MysqlLoadDataWriter().write(
                second, test_schema, ["users", "posts"], mysql_url, upsert=True
            )

            params = _parse_mysql_url(mysql_url)
            conn = pymysql.connect(**params)
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM `users`")
                    assert cur.fetchone()[0] == 1
                    cur.execute("SELECT `email` FROM `users` WHERE `id` = 1")
                    assert cur.fetchone()[0] == "new@test.com"
                    cur.execute(
                        "SELECT COUNT(*) FROM information_schema.tables "
                        "WHERE table_name LIKE '_dbsprout_stg_%'"
                    )
                    assert cur.fetchone()[0] == 0
            finally:
                conn.close()
        finally:
            drop_mysql_tables(mysql_url, test_schema)

    def test_no_orphan_staging_after_success(
        self,
        mysql_url: str,
        test_schema: DatabaseSchema,
    ) -> None:
        """Staging table is dropped after a successful upsert run.

        The failure-path drop (``finally``) is covered deterministically by
        the unit test ``test_upsert_drops_staging_on_merge_failure``; here we
        assert the live success invariant (MySQL ``FOREIGN_KEY_CHECKS=0`` in
        the writer means an FK violation does not raise, so a forced failure
        is exercised only at the unit level).
        """
        create_mysql_tables(mysql_url, test_schema)
        try:
            import pymysql  # noqa: PLC0415

            rows = {"users": [{"id": 1, "email": "u@test.com"}], "posts": []}
            MysqlLoadDataWriter().write(
                rows, test_schema, ["users", "posts"], mysql_url, upsert=True
            )

            params = _parse_mysql_url(mysql_url)
            conn = pymysql.connect(**params)
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT COUNT(*) FROM information_schema.tables "
                        "WHERE table_name LIKE '_dbsprout_stg_%'"
                    )
                    assert cur.fetchone()[0] == 0
            finally:
                conn.close()
        finally:
            drop_mysql_tables(mysql_url, test_schema)


@pytest.mark.integration
@pytest.mark.slow
def test_mysql_load_data_throughput_5k_rows_per_sec(mysql_url: str) -> None:
    """AC: MySQL LOAD DATA >=5K rows/sec (CI-safe; production target 80K+)."""
    _col = lambda name, **kw: ColumnSchema(  # noqa: E731
        name=name, data_type=ColumnType.INTEGER, **kw
    )
    schema = DatabaseSchema(
        tables=[
            TableSchema(
                name="bench",
                columns=[
                    _col("id", primary_key=True, autoincrement=True, nullable=False),
                    _col("a"),
                    _col("b"),
                    _col("c"),
                    ColumnSchema(
                        name="label",
                        data_type=ColumnType.VARCHAR,
                        max_length=100,
                    ),
                ],
                primary_key=["id"],
            )
        ]
    )

    rng = random.Random(42)  # noqa: S311
    n_rows = 50_000
    rows = [
        {
            "id": i + 1,
            "a": rng.randint(0, 1_000_000),
            "b": rng.randint(0, 1_000_000),
            "c": rng.randint(0, 1_000_000),
            "label": f"row_{i + 1}",
        }
        for i in range(n_rows)
    ]

    create_mysql_tables(mysql_url, schema)
    try:
        start = time.monotonic()
        MysqlLoadDataWriter().write({"bench": rows}, schema, ["bench"], mysql_url)
        elapsed = time.monotonic() - start

        throughput = n_rows / elapsed
        assert throughput >= 5_000, (
            f"MySQL LOAD DATA throughput {throughput:.0f} rows/sec < 5K threshold"
        )
    finally:
        drop_mysql_tables(mysql_url, schema)
