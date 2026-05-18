"""Integration tests for PostgreSQL COPY direct insertion.

Requires Docker with PostgreSQL and psycopg3. Auto-skips when either unavailable.
"""

from __future__ import annotations

import random
import time
from typing import Any

import pytest

psycopg = pytest.importorskip("psycopg", reason="psycopg3 not installed (pip install dbsprout[pg])")

from dbsprout.output.pg_copy import PgCopyWriter  # noqa: E402
from dbsprout.schema.models import (  # noqa: E402
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    TableSchema,
)

from .conftest import create_pg_tables, drop_pg_tables  # noqa: E402


@pytest.mark.integration
class TestPgCopyLive:
    """AC: Integration test with Testcontainers PostgreSQL."""

    def test_inserts_correct_row_counts(
        self,
        pg_url: str,
        test_schema: DatabaseSchema,
        test_rows: dict[str, list[dict[str, Any]]],
    ) -> None:
        """100 users + 500 posts inserted, row counts match."""
        create_pg_tables(pg_url, test_schema)
        try:
            PgCopyWriter().write(test_rows, test_schema, ["users", "posts"], pg_url)

            import psycopg  # noqa: PLC0415

            with psycopg.connect(pg_url) as conn, conn.cursor() as cur:
                cur.execute('SELECT COUNT(*) FROM "users"')
                assert cur.fetchone()[0] == 100
                cur.execute('SELECT COUNT(*) FROM "posts"')
                assert cur.fetchone()[0] == 500
        finally:
            drop_pg_tables(pg_url, test_schema)

    def test_fk_integrity(
        self,
        pg_url: str,
        test_schema: DatabaseSchema,
        test_rows: dict[str, list[dict[str, Any]]],
    ) -> None:
        """All FK references resolve via JOIN."""
        create_pg_tables(pg_url, test_schema)
        try:
            PgCopyWriter().write(test_rows, test_schema, ["users", "posts"], pg_url)

            import psycopg  # noqa: PLC0415

            with psycopg.connect(pg_url) as conn, conn.cursor() as cur:
                cur.execute('SELECT COUNT(*) FROM "posts" p JOIN "users" u ON p."user_id" = u."id"')
                assert cur.fetchone()[0] == 500
        finally:
            drop_pg_tables(pg_url, test_schema)

    def test_sequence_reset(
        self,
        pg_url: str,
        test_schema: DatabaseSchema,
        test_rows: dict[str, list[dict[str, Any]]],
    ) -> None:
        """After COPY, sequence value matches MAX(id)."""
        create_pg_tables(pg_url, test_schema)
        try:
            PgCopyWriter().write(test_rows, test_schema, ["users", "posts"], pg_url)

            import psycopg  # noqa: PLC0415

            with psycopg.connect(pg_url) as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT last_value FROM pg_sequences "
                    "WHERE sequencename = pg_get_serial_sequence('users', 'id')::regclass::text"
                )
                seq_val = cur.fetchone()[0]
                assert seq_val == 100
        finally:
            drop_pg_tables(pg_url, test_schema)

    def test_returns_insert_result(
        self,
        pg_url: str,
        test_schema: DatabaseSchema,
        test_rows: dict[str, list[dict[str, Any]]],
    ) -> None:
        """InsertResult has correct counts and positive duration."""
        create_pg_tables(pg_url, test_schema)
        try:
            result = PgCopyWriter().write(test_rows, test_schema, ["users", "posts"], pg_url)
            assert result.tables_inserted == 2
            assert result.total_rows == 600
            assert result.duration_seconds > 0
        finally:
            drop_pg_tables(pg_url, test_schema)

    def test_empty_table_no_error(
        self,
        pg_url: str,
        test_schema: DatabaseSchema,
    ) -> None:
        """Insert with 0 rows for a table succeeds without error."""
        create_pg_tables(pg_url, test_schema)
        try:
            empty_rows: dict[str, list[dict[str, Any]]] = {
                "users": [{"id": 1, "email": "solo@test.com"}],
                "posts": [],
            }
            result = PgCopyWriter().write(empty_rows, test_schema, ["users", "posts"], pg_url)
            assert result.tables_inserted == 1
            assert result.total_rows == 1
        finally:
            drop_pg_tables(pg_url, test_schema)


@pytest.mark.integration
class TestPgCopyUpsertLive:
    """S-094-F1: upsert via TEMP staging table — re-run updates, not duplicates."""

    def test_rerun_updates_not_duplicates(
        self,
        pg_url: str,
        test_schema: DatabaseSchema,
    ) -> None:
        """Second run with upsert=True updates the row, count stays 1."""
        create_pg_tables(pg_url, test_schema)
        try:
            import psycopg  # noqa: PLC0415

            first = {"users": [{"id": 1, "email": "old@test.com"}], "posts": []}
            PgCopyWriter().write(first, test_schema, ["users", "posts"], pg_url, upsert=True)

            second = {"users": [{"id": 1, "email": "new@test.com"}], "posts": []}
            PgCopyWriter().write(second, test_schema, ["users", "posts"], pg_url, upsert=True)

            with psycopg.connect(pg_url) as conn, conn.cursor() as cur:
                cur.execute('SELECT COUNT(*) FROM "users"')
                assert cur.fetchone()[0] == 1
                cur.execute('SELECT "email" FROM "users" WHERE "id" = 1')
                assert cur.fetchone()[0] == "new@test.com"
                # Staging table must not survive the transaction
                cur.execute(
                    "SELECT COUNT(*) FROM information_schema.tables "
                    "WHERE table_name LIKE '_dbsprout_stg_%'"
                )
                assert cur.fetchone()[0] == 0
        finally:
            drop_pg_tables(pg_url, test_schema)

    def test_no_orphan_staging_on_failure(
        self,
        pg_url: str,
        test_schema: DatabaseSchema,
    ) -> None:
        """A mid-merge failure leaves no orphan staging table (TEMP auto-drop)."""
        create_pg_tables(pg_url, test_schema)
        try:
            import psycopg  # noqa: PLC0415

            # FK violation on posts forces the transaction to abort.
            bad = {
                "users": [{"id": 1, "email": "u@test.com"}],
                "posts": [{"id": 1, "user_id": 999, "title": "orphan"}],
            }
            with pytest.raises(RuntimeError):
                PgCopyWriter().write(bad, test_schema, ["users", "posts"], pg_url, upsert=True)

            with psycopg.connect(pg_url) as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM information_schema.tables "
                    "WHERE table_name LIKE '_dbsprout_stg_%'"
                )
                assert cur.fetchone()[0] == 0
        finally:
            drop_pg_tables(pg_url, test_schema)


@pytest.mark.integration
@pytest.mark.slow
def test_pg_copy_throughput_10k_rows_per_sec(pg_url: str) -> None:
    """AC: PG COPY >=10K rows/sec (CI-safe; production target 100K+)."""
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

    create_pg_tables(pg_url, schema)
    try:
        start = time.monotonic()
        PgCopyWriter().write({"bench": rows}, schema, ["bench"], pg_url)
        elapsed = time.monotonic() - start

        throughput = n_rows / elapsed
        assert throughput >= 10_000, f"PG COPY throughput {throughput:.0f} rows/sec < 10K threshold"
    finally:
        drop_pg_tables(pg_url, schema)
