"""Integration tests for PostgreSQL COPY direct insertion.

Requires Docker with PostgreSQL. Auto-skips when Docker unavailable.
"""

from __future__ import annotations

import random
import time
from typing import Any

import pytest

from dbsprout.output.pg_copy import PgCopyWriter
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    TableSchema,
)

from .conftest import create_pg_tables, drop_pg_tables


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
                cur.execute("SELECT currval(pg_get_serial_sequence('users', 'id'))")
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
