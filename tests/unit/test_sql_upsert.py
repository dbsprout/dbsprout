"""Tests for UPSERT SQL generation across all dialects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dbsprout.output.sql_writer import (
    SQLWriter,
    build_upsert,
    get_dialect_config,
)
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    TableSchema,
)

if TYPE_CHECKING:
    from pathlib import Path


def _pg() -> dict[str, str]:
    return get_dialect_config("postgresql")


def _my() -> dict[str, str]:
    return get_dialect_config("mysql")


def _sq() -> dict[str, str]:
    return get_dialect_config("sqlite")


def _ms() -> dict[str, str]:
    return get_dialect_config("mssql")


def _simple_schema(*, has_pk: bool = True) -> DatabaseSchema:
    columns = [
        ColumnSchema(
            name="id",
            data_type=ColumnType.INTEGER,
            nullable=False,
            primary_key=has_pk,
        ),
        ColumnSchema(
            name="name",
            data_type=ColumnType.VARCHAR,
            nullable=False,
        ),
        ColumnSchema(
            name="email",
            data_type=ColumnType.VARCHAR,
            nullable=True,
        ),
    ]
    return DatabaseSchema(
        tables=[
            TableSchema(
                name="users",
                columns=columns,
                primary_key=["id"] if has_pk else [],
            ),
        ],
        dialect="postgresql",
    )


def _composite_pk_schema() -> DatabaseSchema:
    return DatabaseSchema(
        tables=[
            TableSchema(
                name="user_roles",
                columns=[
                    ColumnSchema(
                        name="user_id",
                        data_type=ColumnType.INTEGER,
                        nullable=False,
                        primary_key=True,
                    ),
                    ColumnSchema(
                        name="role_id",
                        data_type=ColumnType.INTEGER,
                        nullable=False,
                        primary_key=True,
                    ),
                    ColumnSchema(
                        name="granted_at",
                        data_type=ColumnType.TIMESTAMP,
                        nullable=True,
                    ),
                ],
                primary_key=["user_id", "role_id"],
            ),
        ],
        dialect="postgresql",
    )


def _all_pk_schema() -> DatabaseSchema:
    """Junction table where all columns are PKs."""
    return DatabaseSchema(
        tables=[
            TableSchema(
                name="tag_items",
                columns=[
                    ColumnSchema(
                        name="tag_id",
                        data_type=ColumnType.INTEGER,
                        nullable=False,
                        primary_key=True,
                    ),
                    ColumnSchema(
                        name="item_id",
                        data_type=ColumnType.INTEGER,
                        nullable=False,
                        primary_key=True,
                    ),
                ],
                primary_key=["tag_id", "item_id"],
            ),
        ],
        dialect="postgresql",
    )


# ── PostgreSQL UPSERT ───────────────────────────────────────────────


class TestPostgresUpsert:
    def test_basic_upsert(self) -> None:
        """Single-row PG UPSERT with ON CONFLICT DO UPDATE SET."""
        rows: list[dict[str, Any]] = [{"id": 1, "name": "Alice", "email": "a@b.com"}]
        result = build_upsert("users", ["id", "name", "email"], rows, _pg(), ["id"])

        assert "ON CONFLICT" in result
        assert '"id"' in result
        assert 'EXCLUDED."name"' in result
        assert 'EXCLUDED."email"' in result
        assert "INSERT INTO" in result

    def test_composite_pk(self) -> None:
        """Composite PK in ON CONFLICT clause."""
        rows: list[dict[str, Any]] = [{"user_id": 1, "role_id": 2, "granted_at": None}]
        result = build_upsert(
            "user_roles",
            ["user_id", "role_id", "granted_at"],
            rows,
            _pg(),
            ["user_id", "role_id"],
        )

        assert '"user_id", "role_id"' in result
        assert 'EXCLUDED."granted_at"' in result

    def test_all_columns_pk(self) -> None:
        """Junction table — all PKs, should use DO NOTHING."""
        rows: list[dict[str, Any]] = [{"tag_id": 1, "item_id": 2}]
        result = build_upsert(
            "tag_items", ["tag_id", "item_id"], rows, _pg(), ["tag_id", "item_id"]
        )

        assert "DO NOTHING" in result
        assert "DO UPDATE" not in result

    def test_batch_upsert(self) -> None:
        """Multiple rows in a single UPSERT statement."""
        rows: list[dict[str, Any]] = [
            {"id": 1, "name": "Alice", "email": "a@b.com"},
            {"id": 2, "name": "Bob", "email": "b@b.com"},
        ]
        result = build_upsert("users", ["id", "name", "email"], rows, _pg(), ["id"])

        assert result.count("(") >= 3  # 2 value rows + column list


# ── MySQL UPSERT ────────────────────────────────────────────────────


class TestMysqlUpsert:
    def test_basic_upsert(self) -> None:
        """MySQL ON DUPLICATE KEY UPDATE syntax."""
        rows: list[dict[str, Any]] = [{"id": 1, "name": "Alice", "email": "a@b.com"}]
        result = build_upsert("users", ["id", "name", "email"], rows, _my(), ["id"])

        assert "ON DUPLICATE KEY UPDATE" in result
        assert "VALUES(`name`)" in result
        assert "VALUES(`email`)" in result

    def test_composite_pk(self) -> None:
        """MySQL UPSERT with composite PK."""
        rows: list[dict[str, Any]] = [{"user_id": 1, "role_id": 2, "granted_at": None}]
        result = build_upsert(
            "user_roles",
            ["user_id", "role_id", "granted_at"],
            rows,
            _my(),
            ["user_id", "role_id"],
        )

        assert "ON DUPLICATE KEY UPDATE" in result
        assert "VALUES(`granted_at`)" in result

    def test_all_columns_pk(self) -> None:
        """All PKs — MySQL INSERT IGNORE."""
        rows: list[dict[str, Any]] = [{"tag_id": 1, "item_id": 2}]
        result = build_upsert(
            "tag_items", ["tag_id", "item_id"], rows, _my(), ["tag_id", "item_id"]
        )

        assert "INSERT IGNORE INTO" in result
        assert "ON DUPLICATE KEY" not in result


# ── SQLite UPSERT ───────────────────────────────────────────────────


class TestSqliteUpsert:
    def test_basic_upsert(self) -> None:
        """SQLite ON CONFLICT DO UPDATE SET with lowercase excluded."""
        rows: list[dict[str, Any]] = [{"id": 1, "name": "Alice", "email": "a@b.com"}]
        result = build_upsert("users", ["id", "name", "email"], rows, _sq(), ["id"])

        assert "ON CONFLICT" in result
        assert 'excluded."name"' in result
        assert 'excluded."email"' in result

    def test_composite_pk(self) -> None:
        """SQLite UPSERT with composite PK."""
        rows: list[dict[str, Any]] = [{"user_id": 1, "role_id": 2, "granted_at": None}]
        result = build_upsert(
            "user_roles",
            ["user_id", "role_id", "granted_at"],
            rows,
            _sq(),
            ["user_id", "role_id"],
        )

        assert '"user_id", "role_id"' in result
        assert 'excluded."granted_at"' in result


# ── SQL Server UPSERT (MERGE) ──────────────────────────────────────


class TestMssqlUpsert:
    def test_basic_merge(self) -> None:
        """SQL Server MERGE syntax."""
        rows: list[dict[str, Any]] = [{"id": 1, "name": "Alice", "email": "a@b.com"}]
        result = build_upsert("users", ["id", "name", "email"], rows, _ms(), ["id"])

        assert "MERGE" in result
        assert "WHEN MATCHED" in result
        assert "WHEN NOT MATCHED" in result
        assert "target" in result.lower() or "AS target" in result


# ── No Primary Key ──────────────────────────────────────────────────


class TestNoPrimaryKey:
    def test_fallback_to_insert(self) -> None:
        """No PK → regular INSERT, no conflict clause."""
        rows: list[dict[str, Any]] = [{"id": 1, "name": "Alice", "email": "a@b.com"}]
        result = build_upsert("users", ["id", "name", "email"], rows, _pg(), [])

        assert "INSERT INTO" in result
        assert "ON CONFLICT" not in result
        assert "MERGE" not in result


# ── SQLWriter integration ───────────────────────────────────────────


class TestUpsertWriter:
    def test_upsert_flag_produces_upsert_sql(self, tmp_path: Path) -> None:
        """SQLWriter with upsert=True should produce UPSERT SQL."""
        schema = _simple_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": 1, "name": "Alice", "email": "a@b.com"}],
        }
        writer = SQLWriter()
        paths = writer.write(data, schema, ["users"], tmp_path, dialect="postgresql", upsert=True)

        content = paths[0].read_text()
        assert "ON CONFLICT" in content
        assert "EXCLUDED" in content

    def test_no_upsert_flag_produces_insert(self, tmp_path: Path) -> None:
        """Default SQLWriter (upsert=False) should produce regular INSERT."""
        schema = _simple_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": 1, "name": "Alice", "email": "a@b.com"}],
        }
        writer = SQLWriter()
        paths = writer.write(data, schema, ["users"], tmp_path, dialect="postgresql")

        content = paths[0].read_text()
        assert "INSERT INTO" in content
        assert "ON CONFLICT" not in content

    def test_upsert_no_pk_fallback(self, tmp_path: Path) -> None:
        """UPSERT on table without PK should fall back to INSERT."""
        schema = _simple_schema(has_pk=False)
        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": 1, "name": "Alice", "email": "a@b.com"}],
        }
        writer = SQLWriter()
        paths = writer.write(data, schema, ["users"], tmp_path, dialect="postgresql", upsert=True)

        content = paths[0].read_text()
        assert "INSERT INTO" in content
        assert "ON CONFLICT" not in content

    def test_upsert_mysql_dialect(self, tmp_path: Path) -> None:
        """SQLWriter with upsert=True and MySQL dialect."""
        schema = _simple_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": 1, "name": "Alice", "email": "a@b.com"}],
        }
        writer = SQLWriter()
        paths = writer.write(data, schema, ["users"], tmp_path, dialect="mysql", upsert=True)

        content = paths[0].read_text()
        assert "ON DUPLICATE KEY UPDATE" in content
