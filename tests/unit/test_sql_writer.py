"""Tests for dbsprout.output.sql_writer — SQL INSERT output writer."""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime, time, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from dbsprout.output.sql_writer import (
    SQLWriter,
    build_insert,
    format_value,
    get_dialect_config,
    quote_identifier,
)
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    TableSchema,
)


def _pg() -> dict[str, str]:
    return get_dialect_config("postgresql")


def _my() -> dict[str, str]:
    return get_dialect_config("mysql")


def _sq() -> dict[str, str]:
    return get_dialect_config("sqlite")


# ── format_value tests ──────────────────────────────────────────────


class TestFormatValueNone:
    def test_none_is_null(self) -> None:
        assert format_value(None, _pg()) == "NULL"
        assert format_value(None, _my()) == "NULL"
        assert format_value(None, _sq()) == "NULL"


class TestFormatValueBool:
    def test_bool_postgresql(self) -> None:
        assert format_value(True, _pg()) == "TRUE"
        assert format_value(False, _pg()) == "FALSE"

    def test_bool_mysql(self) -> None:
        assert format_value(True, _my()) == "1"
        assert format_value(False, _my()) == "0"

    def test_bool_sqlite(self) -> None:
        assert format_value(True, _sq()) == "1"
        assert format_value(False, _sq()) == "0"


class TestFormatValueString:
    def test_standard_escape(self) -> None:
        """PostgreSQL/SQLite use '' for single quote escaping."""
        assert format_value("O'Brien", _pg()) == "'O''Brien'"
        assert format_value("O'Brien", _sq()) == "'O''Brien'"

    def test_mysql_escape(self) -> None:
        """MySQL uses backslash escaping."""
        assert format_value("O'Brien", _my()) == "'O\\'Brien'"

    def test_plain_string(self) -> None:
        assert format_value("hello", _pg()) == "'hello'"

    def test_backslash_in_mysql(self) -> None:
        """MySQL must also escape backslashes."""
        assert format_value("path\\file", _my()) == "'path\\\\file'"


class TestFormatValueNumbers:
    def test_int(self) -> None:
        assert format_value(42, _pg()) == "42"
        assert format_value(0, _pg()) == "0"
        assert format_value(-1, _pg()) == "-1"

    def test_float(self) -> None:
        assert format_value(3.14, _pg()) == "3.14"
        assert format_value(0.0, _pg()) == "0.0"


class TestFormatValueDatetime:
    def test_datetime(self) -> None:
        dt = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        assert format_value(dt, _pg()) == "'2024-01-15 10:30:00+00:00'"

    def test_date(self) -> None:
        d = date(2024, 1, 15)
        assert format_value(d, _pg()) == "'2024-01-15'"

    def test_time(self) -> None:
        t = time(10, 30, 0)
        assert format_value(t, _pg()) == "'10:30:00'"


class TestFormatValueBytes:
    def test_bytes_hex(self) -> None:
        assert format_value(b"\xde\xad\xbe\xef", _pg()) == "X'deadbeef'"


class TestFormatValueJson:
    def test_dict_as_json(self) -> None:
        val: dict[str, Any] = {"key": "value"}
        result = format_value(val, _pg())
        # Should be a quoted JSON string
        assert result.startswith("'")
        assert result.endswith("'")
        parsed = json.loads(result[1:-1])
        assert parsed == {"key": "value"}

    def test_list_as_json(self) -> None:
        val = [1, 2, 3]
        result = format_value(val, _pg())
        assert result.startswith("'")
        parsed = json.loads(result[1:-1])
        assert parsed == [1, 2, 3]


class TestFormatValueUUID:
    def test_uuid(self) -> None:
        u = uuid.UUID("12345678-1234-5678-1234-567812345678")
        result = format_value(u, _pg())
        assert result == "'12345678-1234-5678-1234-567812345678'"


# ── quote_identifier tests ──────────────────────────────────────────


class TestQuoteIdentifier:
    def test_postgresql(self) -> None:
        assert quote_identifier("email", _pg()) == '"email"'

    def test_mysql(self) -> None:
        assert quote_identifier("email", _my()) == "`email`"

    def test_sqlite(self) -> None:
        assert quote_identifier("email", _sq()) == '"email"'


# ── build_insert tests ──────────────────────────────────────────────


class TestBuildInsert:
    def test_multi_row_insert(self) -> None:
        cfg = _pg()
        rows = [
            {"id": 1, "email": "alice@example.com"},
            {"id": 2, "email": "bob@example.com"},
        ]
        columns = ["id", "email"]
        result = build_insert("users", columns, rows, cfg)

        assert 'INSERT INTO "users"' in result
        assert '"id"' in result
        assert '"email"' in result
        assert "(1, 'alice@example.com')" in result
        assert "(2, 'bob@example.com')" in result
        assert result.rstrip().endswith(";")

    def test_mysql_quoting(self) -> None:
        cfg = _my()
        rows = [{"id": 1, "name": "test"}]
        result = build_insert("users", ["id", "name"], rows, cfg)
        assert "INSERT INTO `users`" in result
        assert "`id`" in result


# ── SQLWriter.write tests ───────────────────────────────────────────


def _simple_schema() -> DatabaseSchema:
    return DatabaseSchema(
        tables=[
            TableSchema(
                name="users",
                columns=[
                    ColumnSchema(
                        name="id",
                        data_type=ColumnType.INTEGER,
                        nullable=False,
                        primary_key=True,
                    ),
                    ColumnSchema(name="email", data_type=ColumnType.VARCHAR, nullable=False),
                ],
                primary_key=["id"],
            ),
            TableSchema(
                name="orders",
                columns=[
                    ColumnSchema(
                        name="id",
                        data_type=ColumnType.INTEGER,
                        nullable=False,
                        primary_key=True,
                    ),
                    ColumnSchema(name="user_id", data_type=ColumnType.INTEGER, nullable=False),
                ],
                primary_key=["id"],
                foreign_keys=[
                    ForeignKeySchema(columns=["user_id"], ref_table="users", ref_columns=["id"]),
                ],
            ),
        ],
        dialect="postgresql",
    )


class TestSQLWriterFiles:
    def test_creates_files_with_prefix(self, tmp_path: Path) -> None:
        """Files should be named 001_users.sql, 002_orders.sql."""
        schema = _simple_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": 1, "email": "a@b.com"}],
            "orders": [{"id": 1, "user_id": 1}],
        }
        writer = SQLWriter()
        paths = writer.write(data, schema, ["users", "orders"], tmp_path)

        assert len(paths) == 2
        assert paths[0].name == "001_users.sql"
        assert paths[1].name == "002_orders.sql"
        assert all(p.exists() for p in paths)

    def test_includes_transaction_wrapping(self, tmp_path: Path) -> None:
        """Each file should have BEGIN/COMMIT."""
        schema = _simple_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": 1, "email": "a@b.com"}],
        }
        writer = SQLWriter()
        paths = writer.write(data, schema, ["users"], tmp_path)

        content = paths[0].read_text()
        assert "BEGIN;" in content
        assert "COMMIT;" in content

    def test_batches_rows(self, tmp_path: Path) -> None:
        """batch_size=2 with 3 rows should produce 2 INSERT statements."""
        schema = _simple_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [
                {"id": 1, "email": "a@b.com"},
                {"id": 2, "email": "c@d.com"},
                {"id": 3, "email": "e@f.com"},
            ],
        }
        writer = SQLWriter()
        paths = writer.write(data, schema, ["users"], tmp_path, batch_size=2)

        content = paths[0].read_text()
        insert_count = content.count("INSERT INTO")
        assert insert_count == 2

    def test_all_three_dialects(self, tmp_path: Path) -> None:
        """All 3 dialects should produce valid output."""
        schema = _simple_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": 1, "email": "test@test.com"}],
        }
        writer = SQLWriter()

        for dialect in ("postgresql", "mysql", "sqlite"):
            out = tmp_path / dialect
            out.mkdir()
            paths = writer.write(data, schema, ["users"], out, dialect=dialect)
            content = paths[0].read_text()
            assert "INSERT INTO" in content
            assert "test@test.com" in content

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        """Should create output directory if it doesn't exist."""
        schema = _simple_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": 1, "email": "a@b.com"}],
        }
        out = tmp_path / "nested" / "seeds"
        writer = SQLWriter()
        paths = writer.write(data, schema, ["users"], out)

        assert out.exists()
        assert len(paths) == 1

    def test_empty_table_skipped(self, tmp_path: Path) -> None:
        """Tables with 0 rows should not produce a file."""
        schema = _simple_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [],
            "orders": [{"id": 1, "user_id": 1}],
        }
        writer = SQLWriter()
        paths = writer.write(data, schema, ["users", "orders"], tmp_path)

        assert len(paths) == 1
        assert paths[0].name == "002_orders.sql"
