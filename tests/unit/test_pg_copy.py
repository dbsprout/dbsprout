"""Tests for PostgreSQL COPY output writer."""

from __future__ import annotations

import uuid
from datetime import date, datetime, time, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest

from dbsprout.output.pg_copy import (
    InsertResult,
    PgCopyWriter,
    build_copy_data,
    format_copy_value,
)
from dbsprout.schema.models import ColumnSchema, ColumnType, DatabaseSchema, TableSchema

# ── format_copy_value ────────────────────────────────────────────────


class TestFormatCopyValueNone:
    def test_none_returns_backslash_n(self) -> None:
        assert format_copy_value(None) == "\\N"


class TestFormatCopyValueBool:
    def test_true(self) -> None:
        assert format_copy_value(True) == "t"

    def test_false(self) -> None:
        assert format_copy_value(False) == "f"


class TestFormatCopyValueNumeric:
    def test_int(self) -> None:
        assert format_copy_value(42) == "42"

    def test_negative_int(self) -> None:
        assert format_copy_value(-7) == "-7"

    def test_zero(self) -> None:
        assert format_copy_value(0) == "0"

    def test_float(self) -> None:
        assert format_copy_value(3.14) == "3.14"

    def test_float_nan(self) -> None:
        assert format_copy_value(float("nan")) == "\\N"

    def test_float_inf(self) -> None:
        assert format_copy_value(float("inf")) == "\\N"

    def test_float_neg_inf(self) -> None:
        assert format_copy_value(float("-inf")) == "\\N"

    def test_decimal(self) -> None:
        assert format_copy_value(Decimal("99.99")) == "99.99"

    def test_decimal_nan(self) -> None:
        assert format_copy_value(Decimal("NaN")) == "\\N"

    def test_decimal_inf(self) -> None:
        assert format_copy_value(Decimal("Infinity")) == "\\N"


class TestFormatCopyValueString:
    def test_simple(self) -> None:
        assert format_copy_value("hello") == "hello"

    def test_with_tab(self) -> None:
        assert format_copy_value("a\tb") == "a\\tb"

    def test_with_newline(self) -> None:
        assert format_copy_value("a\nb") == "a\\nb"

    def test_with_carriage_return(self) -> None:
        assert format_copy_value("a\rb") == "a\\rb"

    def test_with_backslash(self) -> None:
        assert format_copy_value("a\\b") == "a\\\\b"

    def test_empty_string(self) -> None:
        assert format_copy_value("") == ""


class TestFormatCopyValueTemporal:
    def test_datetime(self) -> None:
        dt = datetime(2026, 4, 5, 12, 30, 45)
        assert format_copy_value(dt) == "2026-04-05 12:30:45"

    def test_datetime_with_tz(self) -> None:
        dt = datetime(2026, 4, 5, 12, 30, 45, tzinfo=timezone.utc)
        assert format_copy_value(dt) == "2026-04-05 12:30:45+00:00"

    def test_date(self) -> None:
        d = date(2026, 4, 5)
        assert format_copy_value(d) == "2026-04-05"

    def test_time(self) -> None:
        t = time(12, 30, 45)
        assert format_copy_value(t) == "12:30:45"


class TestFormatCopyValueComplex:
    def test_uuid(self) -> None:
        u = uuid.UUID("12345678-1234-5678-1234-567812345678")
        assert format_copy_value(u) == "12345678-1234-5678-1234-567812345678"

    def test_bytes(self) -> None:
        assert format_copy_value(b"\xde\xad\xbe\xef") == "\\\\xdeadbeef"

    def test_bytes_empty(self) -> None:
        assert format_copy_value(b"") == "\\\\x"

    def test_dict_json(self) -> None:
        result = format_copy_value({"key": "val"})
        assert '"key"' in result
        assert '"val"' in result

    def test_list_json(self) -> None:
        result = format_copy_value([1, 2, 3])
        assert result == "[1, 2, 3]"

    def test_dict_with_special_chars(self) -> None:
        result = format_copy_value({"k": "a\tb"})
        assert "\\t" in result


# ── build_copy_data ────────���─────────────────────────────────────────


class TestBuildCopyData:
    def test_single_row(self) -> None:
        columns = ["id", "name"]
        rows: list[dict[str, Any]] = [{"id": 1, "name": "Alice"}]
        result = build_copy_data(columns, rows)
        assert result == "1\tAlice\n"

    def test_multiple_rows(self) -> None:
        columns = ["id", "name"]
        rows: list[dict[str, Any]] = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ]
        result = build_copy_data(columns, rows)
        lines = result.strip().split("\n")
        assert len(lines) == 2
        assert lines[0] == "1\tAlice"
        assert lines[1] == "2\tBob"

    def test_empty_rows(self) -> None:
        result = build_copy_data(["id"], [])
        assert result == ""

    def test_preserves_column_order(self) -> None:
        columns = ["b", "a", "c"]
        rows: list[dict[str, Any]] = [{"a": 1, "b": 2, "c": 3}]
        result = build_copy_data(columns, rows)
        assert result == "2\t1\t3\n"

    def test_mixed_types(self) -> None:
        columns = ["id", "name", "active", "score"]
        rows: list[dict[str, Any]] = [
            {"id": 1, "name": "Alice", "active": True, "score": None},
        ]
        result = build_copy_data(columns, rows)
        assert result == "1\tAlice\tt\t\\N\n"

    def test_newline_terminated(self) -> None:
        columns = ["id"]
        rows: list[dict[str, Any]] = [{"id": 1}]
        result = build_copy_data(columns, rows)
        assert result.endswith("\n")


# ─��� PgCopyWriter ─────────���───────────────────────────────────────────


def _simple_schema() -> DatabaseSchema:
    return DatabaseSchema(
        tables=[
            TableSchema(
                name="users",
                columns=[
                    ColumnSchema(
                        name="id",
                        data_type=ColumnType.INTEGER,
                        primary_key=True,
                        autoincrement=True,
                    ),
                    ColumnSchema(name="email", data_type=ColumnType.VARCHAR),
                ],
                primary_key=["id"],
            ),
            TableSchema(
                name="orders",
                columns=[
                    ColumnSchema(
                        name="id",
                        data_type=ColumnType.INTEGER,
                        primary_key=True,
                        autoincrement=True,
                    ),
                    ColumnSchema(name="user_id", data_type=ColumnType.INTEGER),
                    ColumnSchema(name="total", data_type=ColumnType.DECIMAL),
                ],
                primary_key=["id"],
            ),
        ],
    )


def _simple_data() -> dict[str, list[dict[str, Any]]]:
    return {
        "users": [
            {"id": 1, "email": "a@b.com"},
            {"id": 2, "email": "c@d.com"},
        ],
        "orders": [
            {"id": 1, "user_id": 1, "total": Decimal("9.99")},
        ],
    }


class TestPgCopyWriterConnect:
    def test_connects_with_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_conn = _make_mock_conn()
        mock_connect = _make_mock_connect(mock_conn)
        monkeypatch.setattr("dbsprout.output.pg_copy.psycopg", _mock_psycopg(mock_connect))

        PgCopyWriter().write(
            _simple_data(), _simple_schema(), ["users", "orders"], "postgres://localhost/test"
        )
        mock_connect.assert_called_once_with("postgres://localhost/test")


class TestPgCopyWriterTransaction:
    def test_single_transaction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_conn = _make_mock_conn()
        mock_connect = _make_mock_connect(mock_conn)
        monkeypatch.setattr("dbsprout.output.pg_copy.psycopg", _mock_psycopg(mock_connect))

        PgCopyWriter().write(
            _simple_data(), _simple_schema(), ["users", "orders"], "postgres://localhost/test"
        )
        mock_conn.transaction.assert_called_once()


class TestPgCopyWriterCopy:
    def test_copies_each_table_in_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_conn = _make_mock_conn()
        mock_connect = _make_mock_connect(mock_conn)
        monkeypatch.setattr("dbsprout.output.pg_copy.psycopg", _mock_psycopg(mock_connect))

        PgCopyWriter().write(
            _simple_data(), _simple_schema(), ["users", "orders"], "postgres://localhost/test"
        )
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        copy_calls = cursor.copy.call_args_list
        # 2 tables = 2 copy calls (one batch each)
        assert len(copy_calls) == 2

    def test_skips_empty_tables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_conn = _make_mock_conn()
        mock_connect = _make_mock_connect(mock_conn)
        monkeypatch.setattr("dbsprout.output.pg_copy.psycopg", _mock_psycopg(mock_connect))

        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": 1, "email": "a@b.com"}],
            "orders": [],
        }
        PgCopyWriter().write(data, _simple_schema(), ["users", "orders"], "pg://localhost/test")
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        assert len(cursor.copy.call_args_list) == 1

    def test_batching_splits_large_tables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With batch_size=2 and 5 rows, there should be 3 COPY calls for one table."""
        mock_conn = _make_mock_conn()
        mock_connect = _make_mock_connect(mock_conn)
        monkeypatch.setattr("dbsprout.output.pg_copy.psycopg", _mock_psycopg(mock_connect))

        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": i, "email": f"u{i}@x.com"} for i in range(1, 6)],
        }
        PgCopyWriter().write(data, _simple_schema(), ["users"], "pg://localhost/test", batch_size=2)
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        assert len(cursor.copy.call_args_list) == 3  # 2+2+1


class TestPgCopyWriterSequences:
    def test_resets_sequences(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_conn = _make_mock_conn()
        mock_connect = _make_mock_connect(mock_conn)
        monkeypatch.setattr("dbsprout.output.pg_copy.psycopg", _mock_psycopg(mock_connect))

        PgCopyWriter().write(
            _simple_data(), _simple_schema(), ["users", "orders"], "pg://localhost/test"
        )
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        # Exactly 2 autoincrement PK columns → 2 setval execute calls
        assert cursor.execute.call_count == 2

    def test_skips_sequence_reset_for_none_pk_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If PK column values are all None, sequence reset is skipped."""
        mock_conn = _make_mock_conn()
        mock_connect = _make_mock_connect(mock_conn)
        monkeypatch.setattr("dbsprout.output.pg_copy.psycopg", _mock_psycopg(mock_connect))

        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": None, "email": "a@b.com"}],
        }
        PgCopyWriter().write(data, _simple_schema(), ["users"], "pg://localhost/test")
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        assert cursor.execute.call_count == 0


class TestPgCopyWriterResult:
    def test_returns_insert_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_conn = _make_mock_conn()
        mock_connect = _make_mock_connect(mock_conn)
        monkeypatch.setattr("dbsprout.output.pg_copy.psycopg", _mock_psycopg(mock_connect))

        result = PgCopyWriter().write(
            _simple_data(), _simple_schema(), ["users", "orders"], "pg://localhost/test"
        )
        assert isinstance(result, InsertResult)
        assert result.tables_inserted == 2
        assert result.total_rows == 3
        assert result.duration_seconds >= 0


class TestPgCopyWriterImportError:
    def test_raises_on_missing_psycopg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("dbsprout.output.pg_copy.psycopg", None)

        with pytest.raises(ImportError, match="pip install dbsprout\\[pg\\]"):
            PgCopyWriter().write(
                _simple_data(),
                _simple_schema(),
                ["users", "orders"],
                "pg://localhost/test",
            )


class TestPgCopyWriterErrorHandling:
    def test_db_error_wrapped_with_friendly_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Database errors should be wrapped without leaking connection string."""
        mock_psycopg_mod = MagicMock()
        mock_psycopg_mod.connect.side_effect = Exception("connection refused")
        monkeypatch.setattr("dbsprout.output.pg_copy.psycopg", mock_psycopg_mod)

        with pytest.raises(RuntimeError, match="Database insertion failed"):
            PgCopyWriter().write(
                _simple_data(),
                _simple_schema(),
                ["users"],
                "pg://user:secret@host/db",
            )


# ── Mock helpers ─────────────────────────────────────────────────────


def _make_mock_conn() -> MagicMock:
    """Create a mock psycopg connection with transaction and cursor support."""
    conn = MagicMock()
    # connection as context manager
    conn.__enter__ = Mock(return_value=conn)
    conn.__exit__ = Mock(return_value=False)

    # transaction() returns a context manager
    tx = MagicMock()
    conn.transaction.return_value = tx
    tx.__enter__ = Mock(return_value=tx)
    tx.__exit__ = Mock(return_value=False)

    # cursor() returns a context manager wrapping a mock cursor
    cursor = MagicMock()
    cursor_cm = MagicMock()
    cursor_cm.__enter__ = Mock(return_value=cursor)
    cursor_cm.__exit__ = Mock(return_value=False)
    conn.cursor.return_value = cursor_cm

    # cursor.copy() returns a context manager for writing COPY data
    copy_cm = MagicMock()
    copy_cm.__enter__ = Mock(return_value=copy_cm)
    copy_cm.__exit__ = Mock(return_value=False)
    cursor.copy.return_value = copy_cm

    return conn


def _make_mock_connect(conn: MagicMock) -> MagicMock:
    """Create a mock psycopg.connect function."""
    return MagicMock(return_value=conn)


def _mock_psycopg(connect: MagicMock) -> MagicMock:
    """Create a mock psycopg module."""
    mod = MagicMock()
    mod.connect = connect
    return mod
