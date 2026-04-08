"""Tests for MySQL LOAD DATA output writer."""

from __future__ import annotations

import uuid
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import pytest

from dbsprout.output.models import InsertResult
from dbsprout.output.mysql_load_data import (
    MysqlLoadDataWriter,
    build_load_data_content,
    format_load_data_value,
)
from dbsprout.schema.models import ColumnSchema, ColumnType, DatabaseSchema, TableSchema

# ── format_load_data_value ───────────────────────────────────────────


class TestFormatLoadDataValueNone:
    def test_none(self) -> None:
        assert format_load_data_value(None) == "\\N"


class TestFormatLoadDataValueBool:
    def test_true(self) -> None:
        assert format_load_data_value(True) == "1"

    def test_false(self) -> None:
        assert format_load_data_value(False) == "0"


class TestFormatLoadDataValueNumeric:
    def test_int(self) -> None:
        assert format_load_data_value(42) == "42"

    def test_float(self) -> None:
        assert format_load_data_value(3.14) == "3.14"

    def test_float_nan(self) -> None:
        assert format_load_data_value(float("nan")) == "\\N"

    def test_float_inf(self) -> None:
        assert format_load_data_value(float("inf")) == "\\N"

    def test_decimal(self) -> None:
        assert format_load_data_value(Decimal("99.99")) == "99.99"

    def test_decimal_nan(self) -> None:
        assert format_load_data_value(Decimal("NaN")) == "\\N"


class TestFormatLoadDataValueString:
    def test_simple(self) -> None:
        assert format_load_data_value("hello") == "hello"

    def test_with_tab(self) -> None:
        assert format_load_data_value("a\tb") == "a\\tb"

    def test_with_newline(self) -> None:
        assert format_load_data_value("a\nb") == "a\\nb"

    def test_with_backslash(self) -> None:
        assert format_load_data_value("a\\b") == "a\\\\b"


class TestFormatLoadDataValueTemporal:
    def test_datetime(self) -> None:
        dt = datetime(2026, 4, 5, 12, 30, 45)
        assert format_load_data_value(dt) == "2026-04-05 12:30:45"

    def test_date(self) -> None:
        assert format_load_data_value(date(2026, 4, 5)) == "2026-04-05"

    def test_time(self) -> None:
        assert format_load_data_value(time(12, 30, 45)) == "12:30:45"


class TestFormatLoadDataValueComplex:
    def test_uuid(self) -> None:
        u = uuid.UUID("12345678-1234-5678-1234-567812345678")
        assert format_load_data_value(u) == "12345678-1234-5678-1234-567812345678"

    def test_bytes(self) -> None:
        assert format_load_data_value(b"\xde\xad") == "dead"

    def test_dict_json(self) -> None:
        result = format_load_data_value({"key": "val"})
        assert '"key"' in result
        assert '"val"' in result

    def test_list_json(self) -> None:
        assert format_load_data_value([1, 2]) == "[1, 2]"


# ── build_load_data_content ──────────────────────────────────────────


class TestBuildLoadDataContent:
    def test_single_row(self) -> None:
        columns = ["id", "name"]
        rows: list[dict[str, Any]] = [{"id": 1, "name": "Alice"}]
        assert build_load_data_content(columns, rows) == "1\tAlice\n"

    def test_multiple_rows(self) -> None:
        columns = ["id", "name"]
        rows: list[dict[str, Any]] = [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}]
        result = build_load_data_content(columns, rows)
        assert result == "1\tA\n2\tB\n"

    def test_empty_rows(self) -> None:
        assert build_load_data_content(["id"], []) == ""

    def test_preserves_column_order(self) -> None:
        rows: list[dict[str, Any]] = [{"a": 1, "b": 2, "c": 3}]
        assert build_load_data_content(["b", "a", "c"], rows) == "2\t1\t3\n"

    def test_mysql_bool_format(self) -> None:
        rows: list[dict[str, Any]] = [{"id": 1, "active": True, "deleted": False}]
        result = build_load_data_content(["id", "active", "deleted"], rows)
        assert result == "1\t1\t0\n"


# ── MysqlLoadDataWriter ─────────────────────────────────────────────


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
                ],
                primary_key=["id"],
            ),
        ],
    )


def _simple_data() -> dict[str, list[dict[str, Any]]]:
    return {
        "users": [{"id": 1, "email": "a@b.com"}, {"id": 2, "email": "c@d.com"}],
        "orders": [{"id": 1, "user_id": 1}],
    }


class TestMysqlLoadDataWriterConnect:
    def test_connects_with_parsed_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_conn = _make_mock_conn()
        mock_pymysql = _make_mock_pymysql(mock_conn)
        monkeypatch.setattr("dbsprout.output.mysql_load_data.pymysql", mock_pymysql)

        MysqlLoadDataWriter().write(
            _simple_data(),
            _simple_schema(),
            ["users", "orders"],
            "mysql://user:pass@myhost:3307/mydb",
        )
        call_kwargs = mock_pymysql.connect.call_args[1]
        assert call_kwargs["host"] == "myhost"
        assert call_kwargs["port"] == 3307
        assert call_kwargs["user"] == "user"
        assert call_kwargs["database"] == "mydb"
        assert call_kwargs["local_infile"] is True


class TestMysqlLoadDataWriterFKChecks:
    def test_disables_and_reenables_fk_checks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_conn = _make_mock_conn()
        mock_pymysql = _make_mock_pymysql(mock_conn)
        monkeypatch.setattr("dbsprout.output.mysql_load_data.pymysql", mock_pymysql)

        MysqlLoadDataWriter().write(
            _simple_data(),
            _simple_schema(),
            ["users", "orders"],
            "mysql://localhost/test",
        )
        cur = mock_conn.cursor.return_value
        calls = [str(c) for c in cur.execute.call_args_list]
        assert any("FOREIGN_KEY_CHECKS=0" in c for c in calls)
        assert any("FOREIGN_KEY_CHECKS=1" in c for c in calls)


class TestMysqlLoadDataWriterExecution:
    def test_executes_load_data_per_table(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_conn = _make_mock_conn()
        mock_pymysql = _make_mock_pymysql(mock_conn)
        monkeypatch.setattr("dbsprout.output.mysql_load_data.pymysql", mock_pymysql)

        MysqlLoadDataWriter().write(
            _simple_data(),
            _simple_schema(),
            ["users", "orders"],
            "mysql://localhost/test",
        )
        cur = mock_conn.cursor.return_value
        load_calls = [str(c) for c in cur.execute.call_args_list if "LOAD DATA" in str(c)]
        assert len(load_calls) == 2

    def test_skips_empty_tables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_conn = _make_mock_conn()
        mock_pymysql = _make_mock_pymysql(mock_conn)
        monkeypatch.setattr("dbsprout.output.mysql_load_data.pymysql", mock_pymysql)

        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": 1, "email": "a@b.com"}],
            "orders": [],
        }
        MysqlLoadDataWriter().write(
            data,
            _simple_schema(),
            ["users", "orders"],
            "mysql://localhost/test",
        )
        cur = mock_conn.cursor.return_value
        load_calls = [str(c) for c in cur.execute.call_args_list if "LOAD DATA" in str(c)]
        assert len(load_calls) == 1

    def test_batching_splits_large_tables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_conn = _make_mock_conn()
        mock_pymysql = _make_mock_pymysql(mock_conn)
        monkeypatch.setattr("dbsprout.output.mysql_load_data.pymysql", mock_pymysql)

        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": i, "email": f"u{i}@x.com"} for i in range(1, 6)],
        }
        MysqlLoadDataWriter().write(
            data,
            _simple_schema(),
            ["users"],
            "mysql://localhost/test",
            batch_size=2,
        )
        cur = mock_conn.cursor.return_value
        load_calls = [str(c) for c in cur.execute.call_args_list if "LOAD DATA" in str(c)]
        assert len(load_calls) == 3  # 2+2+1


class TestMysqlLoadDataWriterResult:
    def test_returns_insert_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_conn = _make_mock_conn()
        mock_pymysql = _make_mock_pymysql(mock_conn)
        monkeypatch.setattr("dbsprout.output.mysql_load_data.pymysql", mock_pymysql)

        result = MysqlLoadDataWriter().write(
            _simple_data(),
            _simple_schema(),
            ["users", "orders"],
            "mysql://localhost/test",
        )
        assert isinstance(result, InsertResult)
        assert result.tables_inserted == 2
        assert result.total_rows == 3
        assert result.duration_seconds >= 0


class TestMysqlLoadDataWriterImportError:
    def test_raises_on_missing_pymysql(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("dbsprout.output.mysql_load_data.pymysql", None)

        with pytest.raises(ImportError, match="pip install dbsprout\\[db\\]"):
            MysqlLoadDataWriter().write(
                _simple_data(),
                _simple_schema(),
                ["users"],
                "mysql://localhost/test",
            )


class TestMysqlLoadDataWriterErrorHandling:
    def test_db_error_wrapped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_pymysql = MagicMock()
        mock_pymysql.connect.side_effect = Exception("connection refused")
        monkeypatch.setattr("dbsprout.output.mysql_load_data.pymysql", mock_pymysql)

        with pytest.raises(RuntimeError, match="MySQL insertion failed"):
            MysqlLoadDataWriter().write(
                _simple_data(),
                _simple_schema(),
                ["users"],
                "mysql://localhost/test",
            )

    def test_cleans_up_temp_files_on_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
    ) -> None:
        """_cleanup_temp_files is called even when LOAD DATA fails."""
        mock_pymysql = MagicMock()
        mock_conn = _make_mock_conn()
        mock_pymysql.connect.return_value = mock_conn

        def execute_side_effect(sql: str) -> None:
            if "LOAD DATA" in sql:
                raise Exception("load error")

        mock_conn.cursor.return_value.execute = execute_side_effect
        monkeypatch.setattr("dbsprout.output.mysql_load_data.pymysql", mock_pymysql)

        cleanup_called = False
        original_cleanup = __import__(
            "dbsprout.output.mysql_load_data", fromlist=["_cleanup_temp_files"]
        )._cleanup_temp_files

        def mock_cleanup(paths: list[str]) -> None:
            nonlocal cleanup_called
            cleanup_called = True
            original_cleanup(paths)

        monkeypatch.setattr("dbsprout.output.mysql_load_data._cleanup_temp_files", mock_cleanup)

        with pytest.raises(RuntimeError):
            MysqlLoadDataWriter().write(
                _simple_data(),
                _simple_schema(),
                ["users"],
                "mysql://localhost/test",
            )
        assert cleanup_called


# ── Mock helpers ─────────────────────────────────────────────────────


def _make_mock_conn() -> MagicMock:
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    return conn


def _make_mock_pymysql(conn: MagicMock) -> MagicMock:
    mod = MagicMock()
    mod.connect.return_value = conn
    return mod


# ── local_infile error detection ────────────────────────────────────────


class TestLocalInfileErrorDetection:
    """AC: Detect MySQL local_infile disabled error with specific message."""

    def test_local_infile_error_1148_detected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Error 1148 (LOAD DATA not allowed) produces helpful message."""
        mock_conn = _make_mock_conn()
        mock_pymysql = _make_mock_pymysql(mock_conn)

        op_error = type("OperationalError", (Exception,), {})
        mock_pymysql.err.OperationalError = op_error

        call_count = 0

        def execute_side_effect(sql: str) -> None:
            nonlocal call_count
            call_count += 1
            if "LOAD DATA" in sql:
                raise op_error(1148, "The used command is not allowed")

        mock_conn.cursor.return_value.execute = execute_side_effect
        monkeypatch.setattr("dbsprout.output.mysql_load_data.pymysql", mock_pymysql)

        with pytest.raises(RuntimeError, match="local_infile"):
            MysqlLoadDataWriter().write(
                _simple_data(), _simple_schema(), ["users"], "mysql://localhost/test"
            )

    def test_local_infile_error_3948_detected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Error 3948 (loading local data disabled) produces helpful message."""
        mock_conn = _make_mock_conn()
        mock_pymysql = _make_mock_pymysql(mock_conn)

        op_error = type("OperationalError", (Exception,), {})
        mock_pymysql.err.OperationalError = op_error

        def execute_side_effect(sql: str) -> None:
            if "LOAD DATA" in sql:
                raise op_error(3948, "Loading local data is disabled")

        mock_conn.cursor.return_value.execute = execute_side_effect
        monkeypatch.setattr("dbsprout.output.mysql_load_data.pymysql", mock_pymysql)

        with pytest.raises(RuntimeError, match="local_infile"):
            MysqlLoadDataWriter().write(
                _simple_data(), _simple_schema(), ["users"], "mysql://localhost/test"
            )

    def test_other_operational_error_not_masked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-local_infile OperationalErrors propagate as generic RuntimeError."""
        mock_conn = _make_mock_conn()
        mock_pymysql = _make_mock_pymysql(mock_conn)

        op_error = type("OperationalError", (Exception,), {})
        mock_pymysql.err.OperationalError = op_error

        def execute_side_effect(sql: str) -> None:
            if "LOAD DATA" in sql:
                raise op_error(2003, "Can't connect to MySQL server")

        mock_conn.cursor.return_value.execute = execute_side_effect
        monkeypatch.setattr("dbsprout.output.mysql_load_data.pymysql", mock_pymysql)

        with pytest.raises(RuntimeError, match="insertion failed"):
            MysqlLoadDataWriter().write(
                _simple_data(), _simple_schema(), ["users"], "mysql://localhost/test"
            )
