"""Tests for dbsprout.schema.introspect — MySQL-specific introspection."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import sqlalchemy.types as sa_types
from sqlalchemy.dialects import mysql

from dbsprout.schema.introspect import (
    _build_columns,
    _build_foreign_keys,
    _detect_autoincrement,
)

# ── Autoincrement detection ──────────────────────────────────────────────


class TestMysqlAutoincrement:
    """MySQL AUTO_INCREMENT sets autoincrement=True in col_info."""

    def test_auto_increment_detected(self) -> None:
        col_info: dict[str, Any] = {
            "name": "id",
            "type": sa_types.Integer(),
            "nullable": False,
            "default": None,
            "autoincrement": True,
        }
        assert _detect_autoincrement(col_info, True, ["id"], "mysql") is True  # type: ignore[arg-type]

    def test_non_autoincrement(self) -> None:
        col_info: dict[str, Any] = {
            "name": "age",
            "type": sa_types.Integer(),
            "nullable": True,
            "default": None,
            "autoincrement": False,
        }
        assert _detect_autoincrement(col_info, False, ["id"], "mysql") is False  # type: ignore[arg-type]


# ── MySQL ENUM column ────────────────────────────────────────────────────


class TestMysqlEnumColumn:
    """MySQL ENUM reflected via _build_columns."""

    def test_enum_column(self) -> None:
        mock_inspector = MagicMock()
        mock_inspector.get_check_constraints.return_value = []
        raw_columns: list[Any] = [
            {
                "name": "status",
                "type": mysql.ENUM("active", "inactive"),
                "nullable": False,
                "default": "'active'",
            }
        ]
        columns = _build_columns(raw_columns, [], "mysql", "users", mock_inspector)  # type: ignore[arg-type]
        assert columns[0].data_type.value == "enum"
        assert columns[0].enum_values == ["active", "inactive"]


# ── MySQL SET column ─────────────────────────────────────────────────────


class TestMysqlSetColumn:
    """MySQL SET reflected via _build_columns."""

    def test_set_column(self) -> None:
        mock_inspector = MagicMock()
        mock_inspector.get_check_constraints.return_value = []
        raw_columns: list[Any] = [
            {
                "name": "permissions",
                "type": mysql.SET("read", "write", "admin"),
                "nullable": False,
                "default": None,
            }
        ]
        columns = _build_columns(raw_columns, [], "mysql", "roles", mock_inspector)  # type: ignore[arg-type]
        # SET values become enum_values; _build_columns promotes to ENUM
        assert columns[0].data_type.value == "enum"
        assert columns[0].enum_values == ["admin", "read", "write"]


# ── MySQL column comments ────────────────────────────────────────────────


class TestMysqlColumnComments:
    def test_comment_extracted(self) -> None:
        mock_inspector = MagicMock()
        mock_inspector.get_check_constraints.return_value = []
        raw_columns: list[Any] = [
            {
                "name": "email",
                "type": sa_types.VARCHAR(255),
                "nullable": False,
                "default": None,
                "comment": "Primary contact email",
            }
        ]
        columns = _build_columns(raw_columns, [], "mysql", "users", mock_inspector)  # type: ignore[arg-type]
        assert columns[0].comment == "Primary contact email"


# ── MySQL FK ─────────────────────────────────────────────────────────────


class TestMysqlFk:
    def test_on_update_cascade(self) -> None:
        raw_fks: list[Any] = [
            {
                "name": "fk_order_user",
                "constrained_columns": ["user_id"],
                "referred_table": "users",
                "referred_columns": ["id"],
                "options": {"ondelete": "CASCADE", "onupdate": "CASCADE"},
            }
        ]
        fks = _build_foreign_keys(raw_fks)  # type: ignore[arg-type]
        assert fks[0].on_update == "CASCADE"
        assert fks[0].on_delete == "CASCADE"
        assert fks[0].deferrable is False
        assert fks[0].initially is None
