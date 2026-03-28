"""Tests for dbsprout.schema.introspect — PostgreSQL-specific introspection."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import sqlalchemy.types as sa_types
from sqlalchemy.dialects import postgresql as pg

from dbsprout.schema.introspect import (
    _build_columns,
    _build_foreign_keys,
    _detect_autoincrement,
    _extract_pg_enums,
    _get_raw_type,
)

# ── Autoincrement detection ──────────────────────────────────────────────


class TestPgAutoincrement:
    """PG SERIAL, BIGSERIAL, and IDENTITY autoincrement detection."""

    def test_serial_autoincrement_true(self) -> None:
        col_info: dict[str, Any] = {
            "name": "id",
            "type": sa_types.Integer(),
            "nullable": False,
            "default": "nextval('users_id_seq'::regclass)",
            "autoincrement": True,
        }
        assert _detect_autoincrement(col_info, True, ["id"], "postgresql") is True  # type: ignore[arg-type]

    def test_bigserial_autoincrement_true(self) -> None:
        col_info: dict[str, Any] = {
            "name": "id",
            "type": sa_types.BigInteger(),
            "nullable": False,
            "default": "nextval('events_id_seq'::regclass)",
            "autoincrement": True,
        }
        assert _detect_autoincrement(col_info, True, ["id"], "postgresql") is True  # type: ignore[arg-type]

    def test_identity_column(self) -> None:
        col_info: dict[str, Any] = {
            "name": "id",
            "type": sa_types.Integer(),
            "nullable": False,
            "default": None,
            "autoincrement": False,
            "identity": {"always": False, "start": 1, "increment": 1},
        }
        assert _detect_autoincrement(col_info, True, ["id"], "postgresql") is True  # type: ignore[arg-type]

    def test_identity_always(self) -> None:
        col_info: dict[str, Any] = {
            "name": "id",
            "type": sa_types.BigInteger(),
            "nullable": False,
            "default": None,
            "autoincrement": False,
            "identity": {"always": True, "start": 1, "increment": 1},
        }
        assert _detect_autoincrement(col_info, True, ["id"], "postgresql") is True  # type: ignore[arg-type]

    def test_regular_integer_not_autoincrement(self) -> None:
        col_info: dict[str, Any] = {
            "name": "age",
            "type": sa_types.Integer(),
            "nullable": True,
            "default": None,
        }
        assert _detect_autoincrement(col_info, False, ["id"], "postgresql") is False  # type: ignore[arg-type]

    def test_empty_identity_dict_not_autoincrement(self) -> None:
        col_info: dict[str, Any] = {
            "name": "age",
            "type": sa_types.Integer(),
            "nullable": True,
            "default": None,
            "identity": {},
        }
        assert _detect_autoincrement(col_info, False, ["id"], "postgresql") is False  # type: ignore[arg-type]


# ── PG Enum extraction ──────────────────────────────────────────────────


class TestPgEnumExtraction:
    """Test _extract_pg_enums with mocked Inspector."""

    def test_extracts_enums(self) -> None:
        mock_inspector = MagicMock()
        mock_inspector.get_enums.return_value = [
            {
                "name": "status_enum",
                "schema": "public",
                "labels": ["active", "inactive", "deleted"],
            },
            {"name": "role_enum", "schema": "public", "labels": ["admin", "user", "moderator"]},
        ]
        result = _extract_pg_enums(mock_inspector)
        assert result == {
            "status_enum": ["active", "deleted", "inactive"],
            "role_enum": ["admin", "moderator", "user"],
        }

    def test_empty_enums(self) -> None:
        mock_inspector = MagicMock()
        mock_inspector.get_enums.return_value = []
        result = _extract_pg_enums(mock_inspector)
        assert result == {}

    def test_no_get_enums_method(self) -> None:
        mock_inspector = MagicMock(spec=[])
        result = _extract_pg_enums(mock_inspector)
        assert result == {}


# ── Deferrable FK ────────────────────────────────────────────────────────


class TestPgDeferrableFk:
    def test_deferrable_fk(self) -> None:
        raw_fks: list[Any] = [
            {
                "name": "fk_order_user",
                "constrained_columns": ["user_id"],
                "referred_table": "users",
                "referred_columns": ["id"],
                "options": {"ondelete": "CASCADE", "deferrable": True, "initially": "DEFERRED"},
            }
        ]
        fks = _build_foreign_keys(raw_fks)  # type: ignore[arg-type]
        assert len(fks) == 1
        assert fks[0].deferrable is True

    def test_non_deferrable_fk(self) -> None:
        raw_fks: list[Any] = [
            {
                "name": "fk_product_cat",
                "constrained_columns": ["category_id"],
                "referred_table": "categories",
                "referred_columns": ["id"],
                "options": {"ondelete": "CASCADE"},
            }
        ]
        fks = _build_foreign_keys(raw_fks)  # type: ignore[arg-type]
        assert len(fks) == 1
        assert fks[0].deferrable is False


# ── ON UPDATE FK ─────────────────────────────────────────────────────────


class TestPgOnUpdateFk:
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

    def test_on_update_not_set(self) -> None:
        raw_fks: list[Any] = [
            {
                "name": "fk_order_user",
                "constrained_columns": ["user_id"],
                "referred_table": "users",
                "referred_columns": ["id"],
                "options": {},
            }
        ]
        fks = _build_foreign_keys(raw_fks)  # type: ignore[arg-type]
        assert fks[0].on_update is None


# ── Column comments ──────────────────────────────────────────────────────


class TestPgColumnComments:
    def test_comment_extracted(self) -> None:
        mock_inspector = MagicMock()
        mock_inspector.get_check_constraints.return_value = []
        raw_columns: list[Any] = [
            {
                "name": "email",
                "type": sa_types.VARCHAR(255),
                "nullable": False,
                "default": None,
                "comment": "User email address",
            }
        ]
        columns = _build_columns(raw_columns, [], "postgresql", "users", mock_inspector)  # type: ignore[arg-type]
        assert columns[0].comment == "User email address"

    def test_no_comment(self) -> None:
        mock_inspector = MagicMock()
        mock_inspector.get_check_constraints.return_value = []
        raw_columns: list[Any] = [
            {
                "name": "age",
                "type": sa_types.Integer(),
                "nullable": True,
                "default": None,
            }
        ]
        columns = _build_columns(raw_columns, [], "postgresql", "users", mock_inspector)  # type: ignore[arg-type]
        assert columns[0].comment is None


# ── Raw type for ENUM ────────────────────────────────────────────────────


class TestPgRawTypeEnum:
    def test_named_enum_raw_type(self) -> None:
        sa_type = pg.ENUM("active", "inactive", name="status_enum")
        assert _get_raw_type(sa_type) == "status_enum"

    def test_unnamed_enum_raw_type(self) -> None:
        sa_type = sa_types.Enum("a", "b")
        raw = _get_raw_type(sa_type)
        # Unnamed enum should fall through to compile()
        assert raw != ""

    def test_regular_type_raw_type(self) -> None:
        sa_type = sa_types.VARCHAR(100)
        assert _get_raw_type(sa_type) == "VARCHAR(100)"
