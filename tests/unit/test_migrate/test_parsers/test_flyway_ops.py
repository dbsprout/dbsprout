# tests/unit/test_migrate/test_parsers/test_flyway_ops.py
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest
import sqlglot

from dbsprout.migrate.models import SchemaChangeType
from dbsprout.migrate.parsers import MigrationParseError, MigrationParser
from dbsprout.migrate.parsers.flyway import (
    FlywayMigrationParser,
    _FKLedger,
    _parse_file,
    _split_qualified,
    _strip_quotes,
    _walk_statements,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestScaffold:
    def test_implements_protocol(self) -> None:
        assert isinstance(FlywayMigrationParser(), MigrationParser)

    def test_empty_project_raises(self, tmp_path: Path) -> None:
        with pytest.raises(MigrationParseError, match=r"no V\*__\*\.sql found"):
            FlywayMigrationParser().detect_changes(tmp_path)

    def test_frozen_dataclass(self) -> None:
        parser = FlywayMigrationParser()
        with pytest.raises(dataclasses.FrozenInstanceError):
            parser.dialect = "mysql"  # type: ignore[misc]

    def test_default_dialect_is_postgres(self) -> None:
        assert FlywayMigrationParser().dialect == "postgres"


class TestIdentifierNorm:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ('"Users"', "Users"),
            ("`Users`", "Users"),
            ("[Users]", "Users"),
            ("users", "users"),
        ],
    )
    def test_strip_quotes(self, raw: str, expected: str) -> None:
        assert _strip_quotes(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("users", (None, "users")),
            ("public.users", ("public", "users")),
            ('"app"."orders"', ("app", "orders")),
        ],
    )
    def test_split_qualified(self, raw: str, expected: tuple[str | None, str]) -> None:
        assert _split_qualified(raw) == expected


class TestParseFile:
    def test_parse_error_wrapped(self, tmp_path: Path) -> None:
        file_path = tmp_path / "V1__bad.sql"
        file_path.write_text("THIS IS NOT SQL ${{{;;;", encoding="utf-8")
        with pytest.raises(MigrationParseError, match="could not parse"):
            _parse_file(file_path, dialect="postgres", placeholders={})

    def test_placeholder_applied_before_parse(self, tmp_path: Path) -> None:
        file_path = tmp_path / "V1__ok.sql"
        file_path.write_text("CREATE TABLE ${schema}.users (id INT);", encoding="utf-8")
        stmts = _parse_file(file_path, dialect="postgres", placeholders={"schema": "public"})
        assert len(stmts) == 1

    def test_unresolved_placeholder_raises(self, tmp_path: Path) -> None:
        file_path = tmp_path / "V1__ok.sql"
        file_path.write_text("CREATE TABLE ${schema}.users (id INT);", encoding="utf-8")
        with pytest.raises(MigrationParseError, match="unresolved placeholder"):
            _parse_file(file_path, dialect="postgres", placeholders={})


class TestCreateDropTable:
    def test_create_table(self) -> None:
        stmts = sqlglot.parse(
            "CREATE TABLE authors (id INT PRIMARY KEY, name VARCHAR(120) NOT NULL);",
            read="postgres",
        )
        changes = _walk_statements(stmts, dialect="postgres", ledger=_FKLedger())
        assert len(changes) == 1
        c = changes[0]
        assert c.change_type is SchemaChangeType.TABLE_ADDED
        assert c.table_name == "authors"
        assert c.detail is not None
        cols = c.detail["columns"]
        assert cols[0]["name"] == "id"
        assert cols[0]["primary_key"] is True
        assert cols[1]["name"] == "name"
        assert cols[1]["sql_type"].upper().startswith("VARCHAR")
        assert cols[1]["nullable"] is False

    def test_create_table_with_inline_fk(self) -> None:
        stmts = sqlglot.parse(
            "CREATE TABLE books (id INT, author_id INT REFERENCES authors(id));",
            read="postgres",
        )
        changes = _walk_statements(stmts, dialect="postgres", ledger=_FKLedger())
        assert changes[0].change_type is SchemaChangeType.TABLE_ADDED
        fks = changes[0].detail["foreign_keys"]
        assert fks[0]["ref_table"] == "authors"
        assert fks[0]["local_cols"] == ["author_id"]
        assert fks[0]["remote_cols"] == ["id"]

    def test_create_table_with_schema_prefix(self) -> None:
        stmts = sqlglot.parse("CREATE TABLE app.orders (id INT);", read="postgres")
        changes = _walk_statements(stmts, dialect="postgres", ledger=_FKLedger())
        assert changes[0].table_name == "orders"
        assert changes[0].detail["schema"] == "app"

    def test_drop_table(self) -> None:
        stmts = sqlglot.parse("DROP TABLE authors;", read="postgres")
        changes = _walk_statements(stmts, dialect="postgres", ledger=_FKLedger())
        assert len(changes) == 1
        assert changes[0].change_type is SchemaChangeType.TABLE_REMOVED
        assert changes[0].table_name == "authors"
