# tests/unit/test_migrate/test_parsers/test_flyway_ops.py
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest

from dbsprout.migrate.parsers import MigrationParseError, MigrationParser
from dbsprout.migrate.parsers.flyway import (
    FlywayMigrationParser,
    _parse_file,
    _split_qualified,
    _strip_quotes,
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
