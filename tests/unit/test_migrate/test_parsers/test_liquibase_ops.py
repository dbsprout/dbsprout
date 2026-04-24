# tests/unit/test_migrate/test_parsers/test_liquibase_ops.py
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest

from dbsprout.migrate.parsers import MigrationParseError, MigrationParser
from dbsprout.migrate.parsers.liquibase import LiquibaseMigrationParser, _strip_ns

if TYPE_CHECKING:
    from pathlib import Path


class TestScaffold:
    def test_implements_protocol(self) -> None:
        assert isinstance(LiquibaseMigrationParser(), MigrationParser)

    def test_frozen_dataclass(self) -> None:
        parser = LiquibaseMigrationParser()
        with pytest.raises(dataclasses.FrozenInstanceError):
            parser.changelog_file = "other.xml"  # type: ignore[misc]

    def test_empty_project_raises(self, tmp_path: Path) -> None:
        with pytest.raises(MigrationParseError, match="no Liquibase changelog"):
            LiquibaseMigrationParser().detect_changes(tmp_path)


class TestStripNamespace:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("{http://www.liquibase.org/xml/ns/dbchangelog}createTable", "createTable"),
            ("createTable", "createTable"),
            ("{x}y", "y"),
            ("", ""),
        ],
    )
    def test_strip_ns(self, raw: str, expected: str) -> None:
        assert _strip_ns(raw) == expected


class TestWalkerSkipsUnknownElements:
    def test_unknown_top_level_tag_is_skipped(self, tmp_path: Path) -> None:
        body = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<databaseChangeLog xmlns="http://www.liquibase.org/xml/ns/dbchangelog">\n'
            "  <unknownThing/>\n"
            "</databaseChangeLog>\n"
        )
        (tmp_path / "changelog.xml").write_text(body, encoding="utf-8")
        result = LiquibaseMigrationParser().detect_changes(tmp_path)
        assert result == []
