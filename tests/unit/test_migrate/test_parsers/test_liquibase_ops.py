# tests/unit/test_migrate/test_parsers/test_liquibase_ops.py
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest

from dbsprout.migrate.parsers import MigrationParseError, MigrationParser
from dbsprout.migrate.parsers.liquibase import LiquibaseMigrationParser

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
