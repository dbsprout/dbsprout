# tests/unit/test_migrate/test_parsers/test_flyway_ops.py
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest

from dbsprout.migrate.parsers import MigrationParseError, MigrationParser
from dbsprout.migrate.parsers.flyway import FlywayMigrationParser

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
