from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dbsprout.migrate.parsers import MigrationParseError, MigrationParser
from dbsprout.migrate.parsers.django import DjangoMigrationParser

if TYPE_CHECKING:
    from pathlib import Path


class TestScaffold:
    def test_implements_protocol(self) -> None:
        assert isinstance(DjangoMigrationParser(), MigrationParser)

    def test_empty_project_raises(self, tmp_path: Path) -> None:
        with pytest.raises(MigrationParseError, match=r"no .*migrations"):
            DjangoMigrationParser().detect_changes(tmp_path)

    def test_frozen_dataclass(self) -> None:
        parser = DjangoMigrationParser()
        with pytest.raises((AttributeError, TypeError)):
            parser.foo = "bar"  # type: ignore[attr-defined]
