from __future__ import annotations

from pathlib import Path  # noqa: TC003

import pytest

from dbsprout.migrate.parsers import MigrationParser
from dbsprout.migrate.parsers.alembic import AlembicParser


class TestScaffold:
    def test_implements_protocol(self) -> None:
        assert isinstance(AlembicParser(), MigrationParser)

    def test_empty_project_raises(self, tmp_path: Path) -> None:
        from dbsprout.migrate.parsers import MigrationParseError  # noqa: PLC0415

        with pytest.raises(MigrationParseError):
            AlembicParser().detect_changes(tmp_path)
