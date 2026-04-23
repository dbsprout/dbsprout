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


class TestDiscovery:
    def test_uses_alembic_ini_script_location(self, tmp_path: Path) -> None:
        (tmp_path / "alembic.ini").write_text(
            "[alembic]\nscript_location = alembic\n", encoding="utf-8"
        )
        versions = tmp_path / "alembic" / "versions"
        versions.mkdir(parents=True)
        from dbsprout.migrate.parsers.alembic import _discover_versions_dir  # noqa: PLC0415

        assert _discover_versions_dir(tmp_path) == versions

    def test_fallback_alembic_versions(self, tmp_path: Path) -> None:
        versions = tmp_path / "alembic" / "versions"
        versions.mkdir(parents=True)
        from dbsprout.migrate.parsers.alembic import _discover_versions_dir  # noqa: PLC0415

        assert _discover_versions_dir(tmp_path) == versions

    def test_fallback_migrations_versions(self, tmp_path: Path) -> None:
        versions = tmp_path / "migrations" / "versions"
        versions.mkdir(parents=True)
        from dbsprout.migrate.parsers.alembic import _discover_versions_dir  # noqa: PLC0415

        assert _discover_versions_dir(tmp_path) == versions

    def test_missing_raises(self, tmp_path: Path) -> None:
        from dbsprout.migrate.parsers import MigrationParseError  # noqa: PLC0415
        from dbsprout.migrate.parsers.alembic import _discover_versions_dir  # noqa: PLC0415

        with pytest.raises(MigrationParseError):
            _discover_versions_dir(tmp_path)
