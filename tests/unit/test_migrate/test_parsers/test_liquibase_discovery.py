"""Discovery / include / error-path tests for LiquibaseMigrationParser."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dbsprout.migrate.parsers import MigrationParseError
from dbsprout.migrate.parsers.liquibase import LiquibaseMigrationParser
from tests.unit.test_migrate.test_parsers.conftest import build_liquibase_project

if TYPE_CHECKING:
    from pathlib import Path


class TestFixtureHelper:
    def test_build_liquibase_project_writes_files(self, tmp_path: Path) -> None:
        project = build_liquibase_project(
            tmp_path,
            changelogs={
                "db/changelog/db.changelog-master.xml": "<root/>",
                "db/changelog/01-init.xml": "<child/>",
            },
        )
        assert (project / "db/changelog/db.changelog-master.xml").read_text() == "<root/>"
        assert (project / "db/changelog/01-init.xml").read_text() == "<child/>"


class TestDiscovery:
    def test_explicit_changelog_file(self, tmp_path: Path) -> None:
        project = build_liquibase_project(
            tmp_path,
            changelogs={"my/custom/cl.xml": _EMPTY_CHANGELOG},
        )
        result = LiquibaseMigrationParser(changelog_file="my/custom/cl.xml").detect_changes(project)
        assert result == []

    def test_probe_default_db_changelog(self, tmp_path: Path) -> None:
        project = build_liquibase_project(
            tmp_path,
            changelogs={"db/changelog/db.changelog-master.xml": _EMPTY_CHANGELOG},
        )
        result = LiquibaseMigrationParser().detect_changes(project)
        assert result == []

    def test_probe_default_spring_resources(self, tmp_path: Path) -> None:
        project = build_liquibase_project(
            tmp_path,
            changelogs={
                "src/main/resources/db/changelog/db.changelog-master.xml": _EMPTY_CHANGELOG
            },
        )
        result = LiquibaseMigrationParser().detect_changes(project)
        assert result == []

    def test_probe_default_root(self, tmp_path: Path) -> None:
        project = build_liquibase_project(
            tmp_path,
            changelogs={"changelog.xml": _EMPTY_CHANGELOG},
        )
        result = LiquibaseMigrationParser().detect_changes(project)
        assert result == []

    def test_missing_explicit_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(MigrationParseError, match="no Liquibase changelog"):
            LiquibaseMigrationParser(changelog_file="nope.xml").detect_changes(tmp_path)

    def test_no_probe_match_raises(self, tmp_path: Path) -> None:
        with pytest.raises(MigrationParseError, match="no Liquibase changelog"):
            LiquibaseMigrationParser().detect_changes(tmp_path)


_EMPTY_CHANGELOG = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<databaseChangeLog xmlns="http://www.liquibase.org/xml/ns/dbchangelog"/>\n'
)
