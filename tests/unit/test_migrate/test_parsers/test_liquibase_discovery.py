"""Discovery / include / error-path tests for LiquibaseMigrationParser."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
