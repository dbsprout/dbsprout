"""Discovery / include / error-path tests for LiquibaseMigrationParser."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dbsprout.migrate.models import SchemaChangeType
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


class TestInclude:
    def test_include_relative_to_changelog(self, tmp_path: Path) -> None:
        project = build_liquibase_project(
            tmp_path,
            changelogs={
                "db/changelog/db.changelog-master.xml": (
                    '<?xml version="1.0" encoding="UTF-8"?>\n'
                    '<databaseChangeLog xmlns="http://www.liquibase.org/xml/ns/dbchangelog">\n'
                    '  <include file="01-init.xml" relativeToChangelogFile="true"/>\n'
                    "</databaseChangeLog>\n"
                ),
                "db/changelog/01-init.xml": (
                    '<?xml version="1.0" encoding="UTF-8"?>\n'
                    '<databaseChangeLog xmlns="http://www.liquibase.org/xml/ns/dbchangelog">\n'
                    '  <changeSet id="c1" author="alice">\n'
                    '    <createTable tableName="users"/>\n'
                    "  </changeSet>\n"
                    "</databaseChangeLog>\n"
                ),
            },
        )
        [change] = LiquibaseMigrationParser().detect_changes(project)
        assert change.change_type is SchemaChangeType.TABLE_ADDED
        assert change.table_name == "users"

    def test_include_relative_to_project(self, tmp_path: Path) -> None:
        project = build_liquibase_project(
            tmp_path,
            changelogs={
                "db/changelog/db.changelog-master.xml": (
                    '<?xml version="1.0" encoding="UTF-8"?>\n'
                    '<databaseChangeLog xmlns="http://www.liquibase.org/xml/ns/dbchangelog">\n'
                    '  <include file="db/changelog/01-init.xml"/>\n'
                    "</databaseChangeLog>\n"
                ),
                "db/changelog/01-init.xml": (
                    '<?xml version="1.0" encoding="UTF-8"?>\n'
                    '<databaseChangeLog xmlns="http://www.liquibase.org/xml/ns/dbchangelog">\n'
                    '  <changeSet id="c1" author="alice">\n'
                    '    <createTable tableName="users"/>\n'
                    "  </changeSet>\n"
                    "</databaseChangeLog>\n"
                ),
            },
        )
        [change] = LiquibaseMigrationParser().detect_changes(project)
        assert change.table_name == "users"

    def test_include_missing_raises(self, tmp_path: Path) -> None:
        project = build_liquibase_project(
            tmp_path,
            changelogs={
                "changelog.xml": (
                    '<?xml version="1.0" encoding="UTF-8"?>\n'
                    '<databaseChangeLog xmlns="http://www.liquibase.org/xml/ns/dbchangelog">\n'
                    '  <include file="missing.xml" relativeToChangelogFile="true"/>\n'
                    "</databaseChangeLog>\n"
                ),
            },
        )
        with pytest.raises(MigrationParseError, match="not found"):
            LiquibaseMigrationParser().detect_changes(project)

    def test_include_cycle_detected(self, tmp_path: Path) -> None:
        project = build_liquibase_project(
            tmp_path,
            changelogs={
                "changelog.xml": (
                    '<?xml version="1.0" encoding="UTF-8"?>\n'
                    '<databaseChangeLog xmlns="http://www.liquibase.org/xml/ns/dbchangelog">\n'
                    '  <include file="b.xml" relativeToChangelogFile="true"/>\n'
                    "</databaseChangeLog>\n"
                ),
                "b.xml": (
                    '<?xml version="1.0" encoding="UTF-8"?>\n'
                    '<databaseChangeLog xmlns="http://www.liquibase.org/xml/ns/dbchangelog">\n'
                    '  <include file="changelog.xml" relativeToChangelogFile="true"/>\n'
                    "</databaseChangeLog>\n"
                ),
            },
        )
        with pytest.raises(MigrationParseError, match="cycle"):
            LiquibaseMigrationParser().detect_changes(project)

    def test_include_all_alphabetical(self, tmp_path: Path) -> None:
        project = build_liquibase_project(
            tmp_path,
            changelogs={
                "db/changelog/db.changelog-master.xml": (
                    '<?xml version="1.0" encoding="UTF-8"?>\n'
                    '<databaseChangeLog xmlns="http://www.liquibase.org/xml/ns/dbchangelog">\n'
                    '  <includeAll path="changes" relativeToChangelogFile="true"/>\n'
                    "</databaseChangeLog>\n"
                ),
                "db/changelog/changes/02-second.xml": _wrap_ct("second"),
                "db/changelog/changes/01-first.xml": _wrap_ct("first"),
            },
        )
        changes = LiquibaseMigrationParser().detect_changes(project)
        assert [c.table_name for c in changes] == ["first", "second"]


def _wrap_ct(table_name: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<databaseChangeLog xmlns="http://www.liquibase.org/xml/ns/dbchangelog">\n'
        f'  <changeSet id="c_{table_name}" author="alice">\n'
        f'    <createTable tableName="{table_name}"/>\n'
        "  </changeSet>\n"
        "</databaseChangeLog>\n"
    )


class TestChangesetIdentity:
    def test_duplicate_identity_raises(self, tmp_path: Path) -> None:
        project = build_liquibase_project(
            tmp_path,
            changelogs={
                "db/changelog/db.changelog-master.xml": (
                    '<?xml version="1.0" encoding="UTF-8"?>\n'
                    '<databaseChangeLog xmlns="http://www.liquibase.org/xml/ns/dbchangelog">\n'
                    '  <include file="01-first.xml" relativeToChangelogFile="true"/>\n'
                    '  <include file="02-second.xml" relativeToChangelogFile="true"/>\n'
                    "</databaseChangeLog>\n"
                ),
                "db/changelog/01-first.xml": _wrap_ct_with_identity("alice", "c1", "users"),
                "db/changelog/02-second.xml": _wrap_ct_with_identity("alice", "c1", "accounts"),
            },
        )
        with pytest.raises(MigrationParseError, match="duplicate changeset alice:c1"):
            LiquibaseMigrationParser().detect_changes(project)

    def test_missing_id_raises(self, tmp_path: Path) -> None:
        project = build_liquibase_project(
            tmp_path,
            changelogs={
                "changelog.xml": (
                    '<?xml version="1.0" encoding="UTF-8"?>\n'
                    '<databaseChangeLog xmlns="http://www.liquibase.org/xml/ns/dbchangelog">\n'
                    '  <changeSet author="alice">\n'
                    '    <createTable tableName="users"/>\n'
                    "  </changeSet>\n"
                    "</databaseChangeLog>\n"
                ),
            },
        )
        with pytest.raises(MigrationParseError, match="missing id"):
            LiquibaseMigrationParser().detect_changes(project)

    def test_missing_author_raises(self, tmp_path: Path) -> None:
        project = build_liquibase_project(
            tmp_path,
            changelogs={
                "changelog.xml": (
                    '<?xml version="1.0" encoding="UTF-8"?>\n'
                    '<databaseChangeLog xmlns="http://www.liquibase.org/xml/ns/dbchangelog">\n'
                    '  <changeSet id="c1">\n'
                    '    <createTable tableName="users"/>\n'
                    "  </changeSet>\n"
                    "</databaseChangeLog>\n"
                ),
            },
        )
        with pytest.raises(MigrationParseError, match="missing author"):
            LiquibaseMigrationParser().detect_changes(project)


def _wrap_ct_with_identity(author: str, cs_id: str, table: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<databaseChangeLog xmlns="http://www.liquibase.org/xml/ns/dbchangelog">\n'
        f'  <changeSet id="{cs_id}" author="{author}">\n'
        f'    <createTable tableName="{table}"/>\n'
        "  </changeSet>\n"
        "</databaseChangeLog>\n"
    )


_EMPTY_CHANGELOG = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<databaseChangeLog xmlns="http://www.liquibase.org/xml/ns/dbchangelog"/>\n'
)
