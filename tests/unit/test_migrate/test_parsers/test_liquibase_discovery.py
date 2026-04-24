"""Discovery / include / error-path tests for LiquibaseMigrationParser."""

from __future__ import annotations

from pathlib import Path

import pytest

from dbsprout.migrate.models import SchemaChangeType
from dbsprout.migrate.parsers import MigrationParseError
from dbsprout.migrate.parsers.liquibase import LiquibaseMigrationParser
from tests.unit.test_migrate.test_parsers.conftest import build_liquibase_project


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


class TestFormatRejection:
    @pytest.mark.parametrize(
        "name",
        ["db/changelog/master.yaml", "db/changelog/master.yml", "master.json", "master.sql"],
    )
    def test_non_xml_changelog_raises(self, tmp_path: Path, name: str) -> None:
        project = build_liquibase_project(tmp_path, changelogs={name: "irrelevant"})
        with pytest.raises(MigrationParseError, match="not supported in v1"):
            LiquibaseMigrationParser(changelog_file=name).detect_changes(project)


class TestSecurity:
    def test_xxe_payload_rejected(self, tmp_path: Path) -> None:
        project = build_liquibase_project(
            tmp_path,
            changelogs={
                "changelog.xml": (
                    '<?xml version="1.0" encoding="UTF-8"?>\n'
                    '<!DOCTYPE foo [ <!ENTITY xxe SYSTEM "file:///etc/passwd"> ]>\n'
                    '<databaseChangeLog xmlns="http://www.liquibase.org/xml/ns/dbchangelog">\n'
                    '  <changeSet id="c1" author="alice">\n'
                    '    <createTable tableName="&xxe;"/>\n'
                    "  </changeSet>\n"
                    "</databaseChangeLog>\n"
                ),
            },
        )
        with pytest.raises(MigrationParseError):
            LiquibaseMigrationParser().detect_changes(project)

    def test_oversize_file_skipped(self, tmp_path: Path) -> None:
        body = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<databaseChangeLog xmlns="http://www.liquibase.org/xml/ns/dbchangelog">\n'
            + ("<!-- padding -->\n" * 70_000)
            + "</databaseChangeLog>\n"
        )
        project = build_liquibase_project(
            tmp_path,
            changelogs={"changelog.xml": body},
        )
        assert (project / "changelog.xml").stat().st_size > 1024 * 1024
        result = LiquibaseMigrationParser().detect_changes(project)
        assert result == []

    def test_parser_never_imports_stdlib_etree(self) -> None:
        src = Path("dbsprout/migrate/parsers/liquibase.py").read_text()
        assert "from xml.etree" not in src
        assert "import xml.etree" not in src


_EMPTY_CHANGELOG = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<databaseChangeLog xmlns="http://www.liquibase.org/xml/ns/dbchangelog"/>\n'
)


class TestExtraEdgeCases:
    def test_unknown_extension_raises(self, tmp_path: Path) -> None:
        project = build_liquibase_project(
            tmp_path,
            changelogs={"changelog.txt": "irrelevant"},
        )
        with pytest.raises(
            MigrationParseError,
            match="unknown Liquibase changelog extension",
        ):
            LiquibaseMigrationParser(changelog_file="changelog.txt").detect_changes(project)

    def test_include_all_missing_path_raises(self, tmp_path: Path) -> None:
        body = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<databaseChangeLog xmlns="http://www.liquibase.org/xml/ns/dbchangelog">\n'
            '  <includeAll path="missing_dir" relativeToChangelogFile="true"/>\n'
            "</databaseChangeLog>\n"
        )
        project = build_liquibase_project(
            tmp_path,
            changelogs={"db/changelog/db.changelog-master.xml": body},
        )
        with pytest.raises(MigrationParseError, match="not found at"):
            LiquibaseMigrationParser().detect_changes(project)

    def test_include_all_skips_oversize_child(self, tmp_path: Path) -> None:
        master_body = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<databaseChangeLog xmlns="http://www.liquibase.org/xml/ns/dbchangelog">\n'
            '  <includeAll path="children" relativeToChangelogFile="true"/>\n'
            "</databaseChangeLog>\n"
        )
        child_body = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<databaseChangeLog xmlns="http://www.liquibase.org/xml/ns/dbchangelog">\n'
            "<!-- " + ("x" * (1024 * 1024 + 1000)) + " -->\n"
            "</databaseChangeLog>\n"
        )
        project = build_liquibase_project(
            tmp_path,
            changelogs={
                "db/changelog/db.changelog-master.xml": master_body,
                "db/changelog/children/big.xml": child_body,
            },
        )
        assert (project / "db/changelog/children/big.xml").stat().st_size > 1024 * 1024
        result = LiquibaseMigrationParser().detect_changes(project)
        assert result == []


class TestIncludeAllSecurity:
    def test_include_all_skips_symlinks(self, tmp_path: Path) -> None:
        master_body = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<databaseChangeLog xmlns="http://www.liquibase.org/xml/ns/dbchangelog">\n'
            '  <includeAll path="children" relativeToChangelogFile="true"/>\n'
            "</databaseChangeLog>\n"
        )
        target_body = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<databaseChangeLog xmlns="http://www.liquibase.org/xml/ns/dbchangelog">\n'
            '  <changeSet id="c_sym" author="alice">\n'
            '    <createTable tableName="should_not_appear"/>\n'
            "  </changeSet>\n"
            "</databaseChangeLog>\n"
        )
        project = build_liquibase_project(
            tmp_path,
            changelogs={
                "db/changelog/db.changelog-master.xml": master_body,
                "outside/symtarget.xml": target_body,
            },
        )
        children_dir = project / "db/changelog/children"
        children_dir.mkdir(parents=True, exist_ok=True)
        symlink_path = children_dir / "link.xml"
        symlink_path.symlink_to(project / "outside/symtarget.xml")
        result = LiquibaseMigrationParser().detect_changes(project)
        assert result == []

    def test_include_all_skips_out_of_tree(self, tmp_path: Path) -> None:
        outside_dir = tmp_path / "outside_project"
        outside_dir.mkdir()
        external_child = outside_dir / "external.xml"
        external_child.write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<databaseChangeLog xmlns="http://www.liquibase.org/xml/ns/dbchangelog">\n'
            '  <changeSet id="c_ext" author="alice">\n'
            '    <createTable tableName="should_not_appear"/>\n'
            "  </changeSet>\n"
            "</databaseChangeLog>\n",
            encoding="utf-8",
        )
        project_root = tmp_path / "project"
        project_root.mkdir()
        children_dir = project_root / "db/changelog/children"
        children_dir.mkdir(parents=True, exist_ok=True)
        (children_dir / "escape.xml").symlink_to(external_child)
        master_body = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<databaseChangeLog xmlns="http://www.liquibase.org/xml/ns/dbchangelog">\n'
            '  <includeAll path="children" relativeToChangelogFile="true"/>\n'
            "</databaseChangeLog>\n"
        )
        (project_root / "db/changelog/db.changelog-master.xml").write_text(
            master_body,
            encoding="utf-8",
        )
        result = LiquibaseMigrationParser().detect_changes(project_root)
        assert result == []


class TestIncludePathEscape:
    def test_include_escaping_project_root_raises(self, tmp_path: Path) -> None:
        outside_dir = tmp_path / "outside_project"
        outside_dir.mkdir()
        external_file = outside_dir / "secret.xml"
        external_file.write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<databaseChangeLog xmlns="http://www.liquibase.org/xml/ns/dbchangelog">\n'
            '  <changeSet id="c_ext" author="alice">\n'
            '    <createTable tableName="should_not_appear"/>\n'
            "  </changeSet>\n"
            "</databaseChangeLog>\n",
            encoding="utf-8",
        )
        project_root = tmp_path / "project"
        project_root.mkdir()
        master_body = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<databaseChangeLog xmlns="http://www.liquibase.org/xml/ns/dbchangelog">\n'
            '  <include file="../outside_project/secret.xml" relativeToChangelogFile="true"/>\n'
            "</databaseChangeLog>\n"
        )
        (project_root / "changelog.xml").write_text(master_body, encoding="utf-8")
        with pytest.raises(MigrationParseError, match="escapes project root"):
            LiquibaseMigrationParser().detect_changes(project_root)
