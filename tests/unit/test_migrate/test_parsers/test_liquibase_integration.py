"""End-to-end fixture-walk tests for LiquibaseMigrationParser."""

from __future__ import annotations

from pathlib import Path

from dbsprout.migrate.models import SchemaChangeType
from dbsprout.migrate.parsers.liquibase import LiquibaseMigrationParser

_FIXTURE = Path(__file__).parent / "fixtures" / "liquibase_project_simple"


class TestIntegration:
    def test_fixture_end_to_end(self) -> None:
        changes = LiquibaseMigrationParser().detect_changes(_FIXTURE)
        types = [c.change_type for c in changes]
        assert SchemaChangeType.TABLE_ADDED in types
        assert SchemaChangeType.FOREIGN_KEY_ADDED in types
        assert SchemaChangeType.COLUMN_ADDED in types
        assert SchemaChangeType.INDEX_ADDED in types

    def test_namespace_variants_equivalent(self, tmp_path: Path) -> None:
        def body(xsd_uri: str) -> str:
            return (
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                f'<databaseChangeLog xmlns="{xsd_uri}">\n'
                '  <changeSet id="c1" author="alice">\n'
                '    <createTable tableName="users"/>\n'
                "  </changeSet>\n"
                "</databaseChangeLog>\n"
            )

        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "changelog.xml").write_text(
            body("http://www.liquibase.org/xml/ns/dbchangelog"),
            encoding="utf-8",
        )
        (tmp_path / "b").mkdir()
        (tmp_path / "b" / "changelog.xml").write_text(
            body("http://www.liquibase.org/xml/ns/dbchangelog/dbchangelog-4.24.xsd"),
            encoding="utf-8",
        )
        a = LiquibaseMigrationParser().detect_changes(tmp_path / "a")
        b = LiquibaseMigrationParser().detect_changes(tmp_path / "b")
        assert a == b
