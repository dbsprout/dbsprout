# tests/unit/test_migrate/test_parsers/test_flyway_integration.py
from __future__ import annotations

from pathlib import Path

import pytest

from dbsprout.migrate.models import SchemaChangeType
from dbsprout.migrate.parsers import MigrationParseError
from dbsprout.migrate.parsers.flyway import FlywayMigrationParser

FIXTURES = Path(__file__).parent / "fixtures"


class TestFlywayProjectSimple:
    def test_walks_full_history(self) -> None:
        parser = FlywayMigrationParser()
        changes = parser.detect_changes(FIXTURES / "flyway_project_simple")
        kinds = [c.change_type for c in changes]
        assert SchemaChangeType.TABLE_ADDED in kinds
        assert SchemaChangeType.COLUMN_TYPE_CHANGED in kinds
        assert SchemaChangeType.TABLE_REMOVED in kinds
        # Inline FK in CREATE TABLE is captured in TABLE_ADDED.detail["foreign_keys"]
        books_added = next(
            c
            for c in changes
            if c.change_type is SchemaChangeType.TABLE_ADDED and c.table_name == "books"
        )
        assert books_added.detail is not None
        assert len(books_added.detail["foreign_keys"]) > 0

    def test_duplicate_version_raises(self) -> None:
        parser = FlywayMigrationParser()
        with pytest.raises(MigrationParseError, match="duplicate Flyway version"):
            parser.detect_changes(FIXTURES / "flyway_project_duplicate_version")


class TestDialectSwitch:
    def test_mysql_modify_column(self, tmp_path: Path) -> None:
        from tests.unit.test_migrate.test_parsers.conftest import (  # noqa: PLC0415
            build_flyway_project,
        )

        build_flyway_project(
            tmp_path,
            {"V1__t": "CREATE TABLE t (c INT);"},
        )
        build_flyway_project(
            tmp_path,
            {"V2__alter": "ALTER TABLE t MODIFY COLUMN c BIGINT;"},
        )
        parser = FlywayMigrationParser(dialect="mysql")
        changes = parser.detect_changes(tmp_path)
        kinds = [c.change_type for c in changes]
        assert SchemaChangeType.COLUMN_TYPE_CHANGED in kinds
