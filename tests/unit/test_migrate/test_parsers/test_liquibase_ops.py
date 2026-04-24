# tests/unit/test_migrate/test_parsers/test_liquibase_ops.py
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest

from dbsprout.migrate.models import SchemaChange, SchemaChangeType  # noqa: F401
from dbsprout.migrate.parsers import MigrationParseError, MigrationParser
from dbsprout.migrate.parsers.liquibase import LiquibaseMigrationParser, _strip_ns
from tests.unit.test_migrate.test_parsers.conftest import build_liquibase_project

if TYPE_CHECKING:
    from pathlib import Path


_NS = 'xmlns="http://www.liquibase.org/xml/ns/dbchangelog"'


def _wrap(inner: str) -> str:
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f"<databaseChangeLog {_NS}>\n"
        f'  <changeSet id="c1" author="alice">\n'
        f"    {inner}\n"
        f"  </changeSet>\n"
        f"</databaseChangeLog>\n"
    )


class TestScaffold:
    def test_implements_protocol(self) -> None:
        assert isinstance(LiquibaseMigrationParser(), MigrationParser)

    def test_frozen_dataclass(self) -> None:
        parser = LiquibaseMigrationParser()
        with pytest.raises(dataclasses.FrozenInstanceError):
            parser.changelog_file = "other.xml"  # type: ignore[misc]

    def test_empty_project_raises(self, tmp_path: Path) -> None:
        with pytest.raises(MigrationParseError, match="no Liquibase changelog"):
            LiquibaseMigrationParser().detect_changes(tmp_path)


class TestStripNamespace:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("{http://www.liquibase.org/xml/ns/dbchangelog}createTable", "createTable"),
            ("createTable", "createTable"),
            ("{x}y", "y"),
            ("", ""),
        ],
    )
    def test_strip_ns(self, raw: str, expected: str) -> None:
        assert _strip_ns(raw) == expected


class TestWalkerSkipsUnknownElements:
    def test_unknown_top_level_tag_is_skipped(self, tmp_path: Path) -> None:
        body = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<databaseChangeLog xmlns="http://www.liquibase.org/xml/ns/dbchangelog">\n'
            "  <unknownThing/>\n"
            "</databaseChangeLog>\n"
        )
        (tmp_path / "changelog.xml").write_text(body, encoding="utf-8")
        result = LiquibaseMigrationParser().detect_changes(tmp_path)
        assert result == []


class TestCreateTable:
    def test_simple_create_table(self, tmp_path: Path) -> None:
        project = build_liquibase_project(
            tmp_path,
            changelogs={
                "changelog.xml": _wrap(
                    '<createTable tableName="users">'
                    '<column name="id" type="BIGINT">'
                    '<constraints primaryKey="true" nullable="false"/>'
                    "</column>"
                    '<column name="email" type="VARCHAR(255)"/>'
                    "</createTable>"
                ),
            },
        )
        [change] = LiquibaseMigrationParser().detect_changes(project)
        assert change.change_type is SchemaChangeType.TABLE_ADDED
        assert change.table_name == "users"
        detail = change.detail or {}
        cols = detail["columns"]
        assert cols[0] == {
            "name": "id",
            "sql_type": "BIGINT",
            "nullable": False,
            "default": None,
            "primary_key": True,
        }
        assert cols[1] == {
            "name": "email",
            "sql_type": "VARCHAR(255)",
            "nullable": True,
            "default": None,
            "primary_key": False,
        }
        assert detail["foreign_keys"] == []

    def test_create_table_with_inline_fk(self, tmp_path: Path) -> None:
        project = build_liquibase_project(
            tmp_path,
            changelogs={
                "changelog.xml": _wrap(
                    '<createTable tableName="orders">'
                    '<column name="id" type="BIGINT"/>'
                    '<column name="user_id" type="BIGINT">'
                    '<constraints foreignKeyName="fk_orders_user" references="users(id)"/>'
                    "</column>"
                    "</createTable>"
                ),
            },
        )
        changes = LiquibaseMigrationParser().detect_changes(project)
        assert [c.change_type for c in changes] == [
            SchemaChangeType.TABLE_ADDED,
            SchemaChangeType.FOREIGN_KEY_ADDED,
        ]
        fk = changes[1].detail or {}
        assert fk["constraint_name"] == "fk_orders_user"
        assert fk["local_cols"] == ["user_id"]
        assert fk["ref_table"] == "users"
        assert fk["remote_cols"] == ["id"]

    def test_create_table_with_schema_name(self, tmp_path: Path) -> None:
        project = build_liquibase_project(
            tmp_path,
            changelogs={
                "changelog.xml": _wrap('<createTable schemaName="app" tableName="users"/>'),
            },
        )
        [change] = LiquibaseMigrationParser().detect_changes(project)
        assert change.table_name == "users"
        assert (change.detail or {})["schema"] == "app"


class TestDropTable:
    def test_drop_table(self, tmp_path: Path) -> None:
        project = build_liquibase_project(
            tmp_path,
            changelogs={
                "changelog.xml": _wrap('<dropTable tableName="old_users"/>'),
            },
        )
        [change] = LiquibaseMigrationParser().detect_changes(project)
        assert change.change_type is SchemaChangeType.TABLE_REMOVED
        assert change.table_name == "old_users"


class TestRename:
    def test_rename_table(self, tmp_path: Path) -> None:
        project = build_liquibase_project(
            tmp_path,
            changelogs={
                "changelog.xml": _wrap(
                    '<renameTable oldTableName="users" newTableName="accounts"/>'
                ),
            },
        )
        changes = LiquibaseMigrationParser().detect_changes(project)
        assert [c.change_type for c in changes] == [
            SchemaChangeType.TABLE_REMOVED,
            SchemaChangeType.TABLE_ADDED,
        ]
        assert changes[0].table_name == "users"
        assert (changes[0].detail or {})["rename_of"] == "accounts"
        assert changes[1].table_name == "accounts"
        assert (changes[1].detail or {})["rename_of"] == "users"

    def test_rename_column(self, tmp_path: Path) -> None:
        project = build_liquibase_project(
            tmp_path,
            changelogs={
                "changelog.xml": _wrap(
                    '<renameColumn tableName="users"'
                    ' oldColumnName="fullname" newColumnName="display_name"/>'
                ),
            },
        )
        changes = LiquibaseMigrationParser().detect_changes(project)
        assert [c.change_type for c in changes] == [
            SchemaChangeType.COLUMN_REMOVED,
            SchemaChangeType.COLUMN_ADDED,
        ]
        assert changes[0].column_name == "fullname"
        assert (changes[0].detail or {})["rename_of"] == "display_name"
        assert changes[1].column_name == "display_name"
        assert (changes[1].detail or {})["rename_of"] == "fullname"


class TestAddColumn:
    def test_single_add_column(self, tmp_path: Path) -> None:
        project = build_liquibase_project(
            tmp_path,
            changelogs={
                "changelog.xml": _wrap(
                    '<addColumn tableName="users"><column name="age" type="INTEGER"/></addColumn>'
                ),
            },
        )
        [change] = LiquibaseMigrationParser().detect_changes(project)
        assert change.change_type is SchemaChangeType.COLUMN_ADDED
        assert change.table_name == "users"
        assert change.column_name == "age"
        detail = change.detail or {}
        assert detail["sql_type"] == "INTEGER"
        assert detail["nullable"] is True
        assert detail["default"] is None
        assert detail["primary_key"] is False

    def test_add_column_with_inline_fk(self, tmp_path: Path) -> None:
        project = build_liquibase_project(
            tmp_path,
            changelogs={
                "changelog.xml": _wrap(
                    '<addColumn tableName="orders">'
                    '<column name="user_id" type="BIGINT">'
                    '<constraints foreignKeyName="fk_o_u" references="users(id)"/>'
                    "</column>"
                    "</addColumn>"
                ),
            },
        )
        changes = LiquibaseMigrationParser().detect_changes(project)
        assert [c.change_type for c in changes] == [
            SchemaChangeType.COLUMN_ADDED,
            SchemaChangeType.FOREIGN_KEY_ADDED,
        ]
        assert (changes[1].detail or {})["constraint_name"] == "fk_o_u"

    def test_add_multiple_columns(self, tmp_path: Path) -> None:
        project = build_liquibase_project(
            tmp_path,
            changelogs={
                "changelog.xml": _wrap(
                    '<addColumn tableName="users">'
                    '<column name="age" type="INTEGER"/>'
                    '<column name="bio" type="TEXT"/>'
                    "</addColumn>"
                ),
            },
        )
        changes = LiquibaseMigrationParser().detect_changes(project)
        assert [c.column_name for c in changes] == ["age", "bio"]


class TestDropColumn:
    def test_drop_column_attr_form(self, tmp_path: Path) -> None:
        project = build_liquibase_project(
            tmp_path,
            changelogs={
                "changelog.xml": _wrap('<dropColumn tableName="users" columnName="age"/>'),
            },
        )
        [change] = LiquibaseMigrationParser().detect_changes(project)
        assert change.change_type is SchemaChangeType.COLUMN_REMOVED
        assert change.table_name == "users"
        assert change.column_name == "age"

    def test_drop_column_nested_form(self, tmp_path: Path) -> None:
        project = build_liquibase_project(
            tmp_path,
            changelogs={
                "changelog.xml": _wrap(
                    '<dropColumn tableName="users">'
                    '<column name="age"/>'
                    '<column name="bio"/>'
                    "</dropColumn>"
                ),
            },
        )
        changes = LiquibaseMigrationParser().detect_changes(project)
        assert [c.column_name for c in changes] == ["age", "bio"]


class TestModifyDataType:
    def test_modify_data_type(self, tmp_path: Path) -> None:
        project = build_liquibase_project(
            tmp_path,
            changelogs={
                "changelog.xml": _wrap(
                    '<modifyDataType tableName="users" columnName="email"'
                    ' newDataType="VARCHAR(320)"/>'
                ),
            },
        )
        [change] = LiquibaseMigrationParser().detect_changes(project)
        assert change.change_type is SchemaChangeType.COLUMN_TYPE_CHANGED
        assert change.table_name == "users"
        assert change.column_name == "email"
        assert change.new_value == "VARCHAR(320)"


class TestNullability:
    def test_add_not_null(self, tmp_path: Path) -> None:
        project = build_liquibase_project(
            tmp_path,
            changelogs={
                "changelog.xml": _wrap(
                    '<addNotNullConstraint tableName="users" columnName="email"/>'
                ),
            },
        )
        [change] = LiquibaseMigrationParser().detect_changes(project)
        assert change.change_type is SchemaChangeType.COLUMN_NULLABILITY_CHANGED
        assert change.new_value == "NOT NULL"

    def test_drop_not_null(self, tmp_path: Path) -> None:
        project = build_liquibase_project(
            tmp_path,
            changelogs={
                "changelog.xml": _wrap(
                    '<dropNotNullConstraint tableName="users" columnName="bio"/>'
                ),
            },
        )
        [change] = LiquibaseMigrationParser().detect_changes(project)
        assert change.change_type is SchemaChangeType.COLUMN_NULLABILITY_CHANGED
        assert change.new_value == "NULL"
