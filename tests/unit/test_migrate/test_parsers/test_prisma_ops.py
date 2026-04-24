# tests/unit/test_migrate/test_parsers/test_prisma_ops.py
"""Prisma parser dispatch tests exercising the shared SQL walker via detect_changes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dbsprout.migrate.models import SchemaChangeType
from dbsprout.migrate.parsers import MigrationParseError
from dbsprout.migrate.parsers.prisma import PrismaMigrationParser
from tests.unit.test_migrate.test_parsers.conftest import build_prisma_project

if TYPE_CHECKING:
    from pathlib import Path


class TestCreateTable:
    def test_create_table_emits_table_added_with_columns_and_fks(self, tmp_path: Path) -> None:
        build_prisma_project(
            tmp_path,
            {
                "20240101000000_init": (
                    "CREATE TABLE users (id SERIAL PRIMARY KEY, email TEXT NOT NULL);\n"
                    "CREATE TABLE posts ("
                    "  id SERIAL PRIMARY KEY,"
                    "  user_id INT NOT NULL,"
                    "  FOREIGN KEY (user_id) REFERENCES users (id)"
                    ");"
                ),
            },
        )
        changes = PrismaMigrationParser().detect_changes(tmp_path)
        users, posts = changes[0], changes[1]
        assert users.change_type is SchemaChangeType.TABLE_ADDED
        assert users.table_name == "users"
        assert posts.change_type is SchemaChangeType.TABLE_ADDED
        assert posts.table_name == "posts"
        assert posts.detail is not None
        fks = posts.detail["foreign_keys"]
        assert fks == [
            {
                "constraint_name": None,
                "local_cols": ["user_id"],
                "ref_table": "users",
                "remote_cols": ["id"],
            }
        ]


class TestAlterTable:
    def test_add_column_with_inline_fk_emits_column_and_fk_pair(self, tmp_path: Path) -> None:
        build_prisma_project(
            tmp_path,
            {
                "20240101000000_init": (
                    "CREATE TABLE users (id SERIAL PRIMARY KEY);\n"
                    "CREATE TABLE posts (id SERIAL PRIMARY KEY);\n"
                ),
                "20240102000000_add_author": (
                    "ALTER TABLE posts ADD COLUMN author_id INT REFERENCES users (id);"
                ),
            },
        )
        changes = PrismaMigrationParser().detect_changes(tmp_path)
        added = [c for c in changes if c.change_type is SchemaChangeType.COLUMN_ADDED]
        fks = [c for c in changes if c.change_type is SchemaChangeType.FOREIGN_KEY_ADDED]
        assert any(c.table_name == "posts" and c.column_name == "author_id" for c in added)
        assert any(
            c.table_name == "posts" and c.detail is not None and c.detail["ref_table"] == "users"
            for c in fks
        )

    def test_rename_table_emits_removed_added_pair(self, tmp_path: Path) -> None:
        build_prisma_project(
            tmp_path,
            {
                "20240101000000_init": "CREATE TABLE old_name (id SERIAL PRIMARY KEY);",
                "20240102000000_rename": "ALTER TABLE old_name RENAME TO new_name;",
            },
        )
        changes = PrismaMigrationParser().detect_changes(tmp_path)
        removed = [c for c in changes if c.change_type is SchemaChangeType.TABLE_REMOVED]
        added = [c for c in changes if c.change_type is SchemaChangeType.TABLE_ADDED]
        assert any(c.table_name == "old_name" for c in removed)
        assert any(
            c.table_name == "new_name" and c.detail == {"rename_of": "old_name"} for c in added
        )


class TestFKLedger:
    def test_add_constraint_then_drop_constraint_resolves_via_ledger(self, tmp_path: Path) -> None:
        build_prisma_project(
            tmp_path,
            {
                "20240101000000_init": (
                    "CREATE TABLE users (id SERIAL PRIMARY KEY);\n"
                    "CREATE TABLE posts (id SERIAL PRIMARY KEY, user_id INT);\n"
                ),
                "20240102000000_add_fk": (
                    "ALTER TABLE posts ADD CONSTRAINT fk_posts_users "
                    "FOREIGN KEY (user_id) REFERENCES users (id);"
                ),
                "20240103000000_drop_fk": ("ALTER TABLE posts DROP CONSTRAINT fk_posts_users;"),
            },
        )
        changes = PrismaMigrationParser().detect_changes(tmp_path)
        removed = [c for c in changes if c.change_type is SchemaChangeType.FOREIGN_KEY_REMOVED]
        assert len(removed) == 1
        assert removed[0].table_name == "posts"
        assert removed[0].detail == {"constraint_name": "fk_posts_users"}


class TestIndex:
    def test_create_and_drop_index_emits_paired_changes(self, tmp_path: Path) -> None:
        build_prisma_project(
            tmp_path,
            {
                "20240101000000_init": "CREATE TABLE users (id SERIAL PRIMARY KEY, email TEXT);",
                "20240102000000_ix": "CREATE INDEX users_email_ix ON users (email);",
                "20240103000000_drop_ix": "DROP INDEX users_email_ix;",
            },
        )
        changes = PrismaMigrationParser().detect_changes(tmp_path)
        added = [c for c in changes if c.change_type is SchemaChangeType.INDEX_ADDED]
        removed = [c for c in changes if c.change_type is SchemaChangeType.INDEX_REMOVED]
        assert len(added) == 1
        assert added[0].detail is not None
        assert added[0].detail["index_name"] == "users_email_ix"
        assert added[0].detail["cols"] == ["email"]
        assert len(removed) == 1
        assert removed[0].detail is not None
        assert removed[0].detail["index_name"] == "users_email_ix"


class TestErrorPaths:
    def test_empty_migration_file_yields_no_changes_from_that_file(self, tmp_path: Path) -> None:
        build_prisma_project(
            tmp_path,
            {
                "20240101000000_init": "CREATE TABLE t (id INT);",
                "20240102000000_empty": "",
            },
        )
        changes = PrismaMigrationParser().detect_changes(tmp_path)
        assert len(changes) == 1
        assert changes[0].change_type is SchemaChangeType.TABLE_ADDED
        assert changes[0].table_name == "t"

    def test_sqlglot_parse_error_carries_file_path(self, tmp_path: Path) -> None:
        build_prisma_project(
            tmp_path,
            {"20240101000000_bad": "CREATE @#@# TABLE @###;;"},
        )
        with pytest.raises(MigrationParseError, match=r"could not parse") as ei:
            PrismaMigrationParser().detect_changes(tmp_path)
        assert ei.value.file_path is not None
        assert ei.value.file_path.name == "migration.sql"
