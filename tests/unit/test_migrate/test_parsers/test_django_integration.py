from __future__ import annotations

from pathlib import Path

import pytest

from dbsprout.migrate.models import SchemaChangeType
from dbsprout.migrate.parsers import MigrationParseError
from dbsprout.migrate.parsers.django import DjangoMigrationParser

FIXTURE = Path(__file__).parent / "fixtures" / "django_project_simple"
CYCLE = Path(__file__).parent / "fixtures" / "django_project_cycle"


def test_golden_integration() -> None:
    changes = DjangoMigrationParser().detect_changes(FIXTURE)
    kinds = [c.change_type for c in changes]
    # Order: accounts.User created first (FK target), then blog.Post + alter + add.
    assert kinds[0] is SchemaChangeType.TABLE_ADDED
    assert changes[0].table_name == "accounts_user"
    tables_added = [c.table_name for c in changes if c.change_type is SchemaChangeType.TABLE_ADDED]
    assert "blog_post" in tables_added
    assert any(
        c.change_type is SchemaChangeType.COLUMN_ADDED and c.column_name == "subtitle"
        for c in changes
    )
    assert any(
        c.change_type is SchemaChangeType.FOREIGN_KEY_ADDED
        and c.detail["ref_table"] == "accounts_user"
        for c in changes
    )


def test_cycle_raises_with_both_nodes() -> None:
    with pytest.raises(MigrationParseError) as exc_info:
        DjangoMigrationParser().detect_changes(CYCLE)
    msg = str(exc_info.value)
    assert "cycle" in msg
    assert "app_a" in msg
    assert "app_b" in msg
