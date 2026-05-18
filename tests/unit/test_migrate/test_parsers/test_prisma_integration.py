# tests/unit/test_migrate/test_parsers/test_prisma_integration.py
"""End-to-end Prisma parser walk against the fixtures/prisma_project tree."""

from __future__ import annotations

from pathlib import Path

from dbsprout.migrate.models import SchemaChangeType
from dbsprout.migrate.parsers.prisma import PrismaMigrationParser

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "prisma_project"


def test_sample_prisma_project_emits_expected_change_sequence() -> None:
    changes = PrismaMigrationParser().detect_changes(FIXTURE_ROOT)
    seq = [(c.change_type, c.table_name, c.column_name) for c in changes]

    assert seq == [
        (SchemaChangeType.TABLE_ADDED, "users", None),
        (SchemaChangeType.TABLE_ADDED, "posts", None),
        (SchemaChangeType.COLUMN_ADDED, "posts", "author_id"),
        (SchemaChangeType.FOREIGN_KEY_ADDED, "posts", None),
        (SchemaChangeType.INDEX_ADDED, "posts", None),
    ]

    fk = next(c for c in changes if c.change_type is SchemaChangeType.FOREIGN_KEY_ADDED)
    assert fk.detail == {
        "constraint_name": "fk_posts_author",
        "local_cols": ["author_id"],
        "ref_table": "users",
        "remote_cols": ["id"],
    }

    ix = next(c for c in changes if c.change_type is SchemaChangeType.INDEX_ADDED)
    assert ix.detail == {"index_name": "posts_author_id_ix", "cols": ["author_id"]}
