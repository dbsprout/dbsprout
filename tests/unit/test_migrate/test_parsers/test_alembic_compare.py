"""compare_metadata() tests for the Alembic migration parser.

Covers:
- TestCompareMetadata — live SQLAlchemy MetaData vs live engine comparison,
  including the alembic-missing import-guard path.
- TestTranslateAlembicDiff — direct unit tests for the internal
  _translate_alembic_diff() helper against synthesized diff tuples.

Intentional late imports (# noqa: PLC0415) are preserved so the
`test_raises_when_alembic_missing` case can monkeypatch builtins.__import__.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dbsprout.migrate.parsers.alembic import AlembicParser

if TYPE_CHECKING:
    from pathlib import Path


class TestCompareMetadata:
    def test_raises_when_alembic_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins  # noqa: PLC0415

        from dbsprout.migrate.parsers import MigrationParseError  # noqa: PLC0415

        real_import = builtins.__import__

        def fake_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name.startswith("alembic"):
                raise ImportError("pretend missing")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        import sqlalchemy as sa  # noqa: PLC0415

        md = sa.MetaData()
        with pytest.raises(MigrationParseError, match="alembic"):
            AlembicParser().compare_metadata("sqlite:///:memory:", md)

    def test_adds_missing_table(self, tmp_path: Path) -> None:
        pytest.importorskip("alembic")
        import sqlalchemy as sa  # noqa: PLC0415

        from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

        db_path = tmp_path / "compare.db"
        engine = sa.create_engine(f"sqlite:///{db_path}")
        md_empty = sa.MetaData()
        md_empty.create_all(engine)

        md_with_table = sa.MetaData()
        sa.Table(
            "users",
            md_with_table,
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("name", sa.String(50)),
        )

        changes = AlembicParser().compare_metadata(f"sqlite:///{db_path}", md_with_table)
        kinds = [c.change_type for c in changes]
        assert SchemaChangeType.TABLE_ADDED in kinds
        added = next(c for c in changes if c.change_type == SchemaChangeType.TABLE_ADDED)
        assert added.table_name == "users"


class TestTranslateAlembicDiff:
    """Unit tests exercising every branch of _translate_alembic_diff via synthetic tuples."""

    def _fake_table(self, name: str) -> object:
        return type("FakeTable", (), {"name": name})()

    def _fake_column(self, name: str, type_str: str = "INT") -> object:
        return type(
            "FakeColumn",
            (),
            {"name": name, "type": type_str},
        )()

    def _fake_index(self, name: str, table_name: str) -> object:
        return type(
            "FakeIndex",
            (),
            {"name": name, "table": self._fake_table(table_name)},
        )()

    def _fake_fk(self, name: str | None, table_name: str) -> object:
        return type(
            "FakeFK",
            (),
            {"name": name, "table": self._fake_table(table_name)},
        )()

    def test_add_table(self) -> None:
        from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415
        from dbsprout.migrate.parsers.alembic import _translate_alembic_diff  # noqa: PLC0415

        diff = ("add_table", self._fake_table("users"))
        result = _translate_alembic_diff(diff)
        assert result is not None
        assert result.change_type == SchemaChangeType.TABLE_ADDED
        assert result.table_name == "users"

    def test_remove_table(self) -> None:
        from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415
        from dbsprout.migrate.parsers.alembic import _translate_alembic_diff  # noqa: PLC0415

        diff = ("remove_table", self._fake_table("legacy"))
        result = _translate_alembic_diff(diff)
        assert result is not None
        assert result.change_type == SchemaChangeType.TABLE_REMOVED
        assert result.table_name == "legacy"

    def test_add_column(self) -> None:
        from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415
        from dbsprout.migrate.parsers.alembic import _translate_alembic_diff  # noqa: PLC0415

        diff = ("add_column", None, "users", self._fake_column("email", "VARCHAR(120)"))
        result = _translate_alembic_diff(diff)
        assert result is not None
        assert result.change_type == SchemaChangeType.COLUMN_ADDED
        assert result.table_name == "users"
        assert result.column_name == "email"
        assert result.detail == {"alembic_type": "VARCHAR(120)"}

    def test_remove_column(self) -> None:
        from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415
        from dbsprout.migrate.parsers.alembic import _translate_alembic_diff  # noqa: PLC0415

        diff = ("remove_column", None, "users", self._fake_column("old"))
        result = _translate_alembic_diff(diff)
        assert result is not None
        assert result.change_type == SchemaChangeType.COLUMN_REMOVED
        assert result.table_name == "users"
        assert result.column_name == "old"

    def test_modify_type(self) -> None:
        from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415
        from dbsprout.migrate.parsers.alembic import _translate_alembic_diff  # noqa: PLC0415

        # (verb, schema, table, column, existing_kw, old, new)  # noqa: ERA001
        diff = ("modify_type", None, "users", "email", {}, "VARCHAR(120)", "TEXT")
        result = _translate_alembic_diff(diff)
        assert result is not None
        assert result.change_type == SchemaChangeType.COLUMN_TYPE_CHANGED
        assert result.new_value == "TEXT"

    def test_modify_nullable(self) -> None:
        from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415
        from dbsprout.migrate.parsers.alembic import _translate_alembic_diff  # noqa: PLC0415

        diff = ("modify_nullable", None, "users", "email", {}, True, False)
        result = _translate_alembic_diff(diff)
        assert result is not None
        assert result.change_type == SchemaChangeType.COLUMN_NULLABILITY_CHANGED
        assert result.new_value == "False"

    def test_modify_default(self) -> None:
        from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415
        from dbsprout.migrate.parsers.alembic import _translate_alembic_diff  # noqa: PLC0415

        diff = ("modify_default", None, "users", "active", {}, "1", "0")
        result = _translate_alembic_diff(diff)
        assert result is not None
        assert result.change_type == SchemaChangeType.COLUMN_DEFAULT_CHANGED
        assert result.new_value == "0"

    def test_add_index(self) -> None:
        from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415
        from dbsprout.migrate.parsers.alembic import _translate_alembic_diff  # noqa: PLC0415

        diff = ("add_index", self._fake_index("ix_users_email", "users"))
        result = _translate_alembic_diff(diff)
        assert result is not None
        assert result.change_type == SchemaChangeType.INDEX_ADDED
        assert result.table_name == "users"
        assert result.detail == {"name": "ix_users_email"}

    def test_remove_index(self) -> None:
        from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415
        from dbsprout.migrate.parsers.alembic import _translate_alembic_diff  # noqa: PLC0415

        diff = ("remove_index", self._fake_index("ix_stale", "users"))
        result = _translate_alembic_diff(diff)
        assert result is not None
        assert result.change_type == SchemaChangeType.INDEX_REMOVED
        assert result.table_name == "users"

    def test_add_fk(self) -> None:
        from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415
        from dbsprout.migrate.parsers.alembic import _translate_alembic_diff  # noqa: PLC0415

        diff = ("add_fk", self._fake_fk("fk_posts_user", "posts"))
        result = _translate_alembic_diff(diff)
        assert result is not None
        assert result.change_type == SchemaChangeType.FOREIGN_KEY_ADDED
        assert result.table_name == "posts"
        assert result.detail == {"name": "fk_posts_user"}

    def test_remove_fk(self) -> None:
        from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415
        from dbsprout.migrate.parsers.alembic import _translate_alembic_diff  # noqa: PLC0415

        diff = ("remove_fk", self._fake_fk("fk_posts_user", "posts"))
        result = _translate_alembic_diff(diff)
        assert result is not None
        assert result.change_type == SchemaChangeType.FOREIGN_KEY_REMOVED

    def test_unknown_verb_returns_none(self) -> None:
        from dbsprout.migrate.parsers.alembic import _translate_alembic_diff  # noqa: PLC0415

        assert _translate_alembic_diff(("unknown_verb", None)) is None

    def test_flatten_nested_list(self) -> None:
        from dbsprout.migrate.parsers.alembic import _flatten  # noqa: PLC0415

        a = ("add_table", self._fake_table("a"))
        b = ("add_table", self._fake_table("b"))
        c = ("add_table", self._fake_table("c"))
        nested = [a, [b, [c]]]
        assert _flatten(nested) == [a, b, c]
