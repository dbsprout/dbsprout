from __future__ import annotations

from pathlib import Path

import pytest

from dbsprout.migrate.parsers import MigrationParser
from dbsprout.migrate.parsers.alembic import AlembicParser


class TestScaffold:
    def test_implements_protocol(self) -> None:
        assert isinstance(AlembicParser(), MigrationParser)

    def test_empty_project_raises(self, tmp_path: Path) -> None:
        from dbsprout.migrate.parsers import MigrationParseError  # noqa: PLC0415

        with pytest.raises(MigrationParseError):
            AlembicParser().detect_changes(tmp_path)


class TestDiscovery:
    def test_uses_alembic_ini_script_location(self, tmp_path: Path) -> None:
        (tmp_path / "alembic.ini").write_text(
            "[alembic]\nscript_location = alembic\n", encoding="utf-8"
        )
        versions = tmp_path / "alembic" / "versions"
        versions.mkdir(parents=True)
        from dbsprout.migrate.parsers.alembic import _discover_versions_dir  # noqa: PLC0415

        assert _discover_versions_dir(tmp_path) == versions

    def test_fallback_alembic_versions(self, tmp_path: Path) -> None:
        versions = tmp_path / "alembic" / "versions"
        versions.mkdir(parents=True)
        from dbsprout.migrate.parsers.alembic import _discover_versions_dir  # noqa: PLC0415

        assert _discover_versions_dir(tmp_path) == versions

    def test_fallback_migrations_versions(self, tmp_path: Path) -> None:
        versions = tmp_path / "migrations" / "versions"
        versions.mkdir(parents=True)
        from dbsprout.migrate.parsers.alembic import _discover_versions_dir  # noqa: PLC0415

        assert _discover_versions_dir(tmp_path) == versions

    def test_missing_raises(self, tmp_path: Path) -> None:
        from dbsprout.migrate.parsers import MigrationParseError  # noqa: PLC0415
        from dbsprout.migrate.parsers.alembic import _discover_versions_dir  # noqa: PLC0415

        with pytest.raises(MigrationParseError):
            _discover_versions_dir(tmp_path)


class TestCollectRevisions:
    def test_parses_revision_ids(self, tmp_path: Path) -> None:
        versions = tmp_path / "versions"
        versions.mkdir()
        (versions / "0001_init.py").write_text(
            'revision = "aaaa"\ndown_revision = None\n\ndef upgrade(): pass\n',
            encoding="utf-8",
        )
        (versions / "0002_next.py").write_text(
            'revision = "bbbb"\ndown_revision = "aaaa"\n\ndef upgrade(): pass\n',
            encoding="utf-8",
        )
        from dbsprout.migrate.parsers.alembic import _collect_revisions  # noqa: PLC0415

        revs = _collect_revisions(versions)
        by_id = {r.revision: r for r in revs}
        assert by_id["aaaa"].down_revision is None
        assert by_id["bbbb"].down_revision == "aaaa"

    def test_missing_revision_raises(self, tmp_path: Path) -> None:
        versions = tmp_path / "versions"
        versions.mkdir()
        (versions / "bad.py").write_text(
            "down_revision = None\n\ndef upgrade(): pass\n", encoding="utf-8"
        )
        from dbsprout.migrate.parsers import MigrationParseError  # noqa: PLC0415
        from dbsprout.migrate.parsers.alembic import _collect_revisions  # noqa: PLC0415

        with pytest.raises(MigrationParseError, match="revision"):
            _collect_revisions(versions)

    def test_missing_down_revision_raises(self, tmp_path: Path) -> None:
        versions = tmp_path / "versions"
        versions.mkdir()
        (versions / "bad.py").write_text(
            'revision = "x"\n\ndef upgrade(): pass\n', encoding="utf-8"
        )
        from dbsprout.migrate.parsers import MigrationParseError  # noqa: PLC0415
        from dbsprout.migrate.parsers.alembic import _collect_revisions  # noqa: PLC0415

        with pytest.raises(MigrationParseError, match="down_revision"):
            _collect_revisions(versions)

    def test_size_cap(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        versions = tmp_path / "versions"
        versions.mkdir()
        (versions / "big.py").write_text(
            'revision = "x"\ndown_revision = None\n\ndef upgrade(): pass\n',
            encoding="utf-8",
        )
        from dbsprout.migrate.parsers import MigrationParseError  # noqa: PLC0415
        from dbsprout.migrate.parsers import alembic as am  # noqa: PLC0415

        monkeypatch.setattr(am, "_MAX_REVISION_BYTES", 4)
        with pytest.raises(MigrationParseError, match="too large"):
            am._collect_revisions(versions)


class TestLinearize:
    def _rev(self, rid: str, down: str | None) -> object:
        import ast  # noqa: PLC0415

        from dbsprout.migrate.parsers.alembic import _Revision  # noqa: PLC0415

        return _Revision(
            path=Path(f"{rid}.py"),
            revision=rid,
            down_revision=down,
            module=ast.parse(""),
        )

    def test_empty(self) -> None:
        from dbsprout.migrate.parsers.alembic import _linearize_revisions  # noqa: PLC0415

        assert _linearize_revisions([]) == []

    def test_linear_chain(self) -> None:
        from dbsprout.migrate.parsers.alembic import _linearize_revisions  # noqa: PLC0415

        revs = [self._rev("c", "b"), self._rev("a", None), self._rev("b", "a")]
        ordered = _linearize_revisions(revs)
        assert [r.revision for r in ordered] == ["a", "b", "c"]

    def test_multiple_heads(self) -> None:
        from dbsprout.migrate.parsers import MigrationParseError  # noqa: PLC0415
        from dbsprout.migrate.parsers.alembic import _linearize_revisions  # noqa: PLC0415

        revs = [self._rev("a", None), self._rev("b", "a"), self._rev("c", "a")]
        with pytest.raises(MigrationParseError, match="head"):
            _linearize_revisions(revs)

    def test_multiple_roots(self) -> None:
        from dbsprout.migrate.parsers import MigrationParseError  # noqa: PLC0415
        from dbsprout.migrate.parsers.alembic import _linearize_revisions  # noqa: PLC0415

        revs = [self._rev("a", None), self._rev("b", None)]
        with pytest.raises(MigrationParseError, match=r"head|root"):
            _linearize_revisions(revs)


class TestWalker:
    def _parse_upgrade_body(self, body: str) -> list:
        import ast  # noqa: PLC0415

        from dbsprout.migrate.parsers.alembic import _parse_upgrade, _Revision  # noqa: PLC0415

        src = f'revision = "r"\ndown_revision = None\n\ndef upgrade():\n{body}\n'
        module = ast.parse(src)
        rev = _Revision(path=Path("r.py"), revision="r", down_revision=None, module=module)
        return _parse_upgrade(rev)

    def test_unknown_verb_is_skipped(self) -> None:
        # op.execute is intentionally unmapped
        changes = self._parse_upgrade_body('    op.execute("SELECT 1")')
        assert changes == []

    def test_non_op_call_ignored(self) -> None:
        changes = self._parse_upgrade_body('    print("hi")')
        assert changes == []

    def test_missing_upgrade_raises(self) -> None:
        import ast  # noqa: PLC0415

        from dbsprout.migrate.parsers import MigrationParseError  # noqa: PLC0415
        from dbsprout.migrate.parsers.alembic import _parse_upgrade, _Revision  # noqa: PLC0415

        module = ast.parse('revision = "r"\ndown_revision = None\n')
        rev = _Revision(path=Path("r.py"), revision="r", down_revision=None, module=module)
        with pytest.raises(MigrationParseError, match="upgrade"):
            _parse_upgrade(rev)


class TestTableOps:
    def _run(self, body: str) -> list:
        import ast  # noqa: PLC0415

        from dbsprout.migrate.parsers.alembic import _parse_upgrade, _Revision  # noqa: PLC0415

        src = f'revision = "r"\ndown_revision = None\n\ndef upgrade():\n{body}\n'
        module = ast.parse(src)
        rev = _Revision(path=Path("r.py"), revision="r", down_revision=None, module=module)
        return _parse_upgrade(rev)

    def test_create_table_empty(self) -> None:
        from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

        changes = self._run('    op.create_table("users")')
        assert len(changes) == 1
        assert changes[0].change_type == SchemaChangeType.TABLE_ADDED
        assert changes[0].table_name == "users"

    def test_create_table_with_columns(self) -> None:
        from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

        body = (
            "    op.create_table(\n"
            '        "items",\n'
            '        sa.Column("id", sa.Integer(), nullable=False),\n'
            '        sa.Column("name", sa.String(length=120), nullable=True),\n'
            "    )"
        )
        changes = self._run(body)
        assert len(changes) == 1
        c = changes[0]
        assert c.change_type == SchemaChangeType.TABLE_ADDED
        assert c.table_name == "items"
        assert c.detail is not None
        cols = c.detail["columns"]
        assert cols[0]["name"] == "id"
        assert "Integer" in cols[0]["alembic_type"]
        assert cols[0]["nullable"] is False
        assert cols[1]["name"] == "name"
        assert cols[1]["nullable"] is True

    def test_drop_table(self) -> None:
        from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

        changes = self._run('    op.drop_table("users")')
        assert len(changes) == 1
        assert changes[0].change_type == SchemaChangeType.TABLE_REMOVED
        assert changes[0].table_name == "users"


class TestColumnAddDrop:
    def _run(self, body: str) -> list:
        import ast  # noqa: PLC0415

        from dbsprout.migrate.parsers.alembic import _parse_upgrade, _Revision  # noqa: PLC0415

        src = f'revision = "r"\ndown_revision = None\n\ndef upgrade():\n{body}\n'
        module = ast.parse(src)
        rev = _Revision(path=Path("r.py"), revision="r", down_revision=None, module=module)
        return _parse_upgrade(rev)

    def test_add_column(self) -> None:
        from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

        body = (
            "    op.add_column(\n"
            '        "items",\n'
            '        sa.Column("created_at", sa.DateTime(), nullable=True),\n'
            "    )"
        )
        changes = self._run(body)
        assert len(changes) == 1
        c = changes[0]
        assert c.change_type == SchemaChangeType.COLUMN_ADDED
        assert c.table_name == "items"
        assert c.column_name == "created_at"
        assert c.detail is not None
        assert "DateTime" in c.detail["alembic_type"]
        assert c.detail["nullable"] is True

    def test_drop_column(self) -> None:
        from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

        changes = self._run('    op.drop_column("items", "legacy_col")')
        assert len(changes) == 1
        c = changes[0]
        assert c.change_type == SchemaChangeType.COLUMN_REMOVED
        assert c.table_name == "items"
        assert c.column_name == "legacy_col"


class TestAlterColumn:
    def _run(self, body: str) -> list:
        import ast  # noqa: PLC0415

        from dbsprout.migrate.parsers.alembic import _parse_upgrade, _Revision  # noqa: PLC0415

        src = f'revision = "r"\ndown_revision = None\n\ndef upgrade():\n{body}\n'
        module = ast.parse(src)
        rev = _Revision(path=Path("r.py"), revision="r", down_revision=None, module=module)
        return _parse_upgrade(rev)

    def test_alter_type_only(self) -> None:
        from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

        changes = self._run('    op.alter_column("items", "name", type_=sa.String(length=200))')
        assert len(changes) == 1
        assert changes[0].change_type == SchemaChangeType.COLUMN_TYPE_CHANGED
        assert changes[0].table_name == "items"
        assert changes[0].column_name == "name"
        assert changes[0].new_value is not None
        assert "String" in changes[0].new_value

    def test_alter_nullable_only(self) -> None:
        from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

        changes = self._run('    op.alter_column("items", "name", nullable=False)')
        assert len(changes) == 1
        assert changes[0].change_type == SchemaChangeType.COLUMN_NULLABILITY_CHANGED
        assert changes[0].new_value == "False"

    def test_alter_server_default_only(self) -> None:
        from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

        changes = self._run('    op.alter_column("items", "active", server_default="1")')
        assert len(changes) == 1
        assert changes[0].change_type == SchemaChangeType.COLUMN_DEFAULT_CHANGED
        assert changes[0].new_value is not None
        assert "1" in changes[0].new_value

    def test_alter_multi_axis(self) -> None:
        from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

        changes = self._run('    op.alter_column("items", "n", type_=sa.Text(), nullable=True)')
        kinds = {c.change_type for c in changes}
        assert kinds == {
            SchemaChangeType.COLUMN_TYPE_CHANGED,
            SchemaChangeType.COLUMN_NULLABILITY_CHANGED,
        }

    def test_alter_no_recognized_kw_is_noop(self) -> None:
        changes = self._run('    op.alter_column("items", "n", comment="x")')
        assert changes == []


class TestForeignKey:
    def _run(self, body: str) -> list:
        import ast  # noqa: PLC0415

        from dbsprout.migrate.parsers.alembic import _parse_upgrade, _Revision  # noqa: PLC0415

        src = f'revision = "r"\ndown_revision = None\n\ndef upgrade():\n{body}\n'
        module = ast.parse(src)
        rev = _Revision(path=Path("r.py"), revision="r", down_revision=None, module=module)
        return _parse_upgrade(rev)

    def test_create_fk(self) -> None:
        from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

        body = (
            "    op.create_foreign_key(\n"
            '        "fk_items_owner", "items", "users",\n'
            '        ["owner_id"], ["id"]\n'
            "    )"
        )
        changes = self._run(body)
        assert len(changes) == 1
        c = changes[0]
        assert c.change_type == SchemaChangeType.FOREIGN_KEY_ADDED
        assert c.table_name == "items"
        assert c.detail == {
            "name": "fk_items_owner",
            "ref_table": "users",
            "local_cols": ["owner_id"],
            "remote_cols": ["id"],
        }

    def test_drop_foreign_key_positional(self) -> None:
        from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

        changes = self._run('    op.drop_constraint("fk_items_owner", "items", type_="foreignkey")')
        assert len(changes) == 1
        c = changes[0]
        assert c.change_type == SchemaChangeType.FOREIGN_KEY_REMOVED
        assert c.table_name == "items"
        assert c.detail == {"name": "fk_items_owner"}

    def test_drop_foreign_key_kwarg_table(self) -> None:
        from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

        changes = self._run(
            "    op.drop_constraint(\n"
            '        "fk_items_owner", table_name="items", type_="foreignkey"\n'
            "    )"
        )
        assert len(changes) == 1
        assert changes[0].change_type == SchemaChangeType.FOREIGN_KEY_REMOVED

    def test_drop_non_fk_constraint_skipped(self) -> None:
        # Only foreign-key constraints emit a change; other types are ignored.
        changes = self._run('    op.drop_constraint("uq_items_name", "items", type_="unique")')
        assert changes == []
