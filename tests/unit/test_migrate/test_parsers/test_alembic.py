from __future__ import annotations

from pathlib import Path

import pytest

from dbsprout.migrate.parsers.alembic import AlembicParser


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


class TestIntegrationFixture:
    FIXTURE = Path(__file__).parent / "fixtures" / "alembic_project"

    def test_detect_changes_golden(self) -> None:
        from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

        changes = AlembicParser().detect_changes(self.FIXTURE)
        kinds = [c.change_type for c in changes]
        tables = [c.table_name for c in changes]

        # 0001: create_table(users) + create_index(ix_users_email)
        # 0002: add_column(users.created_at) + alter_column type_+nullable -> 2 changes
        # 0003: create_table(posts) + create_foreign_key
        # 0004: drop_constraint fk + drop_index + drop_column + drop_table
        assert kinds == [
            SchemaChangeType.TABLE_ADDED,
            SchemaChangeType.INDEX_ADDED,
            SchemaChangeType.COLUMN_ADDED,
            SchemaChangeType.COLUMN_TYPE_CHANGED,
            SchemaChangeType.COLUMN_NULLABILITY_CHANGED,
            SchemaChangeType.TABLE_ADDED,
            SchemaChangeType.FOREIGN_KEY_ADDED,
            SchemaChangeType.FOREIGN_KEY_REMOVED,
            SchemaChangeType.INDEX_REMOVED,
            SchemaChangeType.COLUMN_REMOVED,
            SchemaChangeType.TABLE_REMOVED,
        ]
        assert tables == [
            "users",
            "users",
            "users",
            "users",
            "users",
            "posts",
            "posts",
            "posts",
            "users",
            "users",
            "posts",
        ]


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


class TestEdgeBranches:
    """Cover small branches not hit by main-path tests."""

    def test_literal_list_rejects_non_literal(self) -> None:
        import ast  # noqa: PLC0415

        from dbsprout.migrate.parsers import MigrationParseError  # noqa: PLC0415
        from dbsprout.migrate.parsers.alembic import _literal_list  # noqa: PLC0415

        node = ast.parse("x", mode="eval").body  # ast.Name, not List/Tuple
        with pytest.raises(MigrationParseError, match="list/tuple"):
            _literal_list(node)

    def test_literal_rejects_non_string(self) -> None:
        import ast  # noqa: PLC0415

        from dbsprout.migrate.parsers import MigrationParseError  # noqa: PLC0415
        from dbsprout.migrate.parsers.alembic import _literal  # noqa: PLC0415

        node = ast.parse("42", mode="eval").body  # ast.Constant(int)
        with pytest.raises(MigrationParseError, match="string literal"):
            _literal(node)

    def test_create_fk_too_few_args(self) -> None:
        import ast  # noqa: PLC0415

        from dbsprout.migrate.parsers import MigrationParseError  # noqa: PLC0415
        from dbsprout.migrate.parsers.alembic import _parse_upgrade, _Revision  # noqa: PLC0415

        src = (
            'revision = "r"\ndown_revision = None\n\n'
            'def upgrade():\n    op.create_foreign_key("fk", "a", "b")\n'
        )
        module = ast.parse(src)
        rev = _Revision(path=Path("r.py"), revision="r", down_revision=None, module=module)
        with pytest.raises(MigrationParseError, match="create_foreign_key"):
            _parse_upgrade(rev)

    def test_create_index_too_few_args(self) -> None:
        import ast  # noqa: PLC0415

        from dbsprout.migrate.parsers import MigrationParseError  # noqa: PLC0415
        from dbsprout.migrate.parsers.alembic import _parse_upgrade, _Revision  # noqa: PLC0415

        src = (
            'revision = "r"\ndown_revision = None\n\n'
            'def upgrade():\n    op.create_index("ix", "t")\n'
        )
        module = ast.parse(src)
        rev = _Revision(path=Path("r.py"), revision="r", down_revision=None, module=module)
        with pytest.raises(MigrationParseError, match="create_index"):
            _parse_upgrade(rev)

    def test_drop_index_no_table(self) -> None:
        import ast  # noqa: PLC0415

        from dbsprout.migrate.parsers import MigrationParseError  # noqa: PLC0415
        from dbsprout.migrate.parsers.alembic import _parse_upgrade, _Revision  # noqa: PLC0415

        src = 'revision = "r"\ndown_revision = None\n\ndef upgrade():\n    op.drop_index("ix")\n'
        module = ast.parse(src)
        rev = _Revision(path=Path("r.py"), revision="r", down_revision=None, module=module)
        with pytest.raises(MigrationParseError, match="drop_index"):
            _parse_upgrade(rev)

    def test_drop_constraint_no_table(self) -> None:
        import ast  # noqa: PLC0415

        from dbsprout.migrate.parsers import MigrationParseError  # noqa: PLC0415
        from dbsprout.migrate.parsers.alembic import _parse_upgrade, _Revision  # noqa: PLC0415

        src = (
            'revision = "r"\ndown_revision = None\n\n'
            'def upgrade():\n    op.drop_constraint("fk", type_="foreignkey")\n'
        )
        module = ast.parse(src)
        rev = _Revision(path=Path("r.py"), revision="r", down_revision=None, module=module)
        with pytest.raises(MigrationParseError, match="drop_constraint"):
            _parse_upgrade(rev)

    def test_add_column_with_non_column_arg(self) -> None:
        import ast  # noqa: PLC0415

        from dbsprout.migrate.parsers import MigrationParseError  # noqa: PLC0415
        from dbsprout.migrate.parsers.alembic import _parse_upgrade, _Revision  # noqa: PLC0415

        src = (
            'revision = "r"\ndown_revision = None\n\n'
            'def upgrade():\n    op.add_column("t", "not_a_column_call")\n'
        )
        module = ast.parse(src)
        rev = _Revision(path=Path("r.py"), revision="r", down_revision=None, module=module)
        with pytest.raises(MigrationParseError, match="Column"):
            _parse_upgrade(rev)

    def test_skips_dunder_files(self, tmp_path: Path) -> None:
        from dbsprout.migrate.parsers.alembic import _collect_revisions  # noqa: PLC0415

        versions = tmp_path / "versions"
        versions.mkdir()
        (versions / "__init__.py").write_text("# marker\n", encoding="utf-8")
        (versions / "0001.py").write_text(
            'revision = "a"\ndown_revision = None\n\ndef upgrade(): pass\n',
            encoding="utf-8",
        )
        revs = _collect_revisions(versions)
        assert len(revs) == 1
        assert revs[0].revision == "a"

    def test_multiple_upgrade_raises(self) -> None:
        import ast  # noqa: PLC0415

        from dbsprout.migrate.parsers import MigrationParseError  # noqa: PLC0415
        from dbsprout.migrate.parsers.alembic import _parse_upgrade, _Revision  # noqa: PLC0415

        src = 'revision = "r"\ndown_revision = None\n\ndef upgrade(): pass\ndef upgrade(): pass\n'
        module = ast.parse(src)
        rev = _Revision(path=Path("r.py"), revision="r", down_revision=None, module=module)
        with pytest.raises(MigrationParseError, match="Multiple"):
            _parse_upgrade(rev)

    def test_non_constant_revision_ignored(self) -> None:
        import ast  # noqa: PLC0415

        from dbsprout.migrate.parsers import MigrationParseError  # noqa: PLC0415
        from dbsprout.migrate.parsers.alembic import _extract_revision_ids  # noqa: PLC0415

        # revision assigned from a non-constant expression - should raise "missing revision"
        module = ast.parse('revision = f"x"\ndown_revision = None\n')
        with pytest.raises(MigrationParseError, match="revision"):
            _extract_revision_ids(module, Path("r.py"))

    def test_alter_column_server_default_none_literal(self) -> None:
        import ast  # noqa: PLC0415

        from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415
        from dbsprout.migrate.parsers.alembic import _parse_upgrade, _Revision  # noqa: PLC0415

        src = (
            'revision = "r"\ndown_revision = None\n\n'
            'def upgrade():\n    op.alter_column("t", "c", server_default=None)\n'
        )
        module = ast.parse(src)
        rev = _Revision(path=Path("r.py"), revision="r", down_revision=None, module=module)
        changes = _parse_upgrade(rev)
        assert len(changes) == 1
        assert changes[0].change_type == SchemaChangeType.COLUMN_DEFAULT_CHANGED
        assert changes[0].new_value == "None"

    def test_create_table_nullable_non_literal(self) -> None:
        """Covers the ValueError fallback in _extract_column_spec for nullable."""
        import ast  # noqa: PLC0415

        from dbsprout.migrate.parsers.alembic import _parse_upgrade, _Revision  # noqa: PLC0415

        src = (
            'revision = "r"\ndown_revision = None\n\n'
            "def upgrade():\n"
            "    op.create_table(\n"
            '        "t",\n'
            '        sa.Column("c", sa.Integer(), nullable=some_var),\n'
            "    )\n"
        )
        module = ast.parse(src)
        rev = _Revision(path=Path("r.py"), revision="r", down_revision=None, module=module)
        changes = _parse_upgrade(rev)
        assert len(changes) == 1
        cols = changes[0].detail["columns"]  # type: ignore[index]
        assert cols[0]["nullable"] == "some_var"

    def test_column_node_without_name_arg(self) -> None:
        """Covers the 'no args / non-constant name' branch of _extract_column_spec."""
        import ast  # noqa: PLC0415

        from dbsprout.migrate.parsers.alembic import _parse_upgrade, _Revision  # noqa: PLC0415

        src = (
            'revision = "r"\ndown_revision = None\n\n'
            "def upgrade():\n"
            '    op.create_table("t", sa.Column(get_name(), sa.Integer()))\n'
        )
        module = ast.parse(src)
        rev = _Revision(path=Path("r.py"), revision="r", down_revision=None, module=module)
        changes = _parse_upgrade(rev)
        assert len(changes) == 1
        cols = changes[0].detail["columns"]  # type: ignore[index]
        assert cols[0]["name"] == "get_name()"

    def test_create_index_non_literal_unique(self) -> None:
        """Covers the ValueError fallback in _handle_create_index for unique."""
        import ast  # noqa: PLC0415

        from dbsprout.migrate.parsers.alembic import _parse_upgrade, _Revision  # noqa: PLC0415

        src = (
            'revision = "r"\ndown_revision = None\n\n'
            "def upgrade():\n"
            '    op.create_index("ix", "t", ["c"], unique=compute_unique())\n'
        )
        module = ast.parse(src)
        rev = _Revision(path=Path("r.py"), revision="r", down_revision=None, module=module)
        changes = _parse_upgrade(rev)
        assert changes[0].detail["unique"] is False  # type: ignore[index]

    def test_module_without_upgrade_via_detect_changes(self, tmp_path: Path) -> None:
        """End-to-end: versions dir with a revision missing upgrade()."""
        from dbsprout.migrate.parsers import MigrationParseError  # noqa: PLC0415

        versions = tmp_path / "alembic" / "versions"
        versions.mkdir(parents=True)
        (versions / "0001.py").write_text(
            'revision = "a"\ndown_revision = None\n', encoding="utf-8"
        )
        with pytest.raises(MigrationParseError, match="upgrade"):
            AlembicParser().detect_changes(tmp_path)


class TestHandlerErrorsCarryFilePath:
    """§10 contract — handler-raised MigrationParseError carries the revision path."""

    def test_bad_add_column_has_file_path(self, tmp_path: Path) -> None:
        from dbsprout.migrate.parsers import MigrationParseError  # noqa: PLC0415

        versions = tmp_path / "alembic" / "versions"
        versions.mkdir(parents=True)
        rev = versions / "0001.py"
        rev.write_text(
            'revision = "a"\ndown_revision = None\n\n'
            'def upgrade():\n    op.add_column("t", "not_a_column")\n',
            encoding="utf-8",
        )
        with pytest.raises(MigrationParseError) as exc_info:
            AlembicParser().detect_changes(tmp_path)
        assert exc_info.value.file_path == rev

    def test_literal_error_has_file_path(self, tmp_path: Path) -> None:
        from dbsprout.migrate.parsers import MigrationParseError  # noqa: PLC0415

        versions = tmp_path / "alembic" / "versions"
        versions.mkdir(parents=True)
        rev = versions / "0001.py"
        rev.write_text(
            'revision = "a"\ndown_revision = None\n\n'
            "def upgrade():\n    op.drop_table(TABLE_CONST)\n",
            encoding="utf-8",
        )
        with pytest.raises(MigrationParseError) as exc_info:
            AlembicParser().detect_changes(tmp_path)
        assert exc_info.value.file_path == rev
