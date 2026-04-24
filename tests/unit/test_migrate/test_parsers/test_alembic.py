from __future__ import annotations

from pathlib import Path

import pytest

from dbsprout.migrate.parsers.alembic import AlembicParser


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
