from __future__ import annotations

import ast
import unittest.mock
from typing import TYPE_CHECKING

import pytest

from dbsprout.migrate.parsers import MigrationParseError
from dbsprout.migrate.parsers.django import (
    _MAX_MIGRATION_BYTES,
    DjangoMigrationParser,
    _discover_migration_files,
    _extract_dependencies,
    _extract_operations,
    _find_class_assign_value,
    _linearize_migrations,
    _op_name,
    _parse_migration_file,
    _ParsedMigration,
)
from tests.unit.test_migrate.test_parsers.conftest import EMPTY_MIG, build_django_project

if TYPE_CHECKING:
    from pathlib import Path

# Extra bytes added on top of _MAX_MIGRATION_BYTES to produce a clearly oversize file.
_OVERSIZE_PADDING = 100_000


def test_build_django_project_creates_structure(tmp_path: Path) -> None:
    body = "class Migration(migrations.Migration):\n    dependencies = []\n    operations = []\n"
    root = build_django_project(
        tmp_path,
        apps={"blog": [("0001_initial", body)]},
    )
    assert (root / "blog" / "migrations" / "__init__.py").exists()
    assert (root / "blog" / "migrations" / "0001_initial.py").exists()


class TestDiscovery:
    def test_finds_all_apps(self, tmp_path: Path) -> None:
        root = build_django_project(
            tmp_path,
            apps={
                "blog": [("0001_initial", EMPTY_MIG)],
                "accounts": [("0001_initial", EMPTY_MIG)],
            },
        )
        found = _discover_migration_files(root)
        stems = sorted(p.relative_to(root).as_posix() for p in found)
        assert stems == [
            "accounts/migrations/0001_initial.py",
            "blog/migrations/0001_initial.py",
        ]

    def test_skips_init_and_pycache(self, tmp_path: Path) -> None:
        root = build_django_project(tmp_path, apps={"blog": [("0001_initial", EMPTY_MIG)]})
        pyc_dir = root / "blog" / "migrations" / "__pycache__"
        pyc_dir.mkdir()
        (pyc_dir / "0001_initial.cpython-312.pyc").write_text("", encoding="utf-8")
        found = _discover_migration_files(root)
        assert len(found) == 1
        assert found[0].name == "0001_initial.py"

    def test_skips_oversize(self, tmp_path: Path) -> None:
        oversize = "x = '" + "a" * (_MAX_MIGRATION_BYTES + _OVERSIZE_PADDING) + "'\n"
        root = build_django_project(
            tmp_path,
            apps={"blog": [("0001_big", oversize)]},
        )
        found = _discover_migration_files(root)
        assert found == []

    def test_empty_project_raises(self, tmp_path: Path) -> None:
        with pytest.raises(MigrationParseError, match=r"no .*migrations"):
            DjangoMigrationParser().detect_changes(tmp_path)

    def test_no_app_dirs_raises(self, tmp_path: Path) -> None:
        (tmp_path / "manage.py").write_text("", encoding="utf-8")
        with pytest.raises(MigrationParseError, match=r"no .*migrations"):
            DjangoMigrationParser().detect_changes(tmp_path)


class TestWalker:
    def test_extracts_dependencies_and_operations(self, tmp_path: Path) -> None:
        body = (
            "from django.db import migrations\n\n"
            "class Migration(migrations.Migration):\n"
            "    dependencies = [('accounts', '0001_initial')]\n"
            "    operations = [\n"
            "        migrations.CreateModel(name='Post', fields=[]),\n"
            "        migrations.AddField("
            "model_name='Post', name='title', field=models.CharField()),\n"
            "    ]\n"
        )
        root = build_django_project(tmp_path, apps={"blog": [("0001_initial", body)]})
        path = root / "blog" / "migrations" / "0001_initial.py"
        parsed = _parse_migration_file(path, app_label="blog")
        assert parsed is not None
        assert parsed.app_label == "blog"
        assert parsed.name == "0001_initial"
        assert parsed.prefix == 1
        assert parsed.dependencies == (("accounts", "0001_initial"),)
        assert [op.func.attr for op in parsed.operations] == ["CreateModel", "AddField"]

    def test_missing_migration_class_returns_none(self, tmp_path: Path) -> None:
        body = "x = 1\n"
        root = build_django_project(tmp_path, apps={"blog": [("0001_stub", body)]})
        path = root / "blog" / "migrations" / "0001_stub.py"
        assert _parse_migration_file(path, app_label="blog") is None

    def test_syntax_error_raises(self, tmp_path: Path) -> None:
        body = "class Migration(\n"
        root = build_django_project(tmp_path, apps={"blog": [("0001_broken", body)]})
        path = root / "blog" / "migrations" / "0001_broken.py"
        with pytest.raises(MigrationParseError, match="unparseable"):
            _parse_migration_file(path, app_label="blog")

    def test_missing_numeric_prefix_raises(self, tmp_path: Path) -> None:
        body = EMPTY_MIG
        root = build_django_project(tmp_path, apps={"blog": [("initial_setup", body)]})
        path = root / "blog" / "migrations" / "initial_setup.py"
        with pytest.raises(MigrationParseError, match="numeric prefix"):
            _parse_migration_file(path, app_label="blog")

    def test_extracts_ann_assign_dependencies(self, tmp_path: Path) -> None:
        body = (
            "from django.db import migrations\n\n"
            "class Migration(migrations.Migration):\n"
            "    dependencies: list[tuple[str, str]] = [('accounts', '0001_initial')]\n"
            "    operations: list = []\n"
        )
        root = build_django_project(tmp_path, apps={"blog": [("0001_initial", body)]})
        path = root / "blog" / "migrations" / "0001_initial.py"
        parsed = _parse_migration_file(path, app_label="blog")
        assert parsed is not None
        assert parsed.dependencies == (("accounts", "0001_initial"),)
        assert parsed.operations == ()


def _parsed(
    app: str,
    name: str,
    prefix: int,
    deps: tuple[tuple[str, str], ...] = (),
) -> _ParsedMigration:
    from pathlib import Path  # noqa: PLC0415

    return _ParsedMigration(
        path=Path(f"/{app}/migrations/{name}.py"),
        app_label=app,
        name=name,
        prefix=prefix,
        dependencies=deps,
        operations=(),
    )


class TestLinearize:
    def test_in_app_prefix_order(self) -> None:
        a2 = _parsed("blog", "0002_add", 2)
        a1 = _parsed("blog", "0001_initial", 1)
        ordered = _linearize_migrations([a2, a1])
        assert [m.name for m in ordered] == ["0001_initial", "0002_add"]

    def test_cross_app_dependency_ordering(self) -> None:
        blog_1 = _parsed("blog", "0001_initial", 1, deps=(("accounts", "0001_initial"),))
        accounts_1 = _parsed("accounts", "0001_initial", 1)
        ordered = _linearize_migrations([blog_1, accounts_1])
        assert [(m.app_label, m.name) for m in ordered] == [
            ("accounts", "0001_initial"),
            ("blog", "0001_initial"),
        ]

    def test_cycle_raises(self) -> None:
        a = _parsed("app_a", "0001_initial", 1, deps=(("app_b", "0001_initial"),))
        b = _parsed("app_b", "0001_initial", 1, deps=(("app_a", "0001_initial"),))
        with pytest.raises(MigrationParseError, match="cycle"):
            _linearize_migrations([a, b])

    def test_duplicate_prefix_raises(self) -> None:
        x = _parsed("blog", "0001_a", 1)
        y = _parsed("blog", "0001_b", 1)
        with pytest.raises(MigrationParseError, match="duplicate migration prefix"):
            _linearize_migrations([x, y])

    def test_dangling_dep_to_unknown_app_is_ignored(self) -> None:
        m = _parsed("blog", "0001_initial", 1, deps=(("auth", "0001_initial"),))
        ordered = _linearize_migrations([m])
        assert ordered == [m]


class TestDiscoveryEdgeBranches:
    """Coverage for guard branches in _discover_migration_files and _parse_migration_file."""

    def test_stat_oserror_skips_file(self, tmp_path: Path) -> None:
        """File that raises OSError on stat() is silently skipped."""
        root = build_django_project(tmp_path, apps={"blog": [("0001_initial", EMPTY_MIG)]})
        target = root / "blog" / "migrations" / "0001_initial.py"

        # Build a mock path whose stat() raises but whose other methods work normally.
        mock_path = unittest.mock.MagicMock(spec=type(target))
        mock_path.name = target.name
        mock_path.parts = target.parts
        mock_path.resolve.return_value = target.resolve()
        mock_path.stat.side_effect = OSError("stat failed")

        with unittest.mock.patch.object(root.__class__, "rglob", return_value=[mock_path]):
            found = _discover_migration_files(root)
        assert mock_path not in found

    def test_read_text_oserror_raises_parse_error(self, tmp_path: Path) -> None:
        """File that raises OSError on read_text raises MigrationParseError."""
        root = build_django_project(tmp_path, apps={"blog": [("0001_initial", EMPTY_MIG)]})
        path = root / "blog" / "migrations" / "0001_initial.py"
        with (
            unittest.mock.patch.object(
                path.__class__, "read_text", side_effect=OSError("read failed")
            ),
            pytest.raises(MigrationParseError, match="unreadable"),
        ):
            _parse_migration_file(path, app_label="blog")


class TestExtractHelpers:
    """Unit tests for AST extraction helpers, covering non-standard input branches."""

    def _make_cls(self, body: str) -> ast.ClassDef:
        tree = ast.parse(f"class Migration:\n{body}")
        return next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef))

    def test_extract_dependencies_non_list_returns_empty(self) -> None:
        cls = self._make_cls("    dependencies = 'bad'\n")
        assert _extract_dependencies(cls) == ()

    def test_extract_dependencies_non_tuple_element_skipped(self) -> None:
        cls = self._make_cls("    dependencies = ['not_a_tuple']\n")
        assert _extract_dependencies(cls) == ()

    def test_extract_dependencies_non_literal_element_skipped(self) -> None:
        cls = self._make_cls("    dependencies = [(some_var, '0001_initial')]\n")
        assert _extract_dependencies(cls) == ()

    def test_extract_dependencies_non_str_constant_skipped(self) -> None:
        cls = self._make_cls("    dependencies = [(1, 2)]\n")
        assert _extract_dependencies(cls) == ()

    def test_extract_operations_non_list_returns_empty(self) -> None:
        cls = self._make_cls("    operations = 'bad'\n")
        assert _extract_operations(cls) == ()

    def test_find_class_assign_value_returns_none_when_missing(self) -> None:
        cls = self._make_cls("    pass\n")
        assert _find_class_assign_value(cls, "dependencies") is None

    def test_find_class_assign_value_ann_assign_no_value_returns_none(self) -> None:
        """Forward-declaration AnnAssign (name: type with no value) returns None."""
        cls = self._make_cls("    dependencies: list\n")
        assert _find_class_assign_value(cls, "dependencies") is None

    def test_op_name_ast_name(self) -> None:
        """_op_name handles ast.Name func (bare function call)."""
        call = ast.parse("CreateModel()").body[0].value  # type: ignore[attr-defined]
        assert _op_name(call) == "CreateModel"

    def test_op_name_unknown_returns_sentinel(self) -> None:
        """_op_name with an unsupported func type returns '<unknown>'."""
        call = ast.Call(func=ast.Constant(value="not_a_func"), args=[], keywords=[])
        assert _op_name(call) == "<unknown>"
