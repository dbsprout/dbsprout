from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dbsprout.migrate.parsers import MigrationParseError
from dbsprout.migrate.parsers.django import DjangoMigrationParser, _discover_migration_files
from tests.unit.test_migrate.test_parsers.conftest import build_django_project

if TYPE_CHECKING:
    from pathlib import Path


EMPTY_MIG = (
    "from django.db import migrations\n\n"
    "class Migration(migrations.Migration):\n"
    "    dependencies = []\n"
    "    operations = []\n"
)


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
        root = build_django_project(
            tmp_path,
            apps={"blog": [("0001_big", "x = '" + "a" * (1_100_000) + "'\n")]},
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
