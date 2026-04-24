from __future__ import annotations

from typing import TYPE_CHECKING

from tests.unit.test_migrate.test_parsers.conftest import build_django_project

if TYPE_CHECKING:
    from pathlib import Path


def test_build_django_project_creates_structure(tmp_path: Path) -> None:
    body = "class Migration(migrations.Migration):\n    dependencies = []\n    operations = []\n"
    root = build_django_project(
        tmp_path,
        apps={"blog": [("0001_initial", body)]},
    )
    assert (root / "blog" / "migrations" / "__init__.py").exists()
    assert (root / "blog" / "migrations" / "0001_initial.py").exists()
