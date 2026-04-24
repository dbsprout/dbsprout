# tests/unit/test_migrate/test_parsers/test_flyway_discovery.py
from __future__ import annotations

from typing import TYPE_CHECKING

from tests.unit.test_migrate.test_parsers.conftest import build_flyway_project

if TYPE_CHECKING:
    from pathlib import Path


def test_build_flyway_project_creates_structure(tmp_path: Path) -> None:
    root = build_flyway_project(
        tmp_path,
        migrations={"V1__initial": "CREATE TABLE t (id INT);"},
    )
    assert (root / "db" / "migration" / "V1__initial.sql").read_text() == "CREATE TABLE t (id INT);"
