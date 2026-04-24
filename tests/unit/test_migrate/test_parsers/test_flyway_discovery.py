# tests/unit/test_migrate/test_parsers/test_flyway_discovery.py
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dbsprout.migrate.parsers import MigrationParseError
from dbsprout.migrate.parsers.flyway import (
    _discover_migration_files,
    _parse_version,
)
from tests.unit.test_migrate.test_parsers.conftest import build_flyway_project

if TYPE_CHECKING:
    from pathlib import Path


def test_build_flyway_project_creates_structure(tmp_path: Path) -> None:
    root = build_flyway_project(
        tmp_path,
        migrations={"V1__initial": "CREATE TABLE t (id INT);"},
    )
    assert (root / "db" / "migration" / "V1__initial.sql").read_text() == "CREATE TABLE t (id INT);"


EMPTY_SQL = "-- empty\n"


class TestDiscovery:
    def test_default_location_db_migration(self, tmp_path: Path) -> None:
        root = build_flyway_project(tmp_path, {"V1__initial": EMPTY_SQL})
        files = _discover_migration_files(root, None)
        assert len(files) == 1
        assert files[0].name == "V1__initial.sql"

    def test_default_location_maven(self, tmp_path: Path) -> None:
        root = build_flyway_project(
            tmp_path,
            {"V1__initial": EMPTY_SQL},
            location="src/main/resources/db/migration",
        )
        files = _discover_migration_files(root, None)
        assert len(files) == 1

    def test_custom_locations(self, tmp_path: Path) -> None:
        build_flyway_project(tmp_path, {"V1__a": EMPTY_SQL}, location="sql/core")
        build_flyway_project(tmp_path, {"V2__b": EMPTY_SQL}, location="sql/addons")
        files = _discover_migration_files(tmp_path, ("sql/core", "sql/addons"))
        assert [f.name for f in files] == ["V1__a.sql", "V2__b.sql"]

    def test_repeatable_skipped(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        build_flyway_project(tmp_path, {"V1__a": EMPTY_SQL, "R__views": EMPTY_SQL})
        with caplog.at_level("DEBUG", logger="dbsprout.migrate.parsers.flyway"):
            files = _discover_migration_files(tmp_path, None)
        assert [f.name for f in files] == ["V1__a.sql"]
        assert "R__views.sql" in caplog.text

    def test_undo_skipped(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        build_flyway_project(tmp_path, {"V1__a": EMPTY_SQL, "U1__undo_a": EMPTY_SQL})
        with caplog.at_level("DEBUG", logger="dbsprout.migrate.parsers.flyway"):
            files = _discover_migration_files(tmp_path, None)
        assert [f.name for f in files] == ["V1__a.sql"]

    def test_duplicate_version_raises(self, tmp_path: Path) -> None:
        build_flyway_project(tmp_path, {"V1__a": EMPTY_SQL, "V1__b": EMPTY_SQL})
        with pytest.raises(MigrationParseError, match="duplicate Flyway version"):
            _discover_migration_files(tmp_path, None)

    def test_file_over_size_cap_skipped(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        huge = "-- padding\n" * (150_000)  # > 1 MB
        build_flyway_project(tmp_path, {"V1__a": EMPTY_SQL, "V2__huge": huge})
        with caplog.at_level("DEBUG", logger="dbsprout.migrate.parsers.flyway"):
            files = _discover_migration_files(tmp_path, None)
        assert [f.name for f in files] == ["V1__a.sql"]


class TestVersionParsing:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("1", (1,)),
            ("1.1", (1, 1)),
            ("1.1.1", (1, 1, 1)),
            ("2_0_1", (2, 0, 1)),
            ("20240423", (20240423,)),
        ],
    )
    def test_parse(self, raw: str, expected: tuple[int, ...]) -> None:
        assert _parse_version(raw) == expected

    def test_invalid_version_raises(self) -> None:
        with pytest.raises(MigrationParseError, match="invalid Flyway version"):
            _parse_version("1.x.3")

    @pytest.mark.parametrize(
        ("lo", "hi"),
        [
            ("1", "1.1"),
            ("1.1", "1.1.1"),
            ("1.1.1", "2"),
            ("2", "2_0_1"),
        ],
    )
    def test_ordering(self, lo: str, hi: str) -> None:
        from dbsprout.migrate.parsers.flyway import _version_sort_key  # noqa: PLC0415

        assert _version_sort_key(_parse_version(lo)) < _version_sort_key(_parse_version(hi))


# ---------------------------------------------------------------------------
# Placeholder tests (added in Task 4)
# ---------------------------------------------------------------------------
