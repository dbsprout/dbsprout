# tests/unit/test_migrate/test_parsers/test_prisma_discovery.py
"""Prisma parser discovery tests: fixture helper, walking, ordering, defence."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from dbsprout.migrate.parsers import MigrationParseError, MigrationParser
from dbsprout.migrate.parsers.prisma import PrismaMigrationParser
from tests.unit.test_migrate.test_parsers.conftest import build_prisma_project


class TestScaffold:
    def test_frozen_dataclass(self) -> None:
        parser = PrismaMigrationParser()
        with pytest.raises(dataclasses.FrozenInstanceError):
            parser.dialect = "mysql"  # type: ignore[misc]

    def test_implements_protocol(self) -> None:
        assert isinstance(PrismaMigrationParser(), MigrationParser)

    def test_missing_migrations_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(MigrationParseError, match=r"no prisma/migrations/"):
            PrismaMigrationParser().detect_changes(tmp_path)


class TestDiscovery:
    def test_discovers_single_migration(self, tmp_path: Path) -> None:
        from dbsprout.migrate.parsers.prisma import _discover_migrations  # noqa: PLC0415

        build_prisma_project(tmp_path, {"20240101000000_init": "CREATE TABLE t (id INT);"})
        mig_root = tmp_path / "prisma" / "migrations"
        files = _discover_migrations(mig_root, tmp_path.resolve())
        assert len(files) == 1
        assert files[0].name == "migration.sql"
        assert files[0].parent.name == "20240101000000_init"

    def test_orders_by_directory_stem_lexically(self, tmp_path: Path) -> None:
        from dbsprout.migrate.parsers.prisma import _discover_migrations  # noqa: PLC0415

        build_prisma_project(
            tmp_path,
            {
                "20240102000000_second": "SELECT 2;",
                "20240101000000_first": "SELECT 1;",
                "20240103000000_third": "SELECT 3;",
            },
        )
        mig_root = tmp_path / "prisma" / "migrations"
        files = _discover_migrations(mig_root, tmp_path.resolve())
        assert [f.parent.name for f in files] == [
            "20240101000000_first",
            "20240102000000_second",
            "20240103000000_third",
        ]

    def test_empty_migrations_dir_raises(self, tmp_path: Path) -> None:
        build_prisma_project(tmp_path, {})
        with pytest.raises(MigrationParseError, match=r"no migration\.sql found"):
            PrismaMigrationParser().detect_changes(tmp_path)

    def test_subdir_without_migration_sql_raises(self, tmp_path: Path) -> None:
        build_prisma_project(tmp_path, {})
        (tmp_path / "prisma" / "migrations" / "20240101000000_init").mkdir()
        with pytest.raises(MigrationParseError, match=r"has no migration\.sql"):
            PrismaMigrationParser().detect_changes(tmp_path)

    def test_symlinked_migration_skipped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        from dbsprout.migrate.parsers.prisma import _discover_migrations  # noqa: PLC0415

        build_prisma_project(
            tmp_path,
            {"20240101000000_real": "CREATE TABLE r (id INT);"},
        )
        sub = tmp_path / "prisma" / "migrations" / "20240102000000_link"
        sub.mkdir()
        target = tmp_path / "prisma" / "migrations" / "20240101000000_real" / "migration.sql"
        (sub / "migration.sql").symlink_to(target)
        mig_root = tmp_path / "prisma" / "migrations"
        with caplog.at_level("DEBUG", logger="dbsprout.migrate.parsers.prisma"):
            files = _discover_migrations(mig_root, tmp_path.resolve())
        names = [f.parent.name for f in files]
        assert names == ["20240101000000_real"]
        assert "symlink" in caplog.text.lower()

    def test_escaping_symlink_subdir_rejected(self, tmp_path: Path) -> None:
        outside = tmp_path.parent / "outside_prisma_root"
        outside.mkdir(exist_ok=True)
        build_prisma_project(tmp_path, {})
        evil = tmp_path / "prisma" / "migrations" / "20240101000000_evil"
        evil.symlink_to(outside, target_is_directory=True)
        with pytest.raises(MigrationParseError, match=r"escapes project root"):
            PrismaMigrationParser().detect_changes(tmp_path)

    def test_oversize_file_skipped(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        from dbsprout.migrate.parsers.prisma import _discover_migrations  # noqa: PLC0415

        build_prisma_project(tmp_path, {"20240101000000_big": ""})
        big = tmp_path / "prisma" / "migrations" / "20240101000000_big" / "migration.sql"
        big.write_bytes(b"-- " + b"a" * (1024 * 1024 + 1))
        mig_root = tmp_path / "prisma" / "migrations"
        with caplog.at_level("DEBUG", logger="dbsprout.migrate.parsers.prisma"):
            files = _discover_migrations(mig_root, tmp_path.resolve())
        assert files == []
        assert "1 MB" in caplog.text

    def test_contained_symlinked_subdir_skipped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        from dbsprout.migrate.parsers.prisma import _discover_migrations  # noqa: PLC0415

        build_prisma_project(
            tmp_path,
            {"20240101000000_real": "CREATE TABLE r (id INT);"},
        )
        real_sub = tmp_path / "prisma" / "migrations" / "20240101000000_real"
        link_sub = tmp_path / "prisma" / "migrations" / "20240102000000_link"
        link_sub.symlink_to(real_sub, target_is_directory=True)
        mig_root = tmp_path / "prisma" / "migrations"
        with caplog.at_level("DEBUG", logger="dbsprout.migrate.parsers.prisma"):
            files = _discover_migrations(mig_root, tmp_path.resolve())
        assert [f.parent.name for f in files] == ["20240101000000_real"]
        assert "symlinked subdir" in caplog.text

    def test_migration_sql_symlink_target_outside_root_skipped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        from dbsprout.migrate.parsers.prisma import _discover_migrations  # noqa: PLC0415

        outside = tmp_path.parent / "outside_target_dir"
        outside.mkdir(exist_ok=True)
        outside_file = outside / "external_migration.sql"
        outside_file.write_text("SELECT 1;", encoding="utf-8")
        build_prisma_project(tmp_path, {})
        sub = tmp_path / "prisma" / "migrations" / "20240101000000_ext"
        sub.mkdir()
        (sub / "migration.sql").symlink_to(outside_file)
        mig_root = tmp_path / "prisma" / "migrations"
        with caplog.at_level("DEBUG", logger="dbsprout.migrate.parsers.prisma"):
            files = _discover_migrations(mig_root, tmp_path.resolve())
        # Symlink is skipped before out-of-tree check; file list is empty
        assert files == []
        assert "symlinked migration" in caplog.text or "out-of-tree" in caplog.text

    def test_stat_oserror_skipped(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from dbsprout.migrate.parsers import prisma as prisma_mod  # noqa: PLC0415

        build_prisma_project(tmp_path, {"20240101000000_a": "SELECT 1;"})
        mig_root = tmp_path / "prisma" / "migrations"
        bad_sub = mig_root / "20240101000000_a"
        original_is_symlink = Path.is_symlink

        def flaky_is_symlink(self: Path) -> bool:
            if self == bad_sub:
                raise OSError("lstat refused")
            return original_is_symlink(self)

        monkeypatch.setattr(Path, "is_symlink", flaky_is_symlink)
        with caplog.at_level("DEBUG", logger="dbsprout.migrate.parsers.prisma"):
            files = prisma_mod._discover_migrations(mig_root, tmp_path.resolve())
        assert files == []
        assert "cannot stat" in caplog.text


class TestBuildPrismaProjectHelper:
    def test_creates_migration_sql_per_subdirectory(self, tmp_path: Path) -> None:
        root = build_prisma_project(
            tmp_path,
            migrations={
                "20240101000000_init": "CREATE TABLE t (id INT);",
                "20240102000000_next": "ALTER TABLE t ADD COLUMN name TEXT;",
            },
        )
        init_sql = root / "prisma" / "migrations" / "20240101000000_init" / "migration.sql"
        next_sql = root / "prisma" / "migrations" / "20240102000000_next" / "migration.sql"
        assert init_sql.read_text() == "CREATE TABLE t (id INT);"
        assert next_sql.read_text() == "ALTER TABLE t ADD COLUMN name TEXT;"

    def test_writes_migration_lock_with_provider_by_default(self, tmp_path: Path) -> None:
        build_prisma_project(tmp_path, {"20240101000000_init": "SELECT 1;"})
        lock = tmp_path / "prisma" / "migrations" / "migration_lock.toml"
        assert lock.is_file()
        assert 'provider = "postgresql"' in lock.read_text()

    def test_skips_lock_when_provider_none(self, tmp_path: Path) -> None:
        build_prisma_project(
            tmp_path,
            {"20240101000000_init": "SELECT 1;"},
            provider=None,
        )
        lock = tmp_path / "prisma" / "migrations" / "migration_lock.toml"
        assert not lock.exists()

    def test_custom_migrations_dir(self, tmp_path: Path) -> None:
        root = build_prisma_project(
            tmp_path,
            {"20240101000000_init": "SELECT 1;"},
            migrations_dir="custom/migs",
        )
        assert (root / "custom" / "migs" / "20240101000000_init" / "migration.sql").is_file()
