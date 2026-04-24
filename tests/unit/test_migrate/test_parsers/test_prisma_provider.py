# tests/unit/test_migrate/test_parsers/test_prisma_provider.py
"""Prisma parser provider/dialect resolution tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dbsprout.migrate.parsers import MigrationParseError
from dbsprout.migrate.parsers.prisma import PROVIDER_DIALECTS, _resolve_dialect
from tests.unit.test_migrate.test_parsers.conftest import build_prisma_project

if TYPE_CHECKING:
    from pathlib import Path


class TestProviderMap:
    @pytest.mark.parametrize(
        ("provider", "dialect"),
        [
            ("postgresql", "postgres"),
            ("mysql", "mysql"),
            ("sqlite", "sqlite"),
            ("sqlserver", "tsql"),
            ("cockroachdb", "postgres"),
        ],
    )
    def test_known_providers_map_to_dialects(
        self, tmp_path: Path, provider: str, dialect: str
    ) -> None:
        build_prisma_project(tmp_path, {"20240101000000_init": "SELECT 1;"}, provider=provider)
        mig_root = tmp_path / "prisma" / "migrations"
        assert _resolve_dialect(mig_root, override=None) == dialect

    def test_provider_map_exposes_expected_keys(self) -> None:
        assert set(PROVIDER_DIALECTS) >= {
            "postgresql",
            "mysql",
            "sqlite",
            "sqlserver",
            "cockroachdb",
        }


class TestRejections:
    def test_rejects_mongodb_provider(self, tmp_path: Path) -> None:
        build_prisma_project(tmp_path, {"20240101000000_init": "SELECT 1;"}, provider="mongodb")
        mig_root = tmp_path / "prisma" / "migrations"
        with pytest.raises(MigrationParseError, match=r"MongoDB provider"):
            _resolve_dialect(mig_root, override=None)

    def test_unknown_provider_raises(self, tmp_path: Path) -> None:
        build_prisma_project(tmp_path, {"20240101000000_init": "SELECT 1;"}, provider="oracle")
        mig_root = tmp_path / "prisma" / "migrations"
        with pytest.raises(MigrationParseError, match=r"unknown Prisma provider 'oracle'"):
            _resolve_dialect(mig_root, override=None)

    def test_malformed_lock_file_raises(self, tmp_path: Path) -> None:
        build_prisma_project(tmp_path, {"20240101000000_init": "SELECT 1;"}, provider=None)
        mig_root = tmp_path / "prisma" / "migrations"
        (mig_root / "migration_lock.toml").write_text("provider =\n", encoding="utf-8")
        with pytest.raises(MigrationParseError, match=r"invalid migration_lock\.toml"):
            _resolve_dialect(mig_root, override=None)

    def test_missing_provider_key_raises(self, tmp_path: Path) -> None:
        build_prisma_project(tmp_path, {"20240101000000_init": "SELECT 1;"}, provider=None)
        mig_root = tmp_path / "prisma" / "migrations"
        (mig_root / "migration_lock.toml").write_text('other = "value"\n', encoding="utf-8")
        with pytest.raises(MigrationParseError, match=r"missing or non-string provider"):
            _resolve_dialect(mig_root, override=None)

    def test_non_string_provider_raises(self, tmp_path: Path) -> None:
        build_prisma_project(tmp_path, {"20240101000000_init": "SELECT 1;"}, provider=None)
        mig_root = tmp_path / "prisma" / "migrations"
        (mig_root / "migration_lock.toml").write_text("provider = 42\n", encoding="utf-8")
        with pytest.raises(MigrationParseError, match=r"missing or non-string provider"):
            _resolve_dialect(mig_root, override=None)


class TestDefaultsAndOverrides:
    def test_missing_lock_defaults_to_postgres(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        build_prisma_project(tmp_path, {"20240101000000_init": "SELECT 1;"}, provider=None)
        mig_root = tmp_path / "prisma" / "migrations"
        with caplog.at_level("DEBUG", logger="dbsprout.migrate.parsers.prisma"):
            assert _resolve_dialect(mig_root, override=None) == "postgres"
        assert "migration_lock.toml" in caplog.text

    def test_constructor_override_wins(self, tmp_path: Path) -> None:
        build_prisma_project(tmp_path, {"20240101000000_init": "SELECT 1;"}, provider="mysql")
        mig_root = tmp_path / "prisma" / "migrations"
        assert _resolve_dialect(mig_root, override="postgres") == "postgres"

    def test_override_used_when_lock_missing(self, tmp_path: Path) -> None:
        build_prisma_project(tmp_path, {"20240101000000_init": "SELECT 1;"}, provider=None)
        mig_root = tmp_path / "prisma" / "migrations"
        assert _resolve_dialect(mig_root, override="sqlite") == "sqlite"
