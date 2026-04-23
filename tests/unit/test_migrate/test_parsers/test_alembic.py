from __future__ import annotations

from pathlib import Path  # noqa: TC003

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
