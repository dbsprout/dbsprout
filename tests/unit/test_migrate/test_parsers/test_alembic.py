from __future__ import annotations

from pathlib import Path

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


class TestLinearize:
    def _rev(self, rid: str, down: str | None) -> object:
        import ast  # noqa: PLC0415

        from dbsprout.migrate.parsers.alembic import _Revision  # noqa: PLC0415

        return _Revision(
            path=Path(f"{rid}.py"),
            revision=rid,
            down_revision=down,
            module=ast.parse(""),
        )

    def test_empty(self) -> None:
        from dbsprout.migrate.parsers.alembic import _linearize_revisions  # noqa: PLC0415

        assert _linearize_revisions([]) == []

    def test_linear_chain(self) -> None:
        from dbsprout.migrate.parsers.alembic import _linearize_revisions  # noqa: PLC0415

        revs = [self._rev("c", "b"), self._rev("a", None), self._rev("b", "a")]
        ordered = _linearize_revisions(revs)
        assert [r.revision for r in ordered] == ["a", "b", "c"]

    def test_multiple_heads(self) -> None:
        from dbsprout.migrate.parsers import MigrationParseError  # noqa: PLC0415
        from dbsprout.migrate.parsers.alembic import _linearize_revisions  # noqa: PLC0415

        revs = [self._rev("a", None), self._rev("b", "a"), self._rev("c", "a")]
        with pytest.raises(MigrationParseError, match="head"):
            _linearize_revisions(revs)

    def test_multiple_roots(self) -> None:
        from dbsprout.migrate.parsers import MigrationParseError  # noqa: PLC0415
        from dbsprout.migrate.parsers.alembic import _linearize_revisions  # noqa: PLC0415

        revs = [self._rev("a", None), self._rev("b", None)]
        with pytest.raises(MigrationParseError, match=r"head|root"):
            _linearize_revisions(revs)


class TestWalker:
    def _parse_upgrade_body(self, body: str) -> list:
        import ast  # noqa: PLC0415

        from dbsprout.migrate.parsers.alembic import _parse_upgrade, _Revision  # noqa: PLC0415

        src = f'revision = "r"\ndown_revision = None\n\ndef upgrade():\n{body}\n'
        module = ast.parse(src)
        rev = _Revision(path=Path("r.py"), revision="r", down_revision=None, module=module)
        return _parse_upgrade(rev)

    def test_unknown_verb_is_skipped(self) -> None:
        # op.execute is intentionally unmapped
        changes = self._parse_upgrade_body('    op.execute("SELECT 1")')
        assert changes == []

    def test_non_op_call_ignored(self) -> None:
        changes = self._parse_upgrade_body('    print("hi")')
        assert changes == []

    def test_missing_upgrade_raises(self) -> None:
        import ast  # noqa: PLC0415

        from dbsprout.migrate.parsers import MigrationParseError  # noqa: PLC0415
        from dbsprout.migrate.parsers.alembic import _parse_upgrade, _Revision  # noqa: PLC0415

        module = ast.parse('revision = "r"\ndown_revision = None\n')
        rev = _Revision(path=Path("r.py"), revision="r", down_revision=None, module=module)
        with pytest.raises(MigrationParseError, match="upgrade"):
            _parse_upgrade(rev)
