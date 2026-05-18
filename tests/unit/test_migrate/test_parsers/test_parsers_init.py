from __future__ import annotations

from pathlib import Path

from dbsprout.migrate.parsers import MigrationParseError, MigrationParser


def test_error_with_path() -> None:
    err = MigrationParseError("bad file", file_path=Path("/tmp/rev.py"))  # noqa: S108
    assert err.file_path == Path("/tmp/rev.py")  # noqa: S108
    assert "bad file" in str(err)


def test_error_without_path() -> None:
    err = MigrationParseError("multiple heads")
    assert err.file_path is None


def test_protocol_runtime_checkable() -> None:
    class Fake:
        def detect_changes(self, project_path: Path) -> list:
            return []

    assert isinstance(Fake(), MigrationParser)


def test_protocol_rejects_missing_method() -> None:
    class NotAParser:
        pass

    assert not isinstance(NotAParser(), MigrationParser)
