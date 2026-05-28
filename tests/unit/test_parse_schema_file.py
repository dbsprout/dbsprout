import sys
from pathlib import Path

import pytest

from dbsprout.schema.parsers import parse_schema_file


def test_parses_sql_suffix(tmp_path: Path) -> None:
    f = tmp_path / "s.sql"
    f.write_text("CREATE TABLE t (id INTEGER PRIMARY KEY);", encoding="utf-8")
    schema = parse_schema_file(f)
    assert any(t.name == "t" for t in schema.tables)


def test_parses_dbml_suffix(tmp_path: Path) -> None:
    f = tmp_path / "s.dbml"
    f.write_text("Table t {\n  id int [pk]\n}", encoding="utf-8")
    schema = parse_schema_file(f)
    assert any(t.name == "t" for t in schema.tables)


def test_unknown_suffix_falls_back_to_ddl(tmp_path: Path) -> None:
    f = tmp_path / "s.unknown"
    f.write_text("CREATE TABLE t (id INTEGER PRIMARY KEY);", encoding="utf-8")
    schema = parse_schema_file(f)
    assert any(t.name == "t" for t in schema.tables)


def test_missing_file_raises_filenotfounderror(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parse_schema_file(tmp_path / "nope.sql")


def test_too_large_raises_valueerror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = tmp_path / "big.sql"
    f.write_text("-- small", encoding="utf-8")
    monkeypatch.setattr("dbsprout.schema.parsers._MAX_SCHEMA_BYTES", 4)
    with pytest.raises(ValueError, match="too large"):
        parse_schema_file(f)


@pytest.mark.skipif(sys.platform == "win32", reason="symlink creation needs privilege on Windows")
def test_symlink_to_large_file_triggers_size_guard(tmp_path: Path) -> None:
    """AC-4 (S-095): the size guard must follow symlinks to their target.

    A symlink to a 100 MB file must be rejected by the 10 MB guard, which
    requires the guard to inspect the resolved target rather than the link.
    """
    big = tmp_path / "big.sql"
    with big.open("wb") as fh:
        fh.truncate(100 * 1024 * 1024)  # 100 MB sparse file
    link = tmp_path / "link.sql"
    link.symlink_to(big)

    with pytest.raises(ValueError, match="too large"):
        parse_schema_file(link)


@pytest.mark.skipif(
    sys.platform == "win32" or not Path("/dev/zero").exists(),
    reason="needs a POSIX character device and symlink privilege",
)
def test_symlink_to_char_device_is_rejected(tmp_path: Path) -> None:
    """The classic bypass: a symlink to ``/dev/zero`` reports ``st_size=0``.

    The old guard (``path.stat().st_size``) saw 0 and let ``read_text()``
    block / OOM on an unbounded stream. The hardened guard must reject any
    resolved target that is not a regular file.
    """
    link = tmp_path / "evil.sql"
    link.symlink_to("/dev/zero")

    with pytest.raises(ValueError, match="not a regular file"):
        parse_schema_file(link)


@pytest.mark.skipif(sys.platform == "win32", reason="symlink creation needs privilege on Windows")
def test_symlink_to_small_file_parses_normally(tmp_path: Path) -> None:
    """A symlink to a small, valid schema file still parses (no false positive)."""
    target = tmp_path / "target.sql"
    target.write_text("CREATE TABLE t (id INTEGER PRIMARY KEY);", encoding="utf-8")
    link = tmp_path / "link.sql"
    link.symlink_to(target)

    schema = parse_schema_file(link)
    assert any(t.name == "t" for t in schema.tables)
