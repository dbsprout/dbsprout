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
