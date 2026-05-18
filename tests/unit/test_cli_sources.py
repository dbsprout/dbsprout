"""Tests for dbsprout.cli.sources shared schema-source helper."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING

import pytest

from dbsprout.cli.sources import SchemaSource, SchemaSourceError, resolve_schema_source

if TYPE_CHECKING:
    from pathlib import Path


def test_db_flag_wins(tmp_path: Path) -> None:
    src = resolve_schema_source(db="postgresql://u:p@h/db", file=None, output_dir=tmp_path)
    assert src.kind == "db"
    assert src.raw_value == "postgresql://u:p@h/db"
    assert "p" not in src.display_value.split("@")[0].split(":")[-1]
    assert "***" in src.display_value


def test_file_flag_used_when_no_db(tmp_path: Path) -> None:
    src = resolve_schema_source(db=None, file="schema.sql", output_dir=tmp_path)
    assert src.kind == "file"
    assert src.raw_value == "schema.sql"
    assert src.display_value == "schema.sql"


def test_config_fallback_db(tmp_path: Path) -> None:
    (tmp_path / "dbsprout.toml").write_text(
        '[schema]\nsource = "sqlite:///x.db"\n', encoding="utf-8"
    )
    src = resolve_schema_source(db=None, file=None, output_dir=tmp_path)
    assert src.kind == "db"


def test_config_fallback_file(tmp_path: Path) -> None:
    (tmp_path / "dbsprout.toml").write_text('[schema]\nsource = "schema.sql"\n', encoding="utf-8")
    src = resolve_schema_source(db=None, file=None, output_dir=tmp_path)
    assert src.kind == "file"
    assert src.raw_value == "schema.sql"


def test_no_source_raises_schema_source_error(tmp_path: Path) -> None:
    with pytest.raises(SchemaSourceError, match="No schema source"):
        resolve_schema_source(db=None, file=None, output_dir=tmp_path)


def test_schema_source_is_frozen_dataclass() -> None:
    src = SchemaSource(kind="file", raw_value="x", display_value="x")
    with pytest.raises(FrozenInstanceError):
        src.kind = "db"  # type: ignore[misc]


def test_invalid_db_url_yields_invalid_url_display(tmp_path: Path) -> None:
    """Malformed URL string bypasses SA parsing and produces a guard display."""
    src = resolve_schema_source(db="://", file=None, output_dir=tmp_path)
    assert src.kind == "db"
    assert src.display_value == "<invalid URL>"


def test_config_source_windows_path_treated_as_file(tmp_path: Path) -> None:
    """Windows-style drive path (``C:/schema.sql``) must not be mistaken for a URL."""
    (tmp_path / "dbsprout.toml").write_text(
        '[schema]\nsource = "C:/schema.sql"\n', encoding="utf-8"
    )
    src = resolve_schema_source(db=None, file=None, output_dir=tmp_path)
    assert src.kind == "file"
    assert src.raw_value == "C:/schema.sql"
