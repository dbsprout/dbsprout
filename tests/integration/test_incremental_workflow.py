"""End-to-end --incremental workflow against SQLite."""

from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from dbsprout.cli.app import app

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


@pytest.mark.integration
def test_incremental_preserves_unchanged_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Init → alter db (add col) → generate --incremental → unchanged col values preserved."""
    db_path = tmp_path / "app.db"
    conn = sqlite3.connect(db_path)
    conn.executescript("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT NOT NULL);")
    conn.close()
    monkeypatch.chdir(tmp_path)

    db_url = f"sqlite:///{db_path}"

    res = runner.invoke(app, ["init", "--db", db_url, "--output-dir", str(tmp_path)])
    assert res.exit_code == 0, res.output

    seeds_dir = tmp_path / "seeds"
    res = runner.invoke(
        app,
        [
            "generate",
            "--output-format",
            "json",
            "--rows",
            "10",
            "--seed",
            "42",
            "--output-dir",
            str(seeds_dir),
        ],
    )
    assert res.exit_code == 0, res.output
    before_files = list(seeds_dir.glob("*items.json"))
    assert before_files, f"no json output: {list(seeds_dir.iterdir())}"
    before = json.loads(before_files[0].read_text(encoding="utf-8"))

    conn = sqlite3.connect(db_path)
    conn.executescript("ALTER TABLE items ADD COLUMN created_at TEXT;")
    conn.close()

    res = runner.invoke(
        app,
        [
            "generate",
            "--incremental",
            "--db",
            db_url,
            "--output-format",
            "json",
            "--rows",
            "10",
            "--seed",
            "42",
            "--output-dir",
            str(seeds_dir),
        ],
    )
    assert res.exit_code == 0, res.output

    after_files = list(seeds_dir.glob("*items.json"))
    assert after_files
    after = json.loads(after_files[0].read_text(encoding="utf-8"))

    assert len(after) == len(before)
    for row_before, row_after in zip(before, after, strict=True):
        assert row_after["id"] == row_before["id"]
        assert row_after["name"] == row_before["name"]
        assert "created_at" in row_after

    snaps = list((tmp_path / ".dbsprout" / "snapshots").glob("*.json"))
    assert len(snaps) >= 2
