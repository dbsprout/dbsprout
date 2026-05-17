"""Unit tests for ``ReportGenerator.generate`` (Task 5)."""

from __future__ import annotations

import os
import re
from pathlib import Path

from dbsprout.report import ReportGenerator
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    TableSchema,
)
from dbsprout.state import StateDB

from ._fixtures import make_run


def _demo_schema() -> DatabaseSchema:
    users = TableSchema(
        name="users",
        columns=[
            ColumnSchema(
                name="id",
                data_type=ColumnType.INTEGER,
                primary_key=True,
                nullable=False,
            )
        ],
        primary_key=["id"],
    )
    orders = TableSchema(
        name="orders",
        columns=[
            ColumnSchema(
                name="id",
                data_type=ColumnType.INTEGER,
                primary_key=True,
                nullable=False,
            ),
            ColumnSchema(
                name="user_id",
                data_type=ColumnType.INTEGER,
                nullable=False,
            ),
        ],
        primary_key=["id"],
        foreign_keys=[ForeignKeySchema(columns=["user_id"], ref_table="users", ref_columns=["id"])],
    )
    return DatabaseSchema(tables=[users, orders])


class TestReportGenerator:
    def test_default_output_path(self, tmp_path: Path) -> None:
        db = StateDB(tmp_path / "state.db")
        db.record_run(make_run())
        cwd = Path.cwd()
        try:
            os.chdir(tmp_path)
            out = ReportGenerator().generate(db)
        finally:
            os.chdir(cwd)
        assert out == (tmp_path / "seeds" / "report.html")
        assert out.is_file()

    def test_custom_output_path(self, tmp_path: Path) -> None:
        db = StateDB(tmp_path / "state.db")
        db.record_run(make_run())
        target = tmp_path / "nested" / "out.html"
        out = ReportGenerator(output_path=target).generate(db)
        assert out == target
        assert out.is_file()

    def test_output_is_html_and_self_contained(self, tmp_path: Path) -> None:
        db = StateDB(tmp_path / "state.db")
        db.record_run(make_run())
        out = ReportGenerator(output_path=tmp_path / "r.html").generate(db)
        html = out.read_text(encoding="utf-8")
        assert html.lstrip().lower().startswith("<!doctype html>")
        assert not re.search(r'(href|src)\s*=\s*["\']https?://', html)

    def test_output_under_one_megabyte(self, tmp_path: Path) -> None:
        db = StateDB(tmp_path / "state.db")
        db.record_run(make_run())
        out = ReportGenerator(output_path=tmp_path / "r.html").generate(db)
        assert out.stat().st_size < 1_000_000

    def test_no_runs_produces_empty_state_report(self, tmp_path: Path) -> None:
        db = StateDB(tmp_path / "state.db")
        out = ReportGenerator(output_path=tmp_path / "r.html").generate(db)
        html = out.read_text(encoding="utf-8")
        assert html.lstrip().lower().startswith("<!doctype html>")
        assert "no runs" in html.lower() or "no generation" in html.lower()

    def test_uses_newest_run(self, tmp_path: Path) -> None:
        db = StateDB(tmp_path / "state.db")
        db.record_run(make_run(offset_minutes=0, engine="heuristic"))
        db.record_run(make_run(offset_minutes=10, engine="spec_driven"))
        out = ReportGenerator(output_path=tmp_path / "r.html").generate(db)
        html = out.read_text(encoding="utf-8")
        assert "spec_driven" in html


class TestReportGeneratorErd:
    def test_schema_renders_mermaid_erd(self, tmp_path: Path) -> None:
        db = StateDB(tmp_path / "state.db")
        db.record_run(make_run())
        out = ReportGenerator(output_path=tmp_path / "r.html").generate(db, schema=_demo_schema())
        html = out.read_text(encoding="utf-8")
        assert 'class="mermaid"' in html
        assert "erDiagram" in html
        assert "users" in html
        # still self-contained: no external resources
        assert not re.search(r'(href|src)\s*=\s*["\']https?://', html)

    def test_no_schema_keeps_placeholder_and_small(self, tmp_path: Path) -> None:
        db = StateDB(tmp_path / "state.db")
        db.record_run(make_run())
        out = ReportGenerator(output_path=tmp_path / "r.html").generate(db)
        html = out.read_text(encoding="utf-8")
        assert "S-082" in html  # placeholder text retained
        assert 'class="mermaid"' not in html
        assert out.stat().st_size < 1_000_000

    def test_erd_report_still_under_one_megabyte(self, tmp_path: Path) -> None:
        db = StateDB(tmp_path / "state.db")
        db.record_run(make_run())
        out = ReportGenerator(output_path=tmp_path / "r.html").generate(db, schema=_demo_schema())
        assert out.stat().st_size < 1_000_000
