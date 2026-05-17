"""Unit tests for ``ReportGenerator.generate`` (Task 5)."""

from __future__ import annotations

import os
import re
from pathlib import Path

from dbsprout.report import ReportGenerator
from dbsprout.state import StateDB

from ._fixtures import make_run


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
        # S-083: only the canonical Plotly CDN is permitted (see
        # test_template.test_no_external_resources for rationale).
        externals = re.findall(r'(?:href|src)\s*=\s*["\'](https?://[^"\']+)', html)
        for url in externals:
            assert "cdn.plot.ly/" in url, f"unexpected external resource: {url}"

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
