"""Integration test: real StateDB → ReportGenerator → valid HTML (Task 6).

Asserts the Jinja2 render path actually produces well-formed, self-contained
HTML from a *real* persisted StateDB record (not a mocked template surface).
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import TYPE_CHECKING, ClassVar

import pytest

from dbsprout.report import ReportGenerator
from dbsprout.state import StateDB
from tests.unit.test_report._fixtures import make_run

if TYPE_CHECKING:
    from pathlib import Path


class _WellFormedChecker(HTMLParser):
    """Minimal well-formedness check: tags balance for non-void elements."""

    _VOID: ClassVar[frozenset[str]] = frozenset(
        {
            "area",
            "base",
            "br",
            "col",
            "embed",
            "hr",
            "img",
            "input",
            "link",
            "meta",
            "param",
            "source",
            "track",
            "wbr",
        }
    )

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.saw_html = False
        self.saw_body = False

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag == "html":
            self.saw_html = True
        if tag == "body":
            self.saw_body = True
        if tag not in self._VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in self._VOID:
            return
        if tag in self.stack:
            while self.stack and self.stack[-1] != tag:
                self.stack.pop()
            if self.stack:
                self.stack.pop()


@pytest.mark.integration
def test_real_statedb_to_valid_html(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    db.record_run(make_run())

    out = ReportGenerator(output_path=tmp_path / "report.html").generate(db)
    assert out.is_file()

    html = out.read_text(encoding="utf-8")

    # Well-formed enough to parse cleanly.
    checker = _WellFormedChecker()
    checker.feed(html)
    assert checker.saw_html
    assert checker.saw_body
    assert checker.stack == [], f"unclosed tags: {checker.stack}"

    # Required sections + extension-point placeholders present.
    lower = html.lower()
    assert "heuristic" in html  # run summary
    assert "users" in html  # table stats
    assert "orders" in html
    assert "fk_valid" in html  # quality metrics
    for marker in ("erd", "charts"):
        assert marker in lower
    assert "data_preview" in lower or "data-preview" in lower

    # Self-contained + size budget.
    assert out.stat().st_size < 1_000_000
    # S-083: Plotly.js loads from the single canonical CDN
    # (cdn.plot.ly) per S-081/S-083 Technical Notes; an inline bundle
    # (~3.5 MB) would break the < 1 MB cap. Any other external host is
    # still forbidden.
    externals = re.findall(r'(?:href|src)\s*=\s*["\'](https?://[^"\']+)', html)
    for url in externals:
        assert "cdn.plot.ly/" in url, f"unexpected external resource: {url}"


@pytest.mark.integration
def test_empty_statedb_produces_valid_report(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    out = ReportGenerator(output_path=tmp_path / "report.html").generate(db)
    html = out.read_text(encoding="utf-8")

    checker = _WellFormedChecker()
    checker.feed(html)
    assert checker.stack == []
    assert "no runs" in html.lower() or "no generation" in html.lower()
