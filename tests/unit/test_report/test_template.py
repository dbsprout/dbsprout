"""Unit tests for the Jinja2 environment + template (Tasks 3 & 4)."""

from __future__ import annotations

import re

from dbsprout.report.context import build_report_context
from dbsprout.report.env import build_environment, render_report
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    TableSchema,
)
from dbsprout.state.models import QualityResult

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


class TestEnvironment:
    def test_environment_autoescape_on(self) -> None:
        env = build_environment()
        assert env.autoescape is not False

    def test_template_loads_and_renders_doctype(self) -> None:
        html = render_report(build_report_context(make_run()))
        assert html.lstrip().lower().startswith("<!doctype html>")
        assert "<html" in html
        assert "</html>" in html


class TestSelfContained:
    def test_inline_css_and_js_present(self) -> None:
        html = render_report(build_report_context(make_run()))
        assert "<style" in html
        assert "<script" in html

    def test_no_external_resources(self) -> None:
        html = render_report(build_report_context(make_run()))
        # S-083: Plotly.js is loaded from the single canonical CDN
        # (cdn.plot.ly) per S-081/S-083 Technical Notes; an inline
        # bundle (~3.5 MB) would break the < 1 MB cap. Any *other*
        # external host is still forbidden.
        externals = re.findall(r'(?:href|src)\s*=\s*["\'](https?://[^"\']+)', html)
        for url in externals:
            assert "cdn.plot.ly/" in url, f"unexpected external resource: {url}"

    def test_theme_toggle_control_present(self) -> None:
        html = render_report(build_report_context(make_run()))
        assert "data-theme" in html
        # A user-facing toggle control exists.
        assert re.search(r"theme[-_]?toggle", html, re.IGNORECASE)

    def test_under_one_megabyte(self) -> None:
        html = render_report(build_report_context(make_run()))
        assert len(html.encode("utf-8")) < 1_000_000


class TestSections:
    def test_required_sections_and_placeholders(self) -> None:
        html = render_report(build_report_context(make_run()))
        # Run summary + table stats + quality metrics rendered.
        assert "heuristic" in html
        assert "users" in html
        assert "orders" in html
        assert "fk_valid" in html
        # Extension-point placeholders for S-082/083/084.
        for marker in ("erd", "charts", "data_preview", "data-preview"):
            assert marker in html.lower()

    def test_empty_state_renders(self) -> None:
        html = render_report(build_report_context(None))
        assert html.lstrip().lower().startswith("<!doctype html>")
        assert "no runs" in html.lower() or "no generation" in html.lower()


class TestQualityTableRender:
    def test_quality_table_status_classes(self) -> None:
        html = render_report(build_report_context(make_run()))
        assert "fk_valid" in html
        # New classified status-* CSS classes from the S-083 partial.
        assert "status-pass" in html
        assert 'id="quality"' in html

    def test_quality_table_warn_class_for_low_fidelity(self) -> None:
        run = make_run().model_copy(
            update={
                "quality_results": [
                    QualityResult(
                        metric_type="fidelity",
                        metric_name="ks_complement",
                        score=0.5,
                        passed=True,
                    )
                ]
            }
        )
        html = render_report(build_report_context(run))
        assert "status-warn" in html


class TestChartsRender:
    def test_charts_embedded_json_and_plotly(self) -> None:
        html = render_report(build_report_context(make_run()))
        assert 'id="dbsprout-charts"' in html
        assert 'type="application/json"' in html
        # Plotly loaded via the single canonical CDN host.
        assert "cdn.plot.ly" in html
        assert 'id="charts"' in html

    def test_charts_empty_state_no_crash(self) -> None:
        html = render_report(build_report_context(None))
        assert html.lstrip().lower().startswith("<!doctype html>")


class TestErdBlock:
    def test_placeholder_when_no_schema(self) -> None:
        html = render_report(build_report_context(make_run()))
        assert 'id="erd"' in html
        assert "S-082" in html
        assert 'class="mermaid"' not in html

    def test_mermaid_block_when_schema(self) -> None:
        ctx = build_report_context(make_run(), schema=_demo_schema())
        html = render_report(ctx)
        assert 'id="erd"' in html
        assert '<pre class="mermaid">' in html
        assert "erDiagram" in html
        assert "users {" in html
        assert "orders {" in html

    def test_erd_block_no_external_resources(self) -> None:
        ctx = build_report_context(make_run(), schema=_demo_schema())
        html = render_report(ctx)
        # The ERD partial embeds Mermaid offline (no external URL). The only
        # permitted external resource in the whole report is the Plotly CDN
        # (cdn.plot.ly) added by S-083; assert no OTHER external https.
        externals = re.findall(r'(?:href|src)\s*=\s*["\'](https?://[^"\']+)', html)
        for url in externals:
            assert "cdn.plot.ly/" in url, f"unexpected external resource: {url}"

    def test_erd_block_keeps_other_sections(self) -> None:
        ctx = build_report_context(make_run(), schema=_demo_schema())
        html = render_report(ctx)
        # S-083 charts section + S-084 data-preview section untouched by
        # the ERD change. Both S-083 and S-084 now render real partials
        # (not the old placeholder literals), so assert the stable section
        # markers rather than the removed "S-083"/"S-084" literals.
        assert 'id="charts"' in html
        assert 'data-section="data_preview"' in html
        assert "heuristic" in html
        assert "fk_valid" in html
