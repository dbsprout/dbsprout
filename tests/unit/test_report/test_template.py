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
        # No external stylesheet/script/asset references in S-081 skeleton.
        assert not re.search(r'href\s*=\s*["\']https?://', html)
        assert not re.search(r'src\s*=\s*["\']https?://', html)

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
        assert not re.search(r'href\s*=\s*["\']https?://', html)
        assert not re.search(r'src\s*=\s*["\']https?://', html)

    def test_erd_block_keeps_other_sections(self) -> None:
        ctx = build_report_context(make_run(), schema=_demo_schema())
        html = render_report(ctx)
        # S-083 placeholder + S-084 data-preview section untouched by the
        # ERD change (S-084 now renders the data-preview via its partial,
        # so assert the stable section marker rather than the old literal).
        assert "S-083" in html
        assert 'data-section="data_preview"' in html
        assert "heuristic" in html
        assert "fk_valid" in html
