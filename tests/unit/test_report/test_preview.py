"""Unit tests for the data-preview view-model builder (S-084).

The preview builder turns *real* generated rows
(``GenerateResult.tables_data`` shape: ``dict[str, list[dict[str, Any]]]``)
plus a :class:`DatabaseSchema` into a template-ready view-model. The state
DB stores telemetry only, so preview data is supplied at report time.
"""

from __future__ import annotations

from dbsprout.report.context import build_report_context
from dbsprout.report.env import render_report
from dbsprout.report.preview import (
    PREVIEW_ROW_LIMIT,
    TRUNCATE_LIMIT,
    build_table_previews,
)
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    TableSchema,
)

from ._fixtures import make_run


def _schema() -> DatabaseSchema:
    users = TableSchema(
        name="users",
        columns=[
            ColumnSchema(name="id", data_type=ColumnType.INTEGER, primary_key=True, nullable=False),
            ColumnSchema(name="email", data_type=ColumnType.VARCHAR, unique=True, nullable=False),
            ColumnSchema(name="bio", data_type=ColumnType.TEXT, nullable=True),
        ],
        primary_key=["id"],
    )
    orders = TableSchema(
        name="orders",
        columns=[
            ColumnSchema(name="id", data_type=ColumnType.INTEGER, primary_key=True, nullable=False),
            ColumnSchema(name="user_id", data_type=ColumnType.INTEGER, nullable=False),
            ColumnSchema(name="total", data_type=ColumnType.DECIMAL, nullable=True),
        ],
        primary_key=["id"],
        foreign_keys=[
            ForeignKeySchema(columns=["user_id"], ref_table="users", ref_columns=["id"]),
        ],
    )
    return DatabaseSchema(tables=[users, orders], dialect="sqlite")


def _rows(n: int) -> dict[str, list[dict[str, object]]]:
    return {
        "users": [
            {"id": i, "email": f"u{i}@x.io", "bio": ("z" * 80 if i == 0 else None)}
            for i in range(n)
        ],
        "orders": [
            {"id": i, "user_id": i % 3, "total": (None if i == 1 else 9.99)} for i in range(n)
        ],
    }


class TestBuildTablePreviews:
    def test_one_entry_per_table(self) -> None:
        previews = build_table_previews(_schema(), _rows(5))
        names = [p["name"] for p in previews]
        assert names == ["users", "orders"]
        for p in previews:
            assert p["anchor"]
            assert p["columns"]
            assert "rows" in p
            assert "total_rows" in p

    def test_column_constraint_badges(self) -> None:
        previews = build_table_previews(_schema(), _rows(2))
        users = next(p for p in previews if p["name"] == "users")
        cols = {c["name"]: c for c in users["columns"]}
        assert "PK" in cols["id"]["badges"]
        assert "NOT NULL" in cols["id"]["badges"]
        assert "UNIQUE" in cols["email"]["badges"]
        assert cols["id"]["type"] == "integer"
        assert cols["bio"]["type"] == "text"
        orders = next(p for p in previews if p["name"] == "orders")
        ocols = {c["name"]: c for c in orders["columns"]}
        assert "FK" in ocols["user_id"]["badges"]

    def test_sampling_caps_at_limit(self) -> None:
        previews = build_table_previews(_schema(), _rows(50))
        users = next(p for p in previews if p["name"] == "users")
        assert len(users["rows"]) == PREVIEW_ROW_LIMIT == 10
        assert users["total_rows"] == 50

    def test_fewer_rows_than_limit(self) -> None:
        previews = build_table_previews(_schema(), _rows(3))
        users = next(p for p in previews if p["name"] == "users")
        assert len(users["rows"]) == 3
        assert users["total_rows"] == 3

    def test_null_cell_classified(self) -> None:
        previews = build_table_previews(_schema(), _rows(5))
        orders = next(p for p in previews if p["name"] == "orders")
        # orders row index 1 has total=None
        total_cell = orders["rows"][1][2]
        assert total_cell["kind"] == "null"

    def test_long_text_truncated_with_full_value(self) -> None:
        previews = build_table_previews(_schema(), _rows(2))
        users = next(p for p in previews if p["name"] == "users")
        bio_cell = users["rows"][0][2]  # "z" * 80
        assert bio_cell["kind"] == "trunc"
        assert len(bio_cell["display"]) <= TRUNCATE_LIMIT + 1
        assert bio_cell["full"] == "z" * 80
        assert bio_cell["display"].endswith("…")

    def test_fk_cell_links_to_parent(self) -> None:
        previews = build_table_previews(_schema(), _rows(4))
        orders = next(p for p in previews if p["name"] == "orders")
        fk_cell = orders["rows"][0][1]  # user_id
        assert fk_cell["kind"] == "fk"
        assert fk_cell["ref_table"] == "users"
        assert fk_cell["ref_anchor"]
        assert fk_cell["display"] == "0"

    def test_plain_cell(self) -> None:
        previews = build_table_previews(_schema(), _rows(2))
        users = next(p for p in previews if p["name"] == "users")
        email_cell = users["rows"][1][1]
        assert email_cell["kind"] == "plain"
        assert email_cell["display"] == "u1@x.io"

    def test_anchor_is_slug_unique_and_stable(self) -> None:
        previews = build_table_previews(_schema(), _rows(1))
        anchors = [p["anchor"] for p in previews]
        assert len(anchors) == len(set(anchors))
        assert all(a.startswith("table-") for a in anchors)

    def test_missing_table_data_yields_empty_rows(self) -> None:
        previews = build_table_previews(_schema(), {"users": _rows(2)["users"]})
        orders = next(p for p in previews if p["name"] == "orders")
        assert orders["rows"] == []
        assert orders["total_rows"] == 0

    def test_empty_inputs(self) -> None:
        assert build_table_previews(DatabaseSchema(tables=[]), {}) == []


class TestContextWiring:
    def test_preview_added_when_supplied(self) -> None:
        ctx = build_report_context(make_run(), schema=_schema(), tables_data=_rows(3))
        assert "data_preview" in ctx
        assert [p["name"] for p in ctx["data_preview"]] == ["users", "orders"]

    def test_preview_empty_when_not_supplied(self) -> None:
        ctx = build_report_context(make_run())
        assert ctx["data_preview"] == []

    def test_preview_empty_for_none_run(self) -> None:
        ctx = build_report_context(None)
        assert ctx["data_preview"] == []


class TestPartialRender:
    def test_preview_renders_rows_types_and_features(self) -> None:
        html = render_report(
            build_report_context(make_run(), schema=_schema(), tables_data=_rows(50))
        )
        # 10 sample rows for users (data attribute count)
        assert 'data-preview-rows="10"' in html
        assert 'data-preview-total="50"' in html
        # column types in header
        assert "integer" in html
        assert "varchar" in html
        # table of contents linking each table
        assert 'href="#table-users"' in html
        assert 'href="#table-orders"' in html
        # FK cell linked to parent anchor
        assert 'href="#table-users"' in html
        # NULL distinct styling class present
        assert "preview-null" in html
        # truncated long text with tooltip = full value
        assert 'title="' + "z" * 80 + '"' in html
        # responsive scroll wrapper + sortable header
        assert "preview-scroll" in html
        assert "preview-sortable" in html

    def test_no_preview_section_body_when_absent(self) -> None:
        html = render_report(build_report_context(make_run()))
        # Block still present (template marker) but no data rows.
        assert "data-preview" in html
        assert 'data-preview-rows="' not in html

    def test_xss_escaped_in_preview(self) -> None:
        rows = {
            "users": [{"id": 1, "email": "<script>alert(1)</script>", "bio": None}],
            "orders": [],
        }
        html = render_report(build_report_context(make_run(), schema=_schema(), tables_data=rows))
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html
