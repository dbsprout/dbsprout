"""Web ERD interactivity tests (S-091-F1).

Tests cover the server-rendered output that enables client-side interactivity:
- SVG pan/zoom CDN script tag in the HTML.
- Table-detail JSON blob embedded in the page (enables click-to-detail).
- Mermaid click directives per table (wires Mermaid table-click events).
- Filter input and column-toggle controls markup in the HTML.
- Detail panel element in the HTML.
- Correct behaviour for empty-state (no snapshot) and large schemas (22+ tables).

All assertions are against the HTTP response HTML string — no headless browser.
The web stack lives in the optional [web] extra, so we guard with
``pytest.importorskip`` before importing any FastAPI symbols.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")

from fastapi.testclient import TestClient

from dbsprout.migrate.snapshot import SnapshotStore
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    TableSchema,
)

if TYPE_CHECKING:
    from pathlib import Path


# ── fixtures / helpers ────────────────────────────────────────────────


def _make_client(snapshot_dir: Path) -> TestClient:
    from dbsprout.web.app import create_app  # noqa: PLC0415

    app = create_app(
        state_db_path=snapshot_dir.parent / "state.db",
        snapshot_dir=snapshot_dir,
    )
    return TestClient(app)


def _small_schema() -> DatabaseSchema:
    """A 2-table schema: ``orders`` references ``users``."""
    users = TableSchema(
        name="users",
        columns=[
            ColumnSchema(name="id", data_type=ColumnType.INTEGER, primary_key=True, nullable=False),
            ColumnSchema(name="email", data_type=ColumnType.VARCHAR, unique=True),
        ],
        primary_key=["id"],
    )
    orders = TableSchema(
        name="orders",
        columns=[
            ColumnSchema(name="id", data_type=ColumnType.INTEGER, primary_key=True, nullable=False),
            ColumnSchema(name="user_id", data_type=ColumnType.INTEGER, nullable=False),
            ColumnSchema(name="total", data_type=ColumnType.DECIMAL),
        ],
        primary_key=["id"],
        foreign_keys=[
            ForeignKeySchema(columns=["user_id"], ref_table="users", ref_columns=["id"]),
        ],
    )
    return DatabaseSchema(tables=[users, orders], dialect="sqlite")


def _wide_schema(n: int = 22) -> DatabaseSchema:
    """A schema with *n* tables to exercise large-schema controls."""
    tables = [
        TableSchema(
            name=f"table_{i:02d}",
            columns=[
                ColumnSchema(name="id", data_type=ColumnType.INTEGER, primary_key=True),
                ColumnSchema(name="label", data_type=ColumnType.VARCHAR),
            ],
            primary_key=["id"],
        )
        for i in range(n)
    ]
    return DatabaseSchema(tables=tables, dialect="sqlite")


# ── Task 1: Table-detail JSON blob ────────────────────────────────────


def test_table_detail_json_blob_present(tmp_path: Path) -> None:
    """Server embeds a JSON blob with per-table column details."""
    snap_dir = tmp_path / "snapshots"
    SnapshotStore(base_dir=snap_dir).save(_small_schema())

    body = _make_client(snap_dir).get("/schema").text
    assert 'id="erd-table-data"' in body
    assert 'type="application/json"' in body


def test_table_detail_json_blob_contains_tables(tmp_path: Path) -> None:
    """JSON blob contains entries for each table with column metadata."""
    snap_dir = tmp_path / "snapshots"
    SnapshotStore(base_dir=snap_dir).save(_small_schema())

    body = _make_client(snap_dir).get("/schema").text
    # Extract the JSON blob
    start = body.index('id="erd-table-data"')
    json_start = body.index(">", start) + 1
    json_end = body.index("</script>", json_start)
    data = json.loads(body[json_start:json_end])

    assert "users" in data
    assert "orders" in data

    # users table: check columns
    users_cols = {col["name"]: col for col in data["users"]["columns"]}
    assert "id" in users_cols
    assert "email" in users_cols
    assert users_cols["id"]["primary_key"] is True
    assert users_cols["email"]["unique"] is True
    assert users_cols["id"]["nullable"] is False

    # orders table: check FK info
    assert len(data["orders"]["foreign_keys"]) == 1
    fk = data["orders"]["foreign_keys"][0]
    assert fk["ref_table"] == "users"
    assert "user_id" in fk["columns"]


def test_table_detail_json_blob_absent_when_no_snapshot(tmp_path: Path) -> None:
    """Empty state: no JSON blob for table details (no snapshot)."""
    body = _make_client(tmp_path / "snapshots").get("/schema").text
    assert 'id="erd-table-data"' not in body


# ── Task 2: SVG pan/zoom CDN script ──────────────────────────────────


def test_pan_zoom_cdn_script_present_with_snapshot(tmp_path: Path) -> None:
    """SVG pan/zoom library loaded from CDN when snapshot exists."""
    snap_dir = tmp_path / "snapshots"
    SnapshotStore(base_dir=snap_dir).save(_small_schema())

    body = _make_client(snap_dir).get("/schema").text
    assert "svg-pan-zoom" in body


def test_pan_zoom_init_script_present_with_snapshot(tmp_path: Path) -> None:
    """Pan/zoom initialization JavaScript is present in the rendered page."""
    snap_dir = tmp_path / "snapshots"
    SnapshotStore(base_dir=snap_dir).save(_small_schema())

    body = _make_client(snap_dir).get("/schema").text
    assert "svgPanZoom" in body


def test_pan_zoom_absent_when_no_snapshot(tmp_path: Path) -> None:
    """Empty state: no pan/zoom init when there's no ERD to pan/zoom."""
    body = _make_client(tmp_path / "snapshots").get("/schema").text
    assert "svgPanZoom" not in body


# ── Task 3: Filter input + column toggle controls ─────────────────────


def test_filter_input_present_with_snapshot(tmp_path: Path) -> None:
    """Table-name filter input rendered when snapshot exists."""
    snap_dir = tmp_path / "snapshots"
    SnapshotStore(base_dir=snap_dir).save(_small_schema())

    body = _make_client(snap_dir).get("/schema").text
    assert 'id="erd-filter-input"' in body


def test_columns_toggle_present_with_snapshot(tmp_path: Path) -> None:
    """Show/hide-columns toggle rendered when snapshot exists."""
    snap_dir = tmp_path / "snapshots"
    SnapshotStore(base_dir=snap_dir).save(_small_schema())

    body = _make_client(snap_dir).get("/schema").text
    assert 'id="erd-columns-toggle"' in body


def test_controls_absent_when_no_snapshot(tmp_path: Path) -> None:
    """Empty state: no filter/toggle controls rendered."""
    body = _make_client(tmp_path / "snapshots").get("/schema").text
    assert 'id="erd-filter-input"' not in body
    assert 'id="erd-columns-toggle"' not in body


# ── Task 4: Click-to-detail — JS-bound (no Mermaid click directives) ──


def test_no_mermaid_click_directives(tmp_path: Path) -> None:
    """Mermaid 10.9.6's erDiagram parser rejects ``click`` directives (raises
    "Syntax error in text"), breaking the whole diagram. The served source must
    NOT contain them; click-to-detail is bound in JS post-render instead.
    """
    snap_dir = tmp_path / "snapshots"
    SnapshotStore(base_dir=snap_dir).save(_small_schema())

    body = _make_client(snap_dir).get("/schema").text
    assert "erDiagram" in body
    # The breaking Mermaid directive form must be absent from the diagram source.
    assert "click users call" not in body
    assert "click orders call" not in body
    # Click-to-detail is still wired via the JS callback, bound after render.
    assert "erdTableClick" in body


def test_detail_panel_element_present_with_snapshot(tmp_path: Path) -> None:
    """Detail panel div rendered when snapshot exists."""
    snap_dir = tmp_path / "snapshots"
    SnapshotStore(base_dir=snap_dir).save(_small_schema())

    body = _make_client(snap_dir).get("/schema").text
    assert 'id="erd-detail-panel"' in body


def test_detail_panel_absent_when_no_snapshot(tmp_path: Path) -> None:
    """Empty state: no detail panel rendered."""
    body = _make_client(tmp_path / "snapshots").get("/schema").text
    assert 'id="erd-detail-panel"' not in body


# ── Task 5: Edge cases — empty state + large schema ───────────────────


def test_empty_state_still_200_no_broken_js(tmp_path: Path) -> None:
    """Empty state renders 200, has no-snapshot message, no broken JS references."""
    resp = _make_client(tmp_path / "snapshots").get("/schema")
    assert resp.status_code == 200
    body = resp.text
    # Graceful message
    assert "no schema" in body.lower() or "no snapshot" in body.lower()
    # No JS that references erd data that doesn't exist
    assert "svgPanZoom" not in body
    assert 'id="erd-table-data"' not in body


def test_wide_schema_renders_all_entities_without_click_directives(tmp_path: Path) -> None:
    """Large schema: all 22 tables appear as erDiagram entities, and none get a
    Mermaid ``click`` directive (unsupported in 10.9.x — would break the diagram).
    """
    snap_dir = tmp_path / "snapshots"
    schema = _wide_schema(22)
    SnapshotStore(base_dir=snap_dir).save(schema)

    body = _make_client(snap_dir).get("/schema").text
    for i in range(22):
        table_name = f"table_{i:02d}"
        assert table_name in body, f"Missing entity for {table_name}"
        assert f"click {table_name} call" not in body, (
            f"Unexpected Mermaid click directive for {table_name}"
        )


def test_wide_schema_json_blob_has_all_tables(tmp_path: Path) -> None:
    """Large schema: JSON blob contains entries for all 22 tables."""
    snap_dir = tmp_path / "snapshots"
    schema = _wide_schema(22)
    SnapshotStore(base_dir=snap_dir).save(schema)

    body = _make_client(snap_dir).get("/schema").text
    start = body.index('id="erd-table-data"')
    json_start = body.index(">", start) + 1
    json_end = body.index("</script>", json_start)
    data = json.loads(body[json_start:json_end])

    assert len(data) == 22
    for i in range(22):
        assert f"table_{i:02d}" in data


def test_schema_response_200_wide_schema_controls_present(tmp_path: Path) -> None:
    """Large schema: 200 response with filter + toggle controls present."""
    snap_dir = tmp_path / "snapshots"
    SnapshotStore(base_dir=snap_dir).save(_wide_schema(22))

    resp = _make_client(snap_dir).get("/schema")
    assert resp.status_code == 200
    body = resp.text
    assert 'id="erd-filter-input"' in body
    assert 'id="erd-columns-toggle"' in body
    assert 'id="erd-detail-panel"' in body
