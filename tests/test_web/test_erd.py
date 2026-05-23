"""Web schema ERD view tests (S-091).

The ERD view renders the latest ``DatabaseSchema`` snapshot as a Mermaid.js
``erDiagram`` rendered *client-side* (Mermaid loaded from a CDN, no server-side
image generation). It reuses :func:`dbsprout.report.erd.build_erd_mermaid`
(S-082) for the diagram source and reads the snapshot via
:class:`dbsprout.migrate.snapshot.SnapshotStore` (S-051).

The web stack lives in the optional ``[web]`` extra, so the module guards with
``pytest.importorskip("fastapi")`` *before* importing any FastAPI symbols.
"""

from __future__ import annotations

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

    # State DB lives in a sibling temp path; the ERD view does not touch it.
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
            ColumnSchema(name="id", data_type=ColumnType.INTEGER, primary_key=True),
            ColumnSchema(name="email", data_type=ColumnType.VARCHAR),
        ],
        primary_key=["id"],
    )
    orders = TableSchema(
        name="orders",
        columns=[
            ColumnSchema(name="id", data_type=ColumnType.INTEGER, primary_key=True),
            ColumnSchema(name="user_id", data_type=ColumnType.INTEGER),
            ColumnSchema(name="total", data_type=ColumnType.DECIMAL),
        ],
        primary_key=["id"],
        foreign_keys=[
            ForeignKeySchema(columns=["user_id"], ref_table="users", ref_columns=["id"]),
        ],
    )
    return DatabaseSchema(tables=[users, orders], dialect="sqlite")


def _wide_schema(n: int = 22) -> DatabaseSchema:
    """A schema with *n* tables (>= 20) to exercise the perf-sanity AC."""
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


# ── empty-state (missing snapshot) ────────────────────────────────────


def test_schema_route_200_when_no_snapshot(tmp_path: Path) -> None:
    """AC graceful: with no snapshot, /schema still returns 200 (never 500)."""
    client = _make_client(tmp_path / "snapshots")
    resp = client.get("/schema")
    assert resp.status_code == 200
    body = resp.text.lower()
    assert "no schema" in body or "no snapshot" in body


def test_schema_route_200_when_snapshot_dir_missing(tmp_path: Path) -> None:
    """The snapshot dir need not exist yet — still graceful 200."""
    resp = _make_client(tmp_path / "never_created").get("/schema")
    assert resp.status_code == 200


# ── ERD source rendering ──────────────────────────────────────────────


def test_schema_route_embeds_erdiagram(tmp_path: Path) -> None:
    """AC: route renders the Mermaid erDiagram for the latest snapshot."""
    snap_dir = tmp_path / "snapshots"
    SnapshotStore(base_dir=snap_dir).save(_small_schema())

    resp = _make_client(snap_dir).get("/schema")
    assert resp.status_code == 200
    body = resp.text
    assert "erDiagram" in body
    # AC: shows all tables + FK relationships.
    assert "users" in body
    assert "orders" in body


def test_schema_route_shows_columns_and_types(tmp_path: Path) -> None:
    """AC: ERD shows columns and their types (from build_erd_mermaid)."""
    snap_dir = tmp_path / "snapshots"
    SnapshotStore(base_dir=snap_dir).save(_small_schema())

    body = _make_client(snap_dir).get("/schema").text
    assert "email" in body
    assert "varchar" in body
    assert "decimal" in body


# ── client-side Mermaid via CDN ───────────────────────────────────────


def test_schema_route_loads_mermaid_from_cdn(tmp_path: Path) -> None:
    """AC: Mermaid.js loaded from a CDN, rendered client-side."""
    snap_dir = tmp_path / "snapshots"
    SnapshotStore(base_dir=snap_dir).save(_small_schema())

    body = _make_client(snap_dir).get("/schema").text
    assert "mermaid" in body.lower()
    assert "cdn" in body.lower() or "jsdelivr" in body.lower() or "unpkg" in body.lower()
    assert "mermaid.initialize" in body
    assert "startOnLoad" in body


def test_schema_route_uses_pre_mermaid_block(tmp_path: Path) -> None:
    """The diagram source is embedded in a ``<pre class="mermaid">`` element."""
    snap_dir = tmp_path / "snapshots"
    SnapshotStore(base_dir=snap_dir).save(_small_schema())

    body = _make_client(snap_dir).get("/schema").text
    assert 'class="mermaid"' in body


# ── navigation / base template ────────────────────────────────────────


def test_schema_route_extends_base_with_nav(tmp_path: Path) -> None:
    """AC: template inherits base.html nav; the Schema tab is marked active."""
    snap_dir = tmp_path / "snapshots"
    SnapshotStore(base_dir=snap_dir).save(_small_schema())

    body = _make_client(snap_dir).get("/schema").text
    assert 'id="nav-schema"' in body
    assert 'id="nav-home"' in body  # proves base.html was extended


# ── snapshot override + large schema ──────────────────────────────────


def test_schema_route_honors_snapshot_dir_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The snapshot dir is overridable via DBSPROUT_SNAPSHOT_DIR env var."""
    snap_dir = tmp_path / "env_snapshots"
    SnapshotStore(base_dir=snap_dir).save(_small_schema())
    monkeypatch.setenv("DBSPROUT_SNAPSHOT_DIR", str(snap_dir))

    from dbsprout.web.app import create_app  # noqa: PLC0415

    client = TestClient(create_app(state_db_path=tmp_path / "state.db"))
    resp = client.get("/schema")
    assert resp.status_code == 200
    assert "erDiagram" in resp.text


def test_schema_route_handles_wide_schema(tmp_path: Path) -> None:
    """AC: schemas with 20+ tables render without errors."""
    snap_dir = tmp_path / "snapshots"
    SnapshotStore(base_dir=snap_dir).save(_wide_schema(22))

    resp = _make_client(snap_dir).get("/schema")
    assert resp.status_code == 200
    body = resp.text
    assert "erDiagram" in body
    assert "table_00" in body
    assert "table_21" in body
