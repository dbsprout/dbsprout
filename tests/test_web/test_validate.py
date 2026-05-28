"""POST /api/validate integrity-report endpoint tests (S-133).

The web stack lives in the optional ``[web]`` extra, so the module guards with
``pytest.importorskip("fastapi")`` before importing FastAPI symbols (mirrors the
sibling web tests). The endpoint runs the existing
:func:`dbsprout.quality.integrity.validate_integrity` against the workspace's
last generation result and returns a JSON envelope (or an HTMX HTML fragment)
describing FK / UNIQUE / NOT NULL violations.

Tests seed ``app.state.workspace.schema`` and ``app.state.workspace.last_result``
directly, mirroring ``test_preview.py``.
"""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")

from fastapi.testclient import TestClient

from dbsprout.generate.orchestrator import GenerateResult
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    TableSchema,
)

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi import FastAPI


# ── fixtures / helpers ─────────────────────────────────────────────────


def _make_app(tmp_path: Path) -> FastAPI:
    from dbsprout.web.app import create_app  # noqa: PLC0415

    return create_app(state_db_path=tmp_path / "state.db")


def _users_orders_schema() -> DatabaseSchema:
    """Two-table schema: ``users (id PK, email UNIQUE NOT NULL)``,
    ``orders (id PK, user_id FK->users.id, total NOT NULL)``."""
    users = TableSchema(
        name="users",
        columns=[
            ColumnSchema(
                name="id",
                data_type=ColumnType.INTEGER,
                nullable=False,
                primary_key=True,
            ),
            ColumnSchema(
                name="email",
                data_type=ColumnType.VARCHAR,
                nullable=False,
                unique=True,
            ),
        ],
        primary_key=["id"],
    )
    orders = TableSchema(
        name="orders",
        columns=[
            ColumnSchema(
                name="id",
                data_type=ColumnType.INTEGER,
                nullable=False,
                primary_key=True,
            ),
            ColumnSchema(
                name="user_id",
                data_type=ColumnType.INTEGER,
                nullable=False,
            ),
            ColumnSchema(
                name="total",
                data_type=ColumnType.INTEGER,
                nullable=False,
            ),
        ],
        primary_key=["id"],
        foreign_keys=[
            ForeignKeySchema(
                columns=["user_id"],
                ref_table="users",
                ref_columns=["id"],
            )
        ],
    )
    return DatabaseSchema(tables=[users, orders])


def _clean_result() -> GenerateResult:
    """A fully-consistent two-table dataset (no integrity violations)."""
    users = [{"id": i, "email": f"u{i}@example.com"} for i in range(1, 4)]
    orders = [{"id": i, "user_id": i, "total": i * 10} for i in range(1, 4)]
    return GenerateResult(
        tables_data={"users": users, "orders": orders},
        insertion_order=["users", "orders"],
        total_rows=6,
        total_tables=2,
        duration_seconds=0.0,
    )


def _seed(app: FastAPI, schema: DatabaseSchema | None, result: GenerateResult | None) -> None:
    if schema is not None:
        app.state.workspace.set_schema(schema)
    if result is not None:
        app.state.workspace.set_last_result(result)


# ── router seam / registration ─────────────────────────────────────────


def test_validate_router_importable() -> None:
    from fastapi import APIRouter  # noqa: PLC0415

    from dbsprout.web.routers.validate import validate_router  # noqa: PLC0415

    assert isinstance(validate_router, APIRouter)


def test_validate_route_registered_on_app(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    paths = {getattr(r, "path", "") for r in app.routes}
    assert "/api/validate" in paths


# ── 409 NO_RUN ─────────────────────────────────────────────────────────


def test_validate_409_when_no_last_result(tmp_path: Path) -> None:
    """AC: no run yet → 409 with ``{"code": "NO_RUN", "message": ...}``."""
    app = _make_app(tmp_path)
    # Schema may be set but no last_result → still NO_RUN.
    _seed(app, _users_orders_schema(), None)
    resp = TestClient(app).post("/api/validate")
    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert isinstance(detail, dict)
    assert detail["code"] == "NO_RUN"
    assert "message" in detail
    assert "Traceback" not in resp.text


def test_validate_409_when_no_schema_either(tmp_path: Path) -> None:
    """No schema *and* no last_result still surfaces NO_RUN (single guard)."""
    app = _make_app(tmp_path)
    resp = TestClient(app).post("/api/validate")
    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"]["code"] == "NO_RUN"


def test_validate_409_htmx_fragment(tmp_path: Path) -> None:
    """HTMX variant of the NO_RUN response returns an HTML 409 fragment."""
    app = _make_app(tmp_path)
    resp = TestClient(app).post("/api/validate", headers={"HX-Request": "true"})
    assert resp.status_code == 409, resp.text
    assert "text/html" in resp.headers["content-type"].lower()
    body = resp.text
    assert "NO_RUN" in body or "no run" in body.lower() or "generate" in body.lower()


# ── happy path: clean data → no violations ─────────────────────────────


def test_validate_clean_dataset_returns_zero_violations(tmp_path: Path) -> None:
    """AC: JSON envelope on clean data has ``summary.violations == 0``."""
    app = _make_app(tmp_path)
    _seed(app, _users_orders_schema(), _clean_result())
    resp = TestClient(app).post("/api/validate")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # envelope shape — ``summary``/``by_table``/``details`` come from S-133;
    # S-134 additionally guarantees ``fidelity`` + ``detection`` keys (both
    # ``None`` here — no reference rows seeded).
    assert {"summary", "by_table", "details"} <= set(body)
    assert body.get("fidelity") is None
    assert body.get("detection") is None
    assert set(body["summary"]) == {"tables", "rows", "violations"}
    assert body["summary"]["tables"] == 2
    assert body["summary"]["rows"] == 6
    assert body["summary"]["violations"] == 0

    # by_table populated with all tables and zero counts
    by_table: list[dict[str, Any]] = body["by_table"]
    assert {row["table"] for row in by_table} == {"users", "orders"}
    for row in by_table:
        assert row["fk_violations"] == 0
        assert row["unique_violations"] == 0
        assert row["not_null_violations"] == 0
        assert row["check_violations"] == 0

    assert body["details"] == []


# ── fk violation ───────────────────────────────────────────────────────


def test_validate_detects_fk_violation(tmp_path: Path) -> None:
    """AC: a row with an orphan FK is bucketed under ``fk_violations``."""
    app = _make_app(tmp_path)
    schema = _users_orders_schema()
    users = [{"id": 1, "email": "a@example.com"}]
    orders = [
        {"id": 1, "user_id": 1, "total": 10},
        {"id": 2, "user_id": 999, "total": 20},  # orphan FK
    ]
    result = GenerateResult(
        tables_data={"users": users, "orders": orders},
        insertion_order=["users", "orders"],
        total_rows=3,
        total_tables=2,
        duration_seconds=0.0,
    )
    _seed(app, schema, result)
    resp = TestClient(app).post("/api/validate")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["summary"]["violations"] == 1
    orders_row = next(r for r in body["by_table"] if r["table"] == "orders")
    assert orders_row["fk_violations"] == 1
    assert orders_row["unique_violations"] == 0

    # Details contain one entry referencing orders.user_id
    assert len(body["details"]) == 1
    d0 = body["details"][0]
    assert d0["table"] == "orders"
    assert d0["column"] == "user_id"
    assert d0["check"] == "fk_satisfaction"
    assert "passed" in d0
    assert d0["passed"] is False


# ── unique + not-null violations ───────────────────────────────────────


def test_validate_detects_unique_and_not_null_violations(tmp_path: Path) -> None:
    """Duplicate email + NULL total → two violation buckets."""
    app = _make_app(tmp_path)
    schema = _users_orders_schema()
    users = [
        {"id": 1, "email": "dup@example.com"},
        {"id": 2, "email": "dup@example.com"},
    ]
    orders = [{"id": 1, "user_id": 1, "total": None}]
    result = GenerateResult(
        tables_data={"users": users, "orders": orders},
        insertion_order=["users", "orders"],
        total_rows=3,
        total_tables=2,
        duration_seconds=0.0,
    )
    _seed(app, schema, result)
    resp = TestClient(app).post("/api/validate")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    users_row = next(r for r in body["by_table"] if r["table"] == "users")
    orders_row = next(r for r in body["by_table"] if r["table"] == "orders")
    assert users_row["unique_violations"] == 1
    assert users_row["not_null_violations"] == 0
    assert orders_row["not_null_violations"] == 1
    assert body["summary"]["violations"] == 2

    # details should include both
    checks = {(d["check"], d["table"], d["column"]) for d in body["details"]}
    assert ("unique", "users", "email") in checks
    assert ("not_null", "orders", "total") in checks


# ── details cap ────────────────────────────────────────────────────────


def test_validate_details_capped_at_500(tmp_path: Path) -> None:
    """AC: ``details`` is capped at 500 rows; envelope reports the cap meta."""
    app = _make_app(tmp_path)

    # Schema with one single-column-PK table + a UNIQUE column we can spam dups on.
    cols = [
        ColumnSchema(
            name="id",
            data_type=ColumnType.INTEGER,
            nullable=False,
            primary_key=True,
        ),
        ColumnSchema(name="tag", data_type=ColumnType.VARCHAR, nullable=False, unique=True),
    ]
    # Generate 600 tables to exceed the cap (each with one not-null-violation row).
    n_tables = 600
    tables = [TableSchema(name=f"t{i}", columns=cols, primary_key=["id"]) for i in range(n_tables)]
    schema = DatabaseSchema(tables=tables)
    # One row per table with NULL ``tag`` → 600 ``not_null`` failures + 600 ``unique`` (which pass
    # because we exclude NULLs for unique). To be safe make ``tag`` also collide.
    tables_data = {f"t{i}": [{"id": 1, "tag": None}] for i in range(n_tables)}
    result = GenerateResult(
        tables_data=tables_data,
        insertion_order=[f"t{i}" for i in range(n_tables)],
        total_rows=n_tables,
        total_tables=n_tables,
        duration_seconds=0.0,
    )
    _seed(app, schema, result)
    resp = TestClient(app).post("/api/validate")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # summary.violations counts ALL violations (uncapped).
    assert body["summary"]["violations"] >= n_tables  # at least one per table
    assert len(body["details"]) == 500


# ── HTMX fragment response ─────────────────────────────────────────────


def test_validate_htmx_fragment_renders_panel(tmp_path: Path) -> None:
    """AC: HTMX variant returns an HTML fragment with violation rows + S-135 hooks."""
    app = _make_app(tmp_path)
    schema = _users_orders_schema()
    users = [{"id": 1, "email": "a@example.com"}]
    orders = [
        {"id": 1, "user_id": 1, "total": 10},
        {"id": 2, "user_id": 999, "total": 20},
    ]
    result = GenerateResult(
        tables_data={"users": users, "orders": orders},
        insertion_order=["users", "orders"],
        total_rows=3,
        total_tables=2,
        duration_seconds=0.0,
    )
    _seed(app, schema, result)
    resp = TestClient(app).post("/api/validate", headers={"HX-Request": "true"})
    assert resp.status_code == 200, resp.text
    assert "text/html" in resp.headers["content-type"].lower()
    body = resp.text

    # row attributes that S-135 listens on
    assert 'data-table="orders"' in body
    assert 'data-column="user_id"' in body
    # data-row may be empty when we don't have a specific row index — but the
    # attribute is present on every violation row.
    assert "data-row=" in body


def test_validate_htmx_fragment_clean_dataset_renders_pass_state(tmp_path: Path) -> None:
    """Clean data → fragment renders a "no violations" success state."""
    app = _make_app(tmp_path)
    _seed(app, _users_orders_schema(), _clean_result())
    resp = TestClient(app).post("/api/validate", headers={"HX-Request": "true"})
    assert resp.status_code == 200, resp.text
    assert "text/html" in resp.headers["content-type"].lower()
    body = resp.text.lower()
    assert (
        "no violations" in body
        or "all checks passed" in body
        or "0 violations" in body
        or "clean" in body
    )


# ── lazy-import contract ───────────────────────────────────────────────


def test_validate_router_no_eager_generation_import() -> None:
    """Importing the router must not pull the orchestrator / core service eagerly."""
    probe = (
        "import sys\n"
        "import dbsprout.web.routers.validate  # noqa: F401\n"
        "bad = [m for m in ('dbsprout.generate.orchestrator', 'dbsprout.core.service')"
        " if m in sys.modules]\n"
        "print(bad)\n"
    )
    result = subprocess.run(  # noqa: S603 - fixed argv, trusted interpreter
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert result.stdout.strip() == "[]", (
        "importing dbsprout.web.routers.validate eagerly imported generation/core "
        f"modules: {result.stdout.strip()}"
    )
