"""GET /api/preview/{table} bounded sample tests (S-147).

The web stack lives in the optional ``[web]`` extra, so the module guards with
``pytest.importorskip("fastapi")`` before importing FastAPI symbols (mirrors the
sibling web tests). This endpoint is *read-only* over the in-memory
:class:`~dbsprout.web.workspace.Workspace` (``app.state.workspace``, S-111) that
``POST /api/generate`` (S-124) populates with a
:class:`~dbsprout.generate.orchestrator.GenerateResult`.

* ``GET /api/preview/{table}?limit=N`` returns up to ``N`` rows of the
  generated data for that table from ``workspace.last_result.tables_data``;
  ``404`` (friendly JSON, no traceback) when no result is set or the table is
  unknown; ``422`` when ``limit`` is outside ``[1, 1000]``.

Tests seed ``app.state.workspace`` directly — same approach as ``test_schema_view``.
"""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")

from fastapi.testclient import TestClient

from dbsprout.generate.orchestrator import GenerateResult

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi import FastAPI


# ── fixtures / helpers ────────────────────────────────────────────────


def _make_app(tmp_path: Path) -> FastAPI:
    from dbsprout.web.app import create_app  # noqa: PLC0415

    return create_app(state_db_path=tmp_path / "state.db")


def _sample_result(
    rows_per_table: int = 5,
) -> GenerateResult:
    """Build a small ``GenerateResult`` with two tables and *rows_per_table* rows each."""
    users = [{"id": i, "email": f"u{i}@example.com"} for i in range(1, rows_per_table + 1)]
    orders = [{"id": i, "user_id": i, "total": i * 10} for i in range(1, rows_per_table + 1)]
    return GenerateResult(
        tables_data={"users": users, "orders": orders},
        insertion_order=["users", "orders"],
        total_rows=rows_per_table * 2,
        total_tables=2,
        duration_seconds=0.01,
    )


def _seed(app: FastAPI, result: GenerateResult) -> None:
    """Load *result* onto the app's workspace, as POST /api/generate would."""
    app.state.workspace.set_last_result(result)


# ── happy path ────────────────────────────────────────────────────────


def test_preview_returns_rows_up_to_limit(tmp_path: Path) -> None:
    """AC: returns first up-to-N rows from ``last_result.tables_data[table]``."""
    app = _make_app(tmp_path)
    _seed(app, _sample_result(rows_per_table=10))
    resp = TestClient(app).get("/api/preview/users?limit=3")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["table"] == "users"
    assert body["total"] == 10
    assert body["limit"] == 3
    assert len(body["rows"]) == 3
    assert body["rows"][0] == {"id": 1, "email": "u1@example.com"}
    assert body["rows"][2] == {"id": 3, "email": "u3@example.com"}


def test_preview_returns_all_rows_when_limit_exceeds_table_size(tmp_path: Path) -> None:
    """AC: ``len(rows) <= limit`` — fewer rows are fine when the table is smaller."""
    app = _make_app(tmp_path)
    _seed(app, _sample_result(rows_per_table=2))
    resp = TestClient(app).get("/api/preview/users?limit=100")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["rows"]) == 2
    assert body["total"] == 2
    assert body["limit"] == 100


def test_preview_default_limit_is_100(tmp_path: Path) -> None:
    """AC: ``?limit=100`` is the documented default — omitting the param uses it."""
    app = _make_app(tmp_path)
    _seed(app, _sample_result(rows_per_table=250))
    resp = TestClient(app).get("/api/preview/users")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["limit"] == 100
    assert len(body["rows"]) == 100


def test_preview_returns_orders_table(tmp_path: Path) -> None:
    """A second table also works (no accidental hard-coding)."""
    app = _make_app(tmp_path)
    _seed(app, _sample_result(rows_per_table=4))
    resp = TestClient(app).get("/api/preview/orders?limit=2")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["table"] == "orders"
    assert body["rows"][0] == {"id": 1, "user_id": 1, "total": 10}
    assert len(body["rows"]) == 2


# ── unknown table ─────────────────────────────────────────────────────


def test_preview_404_for_unknown_table(tmp_path: Path) -> None:
    """AC: friendly 404 JSON when the table is not in ``last_result`` (no traceback)."""
    app = _make_app(tmp_path)
    _seed(app, _sample_result())
    resp = TestClient(app).get("/api/preview/missing?limit=5")
    assert resp.status_code == 404, resp.text
    detail = resp.json()["detail"]
    assert isinstance(detail, str)
    assert "missing" in detail
    assert "Traceback" not in resp.text


# ── no last_result ────────────────────────────────────────────────────


def test_preview_404_when_no_last_result_loaded(tmp_path: Path) -> None:
    """AC: friendly 404 JSON when no generation has run yet (no traceback)."""
    resp = TestClient(_make_app(tmp_path)).get("/api/preview/users?limit=10")
    assert resp.status_code == 404, resp.text
    detail = resp.json()["detail"]
    assert isinstance(detail, str)
    assert "generate" in detail.lower() or "no result" in detail.lower()
    assert "Traceback" not in resp.text


# ── limit validation ──────────────────────────────────────────────────


def test_preview_422_for_zero_limit(tmp_path: Path) -> None:
    """``limit`` must be a positive int (Pydantic Query ge=1)."""
    app = _make_app(tmp_path)
    _seed(app, _sample_result())
    resp = TestClient(app).get("/api/preview/users?limit=0")
    assert resp.status_code == 422, resp.text


def test_preview_422_for_negative_limit(tmp_path: Path) -> None:
    """``limit`` must be a positive int (Pydantic Query ge=1)."""
    app = _make_app(tmp_path)
    _seed(app, _sample_result())
    resp = TestClient(app).get("/api/preview/users?limit=-5")
    assert resp.status_code == 422, resp.text


def test_preview_422_when_limit_above_cap(tmp_path: Path) -> None:
    """Sane cap (1000) prevents huge payloads."""
    app = _make_app(tmp_path)
    _seed(app, _sample_result())
    resp = TestClient(app).get("/api/preview/users?limit=10001")
    assert resp.status_code == 422, resp.text


def test_preview_422_for_non_integer_limit(tmp_path: Path) -> None:
    """Non-numeric ``limit`` is rejected at the boundary."""
    app = _make_app(tmp_path)
    _seed(app, _sample_result())
    resp = TestClient(app).get("/api/preview/users?limit=abc")
    assert resp.status_code == 422, resp.text


def test_preview_limit_1_returns_single_row(tmp_path: Path) -> None:
    """``limit=1`` is the minimum allowed and returns exactly one row."""
    app = _make_app(tmp_path)
    _seed(app, _sample_result(rows_per_table=5))
    resp = TestClient(app).get("/api/preview/users?limit=1")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["rows"]) == 1


def test_preview_limit_1000_is_accepted(tmp_path: Path) -> None:
    """The documented cap (1000) is itself accepted, not 422."""
    app = _make_app(tmp_path)
    _seed(app, _sample_result(rows_per_table=3))
    resp = TestClient(app).get("/api/preview/users?limit=1000")
    assert resp.status_code == 200, resp.text


# ── empty table edge case ─────────────────────────────────────────────


def test_preview_404_for_unknown_table_when_result_has_no_tables(tmp_path: Path) -> None:
    """Friendly 404 with the "no tables" variant when the result is empty."""
    app = _make_app(tmp_path)
    empty_result = GenerateResult(
        tables_data={},
        insertion_order=[],
        total_rows=0,
        total_tables=0,
        duration_seconds=0.0,
    )
    _seed(app, empty_result)
    resp = TestClient(app).get("/api/preview/anything?limit=10")
    assert resp.status_code == 404, resp.text
    detail = resp.json()["detail"]
    assert "no tables" in detail.lower()
    assert "Traceback" not in resp.text


def test_preview_empty_table_returns_empty_rows(tmp_path: Path) -> None:
    """A table with 0 generated rows still returns 200 with ``rows=[]``."""
    app = _make_app(tmp_path)
    empty_result = GenerateResult(
        tables_data={"users": []},
        insertion_order=["users"],
        total_rows=0,
        total_tables=1,
        duration_seconds=0.0,
    )
    _seed(app, empty_result)
    resp = TestClient(app).get("/api/preview/users?limit=10")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["rows"] == []
    assert body["total"] == 0


# ── router registration / seam ─────────────────────────────────────────


def test_preview_router_importable() -> None:
    from fastapi import APIRouter  # noqa: PLC0415

    from dbsprout.web.routers.preview import preview_router  # noqa: PLC0415

    assert isinstance(preview_router, APIRouter)


def test_preview_route_registered_on_app(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    paths = {getattr(r, "path", "") for r in app.routes}
    assert "/api/preview/{table}" in paths


# ── lazy-import contract ───────────────────────────────────────────────


def test_preview_router_no_eager_generation_import() -> None:
    """Importing the router must not pull the orchestrator / core service eagerly."""
    probe = (
        "import sys\n"
        "import dbsprout.web.routers.preview  # noqa: F401\n"
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
        "importing dbsprout.web.routers.preview eagerly imported generation/core "
        f"modules: {result.stdout.strip()}"
    )
