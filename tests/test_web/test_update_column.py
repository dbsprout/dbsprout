"""``POST /api/update-column`` endpoint tests (S-139).

Wave 2 of the Output & Insertion sub-epic in the Web UI v1.2.0 sprint. The
route lives in :mod:`dbsprout.web.routers.insert` inside the
``# region: POST /api/update-column (S-139)`` block and re-uses the S-137
write-guard HMAC scheme — with the scope hash bound to
``(table, column, row_count)`` instead of ``[(table, row_count), ...]`` —
so a single update-column token can never be replayed against
``POST /api/insert`` (and vice versa).

The tests follow the same pattern as :mod:`tests.test_web.test_insert`
and :mod:`tests.test_web.test_write_guard`: a tiny SQLite fixture, the
synchronous generate shortcut via :func:`dbsprout.core.service.generate`,
and a TestClient. The route is exercised end-to-end so the actual UPDATE
hits the live SQLite file.
"""

from __future__ import annotations

import sqlite3
import time
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")

from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi import FastAPI


# ── helpers (mirror tests/test_web/test_insert.py) ──────────────────────


def _make_app(state_db: Path) -> FastAPI:
    from dbsprout.web.app import create_app  # noqa: PLC0415

    return create_app(state_db_path=state_db)


def _temp_sqlite_with_users(tmp_path: Path) -> str:
    """Create a 1-table SQLite DB with an existing users row + return the URL."""
    db_path = tmp_path / "target.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute(
            "CREATE TABLE posts (id INTEGER PRIMARY KEY, user_id INTEGER REFERENCES users(id))"
        )
        conn.commit()
    finally:
        conn.close()
    return f"sqlite:///{db_path}"


def _connect_and_generate(app: FastAPI, tmp_path: Path) -> str:
    """Wire workspace target + populate ``last_result`` synchronously."""
    from dbsprout.config.models import DBSproutConfig  # noqa: PLC0415
    from dbsprout.core.service import generate as svc_generate  # noqa: PLC0415

    target_url = _temp_sqlite_with_users(tmp_path)
    client = TestClient(app)
    r1 = client.post("/api/connect", json={"url": target_url})
    assert r1.status_code == 200, r1.text

    workspace = app.state.workspace
    schema = workspace.get_schema()
    assert schema is not None
    config = DBSproutConfig()
    result = svc_generate(
        schema,
        config,
        seed=7,
        default_rows=config.generation.default_rows,
        engine="heuristic",
    )
    workspace.set_last_result(result)
    return target_url


def _insert_rows_into_users(target_url: str, ids: list[int]) -> None:
    """Pre-insert rows into ``users`` so the UPDATE has actual targets."""
    db_path = target_url.replace("sqlite:///", "")
    conn = sqlite3.connect(db_path)
    try:
        conn.executemany(
            "INSERT INTO users (id, name) VALUES (?, ?)",
            [(i, f"old-name-{i}") for i in ids],
        )
        conn.commit()
    finally:
        conn.close()


def _bypass_write_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DBSPROUT_DISABLE_WRITE_GUARD", "1")


def _mint_update_column_token(  # noqa: PLR0913 — sibling of preview-route token builder
    app: FastAPI,
    *,
    target_url: str,
    table: str,
    column: str,
    row_count: int,
    exp_offset: int = 300,
) -> str:
    """Mint a valid update-column token by re-using the S-137 helpers.

    The route's scope binder uses :func:`_hash_update_column_scope` —
    callers re-derive the hash here so the test never imports the route's
    internal flow more than necessary.
    """
    import secrets  # noqa: PLC0415

    from dbsprout.web.routers.insert import (  # noqa: PLC0415
        _encode_token,
        _get_issued_registry,
        _get_write_guard_secret,
        _hash_target,
        _hash_update_column_scope,
    )

    secret = _get_write_guard_secret(app)
    nonce = secrets.token_hex(16)
    payload = {
        "target_hash": _hash_target(target_url),
        "scope_hash": _hash_update_column_scope(table=table, column=column, row_count=row_count),
        "exp": int(time.time()) + exp_offset,
        "nonce": nonce,
        # Marker so the validator can refuse insert-style tokens replayed
        # at this endpoint (and vice versa).
        "kind": "update_column",
    }
    token = _encode_token(payload, secret)
    issued = _get_issued_registry(app)
    issued[nonce] = payload["exp"]
    return token


# ── Step 1: route mounted + router importable ───────────────────────────


def test_update_column_router_is_importable() -> None:
    from fastapi import APIRouter  # noqa: PLC0415

    from dbsprout.web.routers.insert import insert_router  # noqa: PLC0415

    assert isinstance(insert_router, APIRouter)


def test_update_column_route_is_mounted(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")
    paths = {route.path for route in app.routes if hasattr(route, "path")}
    assert "/api/update-column" in paths


# ── Step 2: missing-token guard (production-default closed) ─────────────


def test_update_column_without_token_is_403(tmp_path: Path) -> None:
    """Without DBSPROUT_DISABLE_WRITE_GUARD, missing token → 403 WRITE_GUARD_REQUIRED."""
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)
    resp = TestClient(app).post("/api/update-column", json={"table": "users", "column": "name"})
    assert resp.status_code == 403, resp.text
    assert resp.json()["detail"]["code"] == "WRITE_GUARD_REQUIRED"
    assert "Traceback" not in resp.text


def test_update_column_env_bypass_skips_missing_token_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With env var set, missing token reaches the next guard (NOT_FOUND here)."""
    _bypass_write_guard(monkeypatch)
    app = _make_app(tmp_path / "state.db")
    target_url = _connect_and_generate(app, tmp_path)
    # No pre-existing row, but a real generate result exists, so the route
    # goes straight to the writer which raises ColumnUpdateError - but the
    # writer doesn't run yet because the UPDATE has no rows to touch and
    # rows iterable is non-empty (generated rows exist). To prove the env
    # var bypass alone, just assert we don't get a 403 here.
    _insert_rows_into_users(target_url, ids=[1, 2])
    resp = TestClient(app).post("/api/update-column", json={"table": "users", "column": "name"})
    assert resp.status_code != 403, resp.text


# ── Step 3: no-connection guard ─────────────────────────────────────────


def test_update_column_without_connection_is_409(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _bypass_write_guard(monkeypatch)
    app = _make_app(tmp_path / "state.db")
    resp = TestClient(app).post("/api/update-column", json={"table": "users", "column": "name"})
    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"]["code"] == "NO_CONNECTION"


# ── Step 4: no-regen guard ──────────────────────────────────────────────


def test_update_column_without_generate_result_is_409_no_regen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _bypass_write_guard(monkeypatch)
    app = _make_app(tmp_path / "state.db")
    target_url = _temp_sqlite_with_users(tmp_path)
    TestClient(app).post("/api/connect", json={"url": target_url})
    # Note: NO generate result on the workspace.
    resp = TestClient(app).post("/api/update-column", json={"table": "users", "column": "name"})
    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"]["code"] == "NO_REGEN"


def test_update_column_unknown_table_is_409_no_regen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unknown table in the request → 409 NO_REGEN (no rows for that table)."""
    _bypass_write_guard(monkeypatch)
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)
    resp = TestClient(app).post(
        "/api/update-column",
        json={"table": "ghost_table", "column": "name"},
    )
    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"]["code"] == "NO_REGEN"


# ── Step 5: PK-less table → 409 CONSTRAINT_VIOLATION ────────────────────


def test_update_column_pkless_table_is_409_constraint_violation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Seed a schema whose table has no primary key — writer raises
    ColumnUpdateError(code='no_primary_key') and the route maps it to
    a 409 CONSTRAINT_VIOLATION envelope."""
    _bypass_write_guard(monkeypatch)
    app = _make_app(tmp_path / "state.db")
    # Set up workspace by hand: a target URL + a hand-crafted schema with
    # a PK-less ``logs`` table + a matching ``last_result``.
    from dbsprout.generate.orchestrator import GenerateResult  # noqa: PLC0415
    from dbsprout.schema.models import (  # noqa: PLC0415
        ColumnSchema,
        DatabaseSchema,
        TableSchema,
    )

    workspace = app.state.workspace
    target_url = _temp_sqlite_with_users(tmp_path)
    workspace.set_target_url(target_url)
    logs_table = TableSchema(
        name="logs",
        columns=[ColumnSchema(name="message", data_type="text", nullable=True)],
        primary_key=[],
    )
    schema = DatabaseSchema(tables=[logs_table], dialect="sqlite")
    workspace.set_schema(schema)
    workspace.set_last_result(
        GenerateResult(
            tables_data={"logs": [{"message": "hello"}]},
            insertion_order=["logs"],
            total_rows=1,
            duration_seconds=0.0,
            table_timings=(),
        )
    )

    resp = TestClient(app).post(
        "/api/update-column",
        json={"table": "logs", "column": "message"},
    )
    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert detail["code"] == "CONSTRAINT_VIOLATION"
    assert "no primary key" in detail["message"].lower()


# ── Step 6: happy path — SQLite UPDATE landed ───────────────────────────


def test_update_column_happy_path_updates_rows(tmp_path: Path) -> None:
    """End-to-end: token bound to scope, UPDATE actually executes on SQLite."""
    app = _make_app(tmp_path / "state.db")
    target_url = _connect_and_generate(app, tmp_path)

    workspace = app.state.workspace
    # Replace the generated rows with known PK + value pairs we control.
    from dataclasses import replace  # noqa: PLC0415

    current = workspace.get_last_result()
    assert current is not None
    new_rows = [
        {"id": 1, "name": "alice-new"},
        {"id": 2, "name": "bob-new"},
        {"id": 3, "name": "carol-new"},
    ]
    swapped = {**current.tables_data, "users": new_rows}
    workspace.set_last_result(
        replace(
            current,
            tables_data=swapped,
            total_rows=sum(len(rows) for rows in swapped.values()),
        )
    )
    _insert_rows_into_users(target_url, ids=[1, 2, 3])

    token = _mint_update_column_token(
        app, target_url=target_url, table="users", column="name", row_count=3
    )
    resp = TestClient(app).post(
        "/api/update-column",
        json={"table": "users", "column": "name", "confirmation_token": token},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body == {"rows_updated": 3, "table": "users", "column": "name"}

    # Verify the UPDATE actually landed.
    db_path = target_url.replace("sqlite:///", "")
    conn = sqlite3.connect(db_path)
    try:
        rows = dict(conn.execute("SELECT id, name FROM users ORDER BY id").fetchall())
    finally:
        conn.close()
    assert rows == {1: "alice-new", 2: "bob-new", 3: "carol-new"}


# ── Step 7: scope-mismatched token → 403 ────────────────────────────────


def test_update_column_token_bound_to_other_column_is_403(tmp_path: Path) -> None:
    """A token signed for column ``id`` is rejected for column ``name``."""
    app = _make_app(tmp_path / "state.db")
    target_url = _connect_and_generate(app, tmp_path)

    # Mint token for *id* — submit request for *name* → mismatch.
    rows_count = len(app.state.workspace.get_last_result().tables_data["users"])
    token = _mint_update_column_token(
        app,
        target_url=target_url,
        table="users",
        column="id",
        row_count=rows_count,
    )
    resp = TestClient(app).post(
        "/api/update-column",
        json={"table": "users", "column": "name", "confirmation_token": token},
    )
    assert resp.status_code == 403, resp.text
    assert resp.json()["detail"]["code"] == "WRITE_GUARD_REJECTED"


def test_update_column_token_bound_to_other_row_count_is_403(tmp_path: Path) -> None:
    """A token signed for a different row_count is rejected."""
    app = _make_app(tmp_path / "state.db")
    target_url = _connect_and_generate(app, tmp_path)

    real_count = len(app.state.workspace.get_last_result().tables_data["users"])
    token = _mint_update_column_token(
        app,
        target_url=target_url,
        table="users",
        column="name",
        row_count=real_count + 1,
    )
    resp = TestClient(app).post(
        "/api/update-column",
        json={"table": "users", "column": "name", "confirmation_token": token},
    )
    assert resp.status_code == 403, resp.text
    assert resp.json()["detail"]["code"] == "WRITE_GUARD_REJECTED"


# ── Step 8: token single-use ────────────────────────────────────────────


def test_update_column_token_is_single_use(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")
    target_url = _connect_and_generate(app, tmp_path)

    workspace = app.state.workspace
    from dataclasses import replace  # noqa: PLC0415

    current = workspace.get_last_result()
    assert current is not None
    new_rows = [{"id": 10, "name": "first"}]
    swapped = {**current.tables_data, "users": new_rows}
    workspace.set_last_result(
        replace(
            current,
            tables_data=swapped,
            total_rows=sum(len(rows) for rows in swapped.values()),
        )
    )
    _insert_rows_into_users(target_url, ids=[10])

    token = _mint_update_column_token(
        app, target_url=target_url, table="users", column="name", row_count=1
    )
    client = TestClient(app)
    body = {"table": "users", "column": "name", "confirmation_token": token}
    first = client.post("/api/update-column", json=body)
    assert first.status_code == 200, first.text
    second = client.post("/api/update-column", json=body)
    assert second.status_code == 403, second.text
    assert second.json()["detail"]["code"] == "WRITE_GUARD_REJECTED"


# ── Step 9: bogus token ────────────────────────────────────────────────


def test_update_column_bogus_token_is_403(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)
    resp = TestClient(app).post(
        "/api/update-column",
        json={
            "table": "users",
            "column": "name",
            "confirmation_token": "definitely-not-a-real-token",
        },
    )
    assert resp.status_code == 403, resp.text
    assert resp.json()["detail"]["code"] == "WRITE_GUARD_REJECTED"


# ── Step 10: extra fields rejected ──────────────────────────────────────


def test_update_column_extra_field_is_422(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _bypass_write_guard(monkeypatch)
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)
    resp = TestClient(app).post(
        "/api/update-column",
        json={"table": "users", "column": "name", "weird": 1},
    )
    assert resp.status_code == 422


# ── Step 11: unknown column → 404 NOT_FOUND ─────────────────────────────


def test_update_column_unknown_column_is_404_not_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _bypass_write_guard(monkeypatch)
    app = _make_app(tmp_path / "state.db")
    target_url = _connect_and_generate(app, tmp_path)
    _insert_rows_into_users(target_url, ids=[1])
    resp = TestClient(app).post(
        "/api/update-column",
        json={"table": "users", "column": "no_such_column"},
    )
    assert resp.status_code == 404, resp.text
    assert resp.json()["detail"]["code"] == "NOT_FOUND"


# ── Step 12: factory + enum surface (NO_REGEN) ──────────────────────────


def test_no_regen_factory_returns_typed_envelope() -> None:
    from dbsprout.web.errors import WebErrorCode, web_error_no_regen  # noqa: PLC0415

    err = web_error_no_regen()
    payload = err.to_dict()
    assert payload["code"] == "NO_REGEN"
    assert err.status_code == 409
    assert err.code is WebErrorCode.NO_REGEN
    assert "regenerate" in payload["message"].lower() or "regen" in payload["message"].lower()


# ── Step 13: PK-update column → 409 CONSTRAINT_VIOLATION ────────────────


def test_update_column_pk_column_is_409_constraint_violation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Patching a PK column itself is refused with reason='pk_column_update'."""
    _bypass_write_guard(monkeypatch)
    app = _make_app(tmp_path / "state.db")
    target_url = _connect_and_generate(app, tmp_path)
    _insert_rows_into_users(target_url, ids=[1, 2])

    resp = TestClient(app).post(
        "/api/update-column",
        json={"table": "users", "column": "id"},
    )
    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert detail["code"] == "CONSTRAINT_VIOLATION"


# ── Step 14: response shape is exactly {rows_updated, table, column} ────


def test_update_column_response_shape_is_strict(tmp_path: Path) -> None:
    """Verify the success body has exactly the documented keys, no extras."""
    app = _make_app(tmp_path / "state.db")
    target_url = _connect_and_generate(app, tmp_path)

    workspace = app.state.workspace
    from dataclasses import replace  # noqa: PLC0415

    current = workspace.get_last_result()
    assert current is not None
    new_rows = [{"id": 99, "name": "x"}]
    swapped = {**current.tables_data, "users": new_rows}
    workspace.set_last_result(
        replace(
            current,
            tables_data=swapped,
            total_rows=sum(len(rows) for rows in swapped.values()),
        )
    )
    _insert_rows_into_users(target_url, ids=[99])

    token = _mint_update_column_token(
        app, target_url=target_url, table="users", column="name", row_count=1
    )
    resp = TestClient(app).post(
        "/api/update-column",
        json={"table": "users", "column": "name", "confirmation_token": token},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body.keys()) == {"rows_updated", "table", "column"}
