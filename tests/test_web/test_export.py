"""POST /api/export tests (S-140).

Module guards on fastapi presence (optional [web] extra), mirroring sibling
web tests.
"""

from __future__ import annotations

import json as json_module
import sqlite3
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")

from fastapi.testclient import TestClient
from pydantic import ValidationError

from dbsprout.web.routers.export import ExportRequest, _resolve_scope

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi import FastAPI


# ── Task 2: request model + scope helper (pure-logic) ──────────────────


def test_export_request_accepts_all_four_formats() -> None:
    for fmt in ("sql", "csv", "json", "parquet"):
        req = ExportRequest(format=fmt)  # type: ignore[arg-type]
        assert req.format == fmt
        assert req.tables is None


def test_export_request_rejects_unknown_format() -> None:
    with pytest.raises(ValidationError):
        ExportRequest(format="xml")  # type: ignore[arg-type]


def test_export_request_rejects_extra_keys() -> None:
    with pytest.raises(ValidationError):
        ExportRequest(format="sql", surprise="boom")  # type: ignore[call-arg]


def test_export_request_accepts_tables_list() -> None:
    req = ExportRequest(format="sql", tables=["users", "orders"])
    assert req.tables == ["users", "orders"]


def test_resolve_scope_full_when_tables_none() -> None:
    insertion_order = ["users", "orders", "items"]
    scope, missing = _resolve_scope(insertion_order, None)
    assert scope == ["users", "orders", "items"]
    assert missing == []


def test_resolve_scope_full_when_tables_empty() -> None:
    scope, missing = _resolve_scope(["a", "b"], [])
    assert scope == ["a", "b"]
    assert missing == []


def test_resolve_scope_filters_preserving_order() -> None:
    insertion_order = ["users", "orders", "items"]
    scope, missing = _resolve_scope(insertion_order, ["items", "users"])
    assert scope == ["users", "items"]  # preserved insertion order
    assert missing == []


def test_resolve_scope_reports_missing_tables() -> None:
    insertion_order = ["users", "orders"]
    scope, missing = _resolve_scope(insertion_order, ["users", "nope", "gone"])
    assert scope == ["users"]
    assert sorted(missing) == ["gone", "nope"]


# ── Task 3-5: route handler — fixture + integration tests ─────────────


def _make_app(state_db: Path) -> FastAPI:
    from dbsprout.web.app import create_app  # noqa: PLC0415

    return create_app(state_db_path=state_db)


def _temp_sqlite(tmp_path: Path) -> str:
    db_path = tmp_path / "src.db"
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


def _connect_and_generate(app: FastAPI, tmp_path: Path) -> None:
    from dbsprout.config.models import DBSproutConfig  # noqa: PLC0415
    from dbsprout.core.service import generate as svc_generate  # noqa: PLC0415

    target_url = _temp_sqlite(tmp_path)
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


def test_export_no_run_returns_409(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")
    client = TestClient(app)

    r = client.post("/api/export", json={"format": "sql"})

    assert r.status_code == 409
    body = r.json()
    assert body["detail"]["code"] == "NO_RUN"


def test_export_unknown_format_returns_422(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")
    client = TestClient(app)

    r = client.post("/api/export", json={"format": "xml"})

    assert r.status_code == 422


def test_export_unknown_tables_returns_404(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)
    client = TestClient(app)

    r = client.post(
        "/api/export",
        json={"format": "sql", "tables": ["users", "ghost"]},
    )

    assert r.status_code == 404
    body = r.json()
    assert body["detail"]["code"] == "NOT_FOUND"
    assert "ghost" in body["detail"]["message"]


def test_export_multi_table_csv_returns_422(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)
    client = TestClient(app)

    r = client.post("/api/export", json={"format": "csv"})

    assert r.status_code == 422
    body = r.json()
    assert body["detail"]["code"] == "EXPORT_MULTI_TABLE_UNSUPPORTED"


def test_export_multi_table_parquet_returns_422(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)
    client = TestClient(app)

    r = client.post("/api/export", json={"format": "parquet"})

    assert r.status_code == 422
    body = r.json()
    assert body["detail"]["code"] == "EXPORT_MULTI_TABLE_UNSUPPORTED"


# ── Task 4: streaming (sql + csv + json) ───────────────────────────────


def test_export_sql_full_run_streams_combined_script(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)
    client = TestClient(app)

    r = client.post("/api/export", json={"format": "sql"})

    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("application/sql")
    cd = r.headers["content-disposition"]
    assert "attachment" in cd
    assert "dbsprout-export.sql" in cd
    body = r.text
    assert "BEGIN;" in body
    assert "COMMIT;" in body
    assert "INSERT INTO" in body
    assert "-- next table --" in body  # multi-table separator
    assert "users" in body
    assert "posts" in body


def test_export_sql_single_table_uses_table_filename(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)
    client = TestClient(app)

    r = client.post("/api/export", json={"format": "sql", "tables": ["users"]})

    assert r.status_code == 200
    cd = r.headers["content-disposition"]
    assert "users.sql" in cd


def test_export_csv_single_table_streams_rfc4180(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)
    client = TestClient(app)

    r = client.post("/api/export", json={"format": "csv", "tables": ["users"]})

    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    cd = r.headers["content-disposition"]
    assert "users.csv" in cd
    text = r.text
    # First line is the header row; column "id" is part of users.
    assert text.splitlines()[0].startswith("id")


def test_export_json_multi_table_wraps_envelope(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)
    client = TestClient(app)

    r = client.post("/api/export", json={"format": "json"})

    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    cd = r.headers["content-disposition"]
    assert "dbsprout-export.json" in cd
    payload = json_module.loads(r.text)
    assert payload["insertion_order"] == ["users", "posts"]
    assert set(payload["tables"]) == {"users", "posts"}
    assert isinstance(payload["tables"]["users"], list)
    assert isinstance(payload["tables"]["posts"], list)


def test_export_json_single_table_streams_array(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)
    client = TestClient(app)

    r = client.post("/api/export", json={"format": "json", "tables": ["users"]})

    assert r.status_code == 200
    cd = r.headers["content-disposition"]
    assert "users.json" in cd
    payload = json_module.loads(r.text)
    # Single-table json uses the bare array shape (NOT the multi-table wrap).
    assert isinstance(payload, list)
    if payload:
        assert isinstance(payload[0], dict)


# ── Task 5: parquet happy path + missing-dep guard ─────────────────────


def test_export_parquet_single_table_returns_binary_file(tmp_path: Path) -> None:
    pytest.importorskip("polars", reason="parquet writer requires [data] extra")
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)
    client = TestClient(app)

    r = client.post("/api/export", json={"format": "parquet", "tables": ["users"]})

    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/vnd.apache.parquet")
    cd = r.headers["content-disposition"]
    assert "users.parquet" in cd
    # Parquet magic header is "PAR1" at the start of the file.
    assert r.content[:4] == b"PAR1"


def test_export_parquet_without_polars_returns_422(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)
    client = TestClient(app)

    # Simulate the parquet writer raising ImportError (writer guards on pl is None).
    import dbsprout.output.parquet_writer as pw  # noqa: PLC0415

    monkeypatch.setattr(pw, "pl", None)

    r = client.post("/api/export", json={"format": "parquet", "tables": ["users"]})

    assert r.status_code == 422
    body = r.json()
    assert body["detail"]["code"] == "EXPORT_DEPENDENCY_MISSING"
    assert "parquet" in body["detail"]["message"]
    hint = body["detail"]["hint"]
    assert hint
    assert "dbsprout[data]" in hint
