"""POST /api/insert ``method`` field tests (S-141).

S-136 (Wave 1) shipped the insert route with a dialect-aware *auto* policy:
PostgreSQL → ``PgCopyWriter`` (fallback ``SaBatchWriter``); MySQL →
``MysqlLoadDataWriter`` (fallback ``SaBatchWriter``); everything else →
``SaBatchWriter``. S-141 adds an explicit ``method`` field so the Studio
user can pin the strategy:

* ``method="auto"`` (default) — preserves S-136 dispatch *byte-for-byte*.
* ``method="batch"`` — forces ``SaBatchWriter`` regardless of dialect.
* ``method="copy"`` — forces ``PgCopyWriter`` (postgresql) or
  ``MysqlLoadDataWriter`` (mysql). Unsupported dialect (sqlite / mssql /
  unknown) → ``409 METHOD_UNSUPPORTED`` with
  ``{dialect, method, supported: [...]}`` in the envelope.

All edits live inside the ``# region: insert method select (S-141)`` block
in ``dbsprout/web/routers/insert.py`` (Wave 2 of S-139 extends the same
file in *its own* region — these regions never collide).
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Any

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


def _temp_sqlite(tmp_path: Path) -> str:
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
    return target_url


def _bypass_write_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DBSPROUT_DISABLE_WRITE_GUARD", "1")


# ── Request validation: method field shape ─────────────────────────────


def test_insert_request_default_method_is_auto(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Omitting ``method`` defaults to ``"auto"`` (preserves S-136 behaviour)."""
    _bypass_write_guard(monkeypatch)
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)
    resp = TestClient(app).post("/api/insert", json={})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # sqlite + auto → SaBatchWriter (same as S-136 default).
    assert body["writer"] == "SaBatchWriter"
    assert body["method"] == "auto"


def test_insert_explicit_method_auto_matches_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Passing ``method="auto"`` is byte-equivalent to omitting it."""
    _bypass_write_guard(monkeypatch)
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)
    resp = TestClient(app).post("/api/insert", json={"method": "auto"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["writer"] == "SaBatchWriter"
    assert body["method"] == "auto"


def test_insert_method_batch_forces_sa_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``method="batch"`` always selects SaBatchWriter, even when the target
    is postgresql/mysql (i.e. when *auto* would have picked COPY)."""
    _bypass_write_guard(monkeypatch)
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)
    # Swap workspace target to a PG-style URL so *auto* would have picked
    # PgCopyWriter — *batch* must override that.
    app.state.workspace.set_target_url("postgresql://x:y@h/db")
    resp = TestClient(app).post("/api/insert", json={"method": "batch"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["writer"] == "SaBatchWriter"
    assert body["method"] == "batch"


def test_insert_method_invalid_value_is_422(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unknown ``method`` literal is rejected at the request boundary."""
    _bypass_write_guard(monkeypatch)
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)
    resp = TestClient(app).post("/api/insert", json={"method": "turbo"})
    assert resp.status_code == 422


# ── method=copy: unsupported dialects raise typed 409 ──────────────────


def test_insert_method_copy_on_sqlite_is_409_method_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sqlite has no COPY equivalent → 409 METHOD_UNSUPPORTED envelope."""
    _bypass_write_guard(monkeypatch)
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)
    resp = TestClient(app).post("/api/insert", json={"method": "copy"})
    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert detail["code"] == "METHOD_UNSUPPORTED"
    assert detail["dialect"] == "sqlite"
    assert detail["method"] == "copy"
    assert "batch" in detail["supported"]
    # auto is always supported as the universal default
    assert "auto" in detail["supported"]
    # The supported list excludes "copy" for this dialect.
    assert "copy" not in detail["supported"]


def test_insert_method_copy_on_mssql_is_409_method_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mssql has no COPY equivalent → 409 METHOD_UNSUPPORTED."""
    _bypass_write_guard(monkeypatch)
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)
    # Override the workspace target with an MSSQL DSN; the workspace state
    # is the source of truth for dispatch (the actual rows still live in
    # last_result which was generated against sqlite — fine: we never reach
    # the writer because the guard fires first).
    app.state.workspace.set_target_url("mssql+pyodbc://x:y@h/db")
    resp = TestClient(app).post("/api/insert", json={"method": "copy"})
    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert detail["code"] == "METHOD_UNSUPPORTED"
    assert detail["dialect"] == "mssql"


def test_insert_method_copy_on_unknown_dialect_is_409(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unknown / unmapped dialect is also unsupported for COPY."""
    _bypass_write_guard(monkeypatch)
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)
    app.state.workspace.set_target_url("oracle://x:y@h/db")
    resp = TestClient(app).post("/api/insert", json={"method": "copy"})
    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert detail["code"] == "METHOD_UNSUPPORTED"
    assert detail["dialect"] == "oracle"


# ── method=copy: supported dialects → COPY writer ──────────────────────


def test_insert_method_copy_on_postgresql_uses_pg_copy_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PostgreSQL + ``method="copy"`` → PgCopyWriter (when psycopg present).

    The endpoint does NOT execute the writer in this test (job runs on a
    worker thread); we only assert the dispatch returned the right name.
    """
    _bypass_write_guard(monkeypatch)
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)
    app.state.workspace.set_target_url("postgresql://x:y@h/db")

    # Fake psycopg presence for this test only.
    import builtins  # noqa: PLC0415

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "psycopg":
            import types  # noqa: PLC0415

            return types.ModuleType("psycopg")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    resp = TestClient(app).post("/api/insert", json={"method": "copy"})
    # Submit succeeds (job will fail later on the worker thread because the
    # target is a sham — we don't await it). What matters here is the
    # dispatch returned the right writer name in the response.
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["writer"] == "PgCopyWriter"
    assert body["method"] == "copy"


def test_insert_method_copy_on_mysql_uses_load_data_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MySQL + ``method="copy"`` → MysqlLoadDataWriter (when pymysql present)."""
    _bypass_write_guard(monkeypatch)
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)
    app.state.workspace.set_target_url("mysql://x:y@h/db")

    import builtins  # noqa: PLC0415

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pymysql":
            import types  # noqa: PLC0415

            return types.ModuleType("pymysql")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    resp = TestClient(app).post("/api/insert", json={"method": "copy"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["writer"] == "MysqlLoadDataWriter"
    assert body["method"] == "copy"


def test_insert_method_copy_on_postgresql_without_psycopg_is_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``method="copy"`` on PG with psycopg missing → 409 METHOD_UNSUPPORTED.

    Rationale: the *auto* policy silently downgrades to ``SaBatchWriter``,
    but an *explicit* ``method="copy"`` request is the user saying "I want
    COPY or nothing" — silent downgrade would be a footgun.
    """
    _bypass_write_guard(monkeypatch)
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)
    app.state.workspace.set_target_url("postgresql://x:y@h/db")

    import builtins  # noqa: PLC0415

    real_import = builtins.__import__

    def fake_import_no_psycopg(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "psycopg":
            raise ImportError("no psycopg")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import_no_psycopg)
    resp = TestClient(app).post("/api/insert", json={"method": "copy"})
    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert detail["code"] == "METHOD_UNSUPPORTED"
    assert detail["dialect"] == "postgresql"
    assert detail["method"] == "copy"
    # The hint should mention the missing driver, not just "wrong dialect".
    assert "psycopg" in detail.get("hint", "").lower()


def test_insert_method_copy_on_mysql_without_pymysql_is_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``method="copy"`` on MySQL with pymysql missing → 409 METHOD_UNSUPPORTED."""
    _bypass_write_guard(monkeypatch)
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)
    app.state.workspace.set_target_url("mysql://x:y@h/db")

    import builtins  # noqa: PLC0415

    real_import = builtins.__import__

    def fake_import_no_pymysql(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pymysql":
            raise ImportError("no pymysql")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import_no_pymysql)
    resp = TestClient(app).post("/api/insert", json={"method": "copy"})
    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert detail["code"] == "METHOD_UNSUPPORTED"
    assert detail["dialect"] == "mysql"
    assert "pymysql" in detail.get("hint", "").lower()


# ── _resolve_writer helper unit tests ──────────────────────────────────


def test_resolve_writer_auto_matches_select_writer() -> None:
    """``_resolve_writer(dialect, "auto")`` is byte-equivalent to the S-136
    ``_select_writer(url)`` policy. We re-derive the URL from the dialect for
    coverage of every branch."""
    from dbsprout.web.routers.insert import (  # noqa: PLC0415
        _resolve_writer,
        _select_writer,
    )

    cases = [
        ("sqlite", "sqlite:///x.db"),
        ("mssql", "mssql+pyodbc://x:y@h/db"),
        ("oracle", "oracle://x:y@h/db"),
    ]
    for dialect, url in cases:
        auto_writer, auto_name = _resolve_writer(dialect, "auto", url=url)
        ref_writer, ref_name = _select_writer(url)
        assert auto_name == ref_name
        assert type(auto_writer).__name__ == type(ref_writer).__name__


def test_resolve_writer_batch_always_sa_batch() -> None:
    """``_resolve_writer(_, "batch", ...)`` always picks SaBatchWriter."""
    from dbsprout.web.routers.insert import _resolve_writer  # noqa: PLC0415

    for dialect in ("postgresql", "mysql", "sqlite", "mssql", "oracle"):
        _, name = _resolve_writer(dialect, "batch", url=f"{dialect}://x:y@h/db")
        assert name == "SaBatchWriter", dialect


def test_resolve_writer_copy_unsupported_dialect_raises() -> None:
    """``_resolve_writer(<unsupported>, "copy", ...)`` raises HTTPException 409."""
    from fastapi import HTTPException  # noqa: PLC0415

    from dbsprout.web.routers.insert import _resolve_writer  # noqa: PLC0415

    for dialect in ("sqlite", "mssql", "oracle"):
        with pytest.raises(HTTPException) as exc_info:
            _resolve_writer(dialect, "copy", url=f"{dialect}://x:y@h/db")
        assert exc_info.value.status_code == 409
        detail = exc_info.value.detail
        assert isinstance(detail, dict)
        assert detail["code"] == "METHOD_UNSUPPORTED"
        assert detail["dialect"] == dialect
        assert detail["method"] == "copy"


# ── errors.py: METHOD_UNSUPPORTED factory + taxonomy ───────────────────


def test_web_error_code_includes_method_unsupported() -> None:
    """The taxonomy is closed-set; S-141 adds METHOD_UNSUPPORTED."""
    from dbsprout.web.errors import WebErrorCode  # noqa: PLC0415

    assert WebErrorCode.METHOD_UNSUPPORTED.value == "METHOD_UNSUPPORTED"


def test_web_error_method_unsupported_envelope_shape() -> None:
    """The factory carries ``dialect`` + ``method`` + ``supported`` in
    ``to_dict()`` so the response body has them at the top of ``detail``."""
    from dbsprout.web.errors import web_error_method_unsupported  # noqa: PLC0415

    err = web_error_method_unsupported(
        dialect="sqlite",
        method="copy",
        supported=["auto", "batch"],
    )
    payload = err.to_dict()
    assert payload["code"] == "METHOD_UNSUPPORTED"
    assert payload["dialect"] == "sqlite"
    assert payload["method"] == "copy"
    assert payload["supported"] == ["auto", "batch"]
    assert err.status_code == 409
    # A hint is provided so the Studio surfaces something actionable.
    assert payload.get("hint")


# ── Studio template: write_guard_modal.html gains a method <select> ────


def test_write_guard_modal_template_has_method_select() -> None:
    """The Studio confirmation modal exposes a <select> for the method.

    S-141 extends ``templates/studio/write_guard_modal.html`` (S-137) with
    a three-option select wired to the Alpine state and posted as ``method``
    in the ``/api/insert`` body.
    """
    from pathlib import Path  # noqa: PLC0415

    template_path = (
        Path(__file__).resolve().parents[2]
        / "dbsprout"
        / "web"
        / "templates"
        / "studio"
        / "write_guard_modal.html"
    )
    contents = template_path.read_text(encoding="utf-8")
    # Each of the three methods is offered.
    assert 'value="auto"' in contents
    assert 'value="batch"' in contents
    assert 'value="copy"' in contents
    # The select is bound to an Alpine state field named "method".
    assert "x-model" in contents
    assert "method" in contents
    # The POST body includes the method field (any common JS-object-literal
    # form is fine here — the goal is to lock the wire contract).
    contains_post_field = "method:" in contents or "method =" in contents or '"method"' in contents
    assert contains_post_field
    # Surfaces the typed envelope so a 409 METHOD_UNSUPPORTED is rendered
    # without a console-only blow-up.
    assert "METHOD_UNSUPPORTED" in contents


# ── Studio template: Studio shell still mounts the modal ───────────────


def test_studio_shell_still_includes_write_guard_modal() -> None:
    """Sanity check: extending the modal didn't break the page include."""
    from pathlib import Path  # noqa: PLC0415

    template_path = (
        Path(__file__).resolve().parents[2] / "dbsprout" / "web" / "templates" / "studio.html"
    )
    contents = template_path.read_text(encoding="utf-8")
    assert "studio/write_guard_modal.html" in contents
