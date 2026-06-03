"""P2a-2: saved connections — core helpers + ``/api/connections`` router.

Two layers under test:

* **Core** (:mod:`dbsprout.core.connections`) — the pure TOML read/write helpers,
  password stripping, and ``${ENV_VAR}`` resolution. No FastAPI import; these are
  the security-critical primitives (passwords must NEVER be written).
* **Router** (:mod:`dbsprout.web.routers.connections`) — ``GET``/``POST``/``DELETE``
  ``/api/connections`` over a temp ``connections.toml`` selected via the
  ``DBSPROUT_CONNECTIONS_PATH`` env override.

The single load-bearing invariant, asserted from both layers, is that a literal
password is present in **neither** the persisted TOML bytes **nor** any API JSON
body — only an empty password or a preserved ``${ENV_VAR}`` reference survives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dbsprout.core.connections import (
    _strip_password,
    connections_path,
    delete_connection,
    load_connections,
    resolve_connection_url,
    save_connection,
)

if TYPE_CHECKING:
    from pathlib import Path

# ─────────────────────────── core unit tests ───────────────────────────


def test_strip_password_removes_literal_password() -> None:
    stripped = _strip_password("postgresql://user:s3cret@db.example.com:5432/app")
    assert "s3cret" not in stripped
    assert stripped.startswith("postgresql://")
    assert "user" in stripped
    assert "db.example.com" in stripped
    assert "5432" in stripped
    assert "/app" in stripped


def test_strip_password_preserves_env_ref() -> None:
    url = "postgresql://user:${PGPASSWORD}@db:5432/app"
    assert "${PGPASSWORD}" in _strip_password(url)


def test_strip_password_no_password_is_unchanged() -> None:
    url = "postgresql://user@db:5432/app"
    assert "@db" in _strip_password(url)
    assert _strip_password("sqlite:////tmp/x.db") == "sqlite:////tmp/x.db"


def test_strip_password_unparsable_url_fallback() -> None:
    # A garbage string SQLAlchemy can't parse → the stdlib fallback must still
    # never leak anything that looks like a password and must not raise.
    assert isinstance(_strip_password("not a url at all"), str)


def test_resolve_connection_url_substitutes_env() -> None:
    resolved = resolve_connection_url(
        "postgresql://user:${PGPASSWORD}@db:5432/app",
        environ={"PGPASSWORD": "live-secret"},
    )
    assert resolved == "postgresql://user:live-secret@db:5432/app"


def test_resolve_connection_url_missing_env_raises() -> None:
    with pytest.raises(ValueError, match="PGPASSWORD"):
        resolve_connection_url("postgresql://user:${PGPASSWORD}@db/app", environ={})


def test_resolve_connection_url_no_ref_is_passthrough() -> None:
    url = "postgresql://user@db:5432/app"
    assert resolve_connection_url(url, environ={}) == url


def test_save_load_roundtrip(tmp_path: Path) -> None:
    path = connections_path(tmp_path)
    saved = save_connection(path, "prod", "postgresql://user:secret@db:5432/app")
    assert saved.name == "prod"
    assert "secret" not in saved.url

    loaded = load_connections(path)
    assert [c.name for c in loaded] == ["prod"]
    assert "secret" not in loaded[0].url


def test_save_strips_password_from_toml_bytes(tmp_path: Path) -> None:
    path = connections_path(tmp_path)
    save_connection(path, "prod", "postgresql://user:TOPSECRET@db:5432/app")
    assert b"TOPSECRET" not in path.read_bytes()


def test_save_preserves_env_ref_in_toml(tmp_path: Path) -> None:
    path = connections_path(tmp_path)
    save_connection(path, "prod", "postgresql://user:${PGPASSWORD}@db:5432/app")
    assert b"${PGPASSWORD}" in path.read_bytes()
    assert "${PGPASSWORD}" in load_connections(path)[0].url


def test_save_upserts_by_name(tmp_path: Path) -> None:
    path = connections_path(tmp_path)
    save_connection(path, "prod", "postgresql://a@db/app")
    save_connection(path, "prod", "mysql://b@db2/app2")
    loaded = load_connections(path)
    assert len(loaded) == 1
    assert loaded[0].url.startswith("mysql://")


def test_load_missing_file_is_empty(tmp_path: Path) -> None:
    assert load_connections(connections_path(tmp_path)) == []


def test_load_malformed_toml_is_tolerant(tmp_path: Path) -> None:
    path = connections_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("this is = not [valid toml", encoding="utf-8")
    assert load_connections(path) == []


def test_delete_connection(tmp_path: Path) -> None:
    path = connections_path(tmp_path)
    save_connection(path, "prod", "postgresql://user@db/app")
    save_connection(path, "stage", "postgresql://user@db2/app")
    assert delete_connection(path, "prod") is True
    assert [c.name for c in load_connections(path)] == ["stage"]
    assert delete_connection(path, "absent") is False


# ─────────────────────────── router integration tests ───────────────────────────

pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")

from fastapi.testclient import TestClient  # noqa: E402


def _client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from dbsprout.web.app import create_app  # noqa: PLC0415

    monkeypatch.setenv(
        "DBSPROUT_CONNECTIONS_PATH", str(tmp_path / ".dbsprout" / "connections.toml")
    )
    return TestClient(create_app(state_db_path=tmp_path / "state.db"))


def test_list_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    resp = _client(tmp_path, monkeypatch).get("/api/connections")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"connections": []}


def test_save_then_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path, monkeypatch)
    resp = client.post(
        "/api/connections",
        json={"name": "prod", "url": "postgresql://user:secret@db:5432/app"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["name"] == "prod"
    assert "secret" not in body["url"]

    listed = client.get("/api/connections").json()["connections"]
    assert [c["name"] for c in listed] == ["prod"]
    assert "secret" not in listed[0]["url"]


def test_password_never_in_toml_or_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    leak_marker = "HUNTER2-LITERAL"  # a literal probe string, not a real credential
    save = client.post(
        "/api/connections",
        json={"name": "prod", "url": f"postgresql://u:{leak_marker}@db:5432/app"},
    )
    assert leak_marker not in save.text
    assert leak_marker not in client.get("/api/connections").text

    toml_path = tmp_path / ".dbsprout" / "connections.toml"
    assert leak_marker not in toml_path.read_bytes().decode("utf-8")


def test_env_ref_preserved_through_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path, monkeypatch)
    client.post(
        "/api/connections",
        json={"name": "prod", "url": "postgresql://u:${PGPASSWORD}@db/app"},
    )
    listed = client.get("/api/connections").json()["connections"]
    assert "${PGPASSWORD}" in listed[0]["url"]


def test_delete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path, monkeypatch)
    client.post("/api/connections", json={"name": "prod", "url": "postgresql://u@db/app"})
    resp = client.delete("/api/connections/prod")
    assert resp.status_code == 200, resp.text
    assert resp.json()["deleted"] is True
    assert client.get("/api/connections").json() == {"connections": []}


def test_delete_absent_404(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    resp = _client(tmp_path, monkeypatch).delete("/api/connections/nope")
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"]  # typed envelope


def test_save_blank_name_422(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    resp = _client(tmp_path, monkeypatch).post(
        "/api/connections", json={"name": "  ", "url": "postgresql://u@db/app"}
    )
    assert resp.status_code == 422


def test_save_blank_url_422(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    resp = _client(tmp_path, monkeypatch).post("/api/connections", json={"name": "prod", "url": ""})
    assert resp.status_code == 422
