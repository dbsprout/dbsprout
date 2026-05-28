"""Write-guard preview + HMAC token validation tests (S-137).

Wave 2 of the Output & Insertion sub-epic. S-136 wired the
``/api/insert`` route and left the ``_validate_confirmation_token``
stub + a forward-handoff ``# region: write-guard (S-137)`` block.
This story lands the real HMAC-signed, single-use, scope-bound token —
issued by a new ``POST /api/insert/preview`` route — and replaces the
stub validator inside the same region. The taxonomy gains one new
closed code, ``WRITE_GUARD_REJECTED``, distinct from
``WRITE_GUARD_REQUIRED`` (missing vs. rejected).

The TestClient pattern + the ``_connect_and_generate`` helper are
borrowed from ``tests/test_web/test_insert.py`` (already exercised
under S-136).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import sqlite3
import time
from types import SimpleNamespace
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


def _decode_token_unverified(token: str) -> dict[str, object]:
    """Best-effort inspect of the token payload (tests use this to assert structure)."""
    payload_b64, _sig = token.split(".", 1)
    pad = "=" * (-len(payload_b64) % 4)
    raw = base64.urlsafe_b64decode(payload_b64 + pad)
    out: dict[str, object] = json.loads(raw.decode("utf-8"))
    return out


# ── /api/insert/preview happy path ──────────────────────────────────────


def test_preview_endpoint_is_mounted(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")
    paths = {route.path for route in app.routes if hasattr(route, "path")}
    assert "/api/insert/preview" in paths


def test_preview_endpoint_returns_redacted_target_and_scope(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")
    target_url = _connect_and_generate(app, tmp_path)
    resp = TestClient(app).post("/api/insert/preview", json={})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "target" in body
    assert "dialect" in body
    assert body["dialect"] == "sqlite"
    assert "scope" in body
    assert isinstance(body["scope"], list)
    assert "total_rows" in body
    assert "confirmation_token" in body
    assert isinstance(body["confirmation_token"], str)
    assert body["confirmation_token"]
    # FK-safe order preserved
    assert [e["table"] for e in body["scope"]] == ["users", "posts"]
    assert body["total_rows"] == sum(int(e["row_count"]) for e in body["scope"])
    # target is redacted; for sqlite there is no password — assert the body
    # at least matches the raw URL (sqlite has no creds) but never contains a
    # bare password marker like "@" / colon between user+host.
    assert body["target"] == target_url  # sqlite has no credentials to redact


def test_preview_target_password_is_redacted(tmp_path: Path) -> None:
    """For a credential-bearing URL, ``target`` must not echo the password."""
    app = _make_app(tmp_path / "state.db")
    # Use sqlite for the underlying generate, but mock the target URL to
    # carry a fake password so we can assert _redact_url ran.
    _connect_and_generate(app, tmp_path)
    secret_password = "s3cr3tP4ss"  # noqa: S105 — synthetic test value
    fake_url = f"postgresql://alice:{secret_password}@db.example/app"
    # Forcibly override the workspace target with a credential-bearing URL.
    app.state.workspace._target_url = fake_url

    resp = TestClient(app).post("/api/insert/preview", json={})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert secret_password not in body["target"]
    assert "alice" in body["target"]  # username is OK
    # Dialect now reflects the override
    assert body["dialect"] == "postgresql"


def test_preview_subset_scope(tmp_path: Path) -> None:
    """``tables=[...]`` is honoured by the preview just like /api/insert."""
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)
    resp = TestClient(app).post("/api/insert/preview", json={"tables": ["users"]})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert [e["table"] for e in body["scope"]] == ["users"]


# ── /api/insert/preview guard paths ─────────────────────────────────────


def test_preview_without_connection_is_409(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")
    resp = TestClient(app).post("/api/insert/preview", json={})
    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"]["code"] == "NO_CONNECTION"


def test_preview_without_generate_result_is_409(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")
    target_url = _temp_sqlite(tmp_path)
    TestClient(app).post("/api/connect", json={"url": target_url})
    resp = TestClient(app).post("/api/insert/preview", json={})
    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"]["code"] == "NO_RUN"


def test_preview_unknown_table_is_422(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)
    resp = TestClient(app).post("/api/insert/preview", json={"tables": ["nope"]})
    assert resp.status_code == 422, resp.text


def test_preview_extra_field_is_422(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)
    resp = TestClient(app).post("/api/insert/preview", json={"weird": 1})
    assert resp.status_code == 422


# ── token shape + single-app secret ─────────────────────────────────────


def test_token_payload_carries_only_hashes_not_raw_dsn(tmp_path: Path) -> None:
    """The token payload must NEVER carry the raw DSN or password."""
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)
    secret_password = "topsecret"  # noqa: S105
    fake_url = f"postgresql://user:{secret_password}@host/db"
    app.state.workspace._target_url = fake_url

    resp = TestClient(app).post("/api/insert/preview", json={})
    token = resp.json()["confirmation_token"]
    payload = _decode_token_unverified(token)
    blob = json.dumps(payload)
    assert secret_password not in blob
    assert fake_url not in blob
    # Required fields
    assert "target_hash" in payload
    assert "scope_hash" in payload
    assert "exp" in payload
    assert "nonce" in payload
    # hashes are hex digests
    assert isinstance(payload["target_hash"], str)
    assert len(payload["target_hash"]) == 64
    assert isinstance(payload["scope_hash"], str)
    assert len(payload["scope_hash"]) == 64


def test_two_previews_yield_distinct_tokens(tmp_path: Path) -> None:
    """Each preview gets a fresh nonce → tokens differ even for identical scope."""
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)
    client = TestClient(app)
    t1 = client.post("/api/insert/preview", json={}).json()["confirmation_token"]
    t2 = client.post("/api/insert/preview", json={}).json()["confirmation_token"]
    assert t1 != t2


def test_secret_uses_config_when_present(tmp_path: Path) -> None:
    """When ``app.state.config.web.secret_key`` exists, it drives HMAC signing."""
    from dbsprout.web.routers.insert import (  # noqa: PLC0415
        _get_write_guard_secret,
    )

    app = _make_app(tmp_path / "state.db")
    chosen = b"k" * 32
    app.state.config = SimpleNamespace(web=SimpleNamespace(secret_key=chosen))
    assert _get_write_guard_secret(app) == chosen
    # repeated call is stable
    assert _get_write_guard_secret(app) == chosen


def test_secret_is_generated_lazily_when_no_config(tmp_path: Path) -> None:
    """When no config is wired, the secret is generated once + memoised on app.state."""
    from dbsprout.web.routers.insert import (  # noqa: PLC0415
        _get_write_guard_secret,
    )

    app = _make_app(tmp_path / "state.db")
    assert not hasattr(app.state, "write_guard_secret")
    first = _get_write_guard_secret(app)
    assert isinstance(first, bytes)
    assert len(first) >= 32
    # Second call returns the same value (memoised)
    second = _get_write_guard_secret(app)
    assert first == second


# ── /api/insert with HMAC token ─────────────────────────────────────────


def test_insert_accepts_valid_preview_token(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)
    client = TestClient(app)
    token = client.post("/api/insert/preview", json={}).json()["confirmation_token"]
    resp = client.post("/api/insert", json={"confirmation_token": token})
    assert resp.status_code == 200, resp.text
    assert isinstance(resp.json()["job_id"], str)


def test_insert_rejects_bogus_token(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)
    resp = TestClient(app).post(
        "/api/insert", json={"confirmation_token": "definitely-not-a-real-token"}
    )
    assert resp.status_code == 403, resp.text
    assert resp.json()["detail"]["code"] == "WRITE_GUARD_REJECTED"


def test_insert_rejects_expired_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)
    client = TestClient(app)
    token = client.post("/api/insert/preview", json={}).json()["confirmation_token"]
    # Advance the clock past the 5-min TTL.
    from dbsprout.web.routers import insert as insert_module  # noqa: PLC0415

    real_time = time.time
    monkeypatch.setattr(
        insert_module.time,
        "time",
        lambda: real_time() + insert_module._PREVIEW_TOKEN_TTL + 10,
    )
    resp = client.post("/api/insert", json={"confirmation_token": token})
    assert resp.status_code == 403, resp.text
    assert resp.json()["detail"]["code"] == "WRITE_GUARD_REJECTED"


def test_insert_rejects_scope_mismatched_token(tmp_path: Path) -> None:
    """Preview the FULL scope → submit insert for a subset → mismatch → reject."""
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)
    client = TestClient(app)
    token = client.post("/api/insert/preview", json={}).json()["confirmation_token"]
    resp = client.post("/api/insert", json={"confirmation_token": token, "tables": ["users"]})
    assert resp.status_code == 403, resp.text
    assert resp.json()["detail"]["code"] == "WRITE_GUARD_REJECTED"


@pytest.mark.anyio
async def test_insert_rejects_reused_token(tmp_path: Path) -> None:
    """A token is single-use — re-submitting it after the first insert is rejected."""
    import httpx  # noqa: PLC0415
    from httpx import ASGITransport  # noqa: PLC0415

    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        token = (await client.post("/api/insert/preview", json={})).json()["confirmation_token"]
        r1 = await client.post("/api/insert", json={"confirmation_token": token})
        assert r1.status_code == 200, r1.text
        job_id = r1.json()["job_id"]
        # Wait for the active job to clear so we exercise the single-use
        # path (not the single-active 409 path).
        await app.state.job_manager.wait(job_id)
        r2 = await client.post("/api/insert", json={"confirmation_token": token})
        assert r2.status_code == 403, r2.text
        assert r2.json()["detail"]["code"] == "WRITE_GUARD_REJECTED"


def test_disable_env_var_does_not_short_circuit_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``DBSPROUT_DISABLE_WRITE_GUARD`` short-circuits only the *missing-token*
    path. A present-but-invalid token must still be rejected."""
    monkeypatch.setenv("DBSPROUT_DISABLE_WRITE_GUARD", "1")
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)
    resp = TestClient(app).post("/api/insert", json={"confirmation_token": "tampered"})
    assert resp.status_code == 403, resp.text
    assert resp.json()["detail"]["code"] == "WRITE_GUARD_REJECTED"


# ── direct helper unit tests for coverage ───────────────────────────────


def test_hash_target_is_stable_and_hex() -> None:
    from dbsprout.web.routers.insert import _hash_target  # noqa: PLC0415

    a = _hash_target("sqlite:///x.db")
    b = _hash_target("sqlite:///x.db")
    assert a == b
    assert len(a) == 64
    assert int(a, 16) >= 0  # parses as hex


def test_hash_scope_is_order_stable() -> None:
    """``_hash_scope`` normalises ordering to keep tokens scope-stable."""
    from dbsprout.web.routers.insert import _hash_scope  # noqa: PLC0415

    # Same logical scope (table, row_count) presented in different orders
    a = _hash_scope([("users", 3), ("posts", 7)])
    b = _hash_scope([("posts", 7), ("users", 3)])
    assert a == b
    # Different row counts → different hash
    c = _hash_scope([("users", 3), ("posts", 8)])
    assert c != a


def test_token_round_trip_via_helpers(tmp_path: Path) -> None:
    """Encode + decode via the module's helpers (direct unit test for coverage)."""
    from dbsprout.web.routers.insert import (  # noqa: PLC0415
        _decode_token,
        _encode_token,
    )

    secret = b"s" * 32
    payload = {
        "target_hash": "a" * 64,
        "scope_hash": "b" * 64,
        "exp": int(time.time()) + 60,
        "nonce": "nonce-1",
    }
    token = _encode_token(payload, secret)
    out = _decode_token(token, secret)
    assert out == payload
    # Tampered token → None
    assert _decode_token(token + "x", secret) is None
    # Wrong secret → None
    assert _decode_token(token, b"x" * 32) is None
    # Structurally bogus → None
    assert _decode_token("not.a.token", secret) is None
    assert _decode_token("nodotjustgarbage", secret) is None


def test_decode_token_handles_invalid_json_payload() -> None:
    """A signature-valid token whose payload is not JSON returns ``None``."""
    from dbsprout.web.routers.insert import _decode_token  # noqa: PLC0415

    secret = b"s" * 32
    junk = b"not-json-at-all"
    payload_b64 = base64.urlsafe_b64encode(junk).rstrip(b"=").decode("ascii")
    sig = hmac.new(secret, junk, hashlib.sha256).digest()
    sig_b64 = base64.urlsafe_b64encode(sig).rstrip(b"=").decode("ascii")
    token = f"{payload_b64}.{sig_b64}"
    assert _decode_token(token, secret) is None


def test_validate_real_round_trip(tmp_path: Path) -> None:
    """The real validator accepts a freshly-issued token for the matching scope/target."""
    from dbsprout.web.routers.insert import (  # noqa: PLC0415
        _encode_token,
        _get_write_guard_secret,
        _hash_scope,
        _hash_target,
        _validate_confirmation_token,
    )

    app = _make_app(tmp_path / "state.db")
    secret = _get_write_guard_secret(app)
    target_url = "sqlite:///t.db"
    scope_pairs = [("users", 2), ("posts", 3)]
    payload = {
        "target_hash": _hash_target(target_url),
        "scope_hash": _hash_scope(scope_pairs),
        "exp": int(time.time()) + 60,
        "nonce": "live-nonce",
    }
    token = _encode_token(payload, secret)
    # Register the nonce in the issued-set so single-use enforcement passes.
    from dbsprout.web.routers.insert import _get_issued_registry  # noqa: PLC0415

    _get_issued_registry(app)["live-nonce"] = payload["exp"]
    ok = _validate_confirmation_token(
        token,
        scope=[t for t, _ in scope_pairs],
        target_url=target_url,
        app=app,
        row_counts=dict(scope_pairs),
    )
    assert ok is True
    # Single-use: a second validation with the same nonce fails.
    ok2 = _validate_confirmation_token(
        token,
        scope=[t for t, _ in scope_pairs],
        target_url=target_url,
        app=app,
        row_counts=dict(scope_pairs),
    )
    assert ok2 is False


def test_modal_template_renders(tmp_path: Path) -> None:
    """The new modal template loads + renders without error."""
    app = _make_app(tmp_path / "state.db")
    templates = app.state.templates
    # The template must exist and render with the expected mount point.
    rendered = templates.get_template("studio/write_guard_modal.html").render()
    assert "write-guard" in rendered.lower()
    assert "confirm" in rendered.lower()


def test_studio_page_includes_modal(tmp_path: Path) -> None:
    """The Studio page must mount the modal once at page level (mirrors method_picker)."""
    app = _make_app(tmp_path / "state.db")
    client = TestClient(app)
    resp = client.get("/studio")
    assert resp.status_code == 200, resp.text
    # The mount itself is what we lock — the modal element id will live in the template.
    assert "write-guard-modal" in resp.text


def test_write_guard_issued_registry_garbage_collects_expired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Expired nonces are purged lazily on the next preview call."""
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)
    # Pre-seed an expired nonce
    app.state.write_guard_issued = {"old": int(time.time()) - 1000}
    TestClient(app).post("/api/insert/preview", json={})
    assert "old" not in app.state.write_guard_issued


def test_secret_uses_string_config_value(tmp_path: Path) -> None:
    """A ``str`` ``secret_key`` is accepted (encoded to bytes on the fly)."""
    from dbsprout.web.routers.insert import (  # noqa: PLC0415
        _get_write_guard_secret,
    )

    app = _make_app(tmp_path / "state.db")
    app.state.config = SimpleNamespace(
        web=SimpleNamespace(secret_key="text-secret")  # noqa: S106 — synthetic test value
    )
    assert _get_write_guard_secret(app) == b"text-secret"


def test_token_with_non_dict_payload_decodes_to_none() -> None:
    """A signature-valid token whose payload decodes to a non-dict → None."""
    from dbsprout.web.routers.insert import _decode_token  # noqa: PLC0415

    secret = b"s" * 32
    payload = json.dumps([1, 2, 3]).encode("utf-8")  # JSON list, not dict
    sig = hmac.new(secret, payload, hashlib.sha256).digest()
    token = (
        base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")
        + "."
        + base64.urlsafe_b64encode(sig).rstrip(b"=").decode("ascii")
    )
    assert _decode_token(token, secret) is None


def test_validate_rejects_token_with_non_int_exp(tmp_path: Path) -> None:
    """Payload has structurally-typed fields but ``exp`` is a string → rejected."""
    from dbsprout.web.routers.insert import (  # noqa: PLC0415
        _encode_token,
        _get_issued_registry,
        _get_write_guard_secret,
        _hash_scope,
        _hash_target,
        _validate_confirmation_token,
    )

    app = _make_app(tmp_path / "state.db")
    secret = _get_write_guard_secret(app)
    payload = {
        "target_hash": _hash_target("sqlite:///x"),
        "scope_hash": _hash_scope([("users", 1)]),
        "exp": "not-an-int",  # invalid type
        "nonce": "n1",
    }
    token = _encode_token(payload, secret)
    _get_issued_registry(app)["n1"] = int(time.time()) + 60
    ok = _validate_confirmation_token(
        token,
        scope=["users"],
        target_url="sqlite:///x",
        app=app,
        row_counts={"users": 1},
    )
    assert ok is False


def test_validate_rejects_token_with_non_string_nonce(tmp_path: Path) -> None:
    """Payload with a non-string nonce → rejected (isinstance narrow check)."""
    from dbsprout.web.routers.insert import (  # noqa: PLC0415
        _encode_token,
        _get_issued_registry,
        _get_write_guard_secret,
        _hash_scope,
        _hash_target,
        _validate_confirmation_token,
    )

    app = _make_app(tmp_path / "state.db")
    secret = _get_write_guard_secret(app)
    payload = {
        "target_hash": _hash_target("sqlite:///x"),
        "scope_hash": _hash_scope([("users", 1)]),
        "exp": int(time.time()) + 60,
        "nonce": 12345,  # non-string nonce → narrow check fails
    }
    token = _encode_token(payload, secret)
    _get_issued_registry(app)["12345"] = payload["exp"]
    ok = _validate_confirmation_token(
        token,
        scope=["users"],
        target_url="sqlite:///x",
        app=app,
        row_counts={"users": 1},
    )
    assert ok is False


def test_validate_rejects_when_target_hash_mismatches(tmp_path: Path) -> None:
    """Payload signed for ``target-A`` validated against ``target-B`` → rejected."""
    from dbsprout.web.routers.insert import (  # noqa: PLC0415
        _encode_token,
        _get_issued_registry,
        _get_write_guard_secret,
        _hash_scope,
        _hash_target,
        _validate_confirmation_token,
    )

    app = _make_app(tmp_path / "state.db")
    secret = _get_write_guard_secret(app)
    payload = {
        "target_hash": _hash_target("sqlite:///A"),
        "scope_hash": _hash_scope([("users", 1)]),
        "exp": int(time.time()) + 60,
        "nonce": "mismatch-nonce",
    }
    token = _encode_token(payload, secret)
    _get_issued_registry(app)["mismatch-nonce"] = payload["exp"]
    ok = _validate_confirmation_token(
        token,
        scope=["users"],
        target_url="sqlite:///B",  # different target
        app=app,
        row_counts={"users": 1},
    )
    assert ok is False


def test_validate_rejects_missing_required_field(tmp_path: Path) -> None:
    """Payload missing the ``scope_hash`` field → rejected (all(isinstance) fails)."""
    from dbsprout.web.routers.insert import (  # noqa: PLC0415
        _encode_token,
        _get_issued_registry,
        _get_write_guard_secret,
        _hash_target,
        _validate_confirmation_token,
    )

    app = _make_app(tmp_path / "state.db")
    secret = _get_write_guard_secret(app)
    payload = {
        "target_hash": _hash_target("sqlite:///x"),
        # scope_hash intentionally missing → value is None → all(isinstance) fails
        "exp": int(time.time()) + 60,
        "nonce": "n2",
    }
    token = _encode_token(payload, secret)
    _get_issued_registry(app)["n2"] = payload["exp"]
    ok = _validate_confirmation_token(
        token,
        scope=["users"],
        target_url="sqlite:///x",
        app=app,
        row_counts={"users": 1},
    )
    assert ok is False


def test_token_with_wrong_app_secret_is_rejected(tmp_path: Path) -> None:
    """A token signed with secret-A is rejected when validated against secret-B."""
    from dbsprout.web.routers.insert import (  # noqa: PLC0415
        _encode_token,
        _hash_scope,
        _hash_target,
    )

    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)
    target_url = app.state.workspace.peek_target_url()
    assert target_url is not None
    bad_secret = b"x" * 32
    payload = {
        "target_hash": _hash_target(target_url),
        "scope_hash": _hash_scope([("users", 1), ("posts", 1)]),
        "exp": int(time.time()) + 60,
        "nonce": "n-bad",
    }
    forged = _encode_token(payload, bad_secret)
    # Manually register it (so single-use isn't what causes the failure)
    from dbsprout.web.routers.insert import _get_issued_registry  # noqa: PLC0415

    _get_issued_registry(app)["n-bad"] = payload["exp"]
    resp = TestClient(app).post("/api/insert", json={"confirmation_token": forged})
    assert resp.status_code == 403, resp.text
    assert resp.json()["detail"]["code"] == "WRITE_GUARD_REJECTED"
