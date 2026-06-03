"""End-to-end AC checks for the friendly web-error layer (S-116).

These tests drive a real ``TestClient`` through every classified failure path
and assert the five acceptance-criteria invariants:

1. ``POST /api/connect`` and ``POST /api/schema/load`` always map known driver /
   parser exceptions to a typed envelope ``{code, message, hint?, correlation_id}``.
2. HTTP responses use ``4xx`` for caller-actionable errors and ``5xx`` only for
   genuine server faults; bodies never contain ``Traceback`` or ``repr(exc)``.
3. Responses are JSON-only since the P1c-5 cutover — even with ``HX-Request:
   true`` the body is the JSON envelope (the legacy HTML-fragment branch is gone).
4. The generic catch-all returns ``{"code": "INTERNAL", "message": "Unexpected
   error"}`` with the real exception logged server-side and a correlation id
   surfaced to the user.
5. No password / DSN secret appears in any user-facing body.

Router-level unit tests live in ``test_connect.py`` / ``test_schema_load.py``;
classifier behaviour lives in ``test_errors.py``. This file is the *integration*
seam that the AC table maps onto directly.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")

from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from pathlib import Path

_PASSWORD = "supersecret-not-leaked"  # noqa: S105 — synthetic credential used to assert redaction
_TRACEBACK_RE = re.compile(r'Traceback|File "[^"]+", line \d+')


def _client(tmp_path: Path) -> TestClient:
    from dbsprout.web.app import create_app  # noqa: PLC0415

    return TestClient(create_app(state_db_path=tmp_path / "state.db"))


# ── invariant 1: every failure body has the typed envelope ──────────────


@pytest.mark.parametrize(
    ("path", "kwargs"),
    [
        pytest.param(
            "/api/connect",
            {"json": {"url": "redis://localhost:6379/0"}},
            id="connect-unknown-dialect",
        ),
        pytest.param(
            "/api/connect",
            {"json": {"url": f"postgresql://alice:{_PASSWORD}@unreachable.invalid:5432/app"}},
            id="connect-unreachable",
        ),
        pytest.param(
            "/api/schema/load",
            {"files": {"file": ("x.sql", b"", "application/octet-stream")}},
            id="schema-load-empty",
        ),
        pytest.param(
            "/api/schema/load",
            {
                "files": {"file": ("x.sql", b"x" * (6 * 1024 * 1024), "application/octet-stream")},
            },
            id="schema-load-oversize",
        ),
        pytest.param(
            "/api/schema/load",
            {
                "files": {"file": ("x.sql", b"garbage", "application/octet-stream")},
                "data": {"parser": "bogus"},
            },
            id="schema-load-unknown-parser",
        ),
        pytest.param(
            "/api/schema/load",
            {
                "files": {
                    "file": (
                        "x.prisma",
                        b"this is not prisma at all",
                        "application/octet-stream",
                    )
                },
                "data": {"parser": "prisma"},
            },
            id="schema-load-parse-error",
        ),
    ],
)
def test_every_error_body_carries_typed_envelope(
    tmp_path: Path,
    path: str,
    kwargs: dict[str, object],
) -> None:
    client = _client(tmp_path)
    resp = client.post(path, **kwargs)  # type: ignore[arg-type]
    assert 400 <= resp.status_code < 600
    body = resp.json()
    assert "detail" in body
    detail = body["detail"]
    assert isinstance(detail, dict)
    assert detail["code"]
    assert detail["message"]
    assert detail["correlation_id"]
    # Invariant 2: no traceback in any body.
    assert not _TRACEBACK_RE.search(resp.text), resp.text[:500]


# ── invariant 5: password never leaks ──────────────────────────────────


def test_password_never_appears_in_any_failure_body(tmp_path: Path) -> None:
    client = _client(tmp_path)
    resp = client.post(
        "/api/connect",
        json={"url": f"postgresql://alice:{_PASSWORD}@unreachable.invalid:5432/db"},
    )
    assert resp.status_code == 400
    assert _PASSWORD not in resp.text
    # HTMX path: same invariant.
    resp_htmx = client.post(
        "/api/connect",
        json={"url": f"postgresql://alice:{_PASSWORD}@unreachable.invalid:5432/db"},
        headers={"HX-Request": "true"},
    )
    assert _PASSWORD not in resp_htmx.text
    assert not _TRACEBACK_RE.search(resp_htmx.text)


# ── invariant 3: responses are JSON-only, even under HX-Request (P1c-5) ─


def test_htmx_header_still_returns_json_envelope_for_both_routes(tmp_path: Path) -> None:
    """Since the P1c-5 cutover the ``HX-Request`` HTML-fragment branch is gone.

    The error body is the JSON ``{"detail": {...}}`` envelope regardless of the
    ``HX-Request`` header — the React SPA reads JSON, never an HTML fragment.
    """
    client = _client(tmp_path)
    resp_connect = client.post(
        "/api/connect",
        json={"url": "redis://localhost:6379/0"},
        headers={"HX-Request": "true"},
    )
    assert resp_connect.headers["content-type"].startswith("application/json")
    assert resp_connect.json()["detail"]["code"] == "UNKNOWN_DIALECT"

    resp_load = client.post(
        "/api/schema/load",
        files={"file": ("x.sql", b"", "application/octet-stream")},
        headers={"HX-Request": "true"},
    )
    assert resp_load.headers["content-type"].startswith("application/json")
    assert resp_load.json()["detail"]["code"] == "EMPTY_FILE"


# ── invariant 4: INTERNAL fallback is 500 + real exception logged ──────


def test_internal_catch_all_logs_real_exception_but_hides_it_from_body(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unclassified exception → 500 INTERNAL, traceback only in server log."""
    import dbsprout.schema.parsers as parsers_mod  # noqa: PLC0415

    leak_marker = "should-not-leak-to-the-client"

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError(leak_marker)

    monkeypatch.setattr(parsers_mod, "parse_schema_file", _boom)
    client = _client(tmp_path)

    with caplog.at_level(logging.ERROR, logger="dbsprout.web.errors"):
        resp = client.post(
            "/api/schema/load",
            files={"file": ("x.sql", b"CREATE TABLE x (id INTEGER);", "application/octet-stream")},
        )
    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert detail["code"] == "INTERNAL"
    assert detail["message"] == "Unexpected error"
    assert leak_marker not in resp.text
    assert not _TRACEBACK_RE.search(resp.text)

    # Real exception is captured in the server log via ``exc_info``.
    matching = [r for r in caplog.records if r.name == "dbsprout.web.errors"]
    assert matching, "expected at least one log record"
    last = matching[-1]
    assert last.levelno >= logging.ERROR
    # ``exc_info`` carries the original RuntimeError (not the user-facing message).
    assert last.exc_info is not None
    assert last.exc_info[0] is RuntimeError
    assert leak_marker in str(last.exc_info[1])


# ── invariant 1 bis: correlation id on every error body ────────────────


def test_correlation_id_is_unique_per_response(tmp_path: Path) -> None:
    client = _client(tmp_path)
    a = client.post("/api/connect", json={"url": "redis://localhost/"})
    b = client.post("/api/connect", json={"url": "redis://localhost/"})
    assert a.json()["detail"]["correlation_id"] != b.json()["detail"]["correlation_id"]
