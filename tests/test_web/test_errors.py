"""Unit tests for :mod:`dbsprout.web.errors` (S-116).

These tests lock the *shape* of the friendly web-error layer: the closed enum of
error codes, the :class:`WebError` data class, the per-failure classifier
mappings (connect + parse), and the HTMX-aware response renderer. Router-level
wiring lives in ``test_connect.py`` / ``test_schema_load.py`` / ``test_errors_ac.py``.
"""

from __future__ import annotations

import logging
import re

import pytest

pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")

import sqlalchemy as sa
from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import HTMLResponse

from dbsprout.web.app import create_app
from dbsprout.web.errors import (
    WebError,
    WebErrorCode,
    _extract_password,
    _redact,
    classify_connect_error,
    classify_parse_error,
    raise_web_error,
    web_error_empty_file,
    web_error_export_dependency_missing,
    web_error_export_multi_table_unsupported,
    web_error_file_too_large,
    web_error_internal,
    web_error_not_found_tables,
    web_error_unknown_parser,
)

_TRACEBACK_RE = re.compile(r'Traceback|File "[^"]+", line \d+')


def test_web_error_code_enum_is_closed_set() -> None:
    """The enum is a closed taxonomy; adding a member needs a story update."""
    expected = {
        "CONN_REFUSED",
        "AUTH_FAILED",
        "UNKNOWN_DIALECT",
        "MISSING_DRIVER",
        "MALFORMED_URL",
        "PARSE_ERROR",
        "EMPTY_FILE",
        "FILE_TOO_LARGE",
        "UNKNOWN_PARSER",
        "INTERNAL",
        # S-136 insert-route guards.
        "NO_CONNECTION",
        "NO_RUN",
        "WRITE_GUARD_REQUIRED",
        # S-137 write-guard HMAC token validation.
        "WRITE_GUARD_REJECTED",
        # S-131 regenerate-route guards.
        "NO_SCHEMA",
        "NO_SPEC",
        "CONSTRAINT_VIOLATION",
        "NOT_FOUND",
        # S-141 insert-method-select guard.
        "METHOD_UNSUPPORTED",
        # S-140 export-route guards.
        "EXPORT_MULTI_TABLE_UNSUPPORTED",
        "EXPORT_DEPENDENCY_MISSING",
        # S-139 update-column-route guard.
        "NO_REGEN",
    }
    assert {m.name for m in WebErrorCode} == expected
    # Each code is a plain string so it round-trips through JSON unchanged.
    for member in WebErrorCode:
        assert isinstance(member.value, str)
        assert member.value == member.name


def test_web_error_to_dict_omits_unset_hint() -> None:
    """``to_dict()`` carries code, message, correlation_id; ``hint`` only when set."""
    err = WebError(code=WebErrorCode.CONN_REFUSED, message="x", status_code=400)
    payload = err.to_dict()
    assert payload["code"] == "CONN_REFUSED"
    assert payload["message"] == "x"
    assert "correlation_id" in payload
    assert len(payload["correlation_id"]) >= 8
    assert "hint" not in payload


def test_web_error_to_dict_includes_hint_when_set() -> None:
    err = WebError(code=WebErrorCode.AUTH_FAILED, message="x", status_code=400, hint="try again")
    payload = err.to_dict()
    assert payload["hint"] == "try again"


def test_web_error_correlation_id_is_unique_per_instance() -> None:
    a = WebError(code=WebErrorCode.INTERNAL, message="x", status_code=500)
    b = WebError(code=WebErrorCode.INTERNAL, message="x", status_code=500)
    assert a.correlation_id != b.correlation_id


# ---------------------------------------------------------------------------
# classify_connect_error — locks the exception-type → code → status mapping.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("factory", "expected_code", "expected_status", "url"),
    [
        pytest.param(
            lambda: sa.exc.OperationalError(
                "stmt", {}, ConnectionRefusedError("connection refused")
            ),
            "CONN_REFUSED",
            400,
            "postgresql://u:p@localhost:5432/x",
            id="connection-refused",
        ),
        pytest.param(
            lambda: sa.exc.OperationalError(
                "stmt", {}, Exception("password authentication failed for user 'alice'")
            ),
            "AUTH_FAILED",
            400,
            "postgresql://alice:secret@localhost:5432/x",
            id="auth-failed",
        ),
        pytest.param(
            lambda: sa.exc.NoSuchModuleError("plugin foo not loaded"),
            "UNKNOWN_DIALECT",
            400,
            "foo://u:p@h/db",
            id="unknown-dialect",
        ),
        pytest.param(
            lambda: sa.exc.ArgumentError("Could not parse rfc1738 URL from string"),
            "MALFORMED_URL",
            400,
            "::not-a-url::",
            id="malformed-url",
        ),
        pytest.param(
            lambda: ImportError("No module named 'psycopg'"),
            "MISSING_DRIVER",
            400,
            "postgresql+psycopg://u:p@h/db",
            id="missing-driver",
        ),
        pytest.param(
            lambda: ValueError("anything generic"),
            "CONN_REFUSED",
            400,
            "postgresql://u:p@h/db",
            id="value-error-fallback",
        ),
        pytest.param(
            lambda: RuntimeError("boom"),
            "INTERNAL",
            500,
            "postgresql://u:p@h/db",
            id="unexpected-runtime-error",
        ),
    ],
)
def test_classify_connect_error_maps_exception_to_code(
    factory: object,
    expected_code: str,
    expected_status: int,
    url: str,
) -> None:
    err = classify_connect_error(factory(), url)  # type: ignore[operator]
    assert err.code.value == expected_code
    assert err.status_code == expected_status
    # Never include the raw password in the redacted message.
    assert "secret" not in err.message


def test_classify_connect_error_redacts_url_in_message() -> None:
    url = "postgresql://alice:supersecret@db.example.com:5432/app"
    err = classify_connect_error(
        sa.exc.OperationalError("stmt", {}, Exception("connection refused")), url
    )
    assert "supersecret" not in err.message
    # And we don't leak ``repr(exc)`` shape (no "OperationalError(" prefix).
    assert "OperationalError(" not in err.message


def test_classify_connect_error_missing_driver_hint_has_pip_install() -> None:
    err = classify_connect_error(
        ImportError("No module named 'psycopg'"), "postgresql+psycopg://u:p@h/db"
    )
    assert err.hint is not None
    assert "pip install" in err.hint


def test_classify_connect_error_redact_handles_empty_url() -> None:
    """The redact helper returns ``""`` for an empty / None URL — exercise that branch."""
    assert _redact("") == ""
    assert _redact(None) == ""
    assert _extract_password("") is None
    assert _extract_password(None) is None


def test_classify_connect_error_sqlalchemy_name_fallback() -> None:
    """A SQLAlchemy-named exception that didn't pattern-match still maps to CONN_REFUSED."""

    # Custom exception whose name matches the "*Error" + Operational/Interface/... heuristic
    # but whose message is intentionally generic so neither the type-name dispatch nor the
    # message-pattern sweep wins. This forces the name-based fallback branch.
    class OperationalSomethingError(Exception):
        pass

    err = classify_connect_error(
        OperationalSomethingError("nothing specific"),
        "postgresql://u:p@h/db",
    )
    assert err.code is WebErrorCode.CONN_REFUSED
    assert err.status_code == 400


# ---------------------------------------------------------------------------
# classify_parse_error + the small factory helpers (empty / too-large / unknown).
# ---------------------------------------------------------------------------


def test_classify_parse_error_value_error_is_parse_error() -> None:
    err = classify_parse_error(ValueError("bad dbml at line 3"), "schema.dbml")
    assert err.code.value == "PARSE_ERROR"
    assert err.status_code == 400
    assert "line 3" in err.message
    # Filename basename only — no absolute paths.
    assert "/" not in err.message or "schema.dbml" in err.message


def test_classify_parse_error_os_error_is_parse_error() -> None:
    err = classify_parse_error(OSError("disk full"), "x.sql")
    assert err.code.value == "PARSE_ERROR"
    assert err.status_code == 400


def test_classify_parse_error_unknown_exception_is_internal() -> None:
    err = classify_parse_error(RuntimeError("???"), "x")
    assert err.code.value == "INTERNAL"
    assert err.status_code == 500


def test_web_error_empty_file_factory() -> None:
    err = web_error_empty_file("schema.sql")
    assert err.code.value == "EMPTY_FILE"
    assert err.status_code == 400
    assert "schema.sql" in err.message


def test_web_error_file_too_large_factory() -> None:
    err = web_error_file_too_large(5 * 1024 * 1024)
    assert err.code.value == "FILE_TOO_LARGE"
    assert err.status_code == 413
    assert "5 MB" in err.message


def test_web_error_unknown_parser_factory() -> None:
    err = web_error_unknown_parser("xml", ["sql", "dbml", "mermaid"])
    assert err.code.value == "UNKNOWN_PARSER"
    assert err.status_code == 400
    assert "xml" in err.message
    # Allowed list rendered for the user.
    assert "sql" in err.message
    assert "dbml" in err.message


def test_web_error_internal_factory() -> None:
    err = web_error_internal()
    assert err.code.value == "INTERNAL"
    assert err.status_code == 500
    # Hint is intentionally omitted on INTERNAL so we don't speculate.
    assert err.hint is None
    assert err.message == "Unexpected error"


# ---------------------------------------------------------------------------
# raise_web_error / response renderer — HTMX-aware.
# ---------------------------------------------------------------------------


def _bare_scope(*, hx: bool = False) -> dict[str, object]:
    return {
        "type": "http",
        "method": "POST",
        "path": "/api/connect",
        "headers": [(b"hx-request", b"true")] if hx else [],
        "query_string": b"",
        "root_path": "",
        "scheme": "http",
        "server": ("testserver", 80),
        "client": ("testclient", 50000),
    }


def test_raise_web_error_returns_json_envelope_without_htmx_header(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Without HX-Request, the handler raises ``HTTPException`` w/ the envelope."""
    err = WebError(code=WebErrorCode.AUTH_FAILED, message="bad creds", status_code=400)
    request = Request(_bare_scope())
    with (
        caplog.at_level(logging.WARNING, logger="dbsprout.web.errors"),
        pytest.raises(HTTPException) as excinfo,
    ):
        raise_web_error(request, err)
    assert excinfo.value.status_code == 400
    detail = excinfo.value.detail
    assert isinstance(detail, dict)
    assert detail["code"] == "AUTH_FAILED"
    assert detail["correlation_id"] == err.correlation_id
    # Logged exactly once at WARNING for a non-INTERNAL code.
    matching = [r for r in caplog.records if r.name == "dbsprout.web.errors"]
    assert len(matching) == 1
    assert matching[0].levelno == logging.WARNING


def test_raise_web_error_internal_logs_at_exception_level(
    caplog: pytest.LogCaptureFixture,
) -> None:
    err = WebError(code=WebErrorCode.INTERNAL, message="Unexpected error", status_code=500)
    request = Request(_bare_scope())
    try:
        raise RuntimeError("real cause")
    except RuntimeError as exc:
        captured = exc
    with caplog.at_level(logging.ERROR, logger="dbsprout.web.errors"), pytest.raises(HTTPException):
        raise_web_error(request, err, original=captured)
    matching = [r for r in caplog.records if r.name == "dbsprout.web.errors"]
    assert any(r.levelno >= logging.ERROR for r in matching)
    # The traceback is captured by ``exc_info``, not embedded in the user-facing detail.
    detail = matching[0]
    assert "RuntimeError" not in detail.getMessage()


def test_raise_web_error_returns_htmx_fragment_when_header_present() -> None:
    """With HX-Request, the call returns an :class:`HTMLResponse` (no exception)."""
    app = create_app()
    scope = dict(_bare_scope(hx=True))
    scope["app"] = app
    request = Request(scope)
    err = WebError(
        code=WebErrorCode.AUTH_FAILED, message="bad creds", status_code=400, hint="try again"
    )
    response = raise_web_error(request, err)
    assert isinstance(response, HTMLResponse)
    assert response.status_code == 400
    body = response.body.decode("utf-8")
    assert "bad creds" in body
    assert "try again" in body
    assert err.correlation_id in body
    assert "AUTH_FAILED" in body
    # Plain HTML never carries a stack frame.
    assert not _TRACEBACK_RE.search(body)


# ── S-136 factory helpers ───────────────────────────────────────────────


def test_web_error_no_connection_factory() -> None:
    from dbsprout.web.errors import web_error_no_connection  # noqa: PLC0415

    err = web_error_no_connection()
    assert err.code is WebErrorCode.NO_CONNECTION
    assert err.status_code == 409
    assert "connect" in err.message.lower()
    assert err.hint is not None


def test_web_error_no_run_factory() -> None:
    from dbsprout.web.errors import web_error_no_run  # noqa: PLC0415

    err = web_error_no_run()
    assert err.code is WebErrorCode.NO_RUN
    assert err.status_code == 409
    assert "generation" in err.message.lower() or "generate" in err.message.lower()
    assert err.hint is not None


def test_web_error_write_guard_required_factory() -> None:
    from dbsprout.web.errors import web_error_write_guard_required  # noqa: PLC0415

    err = web_error_write_guard_required()
    assert err.code is WebErrorCode.WRITE_GUARD_REQUIRED
    assert err.status_code == 403
    assert "token" in err.message.lower()
    assert err.hint is not None


# ---------------------------------------------------------------------------
# S-140 export-route factory helpers.
# ---------------------------------------------------------------------------


def test_web_error_not_found_tables_envelope() -> None:
    """Surfaces missing-table names via the existing ``NOT_FOUND`` code (404)."""
    err = web_error_not_found_tables(["users", "orders"])
    assert err.code is WebErrorCode.NOT_FOUND
    assert err.status_code == 404
    assert "users" in err.message
    assert "orders" in err.message
    payload = err.to_dict()
    assert payload["code"] == "NOT_FOUND"
    assert "correlation_id" in payload


def test_web_error_export_multi_table_unsupported_envelope() -> None:
    """422 with hint that mentions a single-element ``tables`` subset."""
    err = web_error_export_multi_table_unsupported("csv")
    assert err.code is WebErrorCode.EXPORT_MULTI_TABLE_UNSUPPORTED
    assert err.status_code == 422
    assert "csv" in err.message
    assert err.hint is not None
    assert "tables" in err.hint


def test_web_error_export_dependency_missing_envelope() -> None:
    """422 with a ``pip install`` hint that mentions the missing extra."""
    err = web_error_export_dependency_missing("parquet", "data")
    assert err.code is WebErrorCode.EXPORT_DEPENDENCY_MISSING
    assert err.status_code == 422
    assert "parquet" in err.message
    assert err.hint is not None
    assert "dbsprout[data]" in err.hint
