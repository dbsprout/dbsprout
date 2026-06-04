"""POST /api/schema/load upload + parser dispatch + size cap (S-113).

The web stack lives in the optional ``[web]`` extra, so the module guards with
``pytest.importorskip("fastapi")`` before importing FastAPI symbols (mirrors the
sibling web tests). The route reuses ``dbsprout.schema.parsers.parse_schema_file``
verbatim — these tests only exercise the HTTP boundary (multipart upload, parser
override, size cap, friendly errors) and the workspace write, never the parsing
internals (covered by the parser unit tests).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")

from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from pathlib import Path

# ── minimal valid fixtures, one per file-content format ─────────────────

_SAMPLE_DDL = "CREATE TABLE users (id INTEGER PRIMARY KEY, email VARCHAR(255));"

_SAMPLE_DBML = """
Table users {
  id integer [pk, increment]
  email varchar [not null, unique]
}
"""

_SAMPLE_MERMAID = """
erDiagram
    USERS {
        int id PK
        string email
    }
"""

_SAMPLE_PLANTUML = """
@startuml
entity "users" as users {
  *id : integer <<PK>>
  --
  *email : varchar
}
@enduml
"""

_SAMPLE_PRISMA = """
model User {
  id    Int    @id @default(autoincrement())
  email String @unique
}
"""


def _make_client(state_db: Path) -> TestClient:
    from dbsprout.web.app import create_app  # noqa: PLC0415

    return TestClient(create_app(state_db_path=state_db))


def _post_file(
    client: TestClient,
    *,
    filename: str,
    content: bytes,
    parser: str | None = None,
) -> object:
    files = {"file": (filename, content, "application/octet-stream")}
    data = {"parser": parser} if parser is not None else None
    return client.post("/api/schema/load", files=files, data=data)


# ── happy path: DDL upload ──────────────────────────────────────────────


def test_upload_ddl_returns_summary(tmp_path: Path) -> None:
    client = _make_client(tmp_path / "state.db")
    resp = _post_file(client, filename="schema.sql", content=_SAMPLE_DDL.encode())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["table_count"] >= 1
    assert "users" in body["tables"]
    assert body["source"].startswith("upload:")
    assert "dialect" in body


def test_upload_stores_schema_in_workspace(tmp_path: Path) -> None:
    from dbsprout.web.app import create_app  # noqa: PLC0415

    app = create_app(state_db_path=tmp_path / "state.db")
    client = TestClient(app)
    resp = _post_file(client, filename="schema.sql", content=_SAMPLE_DDL.encode())
    assert resp.status_code == 200, resp.text

    schema = app.state.workspace.get_schema()
    assert schema is not None
    assert "users" in schema.table_names()
    assert app.state.workspace.get_source() == "upload:schema.sql"


# ── multi-format parse coverage ─────────────────────────────────────────


@pytest.mark.parametrize(
    ("filename", "content", "expected_table"),
    [
        ("schema.sql", _SAMPLE_DDL, "users"),
        ("schema.dbml", _SAMPLE_DBML, "users"),
        ("schema.mermaid", _SAMPLE_MERMAID, "users"),
        ("schema.mmd", _SAMPLE_MERMAID, "users"),
        ("schema.puml", _SAMPLE_PLANTUML, "users"),
        # Prisma maps ``model User`` → table ``user`` (parser's own naming).
        ("schema.prisma", _SAMPLE_PRISMA, "user"),
    ],
)
def test_upload_each_format_by_suffix(
    tmp_path: Path, filename: str, content: str, expected_table: str
) -> None:
    client = _make_client(tmp_path / "state.db")
    resp = _post_file(client, filename=filename, content=content.encode())
    assert resp.status_code == 200, resp.text
    assert expected_table in resp.json()["tables"]


@pytest.mark.parametrize(
    ("parser", "content", "expected_table"),
    [
        ("dbml", _SAMPLE_DBML, "users"),
        ("mermaid", _SAMPLE_MERMAID, "users"),
        ("plantuml", _SAMPLE_PLANTUML, "users"),
        ("prisma", _SAMPLE_PRISMA, "user"),
        ("ddl", _SAMPLE_DDL, "users"),
        ("sql", _SAMPLE_DDL, "users"),
        ("DBML", _SAMPLE_DBML, "users"),  # case-insensitive
    ],
)
def test_parser_override_wins_over_filename(
    tmp_path: Path, parser: str, content: str, expected_table: str
) -> None:
    """An explicit ``parser`` overrides a generic / wrong filename suffix."""
    client = _make_client(tmp_path / "state.db")
    resp = _post_file(client, filename="schema.txt", content=content.encode(), parser=parser)
    assert resp.status_code == 200, resp.text
    assert expected_table in resp.json()["tables"]


def test_unknown_suffix_falls_back_to_ddl(tmp_path: Path) -> None:
    """No parser + unknown suffix → DDL fallback (matches parse_schema_file)."""
    client = _make_client(tmp_path / "state.db")
    resp = _post_file(client, filename="schema.unknown", content=_SAMPLE_DDL.encode())
    assert resp.status_code == 200, resp.text
    assert "users" in resp.json()["tables"]


# ── size cap (413) + empty upload (400) ─────────────────────────────────


def test_oversize_upload_rejected_413(tmp_path: Path) -> None:
    from dbsprout.web.routers import schema_load  # noqa: PLC0415

    client = _make_client(tmp_path / "state.db")
    oversize = b"x" * (schema_load._MAX_UPLOAD_BYTES + 1)
    resp = _post_file(client, filename="big.sql", content=oversize)
    assert resp.status_code == 413, resp.text
    detail = resp.json()["detail"]
    # S-116 typed envelope.
    assert isinstance(detail, dict)
    assert detail["code"] == "FILE_TOO_LARGE"
    assert "correlation_id" in detail
    assert "Traceback" not in resp.text


def test_empty_upload_rejected_400(tmp_path: Path) -> None:
    client = _make_client(tmp_path / "state.db")
    resp = _post_file(client, filename="empty.sql", content=b"")
    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    assert isinstance(detail, dict)
    assert detail["code"] == "EMPTY_FILE"
    assert "correlation_id" in detail


# ── friendly errors (unknown parser, unparseable content) ───────────────


def test_unknown_parser_value_rejected_400(tmp_path: Path) -> None:
    client = _make_client(tmp_path / "state.db")
    resp = _post_file(client, filename="schema.sql", content=_SAMPLE_DDL.encode(), parser="bogus")
    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    assert isinstance(detail, dict)
    assert detail["code"] == "UNKNOWN_PARSER"
    assert "correlation_id" in detail
    assert "Traceback" not in resp.text


def test_unparseable_content_rejected_400_no_traceback(tmp_path: Path) -> None:
    """Garbage forced through a strict parser → friendly 400, no traceback leak."""
    client = _make_client(tmp_path / "state.db")
    resp = _post_file(
        client,
        filename="schema.txt",
        content=b"this is not a valid prisma schema at all !!!",
        parser="prisma",
    )
    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    assert isinstance(detail, dict)
    assert detail["code"] == "PARSE_ERROR"
    assert "correlation_id" in detail
    assert "Traceback" not in resp.text


def test_unexpected_parser_exception_returns_internal_500(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S-116: a parser raising an *unexpected* type → INTERNAL/500, no detail leak."""
    import dbsprout.schema.parsers as parsers_mod  # noqa: PLC0415

    leak_marker = "INTERNAL-DETAIL-SHOULD-NOT-LEAK"

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError(leak_marker)

    # Patch the symbol the router imports inside _parse_upload.
    monkeypatch.setattr(parsers_mod, "parse_schema_file", _boom)
    client = _make_client(tmp_path / "state.db")
    resp = _post_file(client, filename="schema.sql", content=_SAMPLE_DDL.encode())
    assert resp.status_code == 500, resp.text
    detail = resp.json()["detail"]
    assert isinstance(detail, dict)
    assert detail["code"] == "INTERNAL"
    assert detail["message"] == "Unexpected error"
    assert "correlation_id" in detail
    assert leak_marker not in resp.text
    assert "Traceback" not in resp.text


def test_schema_load_htmx_request_returns_json_envelope(tmp_path: Path) -> None:
    """JSON-only since P1c-5: parse-failure errors are JSON even under HX-Request."""
    client = _make_client(tmp_path / "state.db")
    files = {"file": ("empty.sql", b"", "application/octet-stream")}
    resp = client.post("/api/schema/load", files=files, headers={"HX-Request": "true"})
    assert resp.status_code == 400
    assert resp.headers["content-type"].startswith("application/json")
    detail = resp.json()["detail"]
    assert detail["code"] == "EMPTY_FILE"
    assert detail["correlation_id"]
    assert "Traceback" not in resp.text


# ── router registration contract ────────────────────────────────────────


def test_router_registered_via_create_app(tmp_path: Path) -> None:
    """The endpoint is reachable through the real create_app wiring (no manual mount)."""
    client = _make_client(tmp_path / "state.db")
    # A bad request still proves the route exists (404 would mean unregistered).
    resp = client.post("/api/schema/load")
    assert resp.status_code != 404, "POST /api/schema/load not registered in create_app"


# ── S-114: auto-detect (extension → content sniff → DDL fallback) ──────


def test_autodetect_dbml_content_via_generic_suffix(tmp_path: Path) -> None:
    """DBML payload + ``.txt`` filename + no parser → content sniff routes to DBML."""
    client = _make_client(tmp_path / "state.db")
    resp = _post_file(client, filename="schema.txt", content=_SAMPLE_DBML.encode())
    assert resp.status_code == 200, resp.text
    assert "users" in resp.json()["tables"]


@pytest.mark.parametrize(
    ("filename", "content", "expected_table"),
    [
        ("schema.sql", _SAMPLE_DDL, "users"),
        ("schema.dbml", _SAMPLE_DBML, "users"),
        ("schema.mermaid", _SAMPLE_MERMAID, "users"),
        ("schema.mmd", _SAMPLE_MERMAID, "users"),
        ("schema.puml", _SAMPLE_PLANTUML, "users"),
        ("schema.plantuml", _SAMPLE_PLANTUML, "users"),
        ("schema.pu", _SAMPLE_PLANTUML, "users"),
        ("schema.prisma", _SAMPLE_PRISMA, "user"),
    ],
)
def test_autodetect_each_format_by_extension(
    tmp_path: Path, filename: str, content: str, expected_table: str
) -> None:
    """Every known suffix maps to its parser without an explicit ``parser`` field."""
    client = _make_client(tmp_path / "state.db")
    resp = _post_file(client, filename=filename, content=content.encode())
    assert resp.status_code == 200, resp.text
    assert expected_table in resp.json()["tables"]


@pytest.mark.parametrize(
    ("content", "expected_table"),
    [
        (_SAMPLE_DBML, "users"),
        (_SAMPLE_MERMAID, "users"),
        (_SAMPLE_PLANTUML, "users"),
        (_SAMPLE_PRISMA, "user"),
    ],
)
def test_autodetect_content_sniff_with_generic_suffix(
    tmp_path: Path, content: str, expected_table: str
) -> None:
    """Generic ``.txt`` filename + no parser → content sniff picks the right parser."""
    client = _make_client(tmp_path / "state.db")
    resp = _post_file(client, filename="schema.txt", content=content.encode())
    assert resp.status_code == 200, resp.text
    assert expected_table in resp.json()["tables"]


def test_autodetect_extensionless_filename_uses_content_sniff(tmp_path: Path) -> None:
    """Filename without an extension → content sniff still routes correctly."""
    client = _make_client(tmp_path / "state.db")
    resp = _post_file(client, filename="schema_no_extension", content=_SAMPLE_PRISMA.encode())
    assert resp.status_code == 200, resp.text
    assert "user" in resp.json()["tables"]


def test_explicit_parser_overrides_autodetect(tmp_path: Path) -> None:
    """An explicit ``parser`` beats both filename and content auto-detect."""
    client = _make_client(tmp_path / "state.db")
    # DBML payload, DBML filename, but force Prisma → Prisma parser rejects → 400.
    resp = _post_file(
        client,
        filename="schema.dbml",
        content=_SAMPLE_DBML.encode(),
        parser="prisma",
    )
    assert resp.status_code == 400, resp.text
    assert "detail" in resp.json()
    assert "Traceback" not in resp.text


def test_autodetect_unknown_content_falls_back_to_ddl(tmp_path: Path) -> None:
    """Unknown suffix + content that none of the sniffers match → DDL fallback.

    DDL parser then rejects the garbage with a friendly 400 (no traceback).
    """
    client = _make_client(tmp_path / "state.db")
    resp = _post_file(
        client,
        filename="schema.unknown",
        content=b"this content matches none of the sniffers !!!",
    )
    assert resp.status_code == 400, resp.text
    assert "detail" in resp.json()
    assert "Traceback" not in resp.text


def test_autodetect_binary_garbage_falls_back_safely(tmp_path: Path) -> None:
    """Non-UTF-8 / binary garbage head should not crash the sniffer."""
    client = _make_client(tmp_path / "state.db")
    # Non-UTF-8 bytes — the sniffer decodes with errors="ignore" and falls through.
    resp = _post_file(
        client,
        filename="schema.bin",
        content=b"\xff\xfe\x00\x01\x02\x03" * 64,
    )
    # Either 400 (DDL parser rejects) — the goal is "no 500, no traceback".
    assert resp.status_code == 400, resp.text
    assert "Traceback" not in resp.text


# ── unit tests for the detection helpers (cheap, no TestClient) ────────


def test_detect_suffix_from_content_returns_none_for_empty_head() -> None:
    """Empty / whitespace head → no sniff match → returns ``None``."""
    from dbsprout.web.routers.schema_load import _detect_suffix_from_content  # noqa: PLC0415

    assert _detect_suffix_from_content(b"") is None
    assert _detect_suffix_from_content(b"   \n\t ") is None


def test_detect_suffix_from_content_returns_none_for_random_bytes() -> None:
    """Random text matching none of the sniffers → ``None``."""
    from dbsprout.web.routers.schema_load import _detect_suffix_from_content  # noqa: PLC0415

    assert _detect_suffix_from_content(b"hello world, just a note") is None


def test_detect_suffix_from_content_matches_each_format() -> None:
    """Each canonical format keyword resolves to its suffix."""
    from dbsprout.web.routers.schema_load import _detect_suffix_from_content  # noqa: PLC0415

    assert _detect_suffix_from_content(_SAMPLE_DBML.encode()) == ".dbml"
    assert _detect_suffix_from_content(_SAMPLE_MERMAID.encode()) == ".mermaid"
    assert _detect_suffix_from_content(_SAMPLE_PLANTUML.encode()) == ".puml"
    assert _detect_suffix_from_content(_SAMPLE_PRISMA.encode()) == ".prisma"


def test_resolve_suffix_without_content_head_falls_back_to_ddl() -> None:
    """When no content_head is passed and filename suffix is unknown → ``.sql``."""
    from dbsprout.web.routers.schema_load import _resolve_suffix  # noqa: PLC0415

    assert _resolve_suffix("schema.unknown", None) == ".sql"
    assert _resolve_suffix(None, None) == ".sql"


def test_resolve_suffix_known_filename_skips_content_sniff() -> None:
    """A known filename suffix wins even when content would sniff differently."""
    from dbsprout.web.routers.schema_load import _resolve_suffix  # noqa: PLC0415

    # DBML content, but the filename says .sql → .sql (filename beats content).
    assert _resolve_suffix("schema.sql", None, content_head=_SAMPLE_DBML.encode()) == ".sql"


def test_resolve_suffix_parser_override_skips_filename_and_content() -> None:
    """Explicit parser override beats both filename and content sniff."""
    from dbsprout.web.routers.schema_load import _resolve_suffix  # noqa: PLC0415

    # DBML filename + DBML content, but parser=prisma → .prisma.
    assert _resolve_suffix("schema.dbml", "prisma", content_head=_SAMPLE_DBML.encode()) == ".prisma"
