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
    assert "detail" in resp.json()
    assert "Traceback" not in resp.text


def test_empty_upload_rejected_400(tmp_path: Path) -> None:
    client = _make_client(tmp_path / "state.db")
    resp = _post_file(client, filename="empty.sql", content=b"")
    assert resp.status_code == 400, resp.text
    assert "detail" in resp.json()


# ── friendly errors (unknown parser, unparseable content) ───────────────


def test_unknown_parser_value_rejected_400(tmp_path: Path) -> None:
    client = _make_client(tmp_path / "state.db")
    resp = _post_file(client, filename="schema.sql", content=_SAMPLE_DDL.encode(), parser="bogus")
    assert resp.status_code == 400, resp.text
    body = resp.json()
    assert "detail" in body
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
    assert "detail" in resp.json()
    assert "Traceback" not in resp.text


# ── router registration contract ────────────────────────────────────────


def test_router_registered_via_create_app(tmp_path: Path) -> None:
    """The endpoint is reachable through the real create_app wiring (no manual mount)."""
    client = _make_client(tmp_path / "state.db")
    # A bad request still proves the route exists (404 would mean unregistered).
    resp = client.post("/api/schema/load")
    assert resp.status_code != 404, "POST /api/schema/load not registered in create_app"
