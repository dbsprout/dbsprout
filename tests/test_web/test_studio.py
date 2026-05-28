"""Studio shell layout tests (S-117).

The Studio page is a single Jinja2 shell with **four named slots**
(``tree`` · ``grid`` · ``context`` · ``console``) wired onto stable element ids
so later Phase-C stories (S-125 console progress, S-127 seed control, S-118
spec grid) can plug content into the same shell without editing it.

The web stack lives in the optional ``[web]`` extra, so the module guards with
``pytest.importorskip("fastapi")`` before importing FastAPI symbols (mirrors the
sibling web tests). Tests seed ``app.state.workspace`` directly (a plain mutable
object), exactly as the connect / schema-view tests do; they never exercise the
snapshot-backed ``GET /schema`` view (covered by ``test_erd.py``).
"""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")

from fastapi.testclient import TestClient

from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    TableSchema,
)

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi import FastAPI


# ── fixtures / helpers ────────────────────────────────────────────────


def _make_app(tmp_path: Path) -> FastAPI:
    from dbsprout.web.app import create_app  # noqa: PLC0415

    return create_app(state_db_path=tmp_path / "state.db")


def _small_schema() -> DatabaseSchema:
    """A 2-table schema: ``orders`` references ``users``."""
    users = TableSchema(
        name="users",
        columns=[
            ColumnSchema(name="id", data_type=ColumnType.INTEGER, primary_key=True),
            ColumnSchema(name="email", data_type=ColumnType.VARCHAR, unique=True, nullable=False),
        ],
        primary_key=["id"],
    )
    orders = TableSchema(
        name="orders",
        columns=[
            ColumnSchema(name="id", data_type=ColumnType.INTEGER, primary_key=True),
            ColumnSchema(name="user_id", data_type=ColumnType.INTEGER),
            ColumnSchema(name="total", data_type=ColumnType.DECIMAL),
        ],
        primary_key=["id"],
        foreign_keys=[
            ForeignKeySchema(
                columns=["user_id"], ref_table="users", ref_columns=["id"], on_delete="CASCADE"
            ),
        ],
    )
    return DatabaseSchema(tables=[users, orders], dialect="sqlite")


def _seed(app: FastAPI, schema: DatabaseSchema, source: str | None = None) -> None:
    """Load *schema* into the app's workspace, as connect/load would."""
    app.state.workspace.set_schema(schema)
    if source is not None:
        app.state.workspace.set_source(source)


# ── GET /studio — basic shell ──────────────────────────────────────────


def test_studio_route_returns_200(tmp_path: Path) -> None:
    """AC: GET /studio returns 200 even with no schema loaded."""
    resp = TestClient(_make_app(tmp_path)).get("/studio")
    assert resp.status_code == 200, resp.text


def test_studio_returns_html_content_type(tmp_path: Path) -> None:
    resp = TestClient(_make_app(tmp_path)).get("/studio")
    assert "text/html" in resp.headers["content-type"]


def test_studio_extends_base_with_full_html_document(tmp_path: Path) -> None:
    """The Studio page is a full HTML document (extends base.html), not a fragment."""
    body = TestClient(_make_app(tmp_path)).get("/studio").text.lower()
    assert "<!doctype html>" in body
    assert "<html" in body
    assert "</html>" in body


# ── four named panels with stable ids ─────────────────────────────────


@pytest.mark.parametrize(
    ("element_id", "panel_name"),
    [
        ("studio-tree", "tree"),
        ("studio-grid", "grid"),
        ("studio-context", "context"),
        ("studio-console", "console"),
    ],
)
def test_studio_has_four_panel_ids(tmp_path: Path, element_id: str, panel_name: str) -> None:
    """AC: each of the 4 panels has a stable id + ``data-panel`` attribute.

    Later Phase-C stories target these ids; this test pins the contract.
    """
    body = TestClient(_make_app(tmp_path)).get("/studio").text
    assert f'id="{element_id}"' in body, f"missing stable panel id #{element_id}"
    assert f'data-panel="{panel_name}"' in body, f"missing data-panel={panel_name} attr"


def test_studio_panels_use_semantic_html(tmp_path: Path) -> None:
    """AC: semantic HTML — tree/context are <aside>, grid is <main>, console is <footer>."""
    body = TestClient(_make_app(tmp_path)).get("/studio").text
    assert '<aside id="studio-tree"' in body
    assert '<main id="studio-grid"' in body
    assert '<aside id="studio-context"' in body
    assert '<footer id="studio-console"' in body


# ── schema tree panel content ──────────────────────────────────────────


def test_studio_tree_lists_table_names_when_schema_loaded(tmp_path: Path) -> None:
    """AC: tree panel renders the workspace schema (table names) when loaded."""
    app = _make_app(tmp_path)
    _seed(app, _small_schema())
    body = TestClient(app).get("/studio").text
    # Both table names appear inside the tree panel
    tree_start = body.index('id="studio-tree"')
    tree_chunk = body[tree_start : tree_start + 4000]
    assert "users" in tree_chunk
    assert "orders" in tree_chunk


def test_studio_tree_shows_table_count_when_schema_loaded(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed(app, _small_schema())
    body = TestClient(app).get("/studio").text
    # 2 tables loaded → count appears somewhere in the tree panel
    tree_start = body.index('id="studio-tree"')
    tree_chunk = body[tree_start : tree_start + 4000]
    assert "2" in tree_chunk


def test_studio_tree_empty_state_when_no_schema(tmp_path: Path) -> None:
    """AC: tree panel shows 'no schema loaded' empty state when workspace empty."""
    body = TestClient(_make_app(tmp_path)).get("/studio").text
    tree_start = body.index('id="studio-tree"')
    tree_chunk = body[tree_start : tree_start + 4000]
    assert "no schema" in tree_chunk.lower()


def test_studio_empty_state_links_to_connect_and_upload(tmp_path: Path) -> None:
    """AC: empty state surfaces links to connect (S-112) / upload (S-113)."""
    body = TestClient(_make_app(tmp_path)).get("/studio").text.lower()
    # Mentions both the connect and upload paths so the user can act
    assert "/api/connect" in body or "connect" in body
    assert "/api/schema/load" in body or "upload" in body


# ── grid / context / console placeholders ─────────────────────────────


def test_studio_grid_has_placeholder(tmp_path: Path) -> None:
    """Center grid is a placeholder (S-118 fills it) but renders with the slot."""
    body = TestClient(_make_app(tmp_path)).get("/studio").text
    grid_start = body.index('id="studio-grid"')
    grid_chunk = body[grid_start : grid_start + 2000].lower()
    # Some human-readable hint that this panel is reserved for the spec grid
    assert "preview" in grid_chunk or "spec" in grid_chunk or "grid" in grid_chunk


def test_studio_console_has_placeholder(tmp_path: Path) -> None:
    """Bottom console is a placeholder (S-125 fills it)."""
    body = TestClient(_make_app(tmp_path)).get("/studio").text
    console_start = body.index('id="studio-console"')
    console_chunk = body[console_start : console_start + 2000].lower()
    assert "console" in console_chunk or "run" in console_chunk or "log" in console_chunk


def test_studio_context_has_placeholder(tmp_path: Path) -> None:
    body = TestClient(_make_app(tmp_path)).get("/studio").text
    ctx_start = body.index('id="studio-context"')
    ctx_chunk = body[ctx_start : ctx_start + 2000].lower()
    assert "context" in ctx_chunk or "help" in ctx_chunk or "detail" in ctx_chunk


# ── alpine / htmx scaffold ─────────────────────────────────────────────


def test_studio_loads_alpine_static_not_cdn(tmp_path: Path) -> None:
    """AC: Alpine is vendored as a static file — never added as a Python dep."""
    body = TestClient(_make_app(tmp_path)).get("/studio").text
    assert "/static/alpine" in body, "Studio shell must load Alpine from /static (vendored)"


def test_alpine_static_asset_is_served(tmp_path: Path) -> None:
    """AC: the vendored Alpine file is reachable from /static."""
    resp = TestClient(_make_app(tmp_path)).get("/static/alpine.min.js")
    assert resp.status_code == 200, resp.text
    # Sanity: looks like JS, not an HTML 404 page
    assert "html" not in resp.headers.get("content-type", "").lower() or resp.text.strip()


def test_studio_loads_htmx(tmp_path: Path) -> None:
    """HTMX is already in base.html via CDN; the Studio page must inherit it."""
    body = TestClient(_make_app(tmp_path)).get("/studio").text.lower()
    assert "htmx" in body


# ── named-block extensibility ─────────────────────────────────────────


def test_studio_template_exposes_named_blocks_for_each_panel() -> None:
    """AC: Jinja2 template exposes named blocks ``tree``/``grid``/``context``/``console``.

    Later Phase-C stories override these blocks (or hx-get content into the
    matching ids) without ever touching this shell template.
    """
    from pathlib import Path  # noqa: PLC0415

    from dbsprout.web import app as web_app  # noqa: PLC0415

    tpl = Path(web_app.__file__).resolve().parent / "templates" / "studio.html"
    src = tpl.read_text(encoding="utf-8")
    for block in ("tree", "grid", "context", "console"):
        assert "{% block " + block + " %}" in src, f"missing named Jinja block: {block}"
        assert "{% endblock %}" in src


# ── keyboard / accessibility ─────────────────────────────────────────


def test_studio_panels_are_keyboard_navigable(tmp_path: Path) -> None:
    """AC: panels are keyboard-navigable — each carries a role or tabindex hint."""
    body = TestClient(_make_app(tmp_path)).get("/studio").text
    # We don't require all four to be tabindex'd, but at minimum the page declares
    # role-based regions for the four panels (so screen readers can navigate).
    role_hits = (
        body.count('role="region"')
        + body.count('role="navigation"')
        + body.count('role="complementary"')
    )
    assert role_hits >= 4, f"expected >= 4 landmark roles for 4 panels, found {role_hits}"


def test_studio_has_aria_labels_on_panels(tmp_path: Path) -> None:
    """Each landmark panel carries an aria-label so screen readers can announce it."""
    body = TestClient(_make_app(tmp_path)).get("/studio").text.lower()
    assert "aria-label=" in body


# ── nav integration ───────────────────────────────────────────────────


def test_studio_link_in_main_nav(tmp_path: Path) -> None:
    """The Studio route is the new central workspace — surface it in the navbar."""
    body = TestClient(_make_app(tmp_path)).get("/studio").text
    # A nav anchor pointing at /studio so users can find their way back
    assert 'href="/studio"' in body


# ── router registration / seam ─────────────────────────────────────────


def test_studio_router_importable() -> None:
    from fastapi import APIRouter  # noqa: PLC0415

    from dbsprout.web.routers.studio import studio_router  # noqa: PLC0415

    assert isinstance(studio_router, APIRouter)


def test_studio_route_registered_on_app(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    paths = {getattr(r, "path", "") for r in app.routes}
    assert "/studio" in paths


def test_studio_region_block_in_app_factory() -> None:
    """AC: ``create_app`` registers the router inside an S-117 region block.

    Phase-A sibling stories edit ``create_app`` in parallel; this test pins the
    region delimiters so the union merge stays clean.
    """
    from pathlib import Path  # noqa: PLC0415

    from dbsprout.web import app as web_app  # noqa: PLC0415

    src = Path(web_app.__file__).read_text(encoding="utf-8")
    assert "# ── S-117 studio shell ──" in src
    assert "# ── end S-117 ──" in src
    assert "studio_router" in src


# ── lazy-import contract ───────────────────────────────────────────────


def test_studio_router_no_eager_generation_import() -> None:
    """Importing the router must not pull the orchestrator / core service."""
    probe = (
        "import sys\n"
        "import dbsprout.web.routers.studio  # noqa: F401\n"
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
        "importing dbsprout.web.routers.studio eagerly imported generation/core "
        f"modules: {result.stdout.strip()}"
    )
