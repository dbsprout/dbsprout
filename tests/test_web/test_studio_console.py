"""Studio live-progress console — server-rendered template + JS asset (S-125).

The console is an Alpine.js component that subscribes to the existing
``/ws/jobs/{job_id}`` WebSocket (S-109) once a job is submitted via
``POST /api/generate`` (S-124). This module covers the *server-side* contract:

* ``_studio_console.html`` renders an Alpine ``x-data`` root with the stable
  element ids the JS module mutates (overall bar, table list, status badge,
  summary line).
* The vendored JS module ``studio_console.js`` is served from ``/static/`` and
  exposes a ``studioConsole`` factory function (used by Alpine ``x-data``).

JS behaviour itself (state transitions, rolling rows/sec, reconnect path) is
verified in ``test_studio_console_js.py`` — a Node subprocess drives the
factory directly without needing a browser.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")

from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from fastapi import FastAPI


def _make_app(tmp_path: Path) -> FastAPI:
    from dbsprout.web.app import create_app  # noqa: PLC0415

    return create_app(state_db_path=tmp_path / "state.db")


# ── Studio console template renders Alpine root with stable element ids ────


def test_studio_console_has_alpine_x_data_root(tmp_path: Path) -> None:
    """AC: console body is an Alpine ``x-data`` component rooted at the panel."""
    body = TestClient(_make_app(tmp_path)).get("/studio").text
    assert 'x-data="studioConsole()"' in body, (
        "Studio console must instantiate the studioConsole Alpine factory"
    )


def test_studio_console_exposes_overall_bar_id(tmp_path: Path) -> None:
    """AC: overall progress bar carries a stable id the JS module mutates."""
    body = TestClient(_make_app(tmp_path)).get("/studio").text
    assert 'id="studio-console-overall"' in body


def test_studio_console_exposes_table_list_id(tmp_path: Path) -> None:
    """AC: per-table list root carries a stable id and an ``x-for`` template."""
    body = TestClient(_make_app(tmp_path)).get("/studio").text
    assert 'id="studio-console-tables"' in body
    assert "x-for" in body, "per-table render must use x-for (keyed) to avoid full re-render"
    # Alpine x-for requires :key — the AC pins stable keys for 100+ tables.
    assert ":key=" in body


def test_studio_console_exposes_status_badge(tmp_path: Path) -> None:
    """AC: status badge id is stable so the JS can swap idle→connecting→…→succeeded."""
    body = TestClient(_make_app(tmp_path)).get("/studio").text
    assert 'id="studio-console-status"' in body


def test_studio_console_exposes_summary_line(tmp_path: Path) -> None:
    """AC: terminal events render a summary line; the slot is reserved up-front."""
    body = TestClient(_make_app(tmp_path)).get("/studio").text
    assert 'id="studio-console-summary"' in body


def test_studio_console_loads_static_js_module(tmp_path: Path) -> None:
    """AC: the JS lives at /static/studio_console.js and is included on /studio."""
    body = TestClient(_make_app(tmp_path)).get("/studio").text
    assert "/static/studio_console.js" in body


def test_studio_console_js_asset_is_served(tmp_path: Path) -> None:
    """AC: the JS asset is reachable from the FastAPI static mount."""
    resp = TestClient(_make_app(tmp_path)).get("/static/studio_console.js")
    assert resp.status_code == 200, resp.text
    # Sanity: looks like JS, not a 404 HTML.
    assert "studioConsole" in resp.text


# ── JS module file shape (without invoking it — that's the Node test) ───


def _studio_console_js_path() -> Path:
    from dbsprout.web import app as web_app  # noqa: PLC0415

    return Path(web_app.__file__).resolve().parent / "static" / "studio_console.js"


def test_studio_console_js_file_exists() -> None:
    assert _studio_console_js_path().is_file(), (
        "studio_console.js must be vendored as a static asset"
    )


def test_studio_console_js_defines_factory() -> None:
    """AC: JS file defines the ``studioConsole`` Alpine factory at the top level."""
    src = _studio_console_js_path().read_text(encoding="utf-8")
    assert "studioConsole" in src
    # The factory must be reachable from Alpine x-data — either attached to
    # ``window`` or exposed as a top-level function. Either form is acceptable.
    assert "window.studioConsole" in src or "function studioConsole" in src, (
        "studioConsole must be a top-level / window-attached factory"
    )


# ── job-start event glue: generate route returns a job_id; the page wires
#    a custom event the Alpine factory listens for, so the submit form
#    doesn't have to know about the WebSocket. ────────────────────────────


def test_studio_console_listens_for_job_start_event(tmp_path: Path) -> None:
    """AC: console listens on the window for a ``studio:job-start`` custom event.

    The submit form dispatches ``studio:job-start`` with ``{jobId}`` once the
    POST /api/generate response is in hand; the console then opens the WS.
    This decouples the console from the form (and from HTMX vs. fetch).
    """
    body = TestClient(_make_app(tmp_path)).get("/studio").text
    assert "studio:job-start" in body, (
        "Studio page must dispatch / listen on 'studio:job-start' to connect the console"
    )
