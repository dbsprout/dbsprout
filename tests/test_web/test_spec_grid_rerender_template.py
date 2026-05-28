"""Server-side contract for the S-132 spec-grid rerender JS asset.

The JS behaviour is exercised end-to-end under Node in
``test_spec_grid_rerender_js.py``. This module covers the *Python* side:

* The ``spec_grid_rerender.js`` file is vendored as a static asset.
* The studio page (`/studio`) loads it via a ``<script>`` tag.
* The page bootstraps ``window.attachSpecGridRerender(...)`` so the
  picker → regen → preview pipeline is live without manual wiring.
* The method-picker template dispatches a ``studio:row-rerendered``
  window event after a successful row swap — the documented seam.
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


def _js_path() -> Path:
    from dbsprout.web import app as web_app  # noqa: PLC0415

    return Path(web_app.__file__).resolve().parent / "static" / "spec_grid_rerender.js"


# ── vendored asset ──────────────────────────────────────────────────────


def test_spec_grid_rerender_js_file_exists() -> None:
    """AC: ``spec_grid_rerender.js`` lives at ``dbsprout/web/static/``."""
    assert _js_path().is_file(), "spec_grid_rerender.js must be vendored as a static asset (S-132)"


def test_spec_grid_rerender_js_exposes_attach_function() -> None:
    """AC: the module exposes ``attachSpecGridRerender`` at top-level."""
    src = _js_path().read_text(encoding="utf-8")
    assert "attachSpecGridRerender" in src, (
        "spec_grid_rerender.js must define attachSpecGridRerender"
    )


def test_spec_grid_rerender_js_is_dual_loadable() -> None:
    """AC: the file is both a browser global and a CommonJS module (Node test path)."""
    src = _js_path().read_text(encoding="utf-8")
    # Same dual-loadable contract as focus_cell.js / studio_console.js.
    assert "module.exports" in src
    assert "attachSpecGridRerender" in src


# ── studio page mounts the script + bootstraps ──────────────────────────


def test_studio_page_loads_spec_grid_rerender_js(tmp_path: Path) -> None:
    """AC: /studio includes ``<script ... src=".../spec_grid_rerender.js">``."""
    body = TestClient(_make_app(tmp_path)).get("/studio").text
    assert "/static/spec_grid_rerender.js" in body


def test_studio_page_bootstraps_attach_spec_grid_rerender(tmp_path: Path) -> None:
    """AC: /studio calls ``window.attachSpecGridRerender(...)`` to wire the listener."""
    body = TestClient(_make_app(tmp_path)).get("/studio").text
    assert "attachSpecGridRerender(" in body


def test_spec_grid_rerender_js_asset_is_served(tmp_path: Path) -> None:
    """AC: the static asset is reachable from the FastAPI mount."""
    resp = TestClient(_make_app(tmp_path)).get("/static/spec_grid_rerender.js")
    assert resp.status_code == 200, resp.text
    assert "attachSpecGridRerender" in resp.text


# ── picker seam ─────────────────────────────────────────────────────────


def test_method_picker_dispatches_row_rerendered_event(tmp_path: Path) -> None:
    """AC: after a successful PUT swap, the picker dispatches ``studio:row-rerendered``.

    The studio page embeds the picker template inline, so we can grep the
    rendered page for the event-dispatch source. The literal event name is
    the documented seam other components rely on.
    """
    body = TestClient(_make_app(tmp_path)).get("/studio").text
    assert "studio:row-rerendered" in body, (
        "method_picker.html must dispatch 'studio:row-rerendered' on swap success"
    )
