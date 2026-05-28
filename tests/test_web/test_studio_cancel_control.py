"""Studio console Cancel-control template tests (S-126).

The full live-tail console is owned by S-125. S-126 only adds the **Cancel
button slot** into the same ``_studio_console.html`` partial: an Alpine-bound
control that POSTs to ``/api/jobs/{job_id}/cancel`` and disables itself on
click. The slot uses stable ids + ``data-testid`` handles so S-125's
wave-merge can union the live-tail content without disturbing the button.

The web stack lives in the optional ``[web]`` extra → ``importorskip("fastapi")``.
We assert against the rendered ``GET /studio`` HTML (not against the file on
disk) so the test exercises Jinja rendering exactly as the dashboard serves it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")

from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi import FastAPI


def _make_app(state_db: Path) -> FastAPI:
    from dbsprout.web.app import create_app  # noqa: PLC0415

    return create_app(state_db_path=state_db)


def _studio_html(app: FastAPI) -> str:
    resp = TestClient(app).get("/studio")
    assert resp.status_code == 200, resp.text
    return resp.text


# ── slot exists with stable handles ───────────────────────────────────


def test_studio_console_has_cancel_button(tmp_path: Path) -> None:
    """The console partial renders a Cancel button with a stable id + testid."""
    html = _studio_html(_make_app(tmp_path / "state.db"))
    assert 'id="studio-cancel-button"' in html
    assert 'data-testid="studio-cancel-button"' in html


def test_studio_cancel_button_targets_cancel_api(tmp_path: Path) -> None:
    """The console template references the cancel endpoint so the Alpine handler
    can POST to it. The literal ``/api/jobs/`` path is enough — the ``{job_id}``
    interpolation happens at click time."""
    html = _studio_html(_make_app(tmp_path / "state.db"))
    assert "/api/jobs/" in html


def test_studio_cancel_button_hidden_by_default(tmp_path: Path) -> None:
    """Idle dashboard has no active job → the Cancel control is hidden (Alpine
    ``x-show`` over a ``jobId`` flag). We assert the directive is present so a
    later regression that drops the gate is caught."""
    html = _studio_html(_make_app(tmp_path / "state.db"))
    # Alpine x-show binding on the cancel control. The exact expression is
    # stable: ``jobId`` is the data field S-125 will populate.
    assert "x-show" in html
    assert "jobId" in html
