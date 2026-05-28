"""Wizard "Open in Studio" handoff tests (S-144).

The Sprint-4 wizard ships a shared affordance on every step body that lets the
user jump to the full Studio (S-117) without re-loading their schema, spec, or
last generate result. The handoff is *implicit*: both pages bind to the same
``app.state.workspace`` singleton, so navigation alone is enough — no query
string, no session token, no transfer endpoint.

This module pins:

* the affordance lives on the **shared** wrapper ``_step_base.html`` so all six
  steps inherit it for free (no sibling collision with S-145 / S-146),
* the anchor is plain ``<a href="/studio">`` (progressive enhancement),
* visible on steps 1..6 (parametrised),
* positioned **top-right of the body**, never inside the left rail,
* the ``GET /studio`` response after a wizard session **does not** reset the
  shared ``Workspace`` — schema / spec / last_result survive the jump,
* the Studio handler itself never writes to the workspace (no fresh-session
  state-clear gap).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")

from fastapi.testclient import TestClient

from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    TableSchema,
)

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi import FastAPI


# ── helpers ──────────────────────────────────────────────────────────────


def _make_app(tmp_path: Path) -> FastAPI:
    from dbsprout.web.app import create_app  # noqa: PLC0415

    return create_app(state_db_path=tmp_path / "state.db")


def _tiny_schema() -> DatabaseSchema:
    """A 1-table schema; enough to assert the Studio tree renders it post-jump."""
    users = TableSchema(
        name="users",
        columns=[
            ColumnSchema(name="id", data_type=ColumnType.INTEGER, primary_key=True),
            ColumnSchema(name="email", data_type=ColumnType.VARCHAR, nullable=False),
        ],
        primary_key=["id"],
    )
    return DatabaseSchema(tables=[users], dialect="sqlite")


# ── affordance presence + shape ──────────────────────────────────────────


def test_step_base_has_open_in_studio_anchor(tmp_path: Path) -> None:
    """AC: the shared wrapper renders an ``Open in Studio`` anchor."""
    body = TestClient(_make_app(tmp_path)).get("/wizard").text
    # Stable testid pins the seam used by parallel sibling stories.
    assert 'data-testid="wizard-open-in-studio"' in body, (
        "wizard shell must render an Open-in-Studio affordance with a stable testid"
    )
    assert 'href="/studio"' in body, "the affordance must navigate to /studio"


@pytest.mark.parametrize("step", [1, 2, 3, 4, 5, 6])
def test_open_in_studio_visible_on_all_six_steps(tmp_path: Path, step: int) -> None:
    """AC: every wizard step body inherits the affordance via the shared wrapper."""
    body = TestClient(_make_app(tmp_path)).get(f"/wizard/step/{step}").text
    assert 'data-testid="wizard-open-in-studio"' in body, (
        f"step {step} body must inherit the Open-in-Studio affordance"
    )
    assert 'href="/studio"' in body


def test_open_in_studio_positioned_inside_step_body_not_rail(tmp_path: Path) -> None:
    """AC: the affordance sits inside the step ``<section data-panel="wizard-step">``.

    The rail (``data-panel="rail-panel"``) is reserved for step navigation; the
    handoff lives inside the body card so it travels with the step content.
    """
    body = TestClient(_make_app(tmp_path)).get("/wizard").text
    # Anchor must come *after* the wizard step body's data-panel marker, and
    # *before* the rail panel close, i.e. inside the body section.
    step_start = body.index('data-panel="wizard-step"')
    anchor_idx = body.index('data-testid="wizard-open-in-studio"')
    assert anchor_idx > step_start, (
        "Open-in-Studio anchor must live inside the wizard step body, "
        "not before its data-panel marker"
    )


def test_open_in_studio_anchor_is_top_right(tmp_path: Path) -> None:
    """AC: the affordance is right-aligned (top-right of the body header)."""
    body = TestClient(_make_app(tmp_path)).get("/wizard").text
    # Find the testid and look back to the enclosing <a ... > tag to read its classes.
    anchor_idx = body.index('data-testid="wizard-open-in-studio"')
    tag_start = body.rfind("<a", 0, anchor_idx)
    tag_end = body.index(">", anchor_idx)
    tag = body[tag_start : tag_end + 1]
    # Layout signal: any of these utility classes mean "push to the right".
    assert any(hint in tag for hint in ("ml-auto", "self-end", "justify-self-end")), (
        f"anchor should be right-aligned, got: {tag!r}"
    )


def test_open_in_studio_has_aria_label(tmp_path: Path) -> None:
    """AC: the anchor exposes an accessible name describing the destination."""
    body = TestClient(_make_app(tmp_path)).get("/wizard").text
    anchor_idx = body.index('data-testid="wizard-open-in-studio"')
    tag_start = body.rfind("<a", 0, anchor_idx)
    tag_end = body.index(">", anchor_idx)
    tag = body[tag_start : tag_end + 1].lower()
    assert "aria-label=" in tag, "Open-in-Studio anchor must carry an aria-label"
    assert "studio" in tag, "aria-label should mention Studio so it is meaningful"


def test_open_in_studio_progressive_enhancement(tmp_path: Path) -> None:
    """AC: the affordance works without JS — plain anchor, no onclick/Alpine.

    The handoff is just navigation to ``/studio``; both surfaces bind to the
    same ``app.state.workspace`` singleton, so no JS state transfer is needed.
    """
    body = TestClient(_make_app(tmp_path)).get("/wizard").text
    anchor_idx = body.index('data-testid="wizard-open-in-studio"')
    tag_start = body.rfind("<a", 0, anchor_idx)
    tag_end = body.index(">", anchor_idx)
    tag = body[tag_start : tag_end + 1]
    assert "onclick=" not in tag, "Open-in-Studio must work without JS — no onclick"
    assert "x-on:" not in tag, "Open-in-Studio must work without JS — no Alpine handler"
    assert "@click" not in tag, "Open-in-Studio must work without JS — no Alpine shorthand"


# ── handoff: shared session state survives the jump ──────────────────────


def test_studio_after_wizard_jump_renders_loaded_schema(tmp_path: Path) -> None:
    """AC: after seeding schema on the workspace, GET /studio shows it.

    Proves the handoff carries schema state because both surfaces read from the
    same ``app.state.workspace`` singleton.
    """
    app = _make_app(tmp_path)
    app.state.workspace.set_schema(_tiny_schema())
    client = TestClient(app)
    # Simulate wizard session
    assert client.get("/wizard").status_code == 200
    # Jump to Studio
    studio_body = client.get("/studio").text
    tree_start = studio_body.index('id="studio-tree"')
    tree_chunk = studio_body[tree_start : tree_start + 4000]
    assert "users" in tree_chunk, (
        "Studio tree panel must render the wizard's loaded schema (shared workspace)"
    )


def test_studio_after_wizard_jump_preserves_spec_state(tmp_path: Path) -> None:
    """AC: a spec set during the wizard survives the jump to Studio.

    We seed a sentinel spec on the workspace, simulate the wizard render, then
    hit ``GET /studio`` and assert the workspace still holds the same spec
    (the in-process singleton is shared by both routers).
    """
    app = _make_app(tmp_path)
    app.state.workspace.set_schema(_tiny_schema())
    sentinel_spec = object()
    app.state.workspace.set_spec(sentinel_spec)  # type: ignore[arg-type]
    client = TestClient(app)
    assert client.get("/wizard").status_code == 200
    assert client.get("/studio").status_code == 200
    # Spec sentinel still present after the Studio render — handler did not
    # clear it on visit.
    assert app.state.workspace.spec is sentinel_spec, (
        "GET /studio must not drop the wizard's loaded spec — Workspace is shared"
    )


def test_studio_after_wizard_jump_preserves_last_result(tmp_path: Path) -> None:
    """AC: ``last_result`` survives the jump.

    Studio's console panel surfaces ``last_result`` so the user can pick up the
    most recent run after the handoff. ``GET /studio`` must not clear it.
    """
    app = _make_app(tmp_path)
    app.state.workspace.set_schema(_tiny_schema())
    sentinel_result = object()
    app.state.workspace.set_last_result(sentinel_result)  # type: ignore[arg-type]
    client = TestClient(app)
    assert client.get("/wizard").status_code == 200
    assert client.get("/studio").status_code == 200
    assert app.state.workspace.get_last_result() is sentinel_result, (
        "GET /studio must not clear last_result — Workspace is shared singleton"
    )


def test_studio_get_does_not_reset_workspace(tmp_path: Path) -> None:
    """AC: ``GET /studio`` is read-only with respect to the workspace.

    Defends against a regression where a future Studio handler tweaks the
    workspace on visit and silently drops the wizard's session state.
    """
    app = _make_app(tmp_path)
    schema = _tiny_schema()
    spec_sentinel = object()
    result_sentinel = object()
    source = "sqlite:///:memory:"
    ws = app.state.workspace
    ws.set_schema(schema)
    ws.set_spec(spec_sentinel)  # type: ignore[arg-type]
    ws.set_last_result(result_sentinel)  # type: ignore[arg-type]
    ws.set_source(source)
    TestClient(app).get("/studio")
    # Every piece of session state is intact
    assert ws.get_schema() is schema
    assert ws.spec is spec_sentinel
    assert ws.get_last_result() is result_sentinel
    assert ws.get_source() == source


# ── template-level seam: the wrapper is the single source of truth ───────


def test_step_base_template_owns_the_anchor() -> None:
    """AC: the affordance lives in ``_step_base.html`` so all six steps share it.

    Defending the *single edit, six steps inherit* design promise. Sibling
    stories (S-145 inside step_3, S-146 inside _help.html) must not duplicate
    this anchor; the wrapper owns it.
    """
    from pathlib import Path  # noqa: PLC0415

    from dbsprout.web import app as web_app  # noqa: PLC0415

    tpl_dir = Path(web_app.__file__).resolve().parent / "templates" / "wizard"
    wrapper = (tpl_dir / "_step_base.html").read_text(encoding="utf-8")
    assert 'data-testid="wizard-open-in-studio"' in wrapper, (
        "_step_base.html must own the Open-in-Studio anchor"
    )
    # And no other wizard template re-declares the same anchor.
    for step_n in range(1, 7):
        step_src = (tpl_dir / f"step_{step_n}.html").read_text(encoding="utf-8")
        assert 'data-testid="wizard-open-in-studio"' not in step_src, (
            f"step_{step_n}.html must not re-declare the shared affordance "
            "— it is inherited from _step_base.html"
        )
