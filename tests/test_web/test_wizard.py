"""Wizard router tests (S-142).

The Sprint-4 guided wizard is a 6-step shell (Connect → Review → Configure →
Generate → Validate → Insert/Export). This story ships the **shell + state**
only — step bodies are placeholders that S-143 (Wave 2) replaces.

Tests pin:

* the 3 routes (``GET /wizard``, ``GET /wizard/step/{n}``, ``POST
  /wizard/step/{n}``),
* the rail contract (6 labelled items, stable element ids, checkmarks read from
  ``WizardState.completed_steps``),
* navigation semantics (next / back / jump with the completed-step guard),
* HTMX vs full-page response shape,
* the ``# ── S-142 wizard ──`` region block in ``create_app`` so parallel
  sibling stories don't collide on the same edit window,
* the lazy-import contract (router import doesn't pull
  ``dbsprout.generate.orchestrator`` / ``dbsprout.core.service``).
"""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")

from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi import FastAPI


# ── helpers ──────────────────────────────────────────────────────────────


def _make_app(tmp_path: Path) -> FastAPI:
    from dbsprout.web.app import create_app  # noqa: PLC0415

    return create_app(state_db_path=tmp_path / "state.db")


def _seed_schema(app: FastAPI) -> None:
    """S-143 — sentinel object on ``workspace.schema`` to satisfy the step-1/2 gate.

    The gating helper only checks ``workspace.get_schema() is not None`` — it
    does not inspect the schema shape — so a bare sentinel is enough to let
    pre-existing navigation tests pass without standing up a real schema.
    """
    app.state.workspace.set_schema(object())  # type: ignore[arg-type]


def _seed_spec(app: FastAPI) -> None:
    """Sentinel on ``workspace.spec`` to satisfy the step-3 gate."""
    app.state.workspace.set_spec(object())  # type: ignore[arg-type]


def _seed_last_result(app: FastAPI) -> None:
    """Sentinel on ``workspace.last_result`` to satisfy step-4 / step-5 gates."""
    app.state.workspace.set_last_result(object())  # type: ignore[arg-type]


def _mark_validated(app: FastAPI) -> None:
    """Stash ``validated=true`` on step_data[5] to satisfy the step-5 → 6 gate."""
    ws = app.state.workspace
    state = ws.wizard_state
    ws.update_wizard_state(step_data={**state.step_data, 5: {"validated": "true"}})


# ── GET /wizard — shell ──────────────────────────────────────────────────


def test_wizard_shell_returns_200(tmp_path: Path) -> None:
    resp = TestClient(_make_app(tmp_path)).get("/wizard")
    assert resp.status_code == 200, resp.text


def test_wizard_shell_returns_html(tmp_path: Path) -> None:
    resp = TestClient(_make_app(tmp_path)).get("/wizard")
    assert "text/html" in resp.headers["content-type"]


def test_wizard_shell_is_full_document(tmp_path: Path) -> None:
    body = TestClient(_make_app(tmp_path)).get("/wizard").text.lower()
    assert "<!doctype html>" in body
    assert "<html" in body
    assert "</html>" in body


def test_wizard_shell_has_rail_and_body_slots(tmp_path: Path) -> None:
    body = TestClient(_make_app(tmp_path)).get("/wizard").text
    assert 'id="wizard-rail"' in body
    assert 'id="wizard-body"' in body


def test_wizard_rail_has_six_labelled_steps(tmp_path: Path) -> None:
    body = TestClient(_make_app(tmp_path)).get("/wizard").text
    for n in range(1, 7):
        assert f'data-step="{n}"' in body, f"missing rail item for step {n}"
    # Labels from the canonical S-142 constant
    for label in (
        "Connect",
        "Review",
        "Configure",
        "Generate",
        "Validate",
        "Insert",
    ):
        assert label in body, f"missing rail label {label!r}"


def test_wizard_shell_loads_body_for_current_step(tmp_path: Path) -> None:
    """Fresh workspace → current_step=1 → body shows Step 1 content."""
    body = TestClient(_make_app(tmp_path)).get("/wizard").text
    body_start = body.index('id="wizard-body"')
    body_chunk = body[body_start : body_start + 4000]
    assert "Step 1" in body_chunk


def test_wizard_rail_marks_completed_steps(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    app.state.workspace.update_wizard_state(completed_steps=frozenset({1, 2}), current_step=3)
    body = TestClient(app).get("/wizard").text
    assert 'data-completed="1"' in body
    assert 'data-completed="2"' in body
    # 3 is the current step, not yet completed
    assert 'data-completed="3"' not in body


def test_wizard_rail_marks_current_step(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    app.state.workspace.update_wizard_state(current_step=4)
    body = TestClient(app).get("/wizard").text
    assert 'data-current="4"' in body


# ── GET /wizard/step/{n} — fragments ─────────────────────────────────────


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_get_step_fragment_200(tmp_path: Path, n: int) -> None:
    resp = TestClient(_make_app(tmp_path)).get(f"/wizard/step/{n}")
    assert resp.status_code == 200, resp.text


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_get_step_fragment_mentions_step_number(tmp_path: Path, n: int) -> None:
    body = TestClient(_make_app(tmp_path)).get(f"/wizard/step/{n}").text
    assert f"Step {n}" in body


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_get_step_fragment_has_back_and_next(tmp_path: Path, n: int) -> None:
    body = TestClient(_make_app(tmp_path)).get(f"/wizard/step/{n}").text
    # Back is suppressed only for step 1, Next is suppressed only for step 6
    if n > 1:
        assert "back" in body.lower()
    if n < 6:
        assert "next" in body.lower()


def test_get_step_fragment_is_not_full_document(tmp_path: Path) -> None:
    """The body fragment is HTMX-friendly — no <html> wrapper."""
    body = TestClient(_make_app(tmp_path)).get("/wizard/step/3").text.lower()
    assert "<!doctype html>" not in body


@pytest.mark.parametrize("n", [0, 7, -1, 99])
def test_get_step_fragment_out_of_range_404(tmp_path: Path, n: int) -> None:
    resp = TestClient(_make_app(tmp_path)).get(f"/wizard/step/{n}")
    assert resp.status_code == 404


# ── POST /wizard/step/{n} — navigation ───────────────────────────────────


def test_post_next_advances_and_marks_completed(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed_schema(app)  # S-143: step-1 gate needs schema for ``next``.
    client = TestClient(app)
    resp = client.post(
        "/wizard/step/1",
        data={"action": "next"},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200, resp.text
    assert app.state.workspace.wizard_state.current_step == 2
    assert 1 in app.state.workspace.wizard_state.completed_steps


def test_post_next_at_step_six_clamps(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    app.state.workspace.update_wizard_state(current_step=6)
    client = TestClient(app)
    resp = client.post(
        "/wizard/step/6",
        data={"action": "next"},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200
    assert app.state.workspace.wizard_state.current_step == 6
    assert 6 in app.state.workspace.wizard_state.completed_steps


def test_post_back_rewinds(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    app.state.workspace.update_wizard_state(current_step=4, completed_steps=frozenset({1, 2, 3}))
    client = TestClient(app)
    resp = client.post(
        "/wizard/step/4",
        data={"action": "back"},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200
    assert app.state.workspace.wizard_state.current_step == 3
    # Back must NOT uncomplete steps
    assert app.state.workspace.wizard_state.completed_steps == frozenset({1, 2, 3})


def test_post_back_at_step_one_clamps(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    client = TestClient(app)
    resp = client.post(
        "/wizard/step/1",
        data={"action": "back"},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200
    assert app.state.workspace.wizard_state.current_step == 1


def test_post_jump_to_completed_step_allowed(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    app.state.workspace.update_wizard_state(current_step=4, completed_steps=frozenset({1, 2, 3}))
    client = TestClient(app)
    resp = client.post(
        "/wizard/step/4",
        data={"action": "jump", "target": "2"},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200
    assert app.state.workspace.wizard_state.current_step == 2


def test_post_jump_to_uncompleted_future_step_rejected(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    client = TestClient(app)
    resp = client.post(
        "/wizard/step/1",
        data={"action": "jump", "target": "5"},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 400
    assert app.state.workspace.wizard_state.current_step == 1


def test_post_jump_without_target_rejected(tmp_path: Path) -> None:
    """``action=jump`` without a ``target`` form field → 400."""
    app = _make_app(tmp_path)
    client = TestClient(app)
    resp = client.post(
        "/wizard/step/1",
        data={"action": "jump"},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 400


def test_post_jump_with_invalid_target_rejected(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    client = TestClient(app)
    resp = client.post(
        "/wizard/step/1",
        data={"action": "jump", "target": "notanint"},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 400


def test_post_jump_with_out_of_range_target_rejected(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    client = TestClient(app)
    resp = client.post(
        "/wizard/step/1",
        data={"action": "jump", "target": "9"},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 400


def test_post_unknown_action_rejected(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    client = TestClient(app)
    resp = client.post(
        "/wizard/step/1",
        data={"action": "warp"},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 400


def test_post_extra_form_fields_round_trip_into_step_data(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed_schema(app)  # S-143: step-1 gate needs schema for ``next``.
    client = TestClient(app)
    client.post(
        "/wizard/step/1",
        data={"action": "next", "db_kind": "postgres", "dsn_hint": "pg://x"},
        headers={"HX-Request": "true"},
    )
    bag = app.state.workspace.wizard_state.step_data.get(1)
    assert bag is not None
    assert bag["db_kind"] == "postgres"
    assert bag["dsn_hint"] == "pg://x"
    # ``action`` and ``target`` are control fields — not stored
    assert "action" not in bag
    assert "target" not in bag


def test_post_step_out_of_range_404(tmp_path: Path) -> None:
    resp = TestClient(_make_app(tmp_path)).post(
        "/wizard/step/0",
        data={"action": "next"},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 404


def test_post_htmx_returns_fragment(tmp_path: Path) -> None:
    """HTMX requests get the new current step's body fragment back."""
    app = _make_app(tmp_path)
    _seed_schema(app)  # S-143: step-1 gate.
    client = TestClient(app)
    resp = client.post(
        "/wizard/step/1",
        data={"action": "next"},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200
    body = resp.text.lower()
    assert "<!doctype html>" not in body
    assert "step 2" in body


def test_post_non_htmx_redirects_to_wizard(tmp_path: Path) -> None:
    """Full-page POST → 303 back to /wizard (PRG pattern)."""
    app = _make_app(tmp_path)
    _seed_schema(app)  # S-143: step-1 gate.
    client = TestClient(app)
    resp = client.post(
        "/wizard/step/1",
        data={"action": "next"},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/wizard"


# ── shell <-> step labels stay in sync ───────────────────────────────────


def test_step_labels_constant_has_six_entries() -> None:
    from dbsprout.web.routers.wizard import STEP_LABELS  # noqa: PLC0415

    assert len(STEP_LABELS) == 6
    assert STEP_LABELS[0].lower().startswith("connect")
    assert STEP_LABELS[5].lower().startswith("insert")


# ── router registration / region block ───────────────────────────────────


def test_wizard_router_importable() -> None:
    from fastapi import APIRouter  # noqa: PLC0415

    from dbsprout.web.routers.wizard import wizard_router  # noqa: PLC0415

    assert isinstance(wizard_router, APIRouter)


def test_wizard_routes_registered_on_app(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    paths = {getattr(r, "path", "") for r in app.routes}
    assert "/wizard" in paths
    assert "/wizard/step/{n}" in paths


def test_wizard_region_block_in_app_factory() -> None:
    """S-143 will plug step bodies in; the region delimiters keep the merge clean."""
    from pathlib import Path  # noqa: PLC0415

    from dbsprout.web import app as web_app  # noqa: PLC0415

    src = Path(web_app.__file__).read_text(encoding="utf-8")
    assert "# ── S-142 wizard ──" in src
    assert "# ── end S-142 ──" in src
    assert "wizard_router" in src


# ── lazy-import contract ─────────────────────────────────────────────────


# ── S-143 step-gating contract ───────────────────────────────────────────


def test_post_next_blocks_step_one_without_schema(tmp_path: Path) -> None:
    """Step 1 → 2 needs ``workspace.get_schema()`` populated.

    No ``HX-Request`` header → JSON envelope branch through ``raise_web_error``.
    """
    app = _make_app(tmp_path)
    client = TestClient(app)
    resp = client.post(
        "/wizard/step/1",
        data={"action": "next"},
        follow_redirects=False,
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["detail"]["code"] == "STEP_GATE_BLOCKED"
    assert body["detail"]["missing"] == ["schema"]
    assert body["detail"]["step"] == 1
    # State must NOT have advanced.
    assert app.state.workspace.wizard_state.current_step == 1


def test_post_next_blocks_step_two_without_schema(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    app.state.workspace.update_wizard_state(current_step=2, completed_steps=frozenset({1}))
    client = TestClient(app)
    resp = client.post(
        "/wizard/step/2",
        data={"action": "next"},
        follow_redirects=False,
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["missing"] == ["schema"]


def test_post_next_blocks_step_three_without_spec(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed_schema(app)
    app.state.workspace.update_wizard_state(current_step=3, completed_steps=frozenset({1, 2}))
    client = TestClient(app)
    resp = client.post(
        "/wizard/step/3",
        data={"action": "next"},
        follow_redirects=False,
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["missing"] == ["spec"]


def test_post_next_blocks_step_four_without_last_result(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed_schema(app)
    _seed_spec(app)
    app.state.workspace.update_wizard_state(current_step=4, completed_steps=frozenset({1, 2, 3}))
    client = TestClient(app)
    resp = client.post(
        "/wizard/step/4",
        data={"action": "next"},
        follow_redirects=False,
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["missing"] == ["last_result"]


def test_post_next_blocks_step_five_without_validated_flag(tmp_path: Path) -> None:
    """Step 5 → 6 needs a validated run (``step_data[5]['validated'] == 'true'``)."""
    app = _make_app(tmp_path)
    _seed_schema(app)
    _seed_spec(app)
    _seed_last_result(app)
    app.state.workspace.update_wizard_state(current_step=5, completed_steps=frozenset({1, 2, 3, 4}))
    client = TestClient(app)
    resp = client.post(
        "/wizard/step/5",
        data={"action": "next"},
        follow_redirects=False,
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["missing"] == ["validation"]


def test_post_next_step_five_with_form_validated_flag_advances(tmp_path: Path) -> None:
    """The Step 5 template POSTs ``validated=true`` alongside ``action=next``.

    That flag is merged into ``step_data[5]`` *before* the gate runs, so a
    user clicking Next on a freshly-validated panel advances cleanly.
    """
    app = _make_app(tmp_path)
    _seed_schema(app)
    _seed_spec(app)
    _seed_last_result(app)
    app.state.workspace.update_wizard_state(current_step=5, completed_steps=frozenset({1, 2, 3, 4}))
    client = TestClient(app)
    resp = client.post(
        "/wizard/step/5",
        data={"action": "next", "validated": "true"},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200
    assert app.state.workspace.wizard_state.current_step == 6


def test_post_next_step_five_with_prepoulated_flag_advances(tmp_path: Path) -> None:
    """The ``validated=true`` flag persisted from an earlier visit also unlocks Next."""
    app = _make_app(tmp_path)
    _seed_schema(app)
    _seed_spec(app)
    _seed_last_result(app)
    app.state.workspace.update_wizard_state(current_step=5, completed_steps=frozenset({1, 2, 3, 4}))
    _mark_validated(app)
    client = TestClient(app)
    resp = client.post(
        "/wizard/step/5",
        data={"action": "next"},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200
    assert app.state.workspace.wizard_state.current_step == 6


def test_post_next_step_six_never_blocks(tmp_path: Path) -> None:
    """Step 6 has no next step — Next always clamps to 6 and never gates."""
    app = _make_app(tmp_path)
    app.state.workspace.update_wizard_state(current_step=6)
    client = TestClient(app)
    resp = client.post(
        "/wizard/step/6",
        data={"action": "next"},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200


def test_post_back_never_blocks_on_gate(tmp_path: Path) -> None:
    """The gate only applies to ``next`` — ``back`` ignores artefact state."""
    app = _make_app(tmp_path)
    app.state.workspace.update_wizard_state(current_step=4, completed_steps=frozenset({1, 2, 3}))
    client = TestClient(app)
    resp = client.post(
        "/wizard/step/4",
        data={"action": "back"},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200


def test_post_step_gate_blocked_htmx_returns_fragment(tmp_path: Path) -> None:
    """An HTMX Next click that gets gated returns the HTML error fragment, not JSON."""
    app = _make_app(tmp_path)
    client = TestClient(app)
    resp = client.post(
        "/wizard/step/1",
        data={"action": "next"},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 400
    # ``raise_web_error`` renders ``error_fragment.html`` when ``HX-Request: true``,
    # so the response is HTML carrying the structured ``data-code`` attribute.
    assert "text/html" in resp.headers["content-type"]
    assert "STEP_GATE_BLOCKED" in resp.text


# ── S-143 step bodies — each template embeds the real Studio surface ─────


def test_step_one_body_includes_connect_and_load_endpoints(tmp_path: Path) -> None:
    body = TestClient(_make_app(tmp_path)).get("/wizard/step/1").text
    assert "/api/connect" in body
    assert "/api/schema/load" in body


def test_step_two_body_includes_schema_erd_endpoint(tmp_path: Path) -> None:
    body = TestClient(_make_app(tmp_path)).get("/wizard/step/2").text
    assert "/api/schema/erd" in body


def test_step_three_body_includes_spec_grid_endpoint(tmp_path: Path) -> None:
    body = TestClient(_make_app(tmp_path)).get("/wizard/step/3").text
    # GET /api/spec with Accept: text/html returns the spec grid fragment.
    assert "/api/spec" in body
    # Method picker must also be present so per-row method swaps work.
    assert "method_picker" in body or "method-picker" in body or "open-method-picker" in body


def test_step_four_body_includes_generate_console(tmp_path: Path) -> None:
    body = TestClient(_make_app(tmp_path)).get("/wizard/step/4").text
    # The shared _studio_console.html surfaces the generate form, cancel, and console.
    assert "studio-generate-btn" in body
    assert "studio-cancel-button" in body
    assert "studio-console-status" in body


def test_step_five_body_triggers_validate_endpoint(tmp_path: Path) -> None:
    body = TestClient(_make_app(tmp_path)).get("/wizard/step/5").text
    assert "/api/validate" in body
    # Hidden ``validated`` flag must be present in the body so the wizard form
    # sends it through to the gate when the user clicks Next.
    assert 'name="validated"' in body
    assert 'value="true"' in body


def test_step_six_body_includes_insert_export_paths(tmp_path: Path) -> None:
    body = TestClient(_make_app(tmp_path)).get("/wizard/step/6").text
    assert "/api/insert/preview" in body
    assert "/api/export" in body
    assert "write-guard-modal" in body


def test_wizard_router_no_eager_generation_import() -> None:
    probe = (
        "import sys\n"
        "import dbsprout.web.routers.wizard  # noqa: F401\n"
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
        "importing dbsprout.web.routers.wizard eagerly imported generation/core "
        f"modules: {result.stdout.strip()}"
    )
