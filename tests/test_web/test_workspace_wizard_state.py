"""Workspace.wizard_state contract (S-142).

The 6-step wizard's state lives on the shared :class:`Workspace` so the
single-user dashboard can survive across stateless HTTP requests (mirrors how
``spec`` and ``last_result`` already live there). Updates flow through
``update_wizard_state(**changes)`` which goes through ``model_copy`` — frozen
models are never mutated in place.

These tests pin:

* default state on construction (``WizardState()``),
* the immutable update path,
* ``reset()`` clearing wizard state back to defaults.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")

from dbsprout.web.wizard_state import WizardState
from dbsprout.web.workspace import Workspace


def test_workspace_exposes_wizard_state_default() -> None:
    ws = Workspace()
    assert isinstance(ws.wizard_state, WizardState)
    assert ws.wizard_state.current_step == 1
    assert ws.wizard_state.completed_steps == frozenset()
    assert ws.wizard_state.step_data == {}


def test_workspace_update_wizard_state_advances() -> None:
    ws = Workspace()
    new = ws.update_wizard_state(current_step=2)
    assert new.current_step == 2
    assert ws.wizard_state is new
    assert ws.wizard_state.current_step == 2


def test_workspace_update_wizard_state_is_immutable_swap() -> None:
    ws = Workspace()
    original = ws.wizard_state
    ws.update_wizard_state(current_step=3)
    assert original.current_step == 1
    assert ws.wizard_state is not original


def test_workspace_update_wizard_state_round_trips_step_data() -> None:
    ws = Workspace()
    ws.update_wizard_state(step_data={1: {"foo": "bar"}})
    assert ws.wizard_state.step_data == {1: {"foo": "bar"}}


def test_workspace_update_wizard_state_marks_completed() -> None:
    ws = Workspace()
    ws.update_wizard_state(completed_steps=frozenset({1, 2}), current_step=3)
    assert ws.wizard_state.completed_steps == frozenset({1, 2})
    assert ws.wizard_state.current_step == 3


def test_workspace_reset_clears_wizard_state() -> None:
    ws = Workspace()
    ws.update_wizard_state(current_step=4, completed_steps=frozenset({1, 2, 3}))
    ws.reset()
    assert ws.wizard_state.current_step == 1
    assert ws.wizard_state.completed_steps == frozenset()
    assert ws.wizard_state.step_data == {}
