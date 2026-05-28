"""WizardState model contract (S-142).

The 6-step wizard's session state lives on the workspace as a *frozen* Pydantic
v2 model so it follows the same immutability discipline as the rest of the
domain (``DataSpec``, ``DatabaseSchema``).

S-143 will round-trip ``step_data`` for the real flows, so the contract for
defaults, bounds, and ``model_copy`` must be locked here before any code is
written.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")

from pydantic import ValidationError

from dbsprout.web.wizard_state import WizardState


def test_wizard_state_defaults() -> None:
    state = WizardState()
    assert state.current_step == 1
    assert state.completed_steps == frozenset()
    assert state.step_data == {}


def test_wizard_state_is_frozen() -> None:
    state = WizardState()
    with pytest.raises(ValidationError):
        state.current_step = 4  # type: ignore[misc]


def test_wizard_state_current_step_lower_bound() -> None:
    with pytest.raises(ValidationError):
        WizardState(current_step=0)


def test_wizard_state_current_step_upper_bound() -> None:
    with pytest.raises(ValidationError):
        WizardState(current_step=7)


def test_wizard_state_current_step_accepts_all_six() -> None:
    for n in range(1, 7):
        assert WizardState(current_step=n).current_step == n


def test_wizard_state_model_copy_returns_new_instance() -> None:
    state = WizardState()
    updated = state.model_copy(update={"current_step": 2})
    assert updated.current_step == 2
    assert state.current_step == 1
    assert updated is not state


def test_wizard_state_completed_steps_is_frozenset() -> None:
    state = WizardState(completed_steps=frozenset({1, 2}))
    assert isinstance(state.completed_steps, frozenset)
    assert state.completed_steps == frozenset({1, 2})


def test_wizard_state_step_data_round_trip() -> None:
    """S-143 forward-compat: per-step bag round-trips through model_copy."""
    state = WizardState(step_data={1: {"db_kind": "postgres"}})
    assert state.step_data[1] == {"db_kind": "postgres"}
    next_state = state.model_copy(update={"step_data": {**state.step_data, 2: {"ok": "1"}}})
    assert next_state.step_data[1] == {"db_kind": "postgres"}
    assert next_state.step_data[2] == {"ok": "1"}
