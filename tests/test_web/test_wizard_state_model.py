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


# ── S-143: step-gating helpers ───────────────────────────────────────────


class _StubWorkspace:
    """Minimal stand-in for the protocol the gating helpers consume.

    Mirrors the four ``get_*`` accessors :meth:`WizardState.can_advance_from`
    inspects (``get_schema``, ``get_spec``, ``get_last_result``) — no need to
    instantiate the full :class:`~dbsprout.web.workspace.Workspace` here.
    """

    def __init__(
        self,
        *,
        schema: object | None = None,
        spec: object | None = None,
        last_result: object | None = None,
    ) -> None:
        self._schema = schema
        self._spec = spec
        self._last_result = last_result

    def get_schema(self) -> object | None:
        return self._schema

    def get_spec(self) -> object | None:
        return self._spec

    def get_last_result(self) -> object | None:
        return self._last_result


def test_can_advance_from_step_one_needs_schema() -> None:
    state = WizardState(current_step=1)
    assert state.can_advance_from(1, _StubWorkspace()) is False
    assert state.can_advance_from(1, _StubWorkspace(schema=object())) is True


def test_can_advance_from_step_two_needs_schema() -> None:
    state = WizardState(current_step=2)
    assert state.can_advance_from(2, _StubWorkspace()) is False
    assert state.can_advance_from(2, _StubWorkspace(schema=object())) is True


def test_can_advance_from_step_three_needs_spec() -> None:
    state = WizardState(current_step=3)
    assert state.can_advance_from(3, _StubWorkspace(schema=object())) is False
    assert state.can_advance_from(3, _StubWorkspace(schema=object(), spec=object())) is True


def test_can_advance_from_step_four_needs_last_result() -> None:
    state = WizardState(current_step=4)
    assert state.can_advance_from(4, _StubWorkspace(spec=object())) is False
    assert state.can_advance_from(4, _StubWorkspace(spec=object(), last_result=object())) is True


def test_can_advance_from_step_five_needs_validated_flag() -> None:
    """Step 5 → 6 also requires a validated run was reviewed (flag in step_data)."""
    base = WizardState(current_step=5)
    # No validated flag — blocked even with last_result.
    assert base.can_advance_from(5, _StubWorkspace(last_result=object())) is False
    # Flag set but no last_result — still blocked.
    flagged = base.model_copy(update={"step_data": {5: {"validated": "true"}}})
    assert flagged.can_advance_from(5, _StubWorkspace()) is False
    # Both present — allowed.
    assert flagged.can_advance_from(5, _StubWorkspace(last_result=object())) is True


def test_can_advance_from_step_six_always_true() -> None:
    """Step 6 is the final step — there is nothing past it to gate."""
    state = WizardState(current_step=6)
    assert state.can_advance_from(6, _StubWorkspace()) is True


def test_can_advance_from_out_of_range_returns_false() -> None:
    state = WizardState()
    assert state.can_advance_from(0, _StubWorkspace()) is False
    assert state.can_advance_from(7, _StubWorkspace()) is False


def test_missing_for_step_one_reports_schema() -> None:
    state = WizardState()
    assert state.missing_for(1, _StubWorkspace()) == ["schema"]
    assert state.missing_for(1, _StubWorkspace(schema=object())) == []


def test_missing_for_step_three_reports_spec() -> None:
    state = WizardState()
    assert state.missing_for(3, _StubWorkspace(schema=object())) == ["spec"]


def test_missing_for_step_four_reports_last_result() -> None:
    state = WizardState()
    assert state.missing_for(4, _StubWorkspace(spec=object())) == ["last_result"]


def test_missing_for_step_five_reports_validation_or_last_result() -> None:
    state = WizardState()
    # No last_result and no flag → reports both.
    assert sorted(state.missing_for(5, _StubWorkspace())) == ["last_result", "validation"]
    # last_result set, flag missing → reports validation only.
    assert state.missing_for(5, _StubWorkspace(last_result=object())) == ["validation"]


def test_missing_for_step_six_is_empty() -> None:
    state = WizardState(current_step=6)
    assert state.missing_for(6, _StubWorkspace()) == []
