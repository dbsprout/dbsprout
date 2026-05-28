"""Wizard step-state model (S-142 + S-143 gating).

The Sprint-4 guided wizard walks a new user through six steps:

  1. Connect      — pick DB live / file / DDL / dbml / mermaid
  2. Review       — verify the loaded schema
  3. Configure    — edit the DataSpec
  4. Generate     — run the pipeline
  5. Validate     — integrity report
  6. Insert / Export — write rows to DB or files

Each step's progress is stored as a frozen Pydantic v2 model on the shared
:class:`~dbsprout.web.workspace.Workspace`. Edits flow through
``Workspace.update_wizard_state(**changes)`` which calls
``WizardState.model_copy(update=...)`` — the model is never mutated in place.

S-143 (Wave 2) adds the step-gating contract: :meth:`WizardState.can_advance_from`
returns ``False`` when the workspace lacks the artefact a Next click needs (no
schema → cannot leave step 1; no spec → cannot leave step 3; no run → cannot
leave step 4; no validated flag → cannot leave step 5). The router blocks the
``next`` action and surfaces :class:`~dbsprout.web.errors.WebErrorCode.STEP_GATE_BLOCKED`
with the list of missing artefacts.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

#: Lowest valid step number (matching the router constant in ``routers/wizard.py``).
_MIN_STEP = 1
#: Highest valid step number — the wizard is six steps long.
_MAX_STEP = 6


class WorkspaceLike(Protocol):
    """Minimal accessor surface :meth:`WizardState.can_advance_from` needs.

    Defined locally as a :class:`typing.Protocol` so this module does NOT have
    to import :class:`~dbsprout.web.workspace.Workspace` — which already imports
    :class:`WizardState`. Avoids a circular import and keeps the contract
    explicit (only ``get_schema`` / ``get_spec`` / ``get_last_result`` are
    consulted; the rest of the workspace is irrelevant to gating).
    """

    def get_schema(self) -> object | None: ...

    def get_spec(self) -> object | None: ...

    def get_last_result(self) -> object | None: ...


class WizardState(BaseModel):
    """Frozen state for the 6-step wizard.

    Attributes:
        current_step: The step the user is currently looking at (1..6 inclusive).
        completed_steps: The set of steps the user has flagged as done; drives
            the rail checkmarks and the jump guard (only completed steps may be
            jumped to backwards). ``frozenset`` so the parent model stays
            immutable.
        step_data: Opaque per-step bag for round-tripping form fields. S-143
            stuffs flow-specific keys; this story stores whatever the POST
            handler passes through, minus the control fields (``action``,
            ``target``).
    """

    model_config = ConfigDict(frozen=True)

    current_step: int = Field(default=1, ge=1, le=6)
    completed_steps: frozenset[int] = Field(default_factory=frozenset)
    step_data: dict[int, dict[str, str]] = Field(default_factory=dict)

    # ── S-143 step-gating helpers ────────────────────────────────────────

    def can_advance_from(self, step: int, workspace: WorkspaceLike) -> bool:
        """Return ``True`` when a Next click from ``step`` should be allowed.

        The gating rules mirror the funnel order: each step's Next is only
        valid once the artefact that the *next* step needs is actually on
        the workspace.

        +------+--------------------------------+
        | step | required artefact              |
        +======+================================+
        | 1    | ``workspace.get_schema()``     |
        | 2    | ``workspace.get_schema()``     |
        | 3    | ``workspace.get_spec()``       |
        | 4    | ``workspace.get_last_result()``|
        | 5    | last_result AND validated flag |
        | 6    | (final step — always ``True``) |
        +------+--------------------------------+

        ``step`` outside ``[1, 6]`` always returns ``False`` — the router
        already 404s out-of-range step numbers, but the helper is defensive
        so callers from tests / future routes do not have to pre-validate.
        """
        return not self.missing_for(step, workspace)

    def missing_for(self, step: int, workspace: WorkspaceLike) -> list[str]:
        """Return the list of missing artefacts blocking advance from ``step``.

        Returns an empty list when :meth:`can_advance_from` would return
        ``True``. The list order is stable so error messages and tests can
        assert exact contents (alphabetical when the same step has more
        than one missing artefact).
        """
        if step < _MIN_STEP or step > _MAX_STEP:
            # Out of range — there is nothing meaningful to advance from, so
            # we treat every artefact as "missing"; the boolean caller turns
            # this into ``can_advance_from == False`` cleanly.
            return ["step_out_of_range"]
        if step == _MAX_STEP:
            # Last step — no "next" exists, nothing can be missing.
            return []

        missing: list[str] = []
        if step in (1, 2):
            if workspace.get_schema() is None:
                missing.append("schema")
        elif step == 3:
            if workspace.get_spec() is None:
                missing.append("spec")
        elif step == 4:
            if workspace.get_last_result() is None:
                missing.append("last_result")
        elif step == 5:
            if workspace.get_last_result() is None:
                missing.append("last_result")
            if self.step_data.get(5, {}).get("validated") != "true":
                missing.append("validation")
        # Alphabetical order keeps tests deterministic.
        missing.sort()
        return missing
