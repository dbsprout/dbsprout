"""Wizard step-state model (S-142).

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

S-143 (Wave 2) populates ``step_data`` with the per-flow keys; this story
ships the shell + storage contract only, so the model is intentionally
schema-light (an opaque ``dict[int, dict[str, str]]`` bag) and ``step_data``
keys are constrained only by the route handlers.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


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
