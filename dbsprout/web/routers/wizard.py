"""Guided wizard router (S-142).

The Sprint-4 guided wizard is a 6-step shell:

  1. Connect      — pick DB live / file / DDL / dbml / mermaid
  2. Review       — verify the loaded schema
  3. Configure    — edit the DataSpec
  4. Generate     — run the pipeline
  5. Validate     — integrity report
  6. Insert / Export — write rows to DB or files

This story (S-142) ships the **shell + step state** only:

* ``GET /wizard`` — full-page shell with the 6-step left rail + a body slot
  populated with the current step's placeholder.
* ``GET /wizard/step/{n}`` — HTMX fragment for the body of step ``n``.
* ``POST /wizard/step/{n}`` — persist step state to
  :class:`~dbsprout.web.workspace.Workspace` and advance / rewind / jump.

S-143 (Wave 2) wires the real per-step flows by replacing each
``templates/wizard/step_{n}.html`` placeholder. The rail, navigation handling,
and ``STEP_LABELS`` constant stay here so S-143 only edits the step body
contents.

This router never imports the orchestrator / core service — it is import-cheap
and `dbsprout serve` lazy-imports it through ``create_app``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import RedirectResponse, Response

from dbsprout.web.errors import raise_web_error, web_error_step_gate_blocked

if TYPE_CHECKING:
    from fastapi.templating import Jinja2Templates

    from dbsprout.web.workspace import Workspace

#: Canonical, **frozen** step labels. The shell (rail) and the step bodies both
#: read from this tuple so S-143 cannot drift. Order matches step number 1..6.
STEP_LABELS: tuple[str, ...] = (
    "Connect",
    "Review",
    "Configure",
    "Generate",
    "Validate",
    "Insert / Export",
)

_MIN_STEP = 1
_MAX_STEP = len(STEP_LABELS)

#: Control fields that drive navigation; they MUST NOT leak into
#: ``WizardState.step_data`` round-tripping.
_CONTROL_FIELDS = frozenset({"action", "target"})

wizard_router = APIRouter()


# ── typed accessors ──────────────────────────────────────────────────────


def _workspace(request: Request) -> Workspace:
    """Typed accessor for the shared session workspace wired in ``app.py``."""
    return cast("Workspace", request.app.state.workspace)


def _templates(request: Request) -> Jinja2Templates:
    """Typed accessor for the shared Jinja2 environment wired in ``app.py``."""
    return cast("Jinja2Templates", request.app.state.templates)


def _is_htmx(request: Request) -> bool:
    """``True`` when the request originated from an HTMX swap."""
    return request.headers.get("HX-Request", "").lower() == "true"


def _clamp_step(n: int) -> int:
    """Squeeze ``n`` into the valid step range — used for next/back nav."""
    return max(_MIN_STEP, min(_MAX_STEP, n))


def _validate_step_in_url(n: int) -> None:
    """Raise 404 when the URL path includes an out-of-range step number."""
    if n < _MIN_STEP or n > _MAX_STEP:
        raise HTTPException(status_code=404, detail="step out of range")


def _shell_context(request: Request) -> dict[str, object]:
    """Render context shared between full shell + body-only responses.

    The shell ``{% include %}``s the current step's body template directly, so
    ``has_back`` / ``has_next`` (consumed by the shared ``_step_base.html``
    wrapper) must already be in scope — Jinja ``include`` inherits the parent
    template's context, but does NOT call the body endpoint's enricher.
    """
    ws = _workspace(request)
    state = ws.wizard_state
    return {
        "step_labels": STEP_LABELS,
        "min_step": _MIN_STEP,
        "max_step": _MAX_STEP,
        "current_step": state.current_step,
        "completed_steps": sorted(state.completed_steps),
        "step": state.current_step,
        "has_back": state.current_step > _MIN_STEP,
        "has_next": state.current_step < _MAX_STEP,
    }


def _step_context(request: Request, step: int) -> dict[str, object]:
    """Render context for a single step body (used by GET / POST handlers)."""
    base = _shell_context(request)
    base["step"] = step
    base["has_back"] = step > _MIN_STEP
    base["has_next"] = step < _MAX_STEP
    return base


# ── GET /wizard — full shell ─────────────────────────────────────────────


@wizard_router.get("/wizard", response_class=Response)
async def wizard_shell(request: Request) -> Response:
    """Render the wizard shell — rail + body slot populated with current step."""
    ctx = _shell_context(request)
    return _templates(request).TemplateResponse(
        request, "wizard/shell.html", {**ctx, "active": "wizard"}
    )


# ── GET /wizard/step/{n} — body fragment ─────────────────────────────────


@wizard_router.get("/wizard/step/{n}", response_class=Response)
async def wizard_step_body(request: Request, n: int) -> Response:
    """Return the HTMX-friendly body fragment for step ``n``."""
    _validate_step_in_url(n)
    ctx = _step_context(request, n)
    return _templates(request).TemplateResponse(request, f"wizard/step_{n}.html", ctx)


# ── POST /wizard/step/{n} — navigation ───────────────────────────────────


def _merge_step_data(
    existing: dict[int, dict[str, str]],
    step: int,
    extras: dict[str, str],
) -> dict[int, dict[str, str]]:
    """Return a new ``step_data`` mapping with ``extras`` merged onto ``step``.

    The frozen ``WizardState`` is replaced wholesale via ``model_copy``; this
    helper keeps the immutable shape clean by never mutating the caller's
    ``existing`` dict.
    """
    if not extras:
        return existing
    merged_step = {**existing.get(step, {}), **extras}
    return {**existing, step: merged_step}


async def _read_extras(
    request: Request,
) -> dict[str, str]:
    """Strip control fields out of the form body to feed ``step_data``."""
    form = await request.form()
    extras: dict[str, str] = {}
    for key, value in form.multi_items():
        if key in _CONTROL_FIELDS:
            continue
        # FastAPI form values are str | UploadFile; we only round-trip text.
        if isinstance(value, str):
            extras[key] = value
    return extras


def _validate_jump_target(
    raw_target: str | None,
    completed: frozenset[int],
) -> int:
    """Parse + permission-check a jump target.

    Returns the target step on success. Raises 400 ``HTTPException`` when the
    target is malformed, out of range, or not in ``completed``.
    """
    if raw_target is None:
        raise HTTPException(status_code=400, detail="target required for jump")
    try:
        target = int(raw_target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="target must be an integer") from exc
    if target < _MIN_STEP or target > _MAX_STEP:
        raise HTTPException(status_code=400, detail="target out of range")
    if target not in completed:
        raise HTTPException(
            status_code=400, detail="cannot jump to a step that is not yet completed"
        )
    return target


@wizard_router.post("/wizard/step/{n}", response_class=Response)
async def wizard_step_submit(
    request: Request,
    n: int,
    action: str = Form(...),
    target: str | None = Form(default=None),
) -> Response:
    """Persist step ``n``'s submission and navigate to the next surface.

    Form contract:

    * ``action`` — ``"next"`` · ``"back"`` · ``"jump"`` (others → 400).
    * ``target`` — required when ``action="jump"``; the target step number.
    * any extra fields are merged into ``WizardState.step_data[n]``.

    HTMX request → 200 with the new step's body fragment.
    Full-page request → 303 redirect to ``/wizard`` (Post-Redirect-Get).
    """
    _validate_step_in_url(n)
    ws = _workspace(request)
    state = ws.wizard_state
    extras = await _read_extras(request)

    if action == "next":
        # S-143 step-gating: the gate evaluates the workspace + any extras the
        # current click carries (notably Step 5's hidden ``validated=true``
        # field), so we fold extras into a *probe* state first and only then
        # ask ``can_advance_from``. Persisting happens after the gate passes,
        # so a blocked Next never mutates the workspace.
        probe = state.model_copy(
            update={"step_data": _merge_step_data(state.step_data, n, extras)},
        )
        if not probe.can_advance_from(n, ws):
            missing = probe.missing_for(n, ws)
            return raise_web_error(
                request,
                web_error_step_gate_blocked(step=n, missing=missing),
            )
        new_completed = state.completed_steps | {n}
        new_current = _clamp_step(n + 1)
    elif action == "back":
        new_completed = state.completed_steps
        new_current = _clamp_step(n - 1)
    elif action == "jump":
        new_current = _validate_jump_target(target, state.completed_steps)
        new_completed = state.completed_steps
    else:
        raise HTTPException(status_code=400, detail=f"unknown action {action!r}")

    ws.update_wizard_state(
        current_step=new_current,
        completed_steps=new_completed,
        step_data=_merge_step_data(state.step_data, n, extras),
    )

    if _is_htmx(request):
        return await wizard_step_body(request, new_current)
    return RedirectResponse(url="/wizard", status_code=303)
