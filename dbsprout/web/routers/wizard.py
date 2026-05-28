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

from dbsprout.web.errors import (
    raise_web_error,
    web_error_llm_unavailable,
    web_error_no_schema,
    web_error_step_gate_blocked,
)

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


#: Step number that owns the auto-heuristic spec entry hook (S-145). Named
#: rather than inlined so future re-ordering of the rail surfaces a single
#: edit point.
_CONFIGURE_STEP = 3


def _ensure_step3_spec(ws: Workspace) -> None:
    """Pre-populate ``workspace.spec`` for Step 3 (Configure) — S-145.

    Order of preference (each step is best-effort, never raises into the
    request path):

    1. ``workspace.spec`` already set → noop (the LLM opt-in path, a prior
       hydrate, or a Studio edit already filled it in).
    2. ``workspace.hydrate_from_cache(schema.schema_hash())`` — re-use the
       S-122 disk cache when available.
    3. ``heuristic_fallback(schema)`` — the existing Sprint-2 mapping,
       same code path ``GET /api/spec`` (S-118) uses on demand.

    No schema on the workspace → noop. The lazy import keeps the wizard
    router import-light: importing the heuristic analyzer pulls in the
    Sprint-2 ``map_columns`` graph but never the LLM provider stack.
    """
    if ws.get_spec() is not None:
        return
    schema = ws.get_schema()
    if schema is None:
        return
    # Step 2: cache hit short-circuits the build.
    if ws.hydrate_from_cache(schema.schema_hash()):
        return
    # Step 3: build via the existing offline / no-LLM path.
    from dbsprout.spec.analyzer import heuristic_fallback  # noqa: PLC0415

    ws.set_spec(heuristic_fallback(schema))


@wizard_router.get("/wizard/step/{n}", response_class=Response)
async def wizard_step_body(request: Request, n: int) -> Response:
    """Return the HTMX-friendly body fragment for step ``n``."""
    _validate_step_in_url(n)
    # S-145: when the user lands on Step 3 (Configure), make sure a spec is
    # already on the workspace so the grid renders immediately — no spinner
    # waiting on heuristic generation. The helper is a no-op when the spec
    # is already populated or when no schema is loaded yet.
    if n == _CONFIGURE_STEP:
        _ensure_step3_spec(_workspace(request))
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


# ── POST /wizard/step/3/llm-spec — opt-in LLM path (S-145) ───────────────


#: Construction-time failures we translate to ``LLM_UNAVAILABLE`` (503).
#: ``ImportError`` covers ``llama-cpp-python`` missing, ``RuntimeError`` covers
#: "no GGUF model on disk", ``OSError`` covers cache-dir / file permission
#: issues — none of them are caller-actionable input errors, so they get the
#: capability-gap envelope rather than a 4xx.
_LLM_BOOT_ERRORS: tuple[type[BaseException], ...] = (
    ImportError,
    RuntimeError,
    OSError,
)


@wizard_router.post("/wizard/step/3/llm-spec", response_model=None)
async def wizard_step3_llm_spec(request: Request) -> Response | dict[str, object]:
    """Opt-in LLM spec build for Step 3 (Configure) — S-145.

    The heuristic spec is already in place from the GET-side entry hook
    (:func:`_ensure_step3_spec`); this endpoint lets the user trade time for
    a (potentially) richer spec by invoking the existing embedded LLM
    provider chain (``EmbeddedProvider`` → ``SpecAnalyzer`` → ``analyze``).
    The new spec replaces the heuristic one on the workspace and is also
    persisted to the disk cache so a page refresh keeps the LLM result.

    Failure modes:

    * No schema loaded → 409 ``NO_SCHEMA`` envelope.
    * Provider construction fails (``llama-cpp-python`` missing, no GGUF
      model, …) → 503 ``LLM_UNAVAILABLE`` envelope; the heuristic spec on
      the workspace is left untouched.

    Response shape:

    * HTMX caller (``HX-Request: true``) → 200 with the re-rendered Step 3
      body fragment so the swap target gets the fresh grid.
    * JSON caller → 200 ``{"ok": true, "schema_hash": "<hash>"}``.

    No model-selection UI is surfaced here (out of scope, see Story
    S-145 § Technical Notes); the existing single embedded-provider chain
    is used as-is.
    """
    ws = _workspace(request)
    schema = ws.get_schema()
    if schema is None:
        return raise_web_error(request, web_error_no_schema())

    # Lazy imports keep the wizard router import-light — the LLM stack is only
    # paid for when the user explicitly opts into the slower path.
    try:
        from dbsprout.spec.analyzer import SpecAnalyzer  # noqa: PLC0415
        from dbsprout.spec.providers.embedded import (  # noqa: PLC0415
            EmbeddedProvider,
        )

        provider = EmbeddedProvider()
        analyzer = SpecAnalyzer(provider)
        new_spec = analyzer.analyze(schema)
    except _LLM_BOOT_ERRORS as exc:
        return raise_web_error(
            request,
            web_error_llm_unavailable(str(exc) or type(exc).__name__),
            original=exc,
        )

    ws.set_spec(new_spec)
    # S-122: persist the LLM-built spec so a reload short-circuits to it.
    ws.persist_spec()

    if _is_htmx(request):
        return await wizard_step_body(request, _CONFIGURE_STEP)
    return {"ok": True, "schema_hash": schema.schema_hash()}
