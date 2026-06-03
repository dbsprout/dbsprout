"""``POST /api/spec/assist`` — LLM-proposed DataSpec (P2b-3).

The legacy wizard's ``POST /wizard/step/3/llm-spec`` route — the opt-in path
that let an LLM *architect* a whole :class:`~dbsprout.spec.models.DataSpec` the
user could then edit — was removed with the rest of the server-rendered UI in
the P1c-5 cutover. This module re-exposes that capability as a JSON endpoint for
the React Workbench.

Flow
----
1. Read the loaded :class:`~dbsprout.schema.models.DatabaseSchema` from the
   per-session :class:`~dbsprout.web.workspace.Workspace` (``app.state.workspace``,
   S-111). No schema ⇒ ``409 NO_SCHEMA`` (the same typed envelope ``GET /api/spec``
   raises).
2. Construct the chosen provider and call its ``generate_spec(schema) -> DataSpec``:

   * ``provider="embedded"`` (default, **offline**) →
     :class:`dbsprout.spec.providers.embedded.EmbeddedProvider` (llama-cpp behind
     the ``[llm]`` extra).
   * ``provider="cloud"`` (opt-in) →
     :class:`dbsprout.spec.providers.cloud.CloudProvider` (litellm/instructor
     behind the ``[cloud]`` extra; needs an API key in the environment).

   Both providers *already* self-cache by ``schema_hash`` via
   :class:`~dbsprout.spec.cache.SpecCache`, so a repeat assist on an unchanged
   schema skips the real model call.
3. Store the proposed spec on the workspace (``set_spec``) so ``GET /api/spec`` +
   the configure grid reflect it, and ``persist_spec`` it to the disk cache
   (keyed by ``schema_hash``) so it survives a server restart.
4. Return a small summary ``{provider, model_used, schema_hash, tables,
   total_columns}``.

Graceful degradation (never a 500)
----------------------------------
The optional extras may be absent (CI installs neither ``[llm]`` nor
``[cloud]``) and the cloud path additionally needs an API key. Any provider
*construction* or *inference* failure — ``ImportError`` (missing extra),
``RuntimeError`` (no GGUF model on disk / no API key), ``OSError`` (denied cache
dir), ``ValueError`` (malformed model output) — is caught and folded into the
kept :func:`dbsprout.web.errors.web_error_llm_unavailable` envelope (503
``LLM_UNAVAILABLE``). The request itself is well-formed; the server simply
cannot serve the optional LLM flow. The existing (heuristic) spec on the
workspace, if any, is left untouched so the user keeps moving offline.

Lazy-import contract
--------------------
``dbsprout serve`` lazy-imports the web layer; importing ``dbsprout.cli.app``
must never pull FastAPI or the LLM stack. This module imports only stdlib +
FastAPI + Pydantic at module level. The provider classes (which themselves
``import`` llama-cpp / litellm only on a real call) and the error factories are
imported **inside the handler**, so importing this router — and ``app.py`` —
pulls neither optional extra. CI imports it cleanly.

Module owns its own :class:`~fastapi.APIRouter` (``spec_assist_router``),
registered by :func:`dbsprout.web.app.create_app` inside a delimited region.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Literal, cast

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from dbsprout.schema.models import DatabaseSchema
    from dbsprout.spec.models import DataSpec
    from dbsprout.web.workspace import Workspace

_log = logging.getLogger(__name__)

spec_assist_router = APIRouter()

#: The provider keys the route accepts. ``embedded`` is the offline default;
#: ``cloud`` is opt-in and needs an API key in the environment.
SpecProvider = Literal["embedded", "cloud"]


class AssistRequest(BaseModel):
    """Body for ``POST /api/spec/assist``.

    ``provider`` defaults to the offline ``embedded`` path; ``cloud`` is opt-in.
    ``extra='forbid'`` keeps the contract tight (mirrors ``GenerateRequest`` /
    ``RegenerateRequest``) — an unknown field yields a 422 from Pydantic.

    P4-11 — cloud key-entry UX. Two *non-secret* fields let the user steer the
    cloud path without ever putting a raw API key on the wire:

    * ``model`` — the litellm model string (e.g. ``gpt-4o-mini``), forwarded to
      ``CloudProvider(model=...)``. ``None`` ⇒ the provider's own default.
    * ``api_key_env`` — the *name* of the environment variable that holds the
      provider key (e.g. ``OPENAI_API_KEY``). The route only checks that this
      variable is **present in the server's own process environment**; litellm
      then reads the value from that same environment on the real call. The key
      *value* is never carried in the request, logged, or persisted — only its
      env-var name (and the model) travel, both non-secret. Ignored for the
      offline ``embedded`` provider.
    """

    model_config = ConfigDict(extra="forbid")

    provider: SpecProvider = Field(default="embedded")
    # ─── P4-11 ─── non-secret cloud steering (model string + env-var *name*)
    model: str | None = Field(default=None)
    api_key_env: str | None = Field(default=None)
    # ─── end P4-11 ───


def _workspace(request: Request) -> Workspace:
    """Typed accessor for the shared session wired in ``app.py``."""
    return cast("Workspace", request.app.state.workspace)


def _build_provider(provider: SpecProvider, model: str | None = None) -> Any:
    """Construct the chosen provider.

    Imported lazily so the module (and ``app.py``) never pull ``llama-cpp`` /
    ``litellm`` at import time. Construction can raise ``ImportError`` when the
    extra is absent — the caller translates that into a typed envelope.

    P4-11 — a non-``None`` *model* is forwarded to ``CloudProvider(model=...)``
    (the offline embedded provider takes no model argument, so it is ignored
    there).
    """
    if provider == "cloud":
        from dbsprout.spec.providers.cloud import CloudProvider  # noqa: PLC0415

        return CloudProvider(model=model) if model else CloudProvider()
    from dbsprout.spec.providers.embedded import EmbeddedProvider  # noqa: PLC0415

    return EmbeddedProvider()


def _summary(provider: SpecProvider, spec: DataSpec) -> dict[str, Any]:
    """Shape the JSON summary returned on a successful proposal."""
    total_columns = sum(len(t.columns) for t in spec.tables)
    return {
        "provider": provider,
        "model_used": spec.model_used,
        "schema_hash": spec.schema_hash,
        "tables": len(spec.tables),
        "total_columns": total_columns,
    }


@spec_assist_router.post("/api/spec/assist", response_model=None)
async def assist_spec(request: Request, body: AssistRequest) -> dict[str, Any]:
    """Propose a full :class:`DataSpec` for the loaded schema via an LLM.

    Returns a summary ``{provider, model_used, schema_hash, tables,
    total_columns}`` on success; the proposed spec is stored on the workspace
    (``GET /api/spec`` then reflects it) and persisted to the disk cache keyed
    by ``schema_hash``.

    Guards (typed envelopes via :mod:`dbsprout.web.errors`):

    * No schema loaded ⇒ ``409 NO_SCHEMA``.
    * Cloud provider with an ``api_key_env`` naming an env var that is **absent
      from the server's process environment** ⇒ ``503 LLM_UNAVAILABLE`` naming
      the variable (P4-11) — caught *before* any provider construction so no
      real API call is attempted.
    * Provider unavailable (missing ``[llm]``/``[cloud]`` extra, no model on
      disk, no API key, malformed model output) ⇒ ``503 LLM_UNAVAILABLE`` —
      never a ``500``; the existing workspace spec is left untouched.
    * Bad ``provider`` value / extra field ⇒ ``422`` (auto from Pydantic).
    """
    from dbsprout.web.errors import (  # noqa: PLC0415
        raise_web_error,
        web_error_llm_unavailable,
        web_error_no_schema,
    )

    workspace = _workspace(request)
    schema: DatabaseSchema | None = workspace.get_schema()
    if schema is None:
        # ``raise_web_error`` is ``NoReturn`` — it always raises an HTTPException.
        raise_web_error(request, web_error_no_schema())

    # ─── P4-11 ─── cloud key-entry guard.
    # When the cloud request references an API key by env-var *name*, verify the
    # variable is present in the server's own process environment before doing
    # any work. This turns a missing key into a clear, actionable 503 (naming the
    # variable) instead of a deep litellm AuthenticationError, and — crucially —
    # the key *value* is never read into a local, logged, or persisted: only its
    # presence is probed via ``os.environ`` membership.
    if body.provider == "cloud" and body.api_key_env and body.api_key_env not in os.environ:
        raise_web_error(
            request,
            web_error_llm_unavailable(
                f"cloud provider key not found — set the {body.api_key_env} "
                f"environment variable on the dbsprout server and retry"
            ),
        )
    # ─── end P4-11 ───

    # Construct + call the provider. Any capability failure degrades to a typed
    # 503 envelope — never a 500, and the existing workspace spec stays in place.
    #
    # We deliberately catch broad ``Exception`` here, not a narrow tuple: the
    # providers are opaque third-party stacks (llama-cpp, litellm + instructor,
    # the cloud SDK underneath) whose failure modes we can't enumerate — a
    # missing ``[llm]``/``[cloud]`` extra raises ``ImportError``, a missing GGUF
    # model raises ``RuntimeError``/``OSError``, but a missing / invalid cloud
    # API key raises litellm's own ``AuthenticationError`` (a subclass of
    # ``Exception``, none of the stdlib types). The AC mandates this path NEVER
    # returns a 500, so any failure is folded into the typed envelope; the real
    # exception is preserved via ``original=exc`` for the server log.
    try:
        provider = _build_provider(body.provider, body.model)
        spec = provider.generate_spec(schema)
    except Exception as exc:  # see comment above; this path must never be a 500
        _log.warning("spec-assist provider %r unavailable: %s", body.provider, exc)
        # ``raise_web_error`` is ``NoReturn``; it re-raises as an HTTPException.
        raise_web_error(
            request,
            web_error_llm_unavailable(str(exc)),
            original=exc,
        )

    # Commit the proposal: store on the workspace (so GET /api/spec + the grid
    # reflect it) and persist to the disk cache keyed by schema_hash.
    workspace.set_spec(spec)
    workspace.persist_spec()

    return _summary(body.provider, spec)
