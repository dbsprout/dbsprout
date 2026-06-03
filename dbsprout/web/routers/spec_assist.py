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
    """

    model_config = ConfigDict(extra="forbid")

    provider: SpecProvider = Field(default="embedded")


def _workspace(request: Request) -> Workspace:
    """Typed accessor for the shared session wired in ``app.py``."""
    return cast("Workspace", request.app.state.workspace)


def _build_provider(provider: SpecProvider) -> Any:
    """Construct the chosen provider.

    Imported lazily so the module (and ``app.py``) never pull ``llama-cpp`` /
    ``litellm`` at import time. Construction can raise ``ImportError`` when the
    extra is absent — the caller translates that into a typed envelope.
    """
    if provider == "cloud":
        from dbsprout.spec.providers.cloud import CloudProvider  # noqa: PLC0415

        return CloudProvider()
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

    # Construct + call the provider. Any capability failure (missing extra,
    # missing model/key, bad model output) degrades to a typed 503 envelope —
    # never a 500, and the existing workspace spec stays in place.
    try:
        provider = _build_provider(body.provider)
        spec = provider.generate_spec(schema)
    except (ImportError, RuntimeError, OSError, ValueError) as exc:
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
