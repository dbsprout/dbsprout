"""Bundled-sample endpoints for the Workbench Start screen (P1a).

GET /api/samples  — list available bundled sample schemas.
POST /api/schema/sample — load a named sample into the session workspace.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from dbsprout.core.samples import list_samples, load_sample
from dbsprout.web.errors import WebError, WebErrorCode, raise_web_error

if TYPE_CHECKING:
    from dbsprout.web.workspace import Workspace

samples_router = APIRouter()


class SampleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)


def _workspace(request: Request) -> Workspace:
    return cast("Workspace", request.app.state.workspace)


@samples_router.get("/api/samples", response_class=JSONResponse)
async def get_samples() -> JSONResponse:
    """Return the list of bundled sample schemas with metadata."""
    samples = [
        {
            "name": s.name,
            "title": s.title,
            "description": s.description,
            "dialect": s.dialect,
            "table_count": s.table_count,
        }
        for s in list_samples()
    ]
    return JSONResponse({"samples": samples})


@samples_router.post("/api/schema/sample", response_class=JSONResponse)
async def load_sample_schema(request: Request, body: SampleRequest) -> Any:
    """Load a named bundled sample schema into the session workspace.

    Returns a JSON summary (``table_count``, ``tables``, ``dialect``, ``source``)
    on success. Responds with ``404 NOT_FOUND`` when *name* is not a registered
    sample — the typed envelope is ``{"detail": {"code": "NOT_FOUND", …}}``.
    """
    try:
        schema = load_sample(body.name)
    except KeyError:
        return raise_web_error(
            request,
            WebError(
                code=WebErrorCode.NOT_FOUND,
                message=f"Unknown sample: {body.name!r}",
                status_code=404,
                hint="Call GET /api/samples for available names.",
            ),
        )
    source = f"sample:{body.name}"
    workspace = _workspace(request)
    workspace.set_schema(schema)
    workspace.set_source(source)
    workspace.hydrate_from_cache(schema.schema_hash())
    return JSONResponse(
        {
            "source": source,
            "table_count": len(schema.tables),
            "tables": [t.name for t in schema.tables],
            "dialect": schema.dialect or "unknown",
        }
    )
