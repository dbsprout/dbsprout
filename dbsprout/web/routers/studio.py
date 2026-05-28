"""Studio shell route (S-117).

The Studio page is the single workspace that arranges DBSprout's four core
panels — **tree** (left, schema), **grid** (centre, live spec preview),
**context** (right, help/details), and **console** (bottom, run log) — into one
HTML shell that later Phase-C stories fill in:

* S-118 spec grid → ``#studio-grid``
* S-125 console progress → ``#studio-console``
* S-127 seed control → ``#studio-context``

This module owns the read-only ``GET /studio`` handler. The handler is *thin*
on purpose: the four panels are exposed as named Jinja2 blocks (``tree`` ·
``grid`` · ``context`` · ``console``) and the tree panel is the only one that
already binds real data — the workspace schema (S-111). The grid / context /
console blocks render placeholders with the stable element ids so later stories
can either override the block or ``hx-get`` content into the matching id.

The schema is read from the in-memory :class:`~dbsprout.web.workspace.Workspace`
on ``app.state.workspace`` (S-111). The tree reuses
:func:`dbsprout.web.routers.schema._schema_tree` so the tree-panel JSON shape is
identical to the schema review API (S-115) — one builder, two surfaces.

This router is import-light: heavy / CLI-adjacent modules (orchestrator, core
service) are not imported here. ``dbsprout.web.app.create_app`` registers it
inside a region-delimited block.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from fastapi import APIRouter, Request
from fastapi.responses import Response

from dbsprout.web.routers.schema import _schema_tree

if TYPE_CHECKING:
    from fastapi.templating import Jinja2Templates

    from dbsprout.web.workspace import Workspace

studio_router = APIRouter()


def _workspace(request: Request) -> Workspace:
    """Typed accessor for the shared session workspace wired in ``app.py``."""
    return cast("Workspace", request.app.state.workspace)


def _templates(request: Request) -> Jinja2Templates:
    """Typed accessor for the shared Jinja2 environment wired in ``app.py``."""
    return cast("Jinja2Templates", request.app.state.templates)


@studio_router.get("/studio", response_class=Response)
async def studio(request: Request) -> Response:
    """Render the Studio shell (tree · grid · context · console).

    Always returns ``200`` — when no schema is loaded the tree panel surfaces a
    friendly empty state with links to ``/api/connect`` (S-112) and
    ``/api/schema/load`` (S-113). The grid / context / console panels render
    placeholders that later Phase-C stories replace.
    """
    workspace = _workspace(request)
    schema = workspace.get_schema()
    tree: dict[str, Any] | None = None
    if schema is not None:
        tree = _schema_tree(schema, workspace.get_source())

    return _templates(request).TemplateResponse(
        request,
        "studio.html",
        {"active": "studio", "tree": tree},
    )
