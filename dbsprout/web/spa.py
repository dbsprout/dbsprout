"""Mount the built single-page Workbench (React/Vite) onto the FastAPI app.

The SPA is built by ``frontend/`` (Vite) into ``dbsprout/web/spa/`` and shipped
inside the wheel (see the hatchling build hook + ``[tool.hatch.build.targets.wheel]``
``artifacts`` in pyproject.toml). It is served under ``/app`` so it coexists with
the legacy dashboard at ``/`` during the rebuild (the legacy UI is removed at the
end of Phase 1).

When the build output is absent — an editable dev checkout where ``npm run build``
has not run, or the Python-only CI job — ``/app`` serves a small placeholder
explaining how to build it. The server never crashes for a missing front-end build.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

if TYPE_CHECKING:
    from fastapi import FastAPI

_SPA_DIR = Path(__file__).resolve().parent / "spa"

_PLACEHOLDER = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>DBSprout Workbench</title></head>
<body style="font-family:system-ui;max-width:40rem;margin:4rem auto;padding:0 1rem">
<h1>DBSprout Workbench</h1>
<p>The front-end build is not present in this install.</p>
<p>Build it with:</p>
<pre>npm --prefix frontend ci
npm --prefix frontend run build</pre>
<p>then reload. (Wheels published from CI already include the build.)</p>
</body></html>
"""


def spa_is_built(spa_dir: Path = _SPA_DIR) -> bool:
    """True when a built ``index.html`` exists in *spa_dir*."""
    return (spa_dir / "index.html").is_file()


def mount_spa(app: FastAPI, spa_dir: Path = _SPA_DIR) -> None:
    """Serve the Workbench SPA at ``/app`` — or a placeholder if not built."""
    if spa_is_built(spa_dir):
        app.mount("/app", StaticFiles(directory=str(spa_dir), html=True), name="spa")
        return

    @app.get("/app", response_class=HTMLResponse)
    async def _spa_placeholder() -> HTMLResponse:
        return HTMLResponse(_PLACEHOLDER)
