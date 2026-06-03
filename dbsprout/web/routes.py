"""Shared route table for the DBSprout web server (S-090; P1c-5 cutover).

After the P1c-5 cutover this shared router carries only the liveness probe
``GET /health``. The legacy home dashboard (``GET /``) and every server-rendered
sibling view were removed; ``GET /`` now redirects to the SPA at ``/app`` (wired
directly in :mod:`dbsprout.web.app`), and all data lives behind the ``/api/*``
JSON routers.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/health", response_class=JSONResponse)
async def health() -> JSONResponse:
    """Liveness probe — cheap JSON, no template."""
    return JSONResponse({"status": "ok"})
