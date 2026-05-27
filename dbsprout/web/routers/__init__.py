"""HTTP API routers for the DBSprout web dashboard.

Each module here owns its own :class:`~fastapi.APIRouter`, registered by
:func:`dbsprout.web.app.create_app` inside a region-delimited ``include_router``
block. Kept intentionally minimal so sibling stories that also add a router under
this package union-merge cleanly.
"""
