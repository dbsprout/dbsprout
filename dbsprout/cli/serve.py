"""`dbsprout serve` command body — launches the FastAPI web dashboard (S-090).

uvicorn and the web app are imported lazily *inside* :func:`serve_command` so
importing the CLI never pulls the heavy optional ``[web]`` extra (preserves the
<500 ms startup budget). The proxy in :mod:`dbsprout.cli.app` delegates here.

The server binds ``127.0.0.1`` by default — the dashboard is a localhost dev
tool, never exposed externally (security by default, per the story AC).
"""

from __future__ import annotations

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8420


def serve_command(*, host: str, port: int, reload: bool) -> None:
    """Launch the DBSprout web dashboard (blocks until interrupted).

    Raises :class:`~dbsprout.errors.MissingDependencyError` with the exact
    ``pip install`` command when the optional ``[web]`` extra is not installed.
    """
    from dbsprout.errors import require_dependency  # noqa: PLC0415

    require_dependency("uvicorn", extra="web")

    import uvicorn  # noqa: PLC0415

    uvicorn.run("dbsprout.web.app:app", host=host, port=port, reload=reload)
