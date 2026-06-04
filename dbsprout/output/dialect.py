"""Shared SQLAlchemy-URL → dialect detection for the direct-insert paths.

Both the CLI direct-insert adapter
(:mod:`dbsprout.cli.commands._direct_insert`) and the web insert router
(:mod:`dbsprout.web.routers.insert`) route generated data to the fastest
available per-dialect writer. They previously each carried a byte-identical
private ``_detect_direct_dialect`` copy of the URL→dialect policy, which was
drift-prone (S-150).

This module is the single source of truth. It lives in ``output/`` because the
per-dialect writers it feeds (PG COPY, MySQL LOAD DATA, SaBatch fallback)
already live there, so both the CLI and the web stage depend on ``output/``
without importing each other (no-cross-stage-imports).

DISTINCT from :func:`dbsprout.schema.parsers.ddl._detect_dialect`, which sniffs
DDL *text*; this one inspects a connection-URL prefix.
"""

from __future__ import annotations


def detect_direct_dialect(url: str) -> str:
    """Detect a database dialect from a connection URL prefix."""
    lower = url.lower()
    if lower.startswith(("postgresql", "postgres")):
        return "postgresql"
    if lower.startswith("mysql"):
        return "mysql"
    if lower.startswith("sqlite"):
        return "sqlite"
    if lower.startswith("mssql"):
        return "mssql"
    return lower.split("://")[0].split("+")[0] if "://" in lower else "unknown"
