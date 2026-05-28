"""In-memory session state for the web dashboard (S-111).

The dashboard serves stateless HTTP requests, but the single-user localhost
flow (load schema → edit spec → generate → view result) needs the loaded
schema, the edited spec, and the last result to survive across requests. A
single :class:`Workspace` instance is wired onto ``app.state.workspace`` by
:func:`dbsprout.web.app.create_app` (one instance — single-user, no locking).

The objects held here are *frozen* (``DatabaseSchema``, ``DataSpec``,
``GenerateResult``); "mutation" means reassigning a new frozen object. Spec
edits go through :meth:`Workspace.update_spec`, which applies
``DataSpec.model_copy(update=...)`` — frozen models are never mutated in place.

The target connection URL is held *privately* and is NEVER persisted to disk or
logged in clear; read it only via :attr:`Workspace.redacted_target`, which masks
``user:password`` using the same approach as :mod:`dbsprout.schema.introspect`
(SQLAlchemy ``render_as_string(hide_password=True)``), with a stdlib userinfo
mask fallback for malformed / non-SQLAlchemy URLs.

This module lives in the web layer; it imports the frozen domain models but must
not import CLI code, and the CLI must never import it at startup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from dbsprout.generate.orchestrator import GenerateResult
    from dbsprout.schema.models import DatabaseSchema
    from dbsprout.spec.models import DataSpec, TableSpec


def _redact_url(url: str) -> str:
    """Return *url* with any password masked; never raises, never leaks.

    Prefers SQLAlchemy ``make_url(...).render_as_string(hide_password=True)``
    (the approach used across the codebase). Falls back to a stdlib
    ``urlsplit`` userinfo mask for URLs SQLAlchemy can't parse. A URL with no
    password is returned unchanged.
    """
    try:
        import sqlalchemy as sa  # noqa: PLC0415

        parsed = sa.engine.make_url(url)
    except Exception:
        return _mask_userinfo(url)
    if not parsed.password:
        return url
    return parsed.render_as_string(hide_password=True)


def _mask_userinfo(url: str) -> str:
    """Stdlib fallback: mask the password in ``scheme://user:pass@host/...``."""
    parts = urlsplit(url)
    if not parts.password:
        return url
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    userinfo = f"{parts.username}:***@" if parts.username else "***@"
    return f"{parts.scheme}://{userinfo}{host}{parts.path}"


class Workspace:
    """Mutable in-memory session state for the single-user dashboard.

    Holds the loaded :class:`~dbsprout.schema.models.DatabaseSchema`, the
    (editable) :class:`~dbsprout.spec.models.DataSpec`, the last
    :class:`~dbsprout.generate.orchestrator.GenerateResult`, a human-readable
    source descriptor, and a *private* target connection URL exposed only via
    :attr:`redacted_target`.
    """

    def __init__(self) -> None:
        self.schema: DatabaseSchema | None = None
        self.spec: DataSpec | None = None
        self.last_result: GenerateResult | None = None
        self.source: str | None = None
        self._target_url: str | None = None

    # ── schema ─────────────────────────────────────────────────────────
    def get_schema(self) -> DatabaseSchema | None:
        return self.schema

    def set_schema(self, schema: DatabaseSchema | None) -> None:
        self.schema = schema

    # ── spec ───────────────────────────────────────────────────────────
    def get_spec(self) -> DataSpec | None:
        return self.spec

    def set_spec(self, spec: DataSpec | None) -> None:
        self.spec = spec

    def update_spec(self, **changes: object) -> DataSpec:
        """Apply an immutable edit to the loaded spec via ``model_copy``.

        Returns the new :class:`~dbsprout.spec.models.DataSpec` and stores it.
        Raises :class:`ValueError` if no spec is loaded.

        ``model_copy(update=...)`` does **not** re-run Pydantic validation — it
        is a low-level structural copy. Callers (the future spec-edit route) are
        responsible for validating ``changes`` at their input boundary before
        calling this; this setter intentionally trusts already-validated input.
        """
        if self.spec is None:
            msg = "no spec loaded; call set_spec() first"
            raise ValueError(msg)
        self.spec = self.spec.model_copy(update=changes)
        return self.spec

    def update_table_row_count(self, table_name: str, row_count: int) -> int:
        """Immutably replace one table's ``row_count`` in the loaded spec (S-121).

        Looks up the matching :class:`~dbsprout.spec.models.TableSpec`, replaces
        it with ``target.model_copy(update={"row_count": row_count})``, and
        stores a fresh :class:`~dbsprout.spec.models.DataSpec` on the workspace
        (table ordering preserved). Returns the new ``row_count``.

        Raises:
            ValueError: if no spec is loaded.
            KeyError: if ``table_name`` is absent from the spec.

        Like :meth:`update_spec`, ``model_copy`` does not re-run Pydantic
        validation; callers (the PUT route in
        :mod:`dbsprout.web.routers.spec`) are responsible for bounds checking
        ``row_count`` at the input boundary before calling.
        """
        if self.spec is None:
            msg = "no spec loaded; call set_spec() first"
            raise ValueError(msg)
        existing_tables = self.spec.tables
        new_tables: list[TableSpec] = []
        replaced = False
        for table_spec in existing_tables:
            if table_spec.table_name == table_name:
                new_tables.append(
                    table_spec.model_copy(update={"row_count": row_count}),
                )
                replaced = True
            else:
                new_tables.append(table_spec)
        if not replaced:
            raise KeyError(table_name)
        self.spec = self.spec.model_copy(update={"tables": new_tables})
        return row_count

    # ── last result ────────────────────────────────────────────────────
    def get_last_result(self) -> GenerateResult | None:
        return self.last_result

    def set_last_result(self, result: GenerateResult | None) -> None:
        self.last_result = result

    # ── source descriptor ──────────────────────────────────────────────
    def get_source(self) -> str | None:
        return self.source

    def set_source(self, descriptor: str | None) -> None:
        self.source = descriptor

    # ── target connection URL (private; redacted accessor only) ─────────
    def set_target_url(self, url: str | None) -> None:
        """Store the raw target URL privately (or clear it when ``None``)."""
        self._target_url = url

    def clear_target_url(self) -> None:
        self._target_url = None

    def peek_target_url(self) -> str | None:
        """Return the *raw* target URL for internal use (e.g. credential scrubbing).

        WARNING: this is the unmasked connection string — callers MUST redact it
        (see :func:`_redact_url`) before logging or returning it anywhere a user
        can see it. The public, always-safe accessor is :attr:`redacted_target`.
        """
        return self._target_url

    @property
    def redacted_target(self) -> str | None:
        """The target URL with credentials masked, or ``None`` if unset."""
        if self._target_url is None:
            return None
        return _redact_url(self._target_url)

    # ── lifecycle ──────────────────────────────────────────────────────
    def reset(self) -> None:
        """Clear all session state back to empty."""
        self.schema = None
        self.spec = None
        self.last_result = None
        self.source = None
        self._target_url = None

    def __repr__(self) -> str:
        """Safe repr — the target URL appears redacted, never in clear.

        Defends against accidental ``log.info(workspace)`` ever leaking the raw
        connection string (the AC requires it is never logged in clear).
        """
        return (
            f"Workspace(schema={'set' if self.schema else None}, "
            f"spec={'set' if self.spec else None}, "
            f"last_result={'set' if self.last_result else None}, "
            f"source={self.source!r}, target={self.redacted_target!r})"
        )
