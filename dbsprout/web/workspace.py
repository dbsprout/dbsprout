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

import logging
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from dbsprout.generate.orchestrator import GenerateResult
    from dbsprout.schema.models import DatabaseSchema
    from dbsprout.spec.cache import SpecCache
    from dbsprout.spec.models import DataSpec, GeneratorConfig, TableSpec

_log = logging.getLogger(__name__)


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

    def __init__(self, spec_cache: SpecCache | None = None) -> None:
        self.schema: DatabaseSchema | None = None
        self.spec: DataSpec | None = None
        self.last_result: GenerateResult | None = None
        # S-131: the materialised seed from the most recent generation run, so
        # downstream regenerate calls can pin to the same RNG state. Set by the
        # generate-router closure on success; read by the regenerate route.
        self.last_seed: int | None = None
        self.source: str | None = None
        self._target_url: str | None = None
        # S-122: optional disk-backed spec cache. Default is ``None`` here so the
        # cache module is only imported (and the ``.dbsprout/cache`` directory only
        # created) on first persist/hydrate — tests that never touch persistence
        # pay nothing, and the default location can still be overridden via
        # :meth:`set_spec_cache` for app-level wiring.
        self._spec_cache: SpecCache | None = spec_cache

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

    def update_column(
        self,
        table: str,
        column: str,
        config: GeneratorConfig,
    ) -> GeneratorConfig:
        """Replace one column's ``GeneratorConfig`` on the loaded spec (S-119).

        The swap is **immutable**: a new ``TableSpec`` is built via
        :meth:`pydantic.BaseModel.model_copy` with the updated ``columns``
        mapping, and a new ``DataSpec`` is built with the table list swapped.
        Both frozen models are replaced, never mutated; the workspace's
        ``spec`` attribute is reassigned to the new ``DataSpec``.

        The router has already validated ``config`` against
        :class:`~dbsprout.spec.models.GeneratorConfig` (Pydantic v2) and run
        the referential-integrity guard
        (:func:`dbsprout.spec.constraints.check_column_update`). This helper
        trusts that contract — it does not re-validate.

        Args:
            table: target ``TableSpec.table_name``.
            column: key in the table's ``columns`` mapping.
            config: the already-validated replacement.

        Returns:
            The freshly-stored :class:`~dbsprout.spec.models.GeneratorConfig`.

        Raises:
            LookupError: when no spec is loaded, the table is not in the
                spec, or the column is not on that table. The router maps
                these to ``404 Not Found`` for the caller.
        """
        if self.spec is None:
            msg = "no spec loaded; call set_spec() first"
            raise LookupError(msg)

        # Locate the target ``TableSpec`` while preserving table order.
        new_tables: list[TableSpec] = []
        found_table = False
        for ts in self.spec.tables:
            if ts.table_name != table:
                new_tables.append(ts)
                continue
            found_table = True
            if column not in ts.columns:
                msg = f"unknown column {column!r} on table {table!r}"
                raise LookupError(msg)
            new_columns = {**ts.columns, column: config}
            new_tables.append(ts.model_copy(update={"columns": new_columns}))

        if not found_table:
            msg = f"unknown table {table!r}"
            raise LookupError(msg)

        self.spec = self.spec.model_copy(update={"tables": new_tables})
        return config

    # ── spec persistence (S-122) ───────────────────────────────────────
    def set_spec_cache(self, cache: SpecCache | None) -> None:
        """Override the disk-backed spec cache used by :meth:`persist_spec`
        and :meth:`hydrate_from_cache`.

        Production code lets :meth:`_get_spec_cache` lazy-initialise the default
        cache (``.dbsprout/cache``); tests inject a tmp-rooted
        :class:`~dbsprout.spec.cache.SpecCache` instead to stay hermetic.
        ``None`` clears the override so the default re-initialises on next use.
        """
        self._spec_cache = cache

    def _get_spec_cache(self) -> SpecCache:
        """Return the active spec cache, lazily creating the default on demand.

        Lazy creation keeps ``Workspace()`` import-cheap — the diskcache
        directory and the SQLite connection are only opened the first time
        persist/hydrate actually fires.
        """
        if self._spec_cache is None:
            from dbsprout.spec.cache import SpecCache  # noqa: PLC0415 — lazy

            self._spec_cache = SpecCache()
        return self._spec_cache

    def persist_spec(self) -> None:
        """Best-effort write of the current spec to the disk cache.

        No-ops cleanly when either ``schema`` or ``spec`` is unset (nothing to
        key, or nothing to persist). Any exception from the underlying cache
        (corrupt file, exhausted disk, permission denied, …) is logged at
        ``WARNING`` and swallowed — Studio edits must succeed even if
        persistence fails. The cache write itself is atomic by virtue of
        :class:`diskcache.Cache`'s SQLite-backed transactional ``set``.
        """
        if self.schema is None or self.spec is None:
            return
        try:
            schema_hash = self.schema.schema_hash()
            self._get_spec_cache().put(schema_hash, self.spec)
        except Exception:
            # Persistence is best-effort: a broken cache must never fail a
            # Studio edit. Log + swallow.
            _log.warning("spec persistence failed; continuing", exc_info=True)

    def hydrate_from_cache(self, schema_hash: str) -> bool:
        """Replace ``self.spec`` with the cached spec for *schema_hash* if any.

        Returns ``True`` on a hit (``self.spec`` is now the cached object) and
        ``False`` on a miss or any cache error (``self.spec`` is left untouched
        so the existing lazy heuristic build path still applies).
        """
        try:
            cached = self._get_spec_cache().get(schema_hash)
        except Exception:
            # Degrade to a clean miss on any cache error so the route can
            # still fall back to the lazy heuristic build.
            _log.warning("spec cache hydration failed; falling back", exc_info=True)
            return False
        if cached is None:
            return False
        self.spec = cached
        return True

    # ── last result ────────────────────────────────────────────────────
    def get_last_result(self) -> GenerateResult | None:
        return self.last_result

    def set_last_result(self, result: GenerateResult | None) -> None:
        self.last_result = result

    # ── last seed (S-131) ──────────────────────────────────────────────
    def get_last_seed(self) -> int | None:
        """Return the seed captured by the most recent generation run, if any.

        Used by the regenerate route (S-131) so a re-roll can pin the global
        RNG to the same state the original run used. Returns ``None`` when no
        run has stored a seed yet.
        """
        return self.last_seed

    def set_last_seed(self, seed: int | None) -> None:
        """Stash the materialised seed of the most recent generation run."""
        self.last_seed = seed

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
        self.last_seed = None
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
