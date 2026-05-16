"""User-facing error hierarchy for DBSprout (S-076).

Every :class:`DBSproutError` carries three plain-language fields — *what*
happened, *why* it happened, and how to *fix* it — so the CLI can render a
consistent, actionable message. ``__str__`` returns a compact single line so
existing ``console.print(f"... {exc}")`` call sites keep working unchanged.
"""

from __future__ import annotations

import importlib


class DBSproutError(Exception):
    """Base class for all user-facing DBSprout errors."""

    def __init__(
        self,
        *,
        what: str,
        why: str,
        fix: str,
        exit_code: int = 1,
    ) -> None:
        self.what = what
        self.why = why
        self.fix = fix
        self.exit_code = exit_code
        super().__init__(f"{what} — {why} Fix: {fix}")

    def __str__(self) -> str:
        return f"{self.what} — {self.why} Fix: {self.fix}"


class ConnectionError(DBSproutError):  # noqa: A001 — intentional user-facing name
    """A database connection could not be established."""


class SchemaError(DBSproutError):
    """A schema could not be parsed or introspected."""


class GenerationError(DBSproutError):
    """Seed-data generation failed for a table or column."""


class ConfigError(DBSproutError):
    """Configuration (TOML) is missing or invalid."""


class ModelError(DBSproutError):
    """An AI model file is missing or could not be loaded."""


class MissingDependencyError(DBSproutError):
    """An optional dependency is not installed.

    The *fix* names the exact ``pip install`` command. When *extra* is
    ``None`` the package ships with core DBSprout and a plain
    ``pip install dbsprout`` reinstall is suggested.
    """

    def __init__(self, *, package: str, extra: str | None) -> None:
        command = "pip install dbsprout" if extra is None else f"pip install dbsprout[{extra}]"
        super().__init__(
            what=f"The optional dependency '{package}' is not installed.",
            why="This feature needs an extra that is not part of the core install.",
            fix=f"Install it with: {command}",
        )
        self.package = package
        self.extra = extra


def require_dependency(
    module: str,
    *,
    extra: str | None,
    package: str | None = None,
) -> None:
    """Import *module*, raising :class:`MissingDependencyError` if absent.

    *package* defaults to *module* and is used only for the user-facing
    message (e.g. import ``cv2`` but tell the user to install ``opencv``).
    """
    try:
        importlib.import_module(module)
    except ImportError as exc:
        raise MissingDependencyError(
            package=package or module,
            extra=extra,
        ) from exc
