"""Saved-connection persistence for the web Workbench (P2a-2).

Reusable database connections are stored in ``.dbsprout/connections.toml`` so a
user can save a target once and reload it. The security-critical contract is
that **passwords are never written**: :func:`_strip_password` removes any literal
password before persistence, and the only credential token that survives is an
``${ENV_VAR}`` reference (resolved to the real secret only at *connect* time via
:func:`resolve_connection_url`, never stored or logged).

The module is pure (no FastAPI / web imports) so it can be unit-tested in
isolation and reused by the CLI. TOML is read through the py3.10
``tomllib``/``tomli`` shim and written with ``tomli_w`` (a core dependency).
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

import tomli_w
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

try:  # py3.11+ ships tomllib in the stdlib; 3.10 falls back to the tomli backport.
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover — Python 3.10
    import tomli as tomllib  # type: ignore[import-not-found]

#: A connection-string token that is an environment-variable reference rather
#: than a literal secret (e.g. ``${PGPASSWORD}``). Such tokens are preserved
#: verbatim on write and resolved only at connect time.
_ENV_REF_RE = re.compile(r"^\$\{[A-Za-z_][A-Za-z0-9_]*\}$")

#: Matches any ``${VAR}`` reference anywhere in a URL (used by the resolver).
_ENV_SUBST_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

#: Top-level TOML table holding the saved connections, keyed by name.
_TABLE = "connections"


class SavedConnection(BaseModel):
    """One persisted connection: a name and a password-stripped URL."""

    model_config = ConfigDict(frozen=True)

    name: str
    url: str


def connections_path(root: Path) -> Path:
    """Return the ``connections.toml`` path under *root*'s ``.dbsprout`` dir."""
    return root / ".dbsprout" / "connections.toml"


def _is_env_ref(token: str) -> bool:
    """True when *token* is a single ``${ENV_VAR}`` reference (not a literal)."""
    return bool(_ENV_REF_RE.fullmatch(token))


def _strip_password(url: str) -> str:
    """Return *url* with any literal password removed.

    An ``${ENV_VAR}``-shaped password is a *reference*, not a secret, so it is
    preserved verbatim; every other password is dropped (empty userinfo
    password). A URL with no password is returned unchanged. Never raises: a
    string SQLAlchemy cannot parse falls back to a stdlib ``urlsplit`` rewrite.
    """
    try:
        import sqlalchemy as sa  # noqa: PLC0415 - lazy; keeps import cost off the hot path

        parsed = sa.engine.make_url(url)
    except Exception:
        return _mask_userinfo_password(url)
    password = parsed.password
    if not password or _is_env_ref(password):
        return url
    # ``set(password=None)`` is a no-op in SQLAlchemy (None means "leave as is");
    # an empty string is what actually clears the credential. The empty literal
    # is the *removal* of a secret, not a hardcoded one — hence the nosec.
    return parsed.set(password="").render_as_string(hide_password=False)  # nosec B106 - clears the password, not a credential


def _mask_userinfo_password(url: str) -> str:
    """Stdlib fallback: drop the password from ``scheme://user:pass@host/...``.

    Used only for strings SQLAlchemy cannot parse. An ``${ENV_VAR}`` password is
    preserved; any other password is removed. A URL with no password (or no
    parseable userinfo) is returned unchanged.
    """
    from urllib.parse import urlsplit  # noqa: PLC0415

    parts = urlsplit(url)
    password = parts.password
    if not password or _is_env_ref(password):
        return url
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    userinfo = f"{parts.username}@" if parts.username else ""
    rest = url.split("@", 1)[1] if "@" in url else f"{host}{parts.path}"
    return f"{parts.scheme}://{userinfo}{rest}" if parts.username else f"{parts.scheme}://{rest}"


def resolve_connection_url(url: str, environ: Mapping[str, str] | None = None) -> str:
    """Substitute ``${VAR}`` references in *url* from *environ* (default ``os.environ``).

    Called only at connect time so the real secret lives in process memory for
    the duration of the connection and never in state or logs. A reference whose
    variable is undefined raises :class:`ValueError` naming the missing variable.
    A URL with no references is returned unchanged.
    """
    env = os.environ if environ is None else environ

    def _replace(match: re.Match[str]) -> str:
        var = match.group(1)
        try:
            return env[var]
        except KeyError as exc:
            msg = f"connection references undefined environment variable: {var}"
            raise ValueError(msg) from exc

    return _ENV_SUBST_RE.sub(_replace, url)


def load_connections(path: Path) -> list[SavedConnection]:
    """Read the saved connections from *path* (missing/malformed file → ``[]``).

    A malformed TOML file or unparsable individual entries are skipped rather
    than raising — a corrupt file should never brick the Start panel.
    """
    if not path.exists():
        return []
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError, UnicodeDecodeError):
        return []
    table = data.get(_TABLE)
    if not isinstance(table, dict):
        return []
    out: list[SavedConnection] = []
    for name, entry in table.items():
        if isinstance(entry, dict) and isinstance(entry.get("url"), str):
            out.append(SavedConnection(name=name, url=entry["url"]))
    return out


def _write_connections(path: Path, connections: list[SavedConnection]) -> None:
    """Serialise *connections* to *path* as ``[connections.<name>]`` tables."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = {c.name: {"url": c.url} for c in connections}
    with path.open("wb") as fh:
        tomli_w.dump({_TABLE: table}, fh)


def save_connection(path: Path, name: str, url: str) -> SavedConnection:
    """Persist a connection under *name*, stripping any literal password.

    Upserts by name (an existing entry with the same name is replaced) and
    returns the stored :class:`SavedConnection` (with the stripped URL).
    """
    stripped = _strip_password(url)
    saved = SavedConnection(name=name, url=stripped)
    existing = [c for c in load_connections(path) if c.name != name]
    _write_connections(path, [*existing, saved])
    return saved


def delete_connection(path: Path, name: str) -> bool:
    """Remove the connection named *name*; return ``True`` iff one was removed."""
    existing = load_connections(path)
    remaining = [c for c in existing if c.name != name]
    if len(remaining) == len(existing):
        return False
    _write_connections(path, remaining)
    return True
