"""Shared helpers for ``dbsprout.cli`` subcommands.

These utilities live outside any individual command module so that multiple
subcommands can reuse them without creating cross-command imports.
"""

from __future__ import annotations


def scrub_secrets(message: str, source_url: str) -> str:
    """Remove any password from *message* by replacing it with ``***``.

    Uses :func:`sqlalchemy.engine.make_url` to extract the password from
    *source_url*. If the URL can't be parsed or carries no password, the
    message is returned unchanged (nothing to scrub). Both the raw password
    and the full source URL are substituted (belt-and-suspenders: some
    SQLAlchemy exceptions embed only the password, others the whole URL).
    """
    import sqlalchemy as sa  # noqa: PLC0415

    try:
        url = sa.engine.make_url(source_url)
    except sa.exc.ArgumentError:
        return message
    password = url.password
    if not password:
        return message
    safe_url = url.render_as_string(hide_password=True)
    return message.replace(password, "***").replace(source_url, safe_url)
