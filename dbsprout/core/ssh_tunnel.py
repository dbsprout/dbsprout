"""SSH-tunnel context manager for connecting to a DB behind a bastion (P2a-3).

When a connect / test request carries an ``ssh`` block, the web layer opens a
local port-forward through the bastion to the remote database, rewrites the
connection URL host/port to the local forward, introspects through it, and tears
the tunnel down with the connection.

Native-dep convention
----------------------
``sshtunnel`` (which pulls ``paramiko`` and crypto deps) lives behind the
optional ``[ssh]`` extra. It is **lazy-imported** here via :func:`_import_sshtunnel`
*inside* the context manager — importing this module pulls nothing heavy, so
``dbsprout/web/routers/connect.py`` (and therefore ``app.py``) stay import-clean
without the extra installed. When the import fails (extra absent), the context
manager raises :class:`SshTunnelUnavailable` so the caller can surface a friendly
``pip install dbsprout[ssh]`` envelope instead of a 500.

Security
--------
* The SSH private key is referenced **by path** (``ssh_pkey=key_path``). This
  module never reads the key bytes, never copies the key into state, and never
  logs the path contents.
* The tunnel target (bastion address + remote DB address) is **scrubbed** out of
  any error this module re-raises — a forwarder start failure surfaces a generic
  message, never the bastion/remote host.
"""

from __future__ import annotations

from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Iterator


class SshTunnelConfig(BaseModel):
    """An SSH bastion descriptor — host/user/key referenced by path.

    Frozen + ``extra='forbid'`` so the request boundary rejects unknown keys.
    ``key_path`` is the filesystem path to the private key; its bytes are never
    read by dbsprout (``paramiko`` opens it itself, inside the forwarder).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    host: str = Field(min_length=1, description="Bastion (jump) host.")
    port: int = Field(default=22, ge=1, le=65535, description="Bastion SSH port.")
    user: str = Field(min_length=1, description="SSH username on the bastion.")
    key_path: str = Field(
        min_length=1,
        description="Path to the SSH private key (referenced by path; never read here).",
    )


class SshTunnelUnavailable(ImportError):  # noqa: N818 — name mirrors the SSH_UNAVAILABLE code
    """Raised when the optional ``[ssh]`` extra (``sshtunnel``) is not installed.

    Subclasses :class:`ImportError` so existing ``except ImportError`` guards in
    the connect path keep working, while the connect handler can catch this exact
    type first to emit the typed ``SSH_UNAVAILABLE`` envelope.
    """


def _import_sshtunnel() -> Any:
    """Import and return the ``sshtunnel`` module (the lazy seam).

    Isolated into a one-line function purely so tests can monkeypatch it: a fake
    module to inject a stub forwarder, or an ``ImportError`` to simulate the
    missing extra. Never called at import time.
    """
    import sshtunnel  # noqa: PLC0415 — lazy: the [ssh] extra is optional

    return sshtunnel


@contextmanager
def open_ssh_tunnel(
    cfg: SshTunnelConfig,
    *,
    remote_host: str,
    remote_port: int,
) -> Iterator[tuple[str, int]]:
    """Open a local port-forward to *remote_host:remote_port* via the bastion.

    Yields the local ``(host, port)`` the forwarder bound; the caller rewrites
    the connection URL to point at it. The forwarder is always stopped on exit
    (success or exception).

    Raises:
        SshTunnelUnavailable: the ``[ssh]`` extra is not installed.
        RuntimeError: the forwarder failed to start — with the tunnel target
            scrubbed out of the message (the bastion/remote address never leaks).
    """
    try:
        sshtunnel = _import_sshtunnel()
    except ImportError as exc:
        # Re-raise as the typed marker WITHOUT echoing any target — the message
        # carries only the (safe) module-missing detail.
        raise SshTunnelUnavailable(str(exc)) from None

    forwarder = sshtunnel.SSHTunnelForwarder(
        (cfg.host, cfg.port),
        ssh_username=cfg.user,
        ssh_pkey=cfg.key_path,
        remote_bind_address=(remote_host, remote_port),
    )
    try:
        try:
            forwarder.start()
        except Exception:
            # Never leak the bastion / remote target into the surfaced message:
            # the original exception's text may embed ``bastion:22 -> db:5432``,
            # so we raise a generic wrapper with ``from None`` to drop the cause
            # entirely (no leak through ``__cause__`` either).
            raise RuntimeError("Failed to open the SSH tunnel to the database host.") from None
        local_host = str(forwarder.local_bind_host)
        local_port = int(forwarder.local_bind_port)
        yield (local_host, local_port)
    finally:
        # Always tear the forwarder down; ignore stop errors on the cleanup path.
        with suppress(Exception):
            forwarder.stop()


__all__ = [
    "SshTunnelConfig",
    "SshTunnelUnavailable",
    "open_ssh_tunnel",
]
