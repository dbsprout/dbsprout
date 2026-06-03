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
  per-kind message via :class:`SshTunnelConnectError`, never the bastion/remote
  host. The coarse failure ``kind`` (``auth`` / ``host`` / ``forward``) is decided
  from the original exception's *shape* (type name + keyword sweep) so the connect
  router can map it to a friendly typed 4xx/502 envelope (P4-9), never a raw 500.
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


#: The three distinguishable bastion-failure kinds (P4-9). ``auth`` is a bad SSH
#: user / rejected key; ``host`` is the bastion being unreachable (DNS / refused /
#: timeout / bad host key); ``forward`` is the bastion reached + authed but the
#: remote-bind / channel open failing. The router maps these to status codes
#: (auth→400, host/forward→502) — see :func:`dbsprout.web.errors.web_error_ssh_tunnel_failed`.
SshTunnelFailureKind = str  # one of "auth" | "host" | "forward"


class SshTunnelConnectError(RuntimeError):
    """Raised when the forwarder fails to ``start()`` against a live bastion (P4-9).

    Carries a coarse :attr:`kind` (``"auth"`` / ``"host"`` / ``"forward"``) decided
    at the raise site from the original exception's *type name* + message — the
    original is dropped via ``from None`` so the bastion / remote target never
    leaks through ``__cause__``. The message is a **generic** per-kind string with
    no target in it; the router turns :attr:`kind` into the friendly typed
    envelope (4xx/502, never a raw 500).

    Subclasses :class:`RuntimeError` so any legacy ``except RuntimeError`` guard
    still catches it, while the connect handler catches this exact type first to
    emit the typed ``SSH_TUNNEL_FAILED`` envelope.
    """

    def __init__(self, kind: SshTunnelFailureKind, message: str) -> None:
        super().__init__(message)
        self.kind = kind


# Type-name → kind. We never import ``sshtunnel`` / ``paramiko``; the *name*
# alone dispatches (mirrors the driver classifier in ``dbsprout.web.errors``).
_START_ERROR_TYPE_KINDS: dict[str, SshTunnelFailureKind] = {
    # paramiko auth failures.
    "AuthenticationException": "auth",
    "BadAuthenticationType": "auth",
    "PasswordRequiredException": "auth",
    "PartialAuthentication": "auth",
    # bastion unreachable / DNS / refused / bad host key.
    "gaierror": "host",
    "ConnectionRefusedError": "host",
    "NoValidConnectionsError": "host",
    "BadHostKeyException": "host",
    "TimeoutError": "host",
    "timeout": "host",
    # sshtunnel forward / channel failures.
    "HandlerSSHTunnelForwarderError": "forward",
    "BaseSSHTunnelForwarderError": "forward",
    "ChannelException": "forward",
}

# Message-keyword → kind, swept case-insensitively when the type name is generic
# (e.g. a bare ``SSHException`` / ``OSError``). Auth wins over host on a tie since
# an auth-worded message is the more specific signal.
_START_ERROR_MESSAGE_KINDS: tuple[tuple[tuple[str, ...], SshTunnelFailureKind], ...] = (
    (("authentication", "auth failed", "private key", "permission denied", "bad password"), "auth"),
    (
        (
            "name or service not known",
            "name resolution",
            "could not resolve",
            "connection refused",
            "no route to host",
            "host is down",
            "timed out",
            "unable to connect",
            "host key",
        ),
        "host",
    ),
)


def _classify_start_error(exc: BaseException) -> SshTunnelFailureKind:
    """Map a forwarder ``start()`` failure to a coarse :data:`SshTunnelFailureKind`.

    Dispatches on ``type(exc).__name__`` first (exact), then sweeps the lowercased
    message for keywords. Anything unrecognised defaults to ``"forward"`` — a
    generic tunnel-could-not-open 502, never a 500. Never reads the bastion target
    out of the exception; only its *shape* informs the kind.
    """
    type_name = type(exc).__name__
    kind = _START_ERROR_TYPE_KINDS.get(type_name)
    if kind is not None:
        return kind
    lowered = str(exc).lower()
    for keywords, mapped in _START_ERROR_MESSAGE_KINDS:
        if any(keyword in lowered for keyword in keywords):
            return mapped
    return "forward"


#: Generic per-kind messages — **no** bastion / remote target ever appears here.
_KIND_MESSAGES: dict[SshTunnelFailureKind, str] = {
    "auth": "SSH authentication to the bastion failed.",
    "host": "Could not reach the SSH bastion host.",
    "forward": "Failed to open the SSH tunnel to the database host.",
}


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
        SshTunnelConnectError: the forwarder failed to start — carries a coarse
            :attr:`~SshTunnelConnectError.kind` (``auth`` / ``host`` / ``forward``)
            with the tunnel target scrubbed out of the message (the bastion /
            remote address never leaks). Subclasses :class:`RuntimeError`.
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
        except Exception as exc:
            # Never leak the bastion / remote target into the surfaced message:
            # the original exception's text may embed ``bastion:22 -> db:5432``,
            # so we classify it into a coarse ``kind`` (auth / host / forward) and
            # raise a typed wrapper carrying ONLY that kind + a generic per-kind
            # message, with ``from None`` to drop the cause entirely (no leak
            # through ``__cause__`` either). The router maps ``kind`` → a friendly
            # 4xx/502 envelope so a live bastion failure is never a raw 500.
            kind = _classify_start_error(exc)
            raise SshTunnelConnectError(kind, _KIND_MESSAGES[kind]) from None
        local_host = str(forwarder.local_bind_host)
        local_port = int(forwarder.local_bind_port)
        yield (local_host, local_port)
    finally:
        # Always tear the forwarder down; ignore stop errors on the cleanup path.
        with suppress(Exception):
            forwarder.stop()


__all__ = [
    "SshTunnelConfig",
    "SshTunnelConnectError",
    "SshTunnelUnavailable",
    "open_ssh_tunnel",
]
