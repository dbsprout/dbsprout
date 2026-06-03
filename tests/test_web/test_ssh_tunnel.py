"""P2a-3 — SSH-tunnel core context manager + connect/test wiring.

Two layers are covered here:

1. ``dbsprout.core.ssh_tunnel`` — a context manager that opens a local
   port-forward to a remote DB host/port through a bastion, yields the local
   ``(host, port)``, and tears the forwarder down on exit. ``sshtunnel`` is
   **lazy-imported** inside the function via the ``_import_sshtunnel`` seam, so
   importing the module never pulls ``sshtunnel`` / ``paramiko``. The mock-based
   tests monkeypatch that seam to inject a fake forwarder, so they run in CI
   **without** the optional ``[ssh]`` extra. The one test that needs the real
   package is guarded with ``pytest.importorskip("sshtunnel")``.

2. ``POST /api/connect`` / ``/api/connect/test`` wiring — with an ``ssh`` block
   the handler opens the tunnel, rewrites the URL host/port to the local
   forward, introspects, and tears down; without it, behaviour is unchanged.
   Missing ``[ssh]`` extra → typed 503 ``SSH_UNAVAILABLE`` (never a 500). The
   bastion host never appears in the response body.

Security invariants asserted: the SSH private key is referenced **by path** and
its bytes are never read by us; the tunnel target (bastion + remote address) is
scrubbed from any error the tunnel re-raises.
"""

from __future__ import annotations

import socket
import sqlite3
import threading
from contextlib import suppress
from typing import TYPE_CHECKING, Any, ClassVar

import pytest

pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")

from fastapi.testclient import TestClient

from dbsprout.core.ssh_tunnel import (
    SshTunnelConfig,
    SshTunnelUnavailable,
    open_ssh_tunnel,
)

if TYPE_CHECKING:
    from pathlib import Path


# ── fakes ───────────────────────────────────────────────────────────────


class _FakeForwarder:
    """Stand-in for ``sshtunnel.SSHTunnelForwarder``.

    Records start/stop calls and the kwargs it was constructed with so the
    tests can assert the key is passed *by path* and the remote target is wired
    through. ``local_bind_host`` / ``local_bind_port`` mimic the real
    forwarder's attributes after ``.start()``.
    """

    instances: ClassVar[list[_FakeForwarder]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.local_bind_host = "127.0.0.1"
        self.local_bind_port = 54321
        _FakeForwarder.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


def _fake_sshtunnel(forwarder_cls: type) -> Any:
    """Build a fake ``sshtunnel`` module exposing ``SSHTunnelForwarder``."""
    import types  # noqa: PLC0415

    mod = types.ModuleType("sshtunnel")
    mod.SSHTunnelForwarder = forwarder_cls  # type: ignore[attr-defined]
    return mod


@pytest.fixture(autouse=True)
def _reset_fake() -> None:
    _FakeForwarder.instances = []


def _cfg(key_path: str = "/home/me/.ssh/id_ed25519") -> SshTunnelConfig:
    return SshTunnelConfig(host="bastion.example.com", port=22, user="deploy", key_path=key_path)


# ── core context manager: success path ──────────────────────────────────


def test_open_tunnel_yields_local_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "dbsprout.core.ssh_tunnel._import_sshtunnel",
        lambda: _fake_sshtunnel(_FakeForwarder),
    )
    with open_ssh_tunnel(_cfg(), remote_host="db.internal", remote_port=5432) as (host, port):
        assert host == "127.0.0.1"
        assert port == 54321
    fwd = _FakeForwarder.instances[-1]
    assert fwd.started is True
    assert fwd.stopped is True  # torn down on normal exit


def test_open_tunnel_passes_key_by_path_and_remote_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dbsprout.core.ssh_tunnel._import_sshtunnel",
        lambda: _fake_sshtunnel(_FakeForwarder),
    )
    with open_ssh_tunnel(_cfg(), remote_host="db.internal", remote_port=5432):
        pass
    fwd = _FakeForwarder.instances[-1]
    # Key referenced by path (never read): ssh_pkey is the literal path string.
    assert fwd.kwargs["ssh_pkey"] == "/home/me/.ssh/id_ed25519"
    # Bastion target + auth wired through.
    assert fwd.args[0] == ("bastion.example.com", 22)
    assert fwd.kwargs["ssh_username"] == "deploy"
    # Remote bind address is the real DB host/port the URL points at.
    assert fwd.kwargs["remote_bind_address"] == ("db.internal", 5432)


def test_open_tunnel_never_reads_key_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tunnel must reference the key by path; it must not read the file itself."""
    monkeypatch.setattr(
        "dbsprout.core.ssh_tunnel._import_sshtunnel",
        lambda: _fake_sshtunnel(_FakeForwarder),
    )
    opened: list[str] = []
    real_open = open

    def _tracking_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr("builtins.open", _tracking_open)
    with open_ssh_tunnel(_cfg("/secret/key"), remote_host="db.internal", remote_port=5432):
        pass
    assert "/secret/key" not in opened


# ── core context manager: teardown on failure ───────────────────────────


def test_open_tunnel_stops_forwarder_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "dbsprout.core.ssh_tunnel._import_sshtunnel",
        lambda: _fake_sshtunnel(_FakeForwarder),
    )
    with (
        pytest.raises(RuntimeError, match="boom"),
        open_ssh_tunnel(_cfg(), remote_host="db.internal", remote_port=5432),
    ):
        raise RuntimeError("boom")
    fwd = _FakeForwarder.instances[-1]
    assert fwd.stopped is True  # finally-block teardown even on error


# ── core context manager: missing extra + error scrubbing ────────────────


def test_open_tunnel_missing_extra_raises_typed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom() -> Any:
        raise ImportError("No module named 'sshtunnel'")

    monkeypatch.setattr("dbsprout.core.ssh_tunnel._import_sshtunnel", _boom)
    with (
        pytest.raises(SshTunnelUnavailable),
        open_ssh_tunnel(_cfg(), remote_host="db.internal", remote_port=5432),
    ):
        pass


def test_open_tunnel_scrubs_target_from_start_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure to start the forwarder must not leak the bastion/remote target."""

    class _ExplodingForwarder(_FakeForwarder):
        def start(self) -> None:
            raise OSError("could not connect to bastion.example.com:22 -> db.internal:5432")

    monkeypatch.setattr(
        "dbsprout.core.ssh_tunnel._import_sshtunnel",
        lambda: _fake_sshtunnel(_ExplodingForwarder),
    )
    with (
        pytest.raises(Exception) as exc_info,  # noqa: PT011 — assert scrubbing, not type
        open_ssh_tunnel(_cfg(), remote_host="db.internal", remote_port=5432),
    ):
        pass
    text = str(exc_info.value)
    assert "bastion.example.com" not in text
    assert "db.internal" not in text


def test_sshtunnel_config_rejects_blank_fields() -> None:
    with pytest.raises(Exception):  # noqa: PT011, B017 — pydantic ValidationError
        SshTunnelConfig(host="", port=22, user="deploy", key_path="/k")


def test_real_sshtunnel_import_seam() -> None:
    """The lazy seam imports the real package when the [ssh] extra is present."""
    pytest.importorskip("sshtunnel", reason="sshtunnel absent (pip install dbsprout[ssh])")
    from dbsprout.core.ssh_tunnel import _import_sshtunnel  # noqa: PLC0415

    mod = _import_sshtunnel()
    assert hasattr(mod, "SSHTunnelForwarder")


# ── endpoint wiring ──────────────────────────────────────────────────────


def _client(tmp_path: Path) -> TestClient:
    from dbsprout.web.app import create_app  # noqa: PLC0415

    return TestClient(create_app(state_db_path=tmp_path / "state.db"))


def _sqlite_url(tmp_path: Path) -> str:
    db = tmp_path / "remote.db"
    conn = sqlite3.connect(db)
    try:
        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        conn.commit()
    finally:
        conn.close()
    return f"sqlite:///{db}"


def _local_tcp_db(tmp_path: Path) -> tuple[str, int, threading.Event]:
    """Open a real localhost TCP port forwarding nothing — used to assert the
    rewritten host/port is what the loader sees.

    We don't actually proxy bytes; the connect tests use sqlite (no TCP), so the
    rewrite target only needs to be a plausible ``(host, port)`` the fake tunnel
    can report. This helper is unused for sqlite but documents the shape.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    host, port = sock.getsockname()
    stop = threading.Event()

    def _serve() -> None:
        sock.settimeout(0.2)
        while not stop.is_set():
            with suppress(OSError):
                conn, _ = sock.accept()
                conn.close()
        sock.close()

    threading.Thread(target=_serve, daemon=True).start()
    return host, port, stop


def test_connect_without_ssh_is_unchanged(tmp_path: Path) -> None:
    """No ssh block → the plain-URL path, byte-identical to today."""
    resp = _client(tmp_path).post("/api/connect", json={"url": _sqlite_url(tmp_path)})
    assert resp.status_code == 200, resp.text
    assert resp.json()["table_count"] == 1


def test_connect_extra_field_still_422(tmp_path: Path) -> None:
    """Relaxing the model to accept `ssh` must NOT open the door to other keys."""
    resp = _client(tmp_path).post("/api/connect", json={"url": "sqlite:///x.db", "rows": 10})
    assert resp.status_code == 422


def test_connect_with_ssh_opens_tunnel_and_introspects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With an ssh block, the handler opens the tunnel, rewrites the URL to the
    local forward, introspects through it, and tears down."""
    from contextlib import contextmanager  # noqa: PLC0415

    url = _sqlite_url(tmp_path)
    teardown: list[bool] = []
    seen_remote: list[tuple[str, int | None]] = []

    @contextmanager
    def _fake_tunnel(cfg: SshTunnelConfig, *, remote_host: str, remote_port: int | None) -> Any:
        seen_remote.append((remote_host, remote_port))
        # sqlite has no host/port; the rewrite is a no-op for the URL but the
        # tunnel still opens. Yield a sentinel local forward.
        try:
            yield ("127.0.0.1", 54321)
        finally:
            teardown.append(True)

    monkeypatch.setattr("dbsprout.web.routers.connect.open_ssh_tunnel", _fake_tunnel)
    resp = _client(tmp_path).post(
        "/api/connect",
        json={
            "url": url,
            "ssh": {
                "host": "bastion.example.com",
                "port": 22,
                "user": "deploy",
                "key_path": "/home/me/.ssh/id",
            },
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["table_count"] == 1
    assert teardown == [True]  # tunnel was torn down
    # The bastion host must never appear in the response body.
    assert "bastion.example.com" not in resp.text


def test_connect_test_with_ssh_opens_tunnel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextlib import contextmanager  # noqa: PLC0415

    url = _sqlite_url(tmp_path)
    teardown: list[bool] = []

    @contextmanager
    def _fake_tunnel(cfg: SshTunnelConfig, *, remote_host: str, remote_port: int | None) -> Any:
        try:
            yield ("127.0.0.1", 54321)
        finally:
            teardown.append(True)

    monkeypatch.setattr("dbsprout.web.routers.connect.open_ssh_tunnel", _fake_tunnel)
    resp = _client(tmp_path).post(
        "/api/connect/test",
        json={
            "url": url,
            "ssh": {"host": "bastion.example.com", "user": "deploy", "key_path": "/k"},
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["ok"] is True
    assert teardown == [True]
    assert "bastion.example.com" not in resp.text


def test_connect_with_ssh_missing_extra_is_503_not_500(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing [ssh] extra → typed 503 SSH_UNAVAILABLE, never an INTERNAL 500."""
    from contextlib import contextmanager  # noqa: PLC0415

    @contextmanager
    def _unavailable(cfg: SshTunnelConfig, *, remote_host: str, remote_port: int | None) -> Any:
        raise SshTunnelUnavailable("No module named 'sshtunnel'")
        yield  # pragma: no cover

    monkeypatch.setattr("dbsprout.web.routers.connect.open_ssh_tunnel", _unavailable)
    resp = _client(tmp_path).post(
        "/api/connect",
        json={
            "url": _sqlite_url(tmp_path),
            "ssh": {"host": "bastion.example.com", "user": "deploy", "key_path": "/k"},
        },
    )
    assert resp.status_code == 503, resp.text
    detail = resp.json()["detail"]
    assert detail["code"] == "SSH_UNAVAILABLE"
    assert "pip install dbsprout[ssh]" in detail["hint"]
    assert "Traceback" not in resp.text


def test_connect_test_with_ssh_missing_extra_is_503(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextlib import contextmanager  # noqa: PLC0415

    @contextmanager
    def _unavailable(cfg: SshTunnelConfig, *, remote_host: str, remote_port: int | None) -> Any:
        raise SshTunnelUnavailable("No module named 'sshtunnel'")
        yield  # pragma: no cover

    monkeypatch.setattr("dbsprout.web.routers.connect.open_ssh_tunnel", _unavailable)
    resp = _client(tmp_path).post(
        "/api/connect/test",
        json={
            "url": _sqlite_url(tmp_path),
            "ssh": {"host": "bastion.example.com", "user": "deploy", "key_path": "/k"},
        },
    )
    assert resp.status_code == 503
    assert resp.json()["detail"]["code"] == "SSH_UNAVAILABLE"


def test_connect_with_ssh_rewrites_tcp_host_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """For a TCP URL the handler must hand the loader the LOCAL forward host/port,
    not the remote one."""
    from contextlib import contextmanager  # noqa: PLC0415

    seen_urls: list[str] = []

    def _fake_load_schema(source: Any) -> Any:
        seen_urls.append(source.raw_value)
        raise ValueError("stop after capturing the rewritten url")

    @contextmanager
    def _fake_tunnel(cfg: SshTunnelConfig, *, remote_host: str, remote_port: int | None) -> Any:
        assert remote_host == "db.internal"
        assert remote_port == 5432
        yield ("127.0.0.1", 6000)

    monkeypatch.setattr("dbsprout.web.routers.connect.open_ssh_tunnel", _fake_tunnel)
    monkeypatch.setattr(
        "dbsprout.web.routers.connect.load_schema", _fake_load_schema, raising=False
    )
    resp = _client(tmp_path).post(
        "/api/connect",
        json={
            "url": "postgresql://u:p@db.internal:5432/app",
            "ssh": {"host": "bastion.example.com", "user": "deploy", "key_path": "/k"},
        },
    )
    # The loader saw the rewritten local host/port, never the remote.
    assert seen_urls, "load_schema was not called"
    rewritten = seen_urls[0]
    assert "127.0.0.1:6000" in rewritten
    assert "db.internal" not in rewritten
    # The failing loader is still a clean 4xx (no creds, no traceback).
    assert resp.status_code == 400
    assert "p@" not in resp.text  # password scrubbed
