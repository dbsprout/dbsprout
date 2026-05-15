"""Tests for the dbsprout models package and CLI."""

from __future__ import annotations

import re
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from dbsprout.cli.app import app
from dbsprout.cli.commands import models as models_mod
from dbsprout.cli.commands.models import _make_client
from dbsprout.models import InstalledModel, ModelEntry, ModelManager, load_registry
from dbsprout.models import manager as manager_mod
from dbsprout.models.manager import DownloadError, _resolve_hf_url

runner = CliRunner()


def _strip_ansi(text: str) -> str:
    return re.compile(r"\x1b\[[0-9;]*m").sub("", text)


class TestModelEntry:
    def test_frozen_and_fields(self) -> None:
        entry = ModelEntry(
            name="qwen2.5-1.5b-instruct",
            description="Qwen2.5 1.5B Instruct (GGUF)",
            repo="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
            filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
            size_bytes=1_000_000_000,
            quantization="Q4_K_M",
            parameters="1.5B",
        )
        assert entry.training == "base"
        assert entry.default is False
        with pytest.raises(ValueError, match="frozen"):
            entry.name = "other"  # type: ignore[misc]


class TestInstalledModel:
    def test_fields(self) -> None:
        im = InstalledModel(name="my-ft", path=Path("/tmp/m.gguf"), size_bytes=42, kind="custom")  # noqa: S108
        assert im.kind == "custom"
        assert im.size_bytes == 42


class TestLoadRegistry:
    def test_bundled_registry_has_default(self) -> None:
        entries = load_registry()
        assert len(entries) >= 1
        names = {e.name for e in entries}
        assert "qwen2.5-1.5b-instruct" in names
        default = [e for e in entries if e.default]
        assert len(default) == 1
        d = default[0]
        assert d.repo == "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
        assert d.filename == "qwen2.5-1.5b-instruct-q4_k_m.gguf"

    def test_load_registry_from_explicit_path(self, tmp_path: Path) -> None:
        manifest = tmp_path / "r.json"
        manifest.write_text(
            '[{"name":"m","description":"d","repo":"r","filename":"f.gguf",'
            '"size_bytes":1,"quantization":"Q4","parameters":"1B"}]',
            encoding="utf-8",
        )
        entries = manager_mod.load_registry(manifest)
        assert entries[0].name == "m"

    def test_invalid_manifest_raises_valueerror(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("not json", encoding="utf-8")
        with pytest.raises(ValueError, match="invalid model registry"):
            manager_mod.load_registry(bad)


class TestModelManagerDiscovery:
    def test_discover_empty(self, tmp_path: Path) -> None:
        mgr = ModelManager(root=tmp_path / ".dbsprout" / "models")
        assert mgr.list_installed() == []

    def test_discover_base_and_custom(self, tmp_path: Path) -> None:
        root = tmp_path / ".dbsprout" / "models"
        (root / "base").mkdir(parents=True)
        (root / "custom").mkdir(parents=True)
        (root / "base" / "qwen2.5-1.5b-instruct-q4_k_m.gguf").write_bytes(b"x" * 10)
        (root / "custom" / "my-ft.gguf").write_bytes(b"y" * 20)

        mgr = ModelManager(root=root)
        installed = {im.name: im for im in mgr.list_installed()}
        assert installed["qwen2.5-1.5b-instruct-q4_k_m.gguf"].kind == "base"
        assert installed["qwen2.5-1.5b-instruct-q4_k_m.gguf"].size_bytes == 10
        assert installed["my-ft"].kind == "custom"
        assert installed["my-ft"].size_bytes == 20

    def test_is_installed_matches_registry_filename(self, tmp_path: Path) -> None:
        root = tmp_path / ".dbsprout" / "models"
        (root / "base").mkdir(parents=True)
        entry = next(e for e in load_registry() if e.default)
        (root / "base" / entry.filename).write_bytes(b"z" * 5)

        mgr = ModelManager(root=root)
        assert mgr.is_installed(entry) is True
        assert mgr.install_path(entry) == root / "base" / entry.filename

    def test_resolve_entry_unknown_raises_keyerror(self, tmp_path: Path) -> None:
        mgr = ModelManager(root=tmp_path)
        with pytest.raises(KeyError):
            mgr.resolve_entry("does-not-exist")


def _mock_client(handler: object) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


class TestDownload:
    def test_resolve_hf_url(self) -> None:
        url = _resolve_hf_url("Org/Repo-GGUF", "model.gguf")
        assert url == "https://huggingface.co/Org/Repo-GGUF/resolve/main/model.gguf"

    def test_download_happy_path(self, tmp_path: Path) -> None:
        entry = next(e for e in load_registry() if e.default)
        body = b"GGUFDATA" * 4

        def handler(request: httpx.Request) -> httpx.Response:
            assert "resolve/main" in str(request.url)
            return httpx.Response(200, content=body, headers={"Content-Length": str(len(body))})

        progress: list[tuple[int, int]] = []
        mgr = ModelManager(root=tmp_path / ".dbsprout" / "models")
        dest = mgr.download(
            entry,
            client=_mock_client(handler),
            progress_cb=lambda done, total: progress.append((done, total)),
        )
        assert dest == mgr.install_path(entry)
        assert dest.read_bytes() == body
        assert not dest.with_suffix(dest.suffix + ".part").exists()
        assert progress
        assert progress[-1][0] == len(body)

    def test_download_resume_206(self, tmp_path: Path) -> None:
        entry = next(e for e in load_registry() if e.default)
        full = b"0123456789ABCDEF"
        mgr = ModelManager(root=tmp_path / ".dbsprout" / "models")
        part = mgr.install_path(entry).with_suffix(mgr.install_path(entry).suffix + ".part")
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(full[:6])

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers.get("Range") == "bytes=6-"
            return httpx.Response(
                206, content=full[6:], headers={"Content-Length": str(len(full) - 6)}
            )

        dest = mgr.download(entry, client=_mock_client(handler))
        assert dest.read_bytes() == full

    def test_download_restart_200_ignores_range(self, tmp_path: Path) -> None:
        entry = next(e for e in load_registry() if e.default)
        mgr = ModelManager(root=tmp_path / ".dbsprout" / "models")
        part = mgr.install_path(entry).with_suffix(mgr.install_path(entry).suffix + ".part")
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(b"STALE")
        full = b"FRESHCONTENT"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=full)

        dest = mgr.download(entry, client=_mock_client(handler))
        assert dest.read_bytes() == full

    def test_download_network_error_keeps_part(self, tmp_path: Path) -> None:
        entry = next(e for e in load_registry() if e.default)
        mgr = ModelManager(root=tmp_path / ".dbsprout" / "models")

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("boom")

        with pytest.raises(DownloadError, match="resume"):
            mgr.download(entry, client=_mock_client(handler))

    def test_download_skip_when_installed(self, tmp_path: Path) -> None:
        entry = next(e for e in load_registry() if e.default)
        mgr = ModelManager(root=tmp_path / ".dbsprout" / "models")
        mgr.install_path(entry).parent.mkdir(parents=True, exist_ok=True)
        mgr.install_path(entry).write_bytes(b"already")

        dest = mgr.download(entry, client=None, force=False)
        assert dest.read_bytes() == b"already"

    def test_download_indeterminate_no_content_length(self, tmp_path: Path) -> None:
        entry = next(e for e in load_registry() if e.default)
        body = b"NOLEN"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=body)

        seen: list[tuple[int, int]] = []
        mgr = ModelManager(root=tmp_path / ".dbsprout" / "models")
        dest = mgr.download(
            entry,
            client=_mock_client(handler),
            progress_cb=lambda d, t: seen.append((d, t)),
        )
        assert dest.read_bytes() == body
        assert seen[-1] == (len(body), len(body))

    def test_download_creates_real_client_when_none(self, tmp_path: Path) -> None:
        # force=False + already installed avoids any network: exercises the
        # client=None branch without opening a real connection.
        entry = next(e for e in load_registry() if e.default)
        mgr = ModelManager(root=tmp_path / ".dbsprout" / "models")
        mgr.install_path(entry).parent.mkdir(parents=True, exist_ok=True)
        mgr.install_path(entry).write_bytes(b"x")
        assert mgr.download(entry).read_bytes() == b"x"


class TestModelsCLIListInfo:
    def test_list_shows_registry_and_custom(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        root = tmp_path / ".dbsprout" / "models"
        (root / "custom").mkdir(parents=True)
        (root / "custom" / "my-ft.gguf").write_bytes(b"q" * 7)

        result = runner.invoke(app, ["models", "list"])
        out = _strip_ansi(result.output)
        assert result.exit_code == 0
        assert "qwen2.5-1.5b-instruct" in out
        assert "my-ft" in out
        assert "Q4_K_M" in out

    def test_info_known_model(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["models", "info", "qwen2.5-1.5b-instruct"])
        out = _strip_ansi(result.output)
        assert result.exit_code == 0
        assert "1.5B" in out
        assert "Q4_K_M" in out
        assert "Qwen/Qwen2.5-1.5B-Instruct-GGUF" in out

    def test_info_unknown_model_exits_nonzero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["models", "info", "nope"])
        out = _strip_ansi(result.output)
        assert result.exit_code != 0
        assert "unknown model" in out.lower()

    def test_info_known_model_installed_shows_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        entry = next(e for e in load_registry() if e.default)
        mgr = ModelManager()
        mgr.install_path(entry).parent.mkdir(parents=True, exist_ok=True)
        mgr.install_path(entry).write_bytes(b"installed")
        result = runner.invoke(app, ["models", "info", entry.name])
        out = _strip_ansi(result.output)
        assert result.exit_code == 0
        assert "yes" in out.lower()

    def test_models_help(self) -> None:
        result = runner.invoke(app, ["models", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "download" in result.output
        assert "info" in result.output


class TestModelsCLIDownload:
    def test_download_unknown_model(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["models", "download", "nope"])
        assert result.exit_code != 0
        assert "unknown model" in _strip_ansi(result.output).lower()

    def test_download_already_installed_skips(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        entry = next(e for e in load_registry() if e.default)
        mgr = ModelManager()
        mgr.install_path(entry).parent.mkdir(parents=True, exist_ok=True)
        mgr.install_path(entry).write_bytes(b"installed")

        result = runner.invoke(app, ["models", "download", entry.name])
        out = _strip_ansi(result.output)
        assert result.exit_code == 0
        assert "already installed" in out.lower()

    def test_download_success(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        body = b"GGUF-BYTES" * 3

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=body, headers={"Content-Length": str(len(body))})

        monkeypatch.setattr(
            models_mod,
            "_make_client",
            lambda: httpx.Client(transport=httpx.MockTransport(handler)),
        )

        entry = next(e for e in load_registry() if e.default)
        result = runner.invoke(app, ["models", "download", entry.name])
        out = _strip_ansi(result.output)
        assert result.exit_code == 0, out
        assert ModelManager().install_path(entry).read_bytes() == body
        assert "downloaded" in out.lower()

    def test_download_force_redownloads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        entry = next(e for e in load_registry() if e.default)
        mgr = ModelManager()
        mgr.install_path(entry).parent.mkdir(parents=True, exist_ok=True)
        mgr.install_path(entry).write_bytes(b"old")
        fresh = b"NEWMODELBYTES"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=fresh, headers={"Content-Length": str(len(fresh))})

        monkeypatch.setattr(
            models_mod,
            "_make_client",
            lambda: httpx.Client(transport=httpx.MockTransport(handler)),
        )

        result = runner.invoke(app, ["models", "download", entry.name, "--force"])
        assert result.exit_code == 0, _strip_ansi(result.output)
        assert mgr.install_path(entry).read_bytes() == fresh

    def test_download_network_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no network")

        monkeypatch.setattr(
            models_mod,
            "_make_client",
            lambda: httpx.Client(transport=httpx.MockTransport(handler)),
        )

        entry = next(e for e in load_registry() if e.default)
        result = runner.invoke(app, ["models", "download", entry.name])
        out = _strip_ansi(result.output)
        assert result.exit_code != 0
        assert "resume" in out.lower()


class TestMakeClient:
    def test_make_client_returns_httpx_client(self) -> None:
        client = _make_client()
        try:
            assert isinstance(client, httpx.Client)
        finally:
            client.close()
