"""Tests for the dbsprout models package and CLI."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from dbsprout.models import InstalledModel, ModelEntry, ModelManager, load_registry
from dbsprout.models import manager as manager_mod


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
