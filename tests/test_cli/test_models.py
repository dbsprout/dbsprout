"""Tests for the dbsprout models package and CLI."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from dbsprout.models import InstalledModel, ModelEntry


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
