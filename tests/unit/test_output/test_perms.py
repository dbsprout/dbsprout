"""Tests for the shared output file-permission helper (S-094)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from dbsprout.output._perms import restrict_file_permissions

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions only")
def test_restrict_file_permissions_sets_0o640(tmp_path: Path) -> None:
    f = tmp_path / "out.sql"
    f.write_text("x", encoding="utf-8")
    os.chmod(f, 0o644)
    restrict_file_permissions(f)
    assert (f.stat().st_mode & 0o777) == 0o640


def test_restrict_file_permissions_missing_file_is_noop(tmp_path: Path) -> None:
    restrict_file_permissions(tmp_path / "does-not-exist.sql")


def test_restrict_file_permissions_non_posix_is_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    f = tmp_path / "out.csv"
    f.write_text("x", encoding="utf-8")
    monkeypatch.setattr("dbsprout.output._perms.os.name", "nt")
    restrict_file_permissions(f)
