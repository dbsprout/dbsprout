"""load_reference_csv hardening: row cap + formula injection (S-094)."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import pytest

from dbsprout.quality.fidelity import load_reference_csv

if TYPE_CHECKING:
    from pathlib import Path


def test_load_reference_csv_signature_drops_table_name() -> None:
    params = list(inspect.signature(load_reference_csv).parameters)
    assert params == ["path"]


def test_load_reference_csv_caps_rows(tmp_path: Path) -> None:
    p = tmp_path / "ref.csv"
    lines = ["v", *[str(i) for i in range(100_005)]]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"100000-row cap"):
        load_reference_csv(p)


def test_load_reference_csv_at_cap_is_ok(tmp_path: Path) -> None:
    p = tmp_path / "ref.csv"
    lines = ["v", *[str(i) for i in range(100_000)]]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rows = load_reference_csv(p)
    assert len(rows) == 100_000


@pytest.mark.parametrize("prefix", ["=", "+", "-", "@"])
def test_load_reference_csv_sanitizes_formula_injection(tmp_path: Path, prefix: str) -> None:
    p = tmp_path / "ref.csv"
    p.write_text(f"name\n{prefix}SUM(A1)\n", encoding="utf-8")
    rows = load_reference_csv(p)
    assert not str(rows[0]["name"]).startswith(prefix)
    assert str(rows[0]["name"]).startswith("'")
