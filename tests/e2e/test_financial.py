"""E2E: financial schema (double-entry ledger, FK chains, enums)."""

from __future__ import annotations

from pathlib import Path

import pytest

from dbsprout.schema.models import ColumnType
from tests.e2e._pipeline import (
    assert_full_integrity,
    assert_programmatic_integrity,
    run_pipeline,
)

FIXTURE = Path(__file__).parent.parent / "fixtures" / "schemas" / "financial.sql"


@pytest.mark.integration
def test_financial_full_integrity(tmp_path: Path) -> None:
    result = run_pipeline(FIXTURE, tmp_path, rows=50, seed=42)
    assert_full_integrity(result)
    assert_programmatic_integrity(result)
    assert len(result.schema.tables) >= 10


@pytest.mark.integration
def test_financial_ledger_fk_chain(tmp_path: Path) -> None:
    result = run_pipeline(FIXTURE, tmp_path, rows=50, seed=42)
    schema = result.schema

    le = schema.get_table("ledger_entries")
    assert le is not None
    refs = {fk.ref_table for fk in le.foreign_keys}
    assert {"transactions", "accounts"} <= refs

    txns = schema.get_table("transactions")
    assert txns is not None
    status = next(c for c in txns.columns if c.name == "status")
    assert status.data_type is ColumnType.ENUM
    assert status.enum_values
