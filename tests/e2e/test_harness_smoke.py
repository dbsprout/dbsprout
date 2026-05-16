"""Temporary self-test for the E2E pipeline harness (removed in Task 8)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.e2e._pipeline import (
    assert_full_integrity,
    assert_programmatic_integrity,
    run_pipeline,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "schemas"


@pytest.mark.integration
def test_harness_runs_full_pipeline(tmp_path: Path) -> None:
    result = run_pipeline(FIXTURES / "_harness.sql", tmp_path, rows=20, seed=42)
    assert result.validate_exit == 0
    assert set(result.seed_data) == {"parent", "child"}
    assert len(result.seed_data["parent"]) == 20
    assert_full_integrity(result)
    assert_programmatic_integrity(result)
