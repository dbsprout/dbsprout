"""CLI startup-time performance benchmark (S-074)."""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

pytestmark = pytest.mark.benchmark


def test_bench_cli_startup(benchmark: BenchmarkFixture) -> None:
    """Time ``dbsprout --help`` via a subprocess (cold import each call)."""

    def _run() -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [sys.executable, "-m", "dbsprout", "--help"],
            capture_output=True,
            check=True,
        )

    result = benchmark(_run)
    assert result.returncode == 0
    assert b"Usage" in result.stdout
