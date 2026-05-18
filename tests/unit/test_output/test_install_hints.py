"""Install-hint consistency across writers + fidelity (S-094)."""

from __future__ import annotations

import inspect
import re

import pytest

from dbsprout.output import mysql_load_data, pg_copy
from dbsprout.quality import fidelity

_QUOTED = re.compile(r'pip install "dbsprout\[[a-z]+\]"')


def test_install_hints_use_quoted_consistent_format() -> None:
    parquet_writer = pytest.importorskip(
        "dbsprout.output.parquet_writer",
        reason="polars not installed ([data] extra)",
    )
    sources = {
        "parquet_writer": inspect.getsource(parquet_writer),
        "pg_copy": inspect.getsource(pg_copy),
        "mysql_load_data": inspect.getsource(mysql_load_data),
        "fidelity": inspect.getsource(fidelity),
    }
    for name, src in sources.items():
        assert "pip install dbsprout[" not in src, f"unquoted hint in {name}"
        assert _QUOTED.search(src), f"no quoted hint found in {name}"
