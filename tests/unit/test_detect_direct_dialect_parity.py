"""Parity + characterization tests for ``detect_direct_dialect`` (S-150).

The SQLAlchemy-URL→dialect mapping was historically duplicated byte-for-byte
in the CLI direct-insert adapter and the web insert router. S-150 consolidates
both copies onto a single shared implementation in
:mod:`dbsprout.output.dialect`.

These tests pin the exact mapping and assert that every call site resolves to
the *same* shared function object, so the two paths can never drift again.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import dbsprout.cli.commands._direct_insert as cli_mod
import dbsprout.web.routers.insert as web_mod
from dbsprout.cli.commands._direct_insert import _detect_direct_dialect as cli_fn
from dbsprout.output.dialect import detect_direct_dialect as shared_fn
from dbsprout.web.routers.insert import _detect_direct_dialect as web_fn

# Full URL→dialect mapping that both insert paths must agree on. This is the
# behavioural contract; it must stay byte-identical across the consolidation.
URL_DIALECT_CASES = [
    ("postgresql://host/db", "postgresql"),
    ("postgres://host/db", "postgresql"),
    ("postgresql+psycopg://host/db", "postgresql"),
    ("POSTGRESQL://HOST/DB", "postgresql"),
    ("mysql://host/db", "mysql"),
    ("mysql+pymysql://host/db", "mysql"),
    ("sqlite:///test.db", "sqlite"),
    ("sqlite:///:memory:", "sqlite"),
    ("mssql+pyodbc://host/db", "mssql"),
    ("oracle://host/db", "oracle"),
    ("oracle+cx_oracle://host/db", "oracle"),
    ("not-a-url", "unknown"),
    ("", "unknown"),
]


@pytest.mark.parametrize(("url", "expected"), URL_DIALECT_CASES)
def test_shared_detect_direct_dialect(url: str, expected: str) -> None:
    """The shared implementation maps each URL to the expected dialect."""
    assert shared_fn(url) == expected


@pytest.mark.parametrize(("url", "expected"), URL_DIALECT_CASES)
def test_cli_and_web_share_one_impl(url: str, expected: str) -> None:
    """CLI and web call sites resolve to the identical shared function."""
    # Same object — single source of truth (AC: CLI and web share one impl).
    assert cli_fn is shared_fn
    assert web_fn is shared_fn

    # Behaviour byte-identical across all three references.
    assert shared_fn(url) == expected
    assert cli_fn(url) == expected
    assert web_fn(url) == expected


def _imports_module(source: str, package: str) -> bool:
    """True if *source* has a real ``import``/``from`` statement for *package*."""
    pattern = rf"^\s*(?:from\s+{re.escape(package)}[\s.]|import\s+{re.escape(package)}[\s.])"
    return bool(re.search(pattern, source, re.MULTILINE))


def test_no_cross_stage_import() -> None:
    """The shared home is neutral: CLI and web don't import each other."""
    cli_src = Path(cli_mod.__file__).read_text(encoding="utf-8")
    web_src = Path(web_mod.__file__).read_text(encoding="utf-8")

    # Neither stage may import the other (prose mentions in docstrings are fine).
    assert not _imports_module(cli_src, "dbsprout.web")
    assert not _imports_module(web_src, "dbsprout.cli")

    # Both depend on the shared output/ home for the dialect helper.
    assert _imports_module(cli_src, "dbsprout.output.dialect")
    assert _imports_module(web_src, "dbsprout.output.dialect")
