"""Unit tests for the shared direct-insert dialect detection (S-150).

Characterization / parity tests: the consolidated ``detect_direct_dialect``
must reproduce the exact behavior previously duplicated byte-for-byte in the
CLI (``dbsprout/cli/commands/_direct_insert.py``) and web
(``dbsprout/web/routers/insert.py``) direct-insert paths.
"""

from __future__ import annotations

import pytest

from dbsprout.output.dialect import detect_direct_dialect

# URL -> expected dialect mapping captured verbatim from the original copies.
_URL_DIALECT_CASES = [
    # PostgreSQL (canonical return is "postgresql", not "postgres")
    ("postgresql://user:pw@host:5432/db", "postgresql"),
    ("postgres://user@host/db", "postgresql"),
    ("postgresql+psycopg://host/db", "postgresql"),
    ("POSTGRESQL://HOST/DB", "postgresql"),
    ("Postgres://Host/DB", "postgresql"),
    # MySQL
    ("mysql://user@host/db", "mysql"),
    ("mysql+pymysql://host/db", "mysql"),
    ("MySQL://Host/DB", "mysql"),
    # SQLite
    ("sqlite:///test.db", "sqlite"),
    ("sqlite+pysqlite:///:memory:", "sqlite"),
    ("SQLITE:///UPPER.DB", "sqlite"),
    # MSSQL
    ("mssql://user@host/db", "mssql"),
    ("mssql+pyodbc://host/db", "mssql"),
    ("MSSQL://HOST/DB", "mssql"),
    # Unsupported-but-URL: scheme prefix echoed back, "+driver" suffix stripped
    ("oracle://host/db", "oracle"),
    ("oracle+cx_oracle://host/db", "oracle"),
    ("cockroachdb://host/db", "cockroachdb"),
    ("vertica+vertica_python://h/db", "vertica"),
    # No "://" at all -> "unknown"
    ("/tmp/data.db", "unknown"),
    ("plainfilename", "unknown"),
    ("", "unknown"),
]


@pytest.mark.parametrize(("url", "expected"), _URL_DIALECT_CASES)
def test_detect_direct_dialect_mapping(url: str, expected: str) -> None:
    assert detect_direct_dialect(url) == expected


def test_detect_direct_dialect_is_case_insensitive() -> None:
    assert detect_direct_dialect("PoStGrEsQl://h/db") == "postgresql"
