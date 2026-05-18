"""MySQL _quote_string control-char escaping (S-094 / S-045 review)."""

from __future__ import annotations

from dbsprout.output.sql_writer import _quote_string, get_dialect_config


def test_mysql_quote_string_escapes_control_chars() -> None:
    cfg = get_dialect_config("mysql")
    out = _quote_string("a\x00b\nc\rd\x1ae", cfg)
    assert "\\0" in out
    assert "\\n" in out
    assert "\\r" in out
    assert "\\Z" in out
    assert "\x00" not in out
    assert "\x1a" not in out


def test_mysql_quote_string_still_escapes_quote_and_backslash() -> None:
    cfg = get_dialect_config("mysql")
    out = _quote_string("a'b\\c", cfg)
    assert "\\'" in out
    assert "\\\\" in out


def test_standard_dialect_quote_string_unchanged() -> None:
    cfg = get_dialect_config("postgresql")
    assert _quote_string("a'b", cfg) == "'a''b'"
    assert _quote_string("a\nb", cfg) == "'a\nb'"
