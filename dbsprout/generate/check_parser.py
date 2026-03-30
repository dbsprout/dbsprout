"""SQL CHECK expression parser — extracts range and enum constraints.

Parses common CHECK patterns (>, >=, <, <=, BETWEEN, IN, AND)
into a CheckConstraint dataclass for generation domain narrowing.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_RE_GTE = re.compile(r">=\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
_RE_GT = re.compile(r">\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
_RE_LTE = re.compile(r"<=\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
_RE_LT = re.compile(r"<\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
_RE_BETWEEN = re.compile(
    r"BETWEEN\s+(-?\d+(?:\.\d+)?)\s+AND\s+(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_RE_IN = re.compile(r"IN\s*\(\s*((?:'[^']*'(?:\s*,\s*'[^']*')*)\s*)\)", re.IGNORECASE)


@dataclass(frozen=True)
class CheckConstraint:
    """Parsed CHECK constraint with domain bounds."""

    column: str
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: list[str] | None = None


def parse_check(column_name: str, expr: str) -> CheckConstraint | None:
    """Parse a SQL CHECK expression into a CheckConstraint.

    Returns ``None`` if the expression cannot be parsed.
    Handles: ``>``, ``>=``, ``<``, ``<=``, ``BETWEEN``, ``IN``,
    and ``AND`` combinations.
    """
    if not expr or not expr.strip():
        return None

    expr = expr.strip()

    # BETWEEN X AND Y
    m = _RE_BETWEEN.search(expr)
    if m:
        return CheckConstraint(
            column=column_name,
            min_value=float(m.group(1)),
            max_value=float(m.group(2)),
        )

    m = _RE_IN.search(expr)
    if m:
        raw = m.group(1)
        values = [v.strip().strip("'") for v in raw.split(",")]
        return CheckConstraint(column=column_name, allowed_values=values)

    # Combined AND: col >= X AND col <= Y
    if re.search(r"\bAND\b", expr, re.IGNORECASE):
        return _parse_and_combination(column_name, expr)

    # Single comparison
    return _parse_single_comparison(column_name, expr)


def _parse_and_combination(column_name: str, expr: str) -> CheckConstraint | None:
    """Parse AND-combined range expressions."""
    parts = re.split(r"\bAND\b", expr, flags=re.IGNORECASE)
    min_val: float | None = None
    max_val: float | None = None

    for part in parts:
        parsed = _parse_single_comparison(column_name, part.strip())
        if parsed is None:
            continue
        if parsed.min_value is not None:
            min_val = parsed.min_value
        if parsed.max_value is not None:
            max_val = parsed.max_value

    if min_val is not None or max_val is not None:
        return CheckConstraint(column=column_name, min_value=min_val, max_value=max_val)
    return None


def _parse_single_comparison(
    column_name: str,
    expr: str,
) -> CheckConstraint | None:
    """Parse a single comparison expression."""
    # >= (must check before >)
    m = _RE_GTE.search(expr)
    if m:
        return CheckConstraint(column=column_name, min_value=float(m.group(1)))

    # > (exclusive → add 1 for integer semantics)
    m = _RE_GT.search(expr)
    if m:
        val = float(m.group(1))
        return CheckConstraint(column=column_name, min_value=val + 1)

    # <= (must check before <)
    m = _RE_LTE.search(expr)
    if m:
        return CheckConstraint(column=column_name, max_value=float(m.group(1)))

    # < (exclusive → subtract 1 for integer semantics)
    m = _RE_LT.search(expr)
    if m:
        val = float(m.group(1))
        return CheckConstraint(column=column_name, max_value=val - 1)

    logger.warning("Cannot parse CHECK expression: %s", expr)
    return None
