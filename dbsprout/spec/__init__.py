"""Spec generation — heuristic and LLM-based column mapping."""

from __future__ import annotations

from dbsprout.spec.heuristics import map_columns
from dbsprout.spec.models import GeneratorMapping
from dbsprout.spec.patterns import PATTERNS

__all__ = [
    "PATTERNS",
    "GeneratorMapping",
    "map_columns",
]
