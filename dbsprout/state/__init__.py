"""State layer: persistent SQLite store for generation-run telemetry.

The CLI writes runs to ``.dbsprout/state.db``; visual surfaces (HTML
report, TUI, web dashboard) read from it with zero coupling.
"""

from __future__ import annotations

from dbsprout.state.db import SCHEMA_VERSION, StateDB
from dbsprout.state.models import LLMCall, QualityResult, RunRecord, TableStats

__all__ = [
    "SCHEMA_VERSION",
    "LLMCall",
    "QualityResult",
    "RunRecord",
    "StateDB",
    "TableStats",
]
