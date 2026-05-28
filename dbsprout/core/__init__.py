"""Core service facade — the single orchestration seam (S-106).

``dbsprout.core.service`` composes the existing five stages (schema → spec →
generate → output → quality) into UI-agnostic entry points that BOTH the CLI
and the future web UI call. Importing ``dbsprout.core`` pulls in NO ``typer``,
``rich``, or ``fastapi`` — the facade sits *above* the stages, exactly where
the CLI sits today, and never below them.
"""

from __future__ import annotations

from dbsprout.core.service import (
    ValidationOutcome,
    WriteOutcome,
    generate,
    load_schema,
    run_validation,
    validate_integrity,
    write_output,
)

__all__ = [
    "ValidationOutcome",
    "WriteOutcome",
    "generate",
    "load_schema",
    "run_validation",
    "validate_integrity",
    "write_output",
]
