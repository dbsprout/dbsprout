"""HTML report module.

Renders a self-contained HTML report (all CSS/JS/data inline, no server,
no external dependencies) from the shared state DB (S-079/S-080). ERD,
distribution charts and data preview are extension-point placeholders
filled by S-082/S-083/S-084.
"""

from __future__ import annotations

from dbsprout.report.context import build_report_context
from dbsprout.report.generator import DEFAULT_OUTPUT_PATH, ReportGenerator

__all__ = [
    "DEFAULT_OUTPUT_PATH",
    "ReportGenerator",
    "build_report_context",
]
