"""PII column name detection for the ``redacted`` privacy tier.

Provides regex-based detection of personally identifiable information
in database column names. Presidio NER is an optional enhancement.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Patterns that indicate PII in column names. Matched case-insensitively
# as substrings. Ordered by specificity (longer patterns first) to avoid
# false positives from short matches.
_PII_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        # Compound patterns (safe as substrings)
        r"social_security",
        r"credit_card",
        r"card_number",
        r"date_of_birth",
        r"bank_account",
        r"routing_number",
        r"national_id",
        r"drivers_license",
        r"medical_record",
        r"email_address",
        r"zip_code",
        r"postal_code",
        # Word-boundary patterns (short, risk false positives without \b)
        r"(?:^|_)passport(?:$|_)",
        r"(?:^|_)phone(?:$|_)",
        r"(?:^|_)mobile(?:$|_)",
        r"(?:^|_)salary(?:$|_)",
        r"(?:^|_)income(?:$|_)",
        r"(?:^|_)address(?:$|_)",
        r"(?:^|_)email(?:$|_)",
        r"(?:^|_)tax_id(?:$|_)",
        r"\bssn\b",
        r"\bdob\b",
        r"\bcvv\b",
    )
)


class PIIDetector:
    """Detects PII column names using regex patterns.

    Uses Presidio NER as an enhancement when ``presidio-analyzer`` is
    installed. Falls back to regex-only detection otherwise.
    """

    def __init__(self) -> None:
        self._use_presidio = False
        try:
            from presidio_analyzer import AnalyzerEngine  # noqa: PLC0415

            self._analyzer = AnalyzerEngine()
            self._use_presidio = True
            logger.info("Presidio analyzer available — using NER-enhanced PII detection")
        except ImportError:
            self._analyzer = None
            logger.debug("Presidio not installed — using regex-only PII detection")

    def is_pii(self, column_name: str) -> bool:
        """Return True if the column name matches a known PII pattern."""
        if self._regex_match(column_name):
            return True
        if self._use_presidio:
            return self._presidio_match(column_name)
        return False

    def detect_pii_columns(self, columns: list[str]) -> set[str]:
        """Scan a list of column names and return those flagged as PII."""
        pii_cols = {col for col in columns if self.is_pii(col)}
        if pii_cols:
            logger.info("PII columns detected: %s", sorted(pii_cols))
        return pii_cols

    def _regex_match(self, column_name: str) -> bool:
        """Check column name against regex PII patterns."""
        return any(pattern.search(column_name) for pattern in _PII_PATTERNS)

    def _presidio_match(self, column_name: str) -> bool:
        """Check column name using Presidio NER (if available)."""
        if self._analyzer is None:
            return False
        # Convert underscores to spaces for NER analysis
        text = column_name.replace("_", " ")
        results = self._analyzer.analyze(text=text, language="en")
        return len(results) > 0
