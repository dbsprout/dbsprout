"""Privacy tier enforcement, PII detection, schema redaction, and audit logging.

Public API:
- ``PrivacyTier`` — enum of privacy tiers (local, redacted, cloud)
- ``PrivacyError`` — raised on tier violations
- ``PrivacyEnforcer`` — validates provider calls against tier
- ``PIIDetector`` — detects PII column names via regex (+ Presidio)
- ``RedactionMap`` — stores original→hashed name mappings
- ``redact_schema`` — returns a redacted schema + mapping
- ``de_redact_spec`` — reverses hashed names in a DataSpec
- ``AuditEvent`` — single audit log entry model
- ``AuditLog`` — append-only JSON Lines audit log
"""

from dbsprout.privacy.audit import AuditEvent, AuditLog
from dbsprout.privacy.enforcer import PrivacyEnforcer, PrivacyError, PrivacyTier
from dbsprout.privacy.pii import PIIDetector
from dbsprout.privacy.redactor import RedactionMap, de_redact_spec, redact_schema

__all__ = [
    "AuditEvent",
    "AuditLog",
    "PIIDetector",
    "PrivacyEnforcer",
    "PrivacyError",
    "PrivacyTier",
    "RedactionMap",
    "de_redact_spec",
    "redact_schema",
]
