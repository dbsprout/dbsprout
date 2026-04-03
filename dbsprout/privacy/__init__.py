"""Privacy tier enforcement and schema redaction.

Public API:
- ``PrivacyTier`` — enum of privacy tiers (local, redacted, cloud)
- ``PrivacyError`` — raised on tier violations
- ``PrivacyEnforcer`` — validates provider calls against tier
- ``redact_schema`` — returns a redacted copy of a DatabaseSchema
"""

from dbsprout.privacy.enforcer import PrivacyEnforcer, PrivacyError, PrivacyTier
from dbsprout.privacy.redactor import redact_schema

__all__ = [
    "PrivacyEnforcer",
    "PrivacyError",
    "PrivacyTier",
    "redact_schema",
]
