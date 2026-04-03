"""Privacy tier enforcement for spec generation providers.

Validates that a provider's locality classification is compatible with
the configured privacy tier before any LLM invocation.
"""

from __future__ import annotations

import logging
from enum import Enum, unique

logger = logging.getLogger(__name__)


@unique
class PrivacyTier(str, Enum):
    """Privacy tier controlling data flow to LLM providers."""

    LOCAL = "local"
    REDACTED = "redacted"
    CLOUD = "cloud"


class PrivacyError(Exception):
    """Raised when a provider call violates the configured privacy tier."""

    def __init__(self, *, tier: PrivacyTier, provider_locality: str) -> None:
        self.tier = tier
        self.provider_locality = provider_locality
        super().__init__(
            f"Privacy violation: tier {tier.value!r} does not allow "
            f"provider with locality {provider_locality!r}"
        )


class PrivacyEnforcer:
    """Validates provider calls against the configured privacy tier.

    Call ``validate_provider()`` before any LLM invocation.
    Raises ``PrivacyError`` if the provider's locality is incompatible
    with the privacy tier.
    """

    def validate_provider(
        self,
        *,
        provider_locality: str,
        tier: PrivacyTier,
    ) -> None:
        """Validate that a provider is allowed under the given tier.

        Parameters
        ----------
        provider_locality:
            The provider's locality classification (``"local"`` or ``"cloud"``).
        tier:
            The configured privacy tier.

        Raises
        ------
        PrivacyError
            If the tier blocks this provider locality.
        """
        logger.info(
            "Privacy check: tier=%s, provider_locality=%s",
            tier.value,
            provider_locality,
        )

        if tier == PrivacyTier.LOCAL and provider_locality == "cloud":
            raise PrivacyError(tier=tier, provider_locality=provider_locality)
