"""Tests for dbsprout.privacy.enforcer — PrivacyTier, PrivacyError, PrivacyEnforcer."""

from __future__ import annotations

import logging

import pytest

from dbsprout.privacy.enforcer import PrivacyEnforcer, PrivacyError, PrivacyTier


class TestPrivacyTier:
    """PrivacyTier enum values and defaults."""

    def test_has_local_value(self) -> None:
        assert PrivacyTier.LOCAL == "local"

    def test_has_redacted_value(self) -> None:
        assert PrivacyTier.REDACTED == "redacted"

    def test_has_cloud_value(self) -> None:
        assert PrivacyTier.CLOUD == "cloud"

    def test_default_is_local(self) -> None:
        assert PrivacyTier.LOCAL.value == "local"

    def test_is_string_enum(self) -> None:
        assert isinstance(PrivacyTier.LOCAL, str)


class TestPrivacyError:
    """PrivacyError exception."""

    def test_inherits_from_exception(self) -> None:
        assert issubclass(PrivacyError, Exception)

    def test_message_contains_tier_and_provider(self) -> None:
        err = PrivacyError(tier=PrivacyTier.LOCAL, provider_locality="cloud")
        assert "local" in str(err)
        assert "cloud" in str(err)


class TestPrivacyEnforcerValidateProvider:
    """PrivacyEnforcer.validate_provider — tier enforcement matrix."""

    def test_local_tier_blocks_cloud_provider(self) -> None:
        enforcer = PrivacyEnforcer()
        with pytest.raises(PrivacyError):
            enforcer.validate_provider(provider_locality="cloud", tier=PrivacyTier.LOCAL)

    def test_local_tier_allows_local_provider(self) -> None:
        enforcer = PrivacyEnforcer()
        enforcer.validate_provider(provider_locality="local", tier=PrivacyTier.LOCAL)

    def test_redacted_tier_allows_cloud_provider(self) -> None:
        enforcer = PrivacyEnforcer()
        enforcer.validate_provider(provider_locality="cloud", tier=PrivacyTier.REDACTED)

    def test_redacted_tier_allows_local_provider(self) -> None:
        enforcer = PrivacyEnforcer()
        enforcer.validate_provider(provider_locality="local", tier=PrivacyTier.REDACTED)

    def test_cloud_tier_allows_cloud_provider(self) -> None:
        enforcer = PrivacyEnforcer()
        enforcer.validate_provider(provider_locality="cloud", tier=PrivacyTier.CLOUD)

    def test_cloud_tier_allows_local_provider(self) -> None:
        enforcer = PrivacyEnforcer()
        enforcer.validate_provider(provider_locality="local", tier=PrivacyTier.CLOUD)

    def test_audit_log_emitted(self, caplog: pytest.LogCaptureFixture) -> None:
        enforcer = PrivacyEnforcer()
        with caplog.at_level(logging.INFO):
            enforcer.validate_provider(provider_locality="local", tier=PrivacyTier.LOCAL)
        assert "local" in caplog.text
