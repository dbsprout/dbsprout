"""Tests for SpecUsage — token/cost accounting surfaced by spec providers (S-080a)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dbsprout.spec.providers.base import (
    SpecProvider,
    SpecUsage,
    UsageReportingProvider,
    read_usage,
)


class TestSpecUsageDefaults:
    def test_defaults_are_honest_zeros(self) -> None:
        usage = SpecUsage()
        assert usage.tokens_sent == 0
        assert usage.tokens_received == 0
        assert usage.cost_usd == 0.0

    def test_accepts_real_values(self) -> None:
        usage = SpecUsage(tokens_sent=120, tokens_received=340, cost_usd=0.0042)
        assert usage.tokens_sent == 120
        assert usage.tokens_received == 340
        assert usage.cost_usd == 0.0042


class TestSpecUsageImmutability:
    def test_frozen(self) -> None:
        usage = SpecUsage(tokens_sent=1)
        with pytest.raises(ValidationError):
            usage.tokens_sent = 2  # type: ignore[misc]

    def test_extra_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            SpecUsage(unknown_field=1)  # type: ignore[call-arg]


class _LegacyProvider:
    def generate_spec(self, schema: object) -> object:
        return object()


class _UsageProvider:
    def generate_spec(self, schema: object) -> object:
        return object()

    def get_last_usage(self) -> SpecUsage | None:
        return SpecUsage(tokens_sent=7, tokens_received=11, cost_usd=0.5)


class TestSpecProviderProtocolBackCompat:
    def test_object_with_only_generate_spec_is_still_a_provider(self) -> None:
        """Existing providers (no get_last_usage) must still satisfy the protocol."""
        assert isinstance(_LegacyProvider(), SpecProvider)

    def test_legacy_provider_is_not_a_usage_reporting_provider(self) -> None:
        assert not isinstance(_LegacyProvider(), UsageReportingProvider)

    def test_usage_provider_satisfies_both_protocols(self) -> None:
        provider = _UsageProvider()
        assert isinstance(provider, SpecProvider)
        assert isinstance(provider, UsageReportingProvider)


class TestReadUsage:
    def test_read_usage_none_for_legacy_provider(self) -> None:
        assert read_usage(_LegacyProvider()) is None

    def test_read_usage_returns_spec_usage(self) -> None:
        usage = read_usage(_UsageProvider())
        assert usage == SpecUsage(tokens_sent=7, tokens_received=11, cost_usd=0.5)

    def test_read_usage_ignores_non_spec_usage_return(self) -> None:
        class WeirdProvider:
            def generate_spec(self, schema: object) -> object:
                return object()

            def get_last_usage(self) -> object:
                return {"tokens_sent": 1}

        assert read_usage(WeirdProvider()) is None

    def test_read_usage_handles_non_callable_attribute(self) -> None:
        class NotCallable:
            get_last_usage = "oops"

        assert read_usage(NotCallable()) is None
