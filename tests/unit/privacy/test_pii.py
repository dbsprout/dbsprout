"""Tests for dbsprout.privacy.pii — PII column name detection."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

from dbsprout.privacy.pii import PIIDetector


class TestPIIDetectorSingleColumn:
    """PIIDetector.is_pii — single column name detection."""

    def test_detects_ssn(self) -> None:
        assert PIIDetector().is_pii("ssn") is True

    def test_detects_social_security(self) -> None:
        assert PIIDetector().is_pii("social_security_number") is True

    def test_detects_credit_card(self) -> None:
        assert PIIDetector().is_pii("credit_card_number") is True

    def test_detects_date_of_birth(self) -> None:
        assert PIIDetector().is_pii("date_of_birth") is True

    def test_detects_dob(self) -> None:
        assert PIIDetector().is_pii("dob") is True

    def test_detects_phone(self) -> None:
        assert PIIDetector().is_pii("phone_number") is True

    def test_detects_passport(self) -> None:
        assert PIIDetector().is_pii("passport_number") is True

    def test_detects_salary(self) -> None:
        assert PIIDetector().is_pii("salary") is True

    def test_detects_bank_account(self) -> None:
        assert PIIDetector().is_pii("bank_account_number") is True

    def test_detects_tax_id(self) -> None:
        assert PIIDetector().is_pii("tax_id") is True

    def test_detects_drivers_license(self) -> None:
        assert PIIDetector().is_pii("drivers_license") is True

    def test_case_insensitive(self) -> None:
        assert PIIDetector().is_pii("SSN") is True
        assert PIIDetector().is_pii("Credit_Card") is True

    def test_ignores_non_pii_created_at(self) -> None:
        assert PIIDetector().is_pii("created_at") is False

    def test_ignores_non_pii_order_total(self) -> None:
        assert PIIDetector().is_pii("order_total") is False

    def test_ignores_non_pii_status(self) -> None:
        assert PIIDetector().is_pii("status") is False

    def test_ignores_non_pii_id(self) -> None:
        assert PIIDetector().is_pii("id") is False

    def test_ignores_non_pii_name(self) -> None:
        """Generic 'name' is not PII — too common in schemas."""
        assert PIIDetector().is_pii("name") is False


class TestPIIDetectorBatch:
    """PIIDetector.detect_pii_columns — batch detection."""

    def test_returns_pii_set(self) -> None:
        columns = ["id", "ssn", "email", "created_at", "phone_number"]
        result = PIIDetector().detect_pii_columns(columns)
        assert isinstance(result, set)
        assert "ssn" in result
        assert "phone_number" in result
        assert "id" not in result
        assert "created_at" not in result

    def test_empty_list(self) -> None:
        assert PIIDetector().detect_pii_columns([]) == set()


class TestPIIDetectorPresidioFallback:
    """Presidio fallback to regex."""

    def test_works_without_presidio(self) -> None:
        """Regex detection works even without Presidio installed."""
        detector = PIIDetector()
        assert detector.is_pii("ssn") is True
        assert detector.is_pii("credit_card") is True

    def test_use_presidio_is_false_without_package(self) -> None:
        """Without presidio-analyzer, _use_presidio is False."""
        detector = PIIDetector()
        # In test env, Presidio is not installed
        assert detector._use_presidio is False


class TestPIIDetectorLogging:
    """PII detection emits audit log."""

    def test_detection_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        detector = PIIDetector()
        with caplog.at_level(logging.INFO):
            detector.detect_pii_columns(["id", "ssn", "email"])
        assert "ssn" in caplog.text
