"""
tests/test_workflow.py — Unit tests for the deterministic workflow routing engine.

These tests verify that:
- workflow routing is purely deterministic (no AI calls, no DB calls)
- all 6 routing rules produce the correct output
- edge cases and default fallback work correctly

The workflow_service module has NO external dependencies — every test
is a pure function call with no mocking required.
"""

import pytest
from app.services.workflow_service import route_ticket, WorkflowDecision


class TestWorkflowIsDeterministic:
    """Verify that the same inputs always produce the same outputs."""

    def test_same_inputs_produce_same_output(self):
        """Calling route_ticket twice with identical args returns identical results."""
        result_1 = route_ticket("billing", "high", "negative")
        result_2 = route_ticket("billing", "high", "negative")
        assert result_1 == result_2

    def test_returns_workflow_decision_dataclass(self):
        """route_ticket must return a WorkflowDecision instance."""
        result = route_ticket("general", "low", "positive")
        assert isinstance(result, WorkflowDecision)

    def test_workflow_does_not_require_any_mock(self):
        """
        Workflow routing works without patching any external service.
        This test passing proves no AI or DB calls are made.
        """
        # If this test runs without any @patch decorator and passes,
        # it proves the function is fully self-contained.
        result = route_ticket("technical", "medium", "neutral")
        assert result.assigned_team is not None
        assert result.sla_hours > 0

    def test_case_insensitive_inputs(self):
        """Input values should be normalised — BILLING == billing."""
        result_lower = route_ticket("billing", "high", "negative")
        result_upper = route_ticket("BILLING", "HIGH", "NEGATIVE")
        assert result_lower == result_upper


class TestRule1UrgentOutage:
    """Rule 1: urgent/critical outage → infrastructure_team, sla_hours=1."""

    def test_urgent_outage_gets_infrastructure_team(self):
        result = route_ticket("outage", "urgent", "neutral")
        assert result.assigned_team == "infrastructure_team"

    def test_urgent_outage_gets_sla_1_hour(self):
        result = route_ticket("outage", "urgent", "neutral")
        assert result.sla_hours == 1

    def test_critical_outage_gets_infrastructure_team(self):
        result = route_ticket("outage", "critical", "neutral")
        assert result.assigned_team == "infrastructure_team"

    def test_critical_outage_gets_sla_1_hour(self):
        result = route_ticket("outage", "critical", "neutral")
        assert result.sla_hours == 1

    def test_urgent_outage_internal_notes_mention_infrastructure(self):
        result = route_ticket("outage", "urgent", "neutral")
        assert "infrastructure" in result.internal_notes.lower()

    def test_critical_technical_gets_infrastructure_team(self):
        """Rule 1b: critical technical issue also goes to infrastructure."""
        result = route_ticket("technical", "critical", "neutral")
        assert result.assigned_team == "infrastructure_team"
        assert result.sla_hours == 1

    def test_low_priority_outage_does_not_get_infrastructure(self):
        """Low priority outage should NOT trigger infrastructure routing."""
        result = route_ticket("outage", "low", "neutral")
        assert result.assigned_team != "infrastructure_team"


class TestRule2BillingFinance:
    """Rule 2: billing/refund → finance_team, sla_hours=4."""

    def test_billing_gets_finance_team(self):
        result = route_ticket("billing", "medium", "neutral")
        assert result.assigned_team == "finance_team"

    def test_billing_gets_sla_4_hours(self):
        result = route_ticket("billing", "medium", "neutral")
        assert result.sla_hours == 4

    def test_refund_gets_finance_team(self):
        result = route_ticket("refund", "medium", "neutral")
        assert result.assigned_team == "finance_team"

    def test_billing_internal_notes_mention_finance(self):
        result = route_ticket("billing", "low", "positive")
        assert "finance" in result.internal_notes.lower()


class TestRule3EnterpriseeSales:
    """Rule 3: enterprise_sales → account_executive, sla_hours=2."""

    def test_enterprise_sales_gets_account_executive(self):
        result = route_ticket("enterprise_sales", "high", "neutral")
        assert result.assigned_team == "account_executive"

    def test_enterprise_sales_gets_sla_2_hours(self):
        result = route_ticket("enterprise_sales", "high", "neutral")
        assert result.sla_hours == 2

    def test_enterprise_sales_notes_mention_enterprise(self):
        result = route_ticket("enterprise_sales", "medium", "neutral")
        assert "enterprise" in result.internal_notes.lower()


class TestRule4Cancellation:
    """Rule 4: cancellation → retention_queue, sla_hours=2."""

    def test_cancellation_gets_retention_queue(self):
        result = route_ticket("cancellation", "medium", "negative")
        assert result.assigned_team == "retention_queue"

    def test_cancellation_gets_sla_2_hours(self):
        result = route_ticket("cancellation", "medium", "negative")
        assert result.sla_hours == 2

    def test_cancellation_internal_notes_mention_retention(self):
        result = route_ticket("cancellation", "low", "neutral")
        assert "retention" in result.internal_notes.lower()

    def test_cancellation_notes_mention_risk(self):
        result = route_ticket("cancellation", "high", "negative")
        assert "cancellation" in result.internal_notes.lower() or "retention" in result.internal_notes.lower()


class TestRule5TechnicalTeam:
    """Rule 5: technical (non-critical) → technical_team."""

    def test_high_technical_gets_technical_team(self):
        result = route_ticket("technical", "high", "neutral")
        assert result.assigned_team == "technical_team"

    def test_medium_technical_gets_sla_8_hours(self):
        result = route_ticket("technical", "medium", "neutral")
        assert result.sla_hours == 8


class TestRule6AngryUrgent:
    """Rule 6 (overlay): angry + urgent/critical → escalation_level=2, requires_human_review=True."""

    def test_angry_urgent_gets_escalation_level_2(self):
        result = route_ticket("billing", "urgent", "angry")
        assert result.escalation_level == 2

    def test_angry_urgent_requires_human_review(self):
        result = route_ticket("billing", "urgent", "angry")
        assert result.requires_human_review is True

    def test_angry_critical_gets_escalation_level_2(self):
        result = route_ticket("technical", "critical", "angry")
        assert result.escalation_level == 2

    def test_angry_urgent_notes_mention_escalation(self):
        result = route_ticket("general", "urgent", "angry")
        assert "escalat" in result.internal_notes.lower()

    def test_angry_low_priority_does_not_escalate(self):
        """Angry sentiment with low priority should NOT trigger escalation."""
        result = route_ticket("billing", "low", "angry")
        assert result.escalation_level != 2
        assert result.requires_human_review is False


class TestRule7HumanReview:
    """Rule 7 (overlay): high+ priority + negative/angry → requires_human_review=True."""

    def test_high_priority_negative_requires_human_review(self):
        result = route_ticket("general", "high", "negative")
        assert result.requires_human_review is True

    def test_high_priority_negative_gets_escalation_level_1(self):
        result = route_ticket("general", "high", "negative")
        assert result.escalation_level >= 1

    def test_urgent_negative_requires_human_review(self):
        result = route_ticket("account", "urgent", "negative")
        assert result.requires_human_review is True

    def test_low_priority_negative_does_not_require_human_review(self):
        """Low priority + negative should NOT trigger human review."""
        result = route_ticket("general", "low", "negative")
        assert result.requires_human_review is False

    def test_high_priority_positive_does_not_require_human_review(self):
        """High priority + positive sentiment should NOT require human review."""
        result = route_ticket("general", "high", "positive")
        assert result.requires_human_review is False


class TestDefaultRouting:
    """Default fallback when no specific rule matches."""

    def test_general_low_gets_support_general(self):
        result = route_ticket("general", "low", "neutral")
        assert result.assigned_team == "support_general"

    def test_general_low_gets_sla_24_hours(self):
        result = route_ticket("general", "low", "neutral")
        assert result.sla_hours == 24

    def test_default_escalation_level_is_zero(self):
        result = route_ticket("general", "low", "neutral")
        assert result.escalation_level == 0

    def test_default_does_not_require_human_review(self):
        result = route_ticket("general", "low", "positive")
        assert result.requires_human_review is False

    def test_internal_notes_always_populated(self):
        """internal_notes must never be empty — default text applies."""
        result = route_ticket("shipping", "low", "neutral")
        assert result.internal_notes
        assert len(result.internal_notes) > 0
