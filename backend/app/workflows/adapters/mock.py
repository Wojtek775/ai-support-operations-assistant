"""
adapters/mock.py — Mock adapters for tests.

MockTriageAdapter     — deterministic TriageOutput, no network calls.
MockResolutionAdapter — deterministic ResolutionOutput, no network calls.
MockReviewerAdapter   — deterministic ReviewOutput, no network calls.

All support:
    - raise_error=True  → raise the corresponding domain error
    - capture_calls     → list of kwargs passed to the method (for assertion)
"""

from __future__ import annotations

from app.models.ticket import TicketClassification, TicketPriority, TicketSentiment
from app.workflows.adapters.base import ResolutionOutput, ReviewOutput, TriageOutput


class MockTriageAdapter:
    """
    Deterministic mock implementing the LLMAdapter Protocol.
    Instantiate with the desired output values for each test scenario.
    """

    def __init__(
        self,
        classification: TicketClassification = TicketClassification.TECHNICAL,
        priority: TicketPriority = TicketPriority.MEDIUM,
        sentiment: TicketSentiment = TicketSentiment.NEUTRAL,
        summary: str = "Customer reports a technical issue.",
        confidence: float = 0.92,
        risk_flags: list[str] | None = None,
        raise_error: bool = False,
        error_message: str = "Mock LLM error",
    ) -> None:
        self._output = TriageOutput(
            classification=classification,
            priority=priority,
            sentiment=sentiment,
            summary=summary,
            confidence=confidence,
            risk_flags=risk_flags or [],
        )
        self._raise_error = raise_error
        self._error_message = error_message

    async def triage(self, customer_name: str, email: str, message: str) -> TriageOutput:
        if self._raise_error:
            from app.workflows.errors import TriageModelError
            raise TriageModelError(self._error_message)
        return self._output


class MockResolutionAdapter:
    """
    Deterministic mock implementing the ResolutionAdapter Protocol.
    Instantiate with the desired output values for each test scenario.

    call_args_list  — every set of kwargs passed to generate_resolution().
                      Use in tests to assert the adapter received correct data.
    """

    def __init__(
        self,
        suggested_response: str = (
            "Thank you for contacting us. We have received your request "
            "and our team will get back to you within the agreed SLA window."
        ),
        recommended_action: str = (
            "Review the ticket, assign to appropriate team, and follow up within SLA."
        ),
        tone: str = "empathetic",
        confidence: float = 0.90,
        raise_error: bool = False,
        error_message: str = "Mock resolution error",
    ) -> None:
        self._output = ResolutionOutput(
            suggested_response=suggested_response,
            recommended_action=recommended_action,
            tone=tone,  # type: ignore[arg-type]
            confidence=confidence,
        )
        self._raise_error = raise_error
        self._error_message = error_message
        # Records every call for test assertions
        self.call_args_list: list[dict] = []

    async def generate_resolution(
        self,
        *,
        ticket: dict,
        triage: dict,
        routing_decision: dict,
        context: list[str],
        reviewer_notes: str | None = None,
    ) -> ResolutionOutput:
        self.call_args_list.append(
            {
                "ticket": ticket,
                "triage": triage,
                "routing_decision": routing_decision,
                "context": context,
                "reviewer_notes": reviewer_notes,
            }
        )
        if self._raise_error:
            from app.workflows.errors import ResolutionModelError
            raise ResolutionModelError(self._error_message)
        return self._output


class MockReviewerAdapter:
    """
    Deterministic mock implementing the ReviewerAdapter Protocol.
    Instantiate with the desired decision for each test scenario.

    call_args_list — every set of kwargs passed to review(). Use in tests
                     to assert the adapter received the correct resolution data.
    """

    def __init__(
        self,
        decision: str = "APPROVED",
        confidence: float = 0.95,
        issues: list[str] | None = None,
        notes: str = "Quality check passed.",
        raise_error: bool = False,
        error_message: str = "Mock reviewer error",
    ) -> None:
        self._output = ReviewOutput(
            decision=decision,   # type: ignore[arg-type]
            confidence=confidence,
            issues=issues or [],
            notes=notes,
        )
        self._raise_error = raise_error
        self._error_message = error_message
        # Records every call for test assertions
        self.call_args_list: list[dict] = []

    async def review(
        self,
        *,
        ticket: dict,
        triage: dict,
        routing_decision: dict,
        resolution: dict,
    ) -> ReviewOutput:
        self.call_args_list.append(
            {
                "ticket": ticket,
                "triage": triage,
                "routing_decision": routing_decision,
                "resolution": resolution,
            }
        )
        if self._raise_error:
            from app.workflows.errors import ReviewerModelError
            raise ReviewerModelError(self._error_message)
        return self._output
