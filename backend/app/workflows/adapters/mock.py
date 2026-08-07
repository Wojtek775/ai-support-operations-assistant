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
from app.workflows.adapters.base import ResolutionOutput, ReviewOutput, SupervisorOutput, TriageOutput


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


class MockSupervisorAdapter:
    """
    Deterministic mock implementing the SupervisorAdapter Protocol.

    Follows the standard happy-path sequence:
        triage → routing → resolution → reviewer → FINISH

    Honours overrides:
      - If requires_human_review is True in state_summary → routes to human_review
      - If review_decision is "RETRY" → routes back to resolution
      - If review_decision is "HUMAN_REVIEW" → routes to human_review
      - If review_decision is "APPROVED" (and resolution done) → FINISH

    Optionally accepts a fixed sequence list for precise test control:
        sequence=["triage", "routing", "resolution", "FINISH"]
    """

    def __init__(
        self,
        sequence: list[str] | None = None,
    ) -> None:
        # If a fixed sequence is provided, we replay it step by step.
        # Otherwise, we use the smart decision logic.
        self._sequence = sequence
        self._call_count = 0
        self.call_args_list: list[dict] = []

    async def supervise(
        self,
        *,
        message: str,
        customer_name: str,
        completed_agents: list[str],
        state_summary: dict,
    ) -> SupervisorOutput:
        self.call_args_list.append(
            {
                "message": message,
                "customer_name": customer_name,
                "completed_agents": list(completed_agents),
                "state_summary": dict(state_summary),
            }
        )

        # Fixed sequence mode (for precise test control)
        if self._sequence is not None:
            idx = self._call_count
            self._call_count += 1
            if idx < len(self._sequence):
                next_agent = self._sequence[idx]
            else:
                next_agent = "FINISH"
            return SupervisorOutput(
                next_agent=next_agent,  # type: ignore[arg-type]
                reasoning=f"Mock sequence step {idx}: {next_agent}",
            )

        # Smart mode: decide based on completed_agents + state_summary
        self._call_count += 1
        requires_review = (state_summary.get("routing_decision") or {}).get(
            "requires_human_review", False
        )
        review_decision = state_summary.get("review_decision")

        if "triage" not in completed_agents:
            return SupervisorOutput(next_agent="triage", reasoning="Mock: starting with triage.")
        if "routing" not in completed_agents:
            return SupervisorOutput(next_agent="routing", reasoning="Mock: routing after triage.")
        if requires_review:
            return SupervisorOutput(next_agent="human_review", reasoning="Mock: requires_human_review.")
        if "resolution" not in completed_agents:
            return SupervisorOutput(next_agent="resolution", reasoning="Mock: generating resolution.")
        if review_decision == "RETRY":
            return SupervisorOutput(next_agent="resolution", reasoning="Mock: reviewer requested retry.")
        if review_decision == "HUMAN_REVIEW":
            return SupervisorOutput(next_agent="human_review", reasoning="Mock: reviewer escalated.")
        if "reviewer" not in completed_agents:
            return SupervisorOutput(next_agent="reviewer", reasoning="Mock: quality review.")
        return SupervisorOutput(next_agent="FINISH", reasoning="Mock: all agents completed.")
