"""
workflows/graph.py — LangGraph graph definitions.

ETAP 4 (current):
    START -> triage -> routing -> conditional
      ├── human_review → human_review_gate → END   (interrupt)
      └── no_review    → resolution → reviewer → conditional
                                        ├── APPROVED      → END
                                        ├── RETRY (1st)   → resolution (retry_count+1)
                                        ├── RETRY (2nd+)  → human_review_gate (interrupt)
                                        └── HUMAN_REVIEW  → human_review_gate (interrupt)

Resolution Agent runs only for workflows that do NOT require human review.
human-review workflows are paused at human_review_gate (interrupt).
Reviewer runs after every resolution call.
Max 1 retry: second RETRY routes to human_review_gate.

Checkpointer:
- InMemorySaver  — tests (passed as argument, default)
- AsyncSqliteSaver — production (managed in lifespan, not yet wired)

thread_id == workflow_id (one UUID serves both roles).
"""

from __future__ import annotations

from datetime import datetime, timezone

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.workflows.adapters.base import LLMAdapter, ResolutionAdapter, ReviewerAdapter
from app.workflows.nodes.human_review_gate import _needs_human_review, human_review_gate_node
from app.workflows.nodes.resolution import ResolutionNode
from app.workflows.nodes.reviewer import ReviewerNode
from app.workflows.nodes.routing import routing_node
from app.workflows.nodes.triage import TriageNode
from app.workflows.state import WorkflowState


# ---------------------------------------------------------------------------
# Conditional edge: after routing
# ---------------------------------------------------------------------------

def _route_after_routing(state: WorkflowState) -> str:
    """
    Conditional edge: decides whether human review is needed BEFORE
    entering human_review_gate (which calls interrupt()).

    Two sources of truth are combined:
    1. routing_decision.requires_human_review — set by workflow_service rule engine
    2. _needs_human_review() — checks confidence, priority, risk_flags, escalation

    If EITHER is True → route to human_review_gate (which will interrupt).
    Otherwise → resolution (Agent 3).

    Returns:
        "human_review" — ticket needs human attention (interrupt will fire)
        "resolution"   — ticket goes to Resolution Agent, then reviewer
    """
    routing = state.get("routing_decision") or {}
    if routing.get("requires_human_review", False):
        return "human_review"
    needs_review, _ = _needs_human_review(state)
    if needs_review:
        return "human_review"
    return "resolution"


# ---------------------------------------------------------------------------
# Conditional edge: after reviewer
# ---------------------------------------------------------------------------

def _route_after_reviewer(state: WorkflowState) -> str:
    """
    Conditional edge after the Quality Reviewer node.

    Decision matrix:
        APPROVED     → END (workflow completes, TicketRecord will be created)
        HUMAN_REVIEW → human_review_gate (interrupt, no TicketRecord)
        RETRY        → if retry_count < 1: resolution (retry with notes)
                       if retry_count >= 1: human_review_gate (max retries exceeded)
        failed state → human_review_gate (safety net for reviewer error)

    Returns:
        "approved"     — proceed to END
        "retry"        — re-run resolution with reviewer feedback
        "human_review" — pause for human (interrupt)
    """
    # If reviewer node itself failed, route to human review
    if state.get("workflow_status") == "failed":
        return "human_review"

    decision = state.get("review_decision")
    if decision == "APPROVED":
        return "approved"
    if decision == "HUMAN_REVIEW":
        return "human_review"
    if decision == "RETRY":
        # retry_count is incremented BEFORE the second resolution call
        # so we check the current count here (before increment)
        retry_count = state.get("retry_count", 0)
        if retry_count < 1:
            return "retry"
        return "human_review"  # max retries exceeded → escalate
    # Unknown decision → safety net
    return "human_review"


# ---------------------------------------------------------------------------
# Retry bump node (increments retry_count before re-running resolution)
# ---------------------------------------------------------------------------

async def _increment_retry_node(state: WorkflowState) -> dict:
    """
    Lightweight node that increments retry_count by 1 before re-running
    the resolution node. Keeps the retry counter accurate without cluttering
    the resolution node itself.
    """
    return {
        "retry_count": state.get("retry_count", 0) + 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------

def build_triage_only_graph(
    llm_adapter: LLMAdapter,
    checkpointer=None,
):
    """
    Minimal graph: START -> triage -> END.
    Kept for backward-compatibility with test_triage_graph_minimal.py.
    """
    if checkpointer is None:
        checkpointer = InMemorySaver()

    triage_node = TriageNode(llm_adapter)

    builder = StateGraph(WorkflowState)
    builder.add_node("triage", triage_node)
    builder.add_edge(START, "triage")
    builder.add_edge("triage", END)

    return builder.compile(checkpointer=checkpointer)


def build_full_graph(
    llm_adapter: LLMAdapter,
    resolution_adapter: ResolutionAdapter,
    reviewer_adapter: ReviewerAdapter | None = None,
    checkpointer=None,
):
    """
    Full graph for ETAP 4:
        START -> triage -> routing -> conditional
          ├── human_review → human_review_gate → END
          └── resolution   → reviewer → conditional
                              ├── approved     → END
                              ├── retry        → _increment_retry → resolution
                              └── human_review → human_review_gate → END

    Args:
        llm_adapter:        Injected Triage LLM adapter.
        resolution_adapter: Injected Resolution LLM adapter.
        reviewer_adapter:   Injected Quality Reviewer adapter.
                            If None, a MockReviewerAdapter(decision="APPROVED") is used.
        checkpointer:       LangGraph checkpointer. Defaults to InMemorySaver.

    Returns:
        Compiled LangGraph graph with interrupt support.
    """
    if checkpointer is None:
        checkpointer = InMemorySaver()

    if reviewer_adapter is None:
        from app.workflows.adapters.mock import MockReviewerAdapter
        reviewer_adapter = MockReviewerAdapter(decision="APPROVED")

    triage_node = TriageNode(llm_adapter)
    resolution_node = ResolutionNode(resolution_adapter)
    reviewer_node = ReviewerNode(reviewer_adapter)

    builder = StateGraph(WorkflowState)

    # Nodes
    builder.add_node("triage", triage_node)
    builder.add_node("routing", routing_node)
    builder.add_node("human_review_gate", human_review_gate_node)
    builder.add_node("resolution", resolution_node)
    builder.add_node("reviewer", reviewer_node)
    builder.add_node("increment_retry", _increment_retry_node)

    # Edges
    builder.add_edge(START, "triage")
    builder.add_edge("triage", "routing")

    # After routing: human review or resolution
    builder.add_conditional_edges(
        "routing",
        _route_after_routing,
        {
            "human_review": "human_review_gate",
            "resolution": "resolution",
        },
    )

    # human_review_gate pauses via interrupt(); if resumed it goes to END
    builder.add_edge("human_review_gate", END)

    # resolution → reviewer
    builder.add_edge("resolution", "reviewer")

    # After reviewer: approved/retry/human_review
    builder.add_conditional_edges(
        "reviewer",
        _route_after_reviewer,
        {
            "approved": END,
            "retry": "increment_retry",
            "human_review": "human_review_gate",
        },
    )

    # retry path: increment counter → re-run resolution
    builder.add_edge("increment_retry", "resolution")

    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# State factory
# ---------------------------------------------------------------------------

def make_initial_state(
    workflow_id: str,
    customer_name: str,
    email: str,
    message: str,
) -> WorkflowState:
    """
    Create the initial WorkflowState for a new workflow run.
    All optional fields start as None / empty lists.
    """
    now = datetime.now(timezone.utc).isoformat()
    return WorkflowState(
        workflow_id=workflow_id,
        ticket_id=None,
        customer_name=customer_name,
        email=email,
        message=message,
        classification=None,
        priority=None,
        sentiment=None,
        summary=None,
        confidence=None,
        risk_flags=[],
        routing_decision=None,
        suggested_response=None,
        recommended_action=None,
        resolution_tone=None,
        resolution_confidence=None,
        review_decision=None,
        review_confidence=None,
        review_issues=[],
        reviewer_notes=None,
        requires_human_review=False,
        human_review_reason=None,
        human_review_status=None,
        workflow_status="running",
        current_node=None,
        retry_count=0,
        execution_trace=[],
        errors=[],
        started_at=now,
        updated_at=now,
        completed_at=None,
    )
