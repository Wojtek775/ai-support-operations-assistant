"""
nodes/human_review_gate.py — Human Review Gate node.

Uses langgraph.types.interrupt() to pause workflow execution
when the ticket requires human attention.

Decision criteria are DETERMINISTIC (Python rules, not LLM).
This node is skipped entirely for simple/low-risk tickets,
enabling the conditional edge in graph.py to bypass it.

The interrupt() call:
- Saves the full checkpoint via the configured checkpointer
- Raises an internal exception that LangGraph catches
- The graph result will contain '__interrupt__' key
- Resumption: graph.ainvoke(Command(resume=decision), config=config)
"""

from __future__ import annotations

from datetime import datetime, timezone

from langgraph.types import interrupt

from app.config import settings
from app.utils.logger import logger
from app.workflows.state import WorkflowState


def _needs_human_review(state: WorkflowState) -> tuple[bool, str]:
    """
    Evaluate whether the ticket requires human review before processing.

    Returns (needs_review: bool, reason: str).
    Reasons are additive — first matching rule wins for clarity.
    """
    confidence = state.get("confidence")
    # Safely handle None — treat missing confidence as low confidence
    low_confidence = confidence is None or confidence < settings.workflow_confidence_threshold

    if low_confidence:
        return True, f"Low confidence score: {confidence} (threshold: {settings.workflow_confidence_threshold})"

    if state.get("priority") == "critical":
        return True, "Critical priority ticket requires human review"

    if state.get("sentiment") == "angry":
        routing = state.get("routing_decision") or {}
        if routing.get("escalation_level", 0) >= 2:
            return True, "Angry customer with escalation level 2 — immediate human review"

    if "security" in (state.get("risk_flags") or []):
        return True, "Security risk flag detected"

    routing = state.get("routing_decision") or {}
    if routing.get("escalation_level", 0) >= 2:
        return True, "Escalation level 2 — human review required"

    return False, ""


async def human_review_gate_node(state: WorkflowState) -> dict:
    """
    Evaluate human review need and either pass through or interrupt.

    If interrupt() is called, LangGraph saves checkpoint and suspends.
    Resumption via Command(resume=human_decision) continues from here.

    Interrupt is triggered by:
    1. _needs_human_review() deterministic rules (CRITICAL priority, low conf, etc.)
    2. review_decision == "HUMAN_REVIEW" (reviewer explicitly escalates)
    3. review_decision == "RETRY" when max retries exceeded (routed here by graph)
    """
    if state.get("workflow_status") == "failed":
        return {}

    now = datetime.now(timezone.utc).isoformat()
    needs_review, reason = _needs_human_review(state)

    # Additional triggers: reviewer escalation paths
    review_decision = state.get("review_decision")
    if not needs_review and review_decision in ("HUMAN_REVIEW", "RETRY"):
        needs_review = True
        if review_decision == "HUMAN_REVIEW":
            reason = "Quality reviewer flagged ticket for human review"
        else:
            # RETRY with retry_count >= 1 means max retries exceeded
            reason = f"Max review retries exceeded (retry_count={state.get('retry_count', 0)})"

    trace_entry = {
        "node": "human_review_gate",
        "started_at": now,
        "needs_human_review": needs_review,
        "reason": reason,
    }

    if needs_review:
        logger.info(
            f"[human_review_gate] workflow_id={state['workflow_id']} "
            f"INTERRUPTING — reason: {reason}"
        )
        trace_entry["finished_at"] = datetime.now(timezone.utc).isoformat()
        trace_entry["status"] = "interrupted"

        # interrupt() pauses graph execution here.
        # The return value of interrupt() is the human decision
        # passed via Command(resume=...) when the workflow is resumed.
        # In ETAP 2, we don't process the decision yet — that's ETAP 4.
        _human_decision = interrupt({
            "reason": reason,
            "workflow_id": state["workflow_id"],
            "classification": state.get("classification"),
            "priority": state.get("priority"),
        })

        # Code below runs only after Command(resume=...) in ETAP 4
        return {
            "requires_human_review": True,
            "human_review_reason": reason,
            "human_review_status": "approved" if _human_decision else "rejected",
            "workflow_status": "running",
            "current_node": "human_review_gate",
            "execution_trace": state.get("execution_trace", []) + [trace_entry],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    # No review needed — pass through
    finished_at = datetime.now(timezone.utc).isoformat()
    trace_entry["finished_at"] = finished_at
    trace_entry["status"] = "passed"
    logger.info(f"[human_review_gate] workflow_id={state['workflow_id']} — passed, no review needed")

    return {
        "requires_human_review": False,
        "human_review_reason": None,
        "human_review_status": None,
        "current_node": "human_review_gate",
        "execution_trace": state.get("execution_trace", []) + [trace_entry],
        "updated_at": finished_at,
    }
