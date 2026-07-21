"""
nodes/reviewer.py — Quality Reviewer Node for LangGraph workflow.

Agent 4 in the pipeline (after resolution):
  - Evaluates suggested_response and recommended_action for quality/safety.
  - Returns one of: APPROVED, RETRY, HUMAN_REVIEW.
  - On RETRY: stores reviewer notes in state for the next resolution call.
  - On HUMAN_REVIEW: calls interrupt() — same as human_review_gate.
  - On error: sets workflow_status=failed, failed_node="reviewer".

Max retries: 1 (enforced by graph conditional — if retry_count >= 1 and
decision is RETRY, the graph routes to human_review_gate instead).

Single responsibility: call the reviewer adapter, return partial state.
Does not know about DB, routing rules, or HTTP.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.utils.logger import logger
from app.workflows.adapters.base import ReviewerAdapter
from app.workflows.errors import ReviewerModelError
from app.workflows.state import WorkflowState


class ReviewerNode:
    """
    Agent 4 — reviews the generated resolution for quality, safety and completeness.

    Adapter injected via constructor (dependency injection).
    """

    def __init__(self, reviewer_adapter: ReviewerAdapter) -> None:
        self._adapter = reviewer_adapter

    async def __call__(self, state: WorkflowState) -> dict:
        # If a previous node failed, pass through without calling LLM
        if state.get("workflow_status") == "failed":
            return {}

        now = datetime.now(timezone.utc).isoformat()
        trace_entry: dict = {
            "node": "reviewer",
            "started_at": now,
            "workflow_id": state["workflow_id"],
        }
        logger.info(f"[reviewer_node] workflow_id={state['workflow_id']}")

        ticket = {
            "customer_name": state.get("customer_name", ""),
            "email": state.get("email", ""),
            "message": state.get("message", ""),
        }
        triage = {
            "classification": state.get("classification"),
            "priority": state.get("priority"),
            "sentiment": state.get("sentiment"),
            "summary": state.get("summary"),
            "confidence": state.get("confidence"),
            "risk_flags": state.get("risk_flags") or [],
        }
        routing_decision = state.get("routing_decision") or {}
        resolution = {
            "suggested_response": state.get("suggested_response"),
            "recommended_action": state.get("recommended_action"),
            "tone": state.get("resolution_tone"),
            "confidence": state.get("resolution_confidence"),
        }

        try:
            result = await self._adapter.review(
                ticket=ticket,
                triage=triage,
                routing_decision=routing_decision,
                resolution=resolution,
            )
        except ReviewerModelError as exc:
            finished_at = datetime.now(timezone.utc).isoformat()
            error_msg = str(exc)
            errors_entry = f"reviewer_node failed: {error_msg}"
            logger.error(f"[reviewer_node] {errors_entry}")
            trace_entry["finished_at"] = finished_at
            trace_entry["status"] = "error"
            trace_entry["error"] = errors_entry
            return {
                "workflow_status": "failed",
                "current_node": "reviewer",
                "errors": state.get("errors", []) + [errors_entry],
                "execution_trace": state.get("execution_trace", []) + [
                    {
                        **trace_entry,
                        "error_code": "ReviewerModelError",
                        "error_message": error_msg,
                        "failed_node": "reviewer",
                    }
                ],
                "updated_at": finished_at,
            }

        finished_at = datetime.now(timezone.utc).isoformat()
        trace_entry["finished_at"] = finished_at
        trace_entry["status"] = "ok"
        trace_entry["review_decision"] = result.decision
        trace_entry["review_confidence"] = result.confidence
        trace_entry["review_issues"] = result.issues

        logger.info(
            f"[reviewer_node] decision={result.decision} "
            f"confidence={result.confidence:.2f} issues={len(result.issues)}"
        )

        return {
            "review_decision": result.decision,
            "review_confidence": result.confidence,
            "review_issues": result.issues,
            "reviewer_notes": result.notes,
            "workflow_status": "running",
            "current_node": "reviewer",
            "execution_trace": state.get("execution_trace", []) + [trace_entry],
            "updated_at": finished_at,
        }
