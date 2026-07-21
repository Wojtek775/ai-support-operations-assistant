"""
nodes/resolution.py — Resolution Node for LangGraph workflow.

Agent 3 in the pipeline (after triage → routing → conditional):
  - Receives full triage output and routing decision from state.
  - Calls ResolutionAdapter.generate_resolution() to produce:
      suggested_response, recommended_action, tone, confidence.
  - Saves results to state (partial update pattern).
  - Adds execution_trace entry.
  - Stores error contract in execution_trace[-1] on failure.

Single responsibility: call the resolution adapter, return partial state.
Does not know about DB, routing rules, or HTTP.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.utils.logger import logger
from app.workflows.adapters.base import ResolutionAdapter
from app.workflows.errors import ResolutionModelError
from app.workflows.state import WorkflowState


class ResolutionNode:
    """
    Agent 3 — generates the customer-facing response and recommended action.

    Adapter injected via constructor (dependency injection).
    """

    def __init__(self, resolution_adapter: ResolutionAdapter) -> None:
        self._adapter = resolution_adapter

    async def __call__(self, state: WorkflowState) -> dict:
        now = datetime.now(timezone.utc).isoformat()
        trace_entry: dict = {
            "node": "resolution",
            "started_at": now,
            "workflow_id": state["workflow_id"],
        }
        logger.info(f"[resolution_node] workflow_id={state['workflow_id']}")

        # Build structured dicts for the adapter contract
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

        try:
            result = await self._adapter.generate_resolution(
                ticket=ticket,
                triage=triage,
                routing_decision=routing_decision,
                context=[],  # RAG context stub — populated in future ETAP
                reviewer_notes=state.get("reviewer_notes"),
            )
        except ResolutionModelError as exc:
            # Controlled failure — store contract fields in execution_trace[-1]
            # so _persist_failed_from_state can read them without LangGraph
            # dropping unknown top-level state keys.
            finished_at = datetime.now(timezone.utc).isoformat()
            error_msg = str(exc)
            errors_entry = f"resolution_node failed: {error_msg}"
            logger.error(f"[resolution_node] {errors_entry}")
            trace_entry["finished_at"] = finished_at
            trace_entry["status"] = "error"
            trace_entry["error"] = errors_entry
            return {
                "workflow_status": "failed",
                "current_node": "resolution",
                "errors": state.get("errors", []) + [errors_entry],
                "execution_trace": state.get("execution_trace", []) + [
                    {
                        **trace_entry,
                        "error_code": "ResolutionModelError",
                        "error_message": error_msg,
                        "failed_node": "resolution",
                    }
                ],
                "updated_at": finished_at,
            }

        finished_at = datetime.now(timezone.utc).isoformat()
        trace_entry["finished_at"] = finished_at
        trace_entry["status"] = "ok"
        trace_entry["tone"] = result.tone
        trace_entry["confidence"] = result.confidence

        logger.info(
            f"[resolution_node] resolution complete "
            f"tone={result.tone} conf={result.confidence:.2f}"
        )

        return {
            "suggested_response": result.suggested_response,
            "recommended_action": result.recommended_action,
            "resolution_tone": result.tone,
            "resolution_confidence": result.confidence,
            "workflow_status": "running",
            "current_node": "resolution",
            "execution_trace": state.get("execution_trace", []) + [trace_entry],
            "updated_at": finished_at,
        }
