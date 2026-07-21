"""
nodes/triage.py — Triage Node for LangGraph workflow.

Calls the LLM adapter to classify the ticket.
Returns only the fields it modifies (partial state update).
The adapter is injected via constructor for testability.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.utils.logger import logger
from app.workflows.adapters.base import LLMAdapter
from app.workflows.errors import TriageModelError
from app.workflows.state import WorkflowState


class TriageNode:
    """
    Agent 1 — classifies the ticket, detects priority, sentiment, and risk.

    Single responsibility: call the LLM adapter and return structured output.
    Does not know about routing, persistence, or HTTP.
    """

    def __init__(self, llm_adapter: LLMAdapter) -> None:
        self._llm = llm_adapter

    async def __call__(self, state: WorkflowState) -> dict:
        now = datetime.now(timezone.utc).isoformat()
        trace_entry: dict = {
            "node": "triage",
            "started_at": now,
            "workflow_id": state["workflow_id"],
        }
        logger.info(f"[triage_node] workflow_id={state['workflow_id']}")

        try:
            result = await self._llm.triage(
                customer_name=state["customer_name"],
                email=state["email"],
                message=state["message"],
            )
        except TriageModelError as exc:
            # TriageModelError is a domain-level error (LLM unavailable, bad response).
            # We capture it as a controlled failed state so the graph stops cleanly
            # and the service can persist WorkflowRunRecord(status="failed") without
            # a DB IntegrityError on classification=None.
            #
            # Unexpected infrastructure errors (network timeouts outside the adapter,
            # programming bugs) are NOT caught here — they propagate to the application
            # service which stores them via _persist_failed(error=exc).
            finished_at = datetime.now(timezone.utc).isoformat()
            error_msg = str(exc)
            # Keep "triage_node failed: <msg>" format in errors[] for backward compat
            # with existing tests (test_triage_graph_minimal.py).
            errors_entry = f"triage_node failed: {error_msg}"
            logger.error(f"[triage_node] {errors_entry}")
            trace_entry["finished_at"] = finished_at
            trace_entry["status"] = "error"
            trace_entry["error"] = errors_entry
            return {
                # --- Explicit failure contract fields (read by _persist_failed_from_state) ---
                "workflow_status": "failed",
                # error_code and error_message are stored in execution_trace so that
                # LangGraph's state merging does not drop them.
                # _persist_failed_from_state reads them from execution_trace[-1].
                "current_node": "triage",
                "errors": state.get("errors", []) + [errors_entry],
                "execution_trace": state.get("execution_trace", []) + [
                    {**trace_entry, "error_code": "TriageModelError", "error_message": error_msg, "failed_node": "triage"}
                ],
                "updated_at": finished_at,
            }

        finished_at = datetime.now(timezone.utc).isoformat()
        trace_entry["finished_at"] = finished_at
        trace_entry["status"] = "ok"
        trace_entry["confidence"] = result.confidence

        conf_str = f"{result.confidence:.2f}" if result.confidence is not None else "None"
        logger.info(
            f"[triage_node] cls={result.classification.value} "
            f"pri={result.priority.value} conf={conf_str}"
        )

        return {
            "classification": result.classification.value,
            "priority": result.priority.value,
            "sentiment": result.sentiment.value,
            "summary": result.summary,
            "confidence": result.confidence,
            "risk_flags": list(result.risk_flags),
            "workflow_status": "running",
            "current_node": "triage",
            "execution_trace": state.get("execution_trace", []) + [trace_entry],
            "updated_at": finished_at,
        }
