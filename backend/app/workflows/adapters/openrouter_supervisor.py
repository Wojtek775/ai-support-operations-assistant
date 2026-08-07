"""
adapters/openrouter_supervisor.py — OpenRouter LLM adapter for the Supervisor Agent.

The Supervisor receives a structured view of the current workflow state and
decides which specialist agent should run next (or signals FINISH).

Prompt strategy:
  - Provides a clear list of available agents with their purpose
  - Tells the LLM what has already been completed
  - Asks for a JSON decision: {"next_agent": "...", "reasoning": "..."}
  - Validates response with Pydantic (SupervisorOutput)
  - Falls back to deterministic logic on LLM failure (safe degradation)

This is intentionally a thin wrapper: the supervisor LLM does ONE thing —
choose the next agent. Business logic stays in the nodes and routing rules.
"""

from __future__ import annotations

import json
import re

import httpx

from app.utils.logger import logger
from app.workflows.adapters.base import SupervisorOutput

# ---------------------------------------------------------------------------
# Supervisor system prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a Supervisor Agent coordinating an AI-powered customer support workflow.
Your ONLY job is to decide which specialist agent should run next.

Available agents:
- "triage"      — Classify the ticket: category, priority, sentiment, summary
- "routing"     — Determine which team handles it and the SLA (runs AFTER triage)
- "resolution"  — Generate the customer response and recommended action (runs AFTER routing)
- "reviewer"    — Quality-check the generated response (runs AFTER resolution)
- "human_review"— Escalate to a human agent (use when confidence is low or priority is critical)
- "FINISH"      — Workflow is complete (use when reviewer approved OR human review is queued)

Rules:
1. Do NOT run the same agent twice unless reviewer asked for a RETRY.
2. If routing_decision.requires_human_review is true, choose "human_review" then "FINISH".
3. If reviewer decision is "APPROVED", choose "FINISH".
4. If reviewer decision is "RETRY", choose "resolution" again.
5. Normal happy path: triage → routing → resolution → reviewer → FINISH.

Respond with ONLY valid JSON in this exact format (no markdown, no explanation):
{"next_agent": "<agent_name>", "reasoning": "<brief explanation, max 100 words>"}
"""


def _build_user_message(
    message: str,
    customer_name: str,
    completed_agents: list[str],
    state_summary: dict,
) -> str:
    return (
        f"Customer: {customer_name}\n"
        f"Message: {message[:300]}\n\n"
        f"Agents already completed: {completed_agents or ['none']}\n"
        f"Current state:\n{json.dumps(state_summary, indent=2, default=str)}\n\n"
        "Which agent should run next?"
    )


def _parse_supervisor_json(raw: str) -> SupervisorOutput | None:
    """
    Try to extract and validate JSON from the LLM response.
    Handles markdown code fences and stray text around the JSON block.
    """
    # Strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    # Find the first {...} block
    match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
        return SupervisorOutput(**data)
    except Exception:
        return None


def _fallback_decision(completed_agents: list[str], state_summary: dict) -> SupervisorOutput:
    """
    Deterministic fallback when the LLM fails or returns invalid JSON.
    Mirrors the happy-path ordering: triage → routing → resolution → reviewer → FINISH.
    """
    requires_review = (state_summary.get("routing_decision") or {}).get(
        "requires_human_review", False
    )
    review_decision = state_summary.get("review_decision")

    if "triage" not in completed_agents:
        return SupervisorOutput(next_agent="triage", reasoning="Fallback: starting triage.")
    if "routing" not in completed_agents:
        return SupervisorOutput(next_agent="routing", reasoning="Fallback: routing after triage.")
    if requires_review:
        return SupervisorOutput(next_agent="human_review", reasoning="Fallback: requires_human_review flag set.")
    if "resolution" not in completed_agents:
        return SupervisorOutput(next_agent="resolution", reasoning="Fallback: generating resolution.")
    if "reviewer" not in completed_agents:
        return SupervisorOutput(next_agent="reviewer", reasoning="Fallback: quality review.")
    if review_decision == "RETRY":
        return SupervisorOutput(next_agent="resolution", reasoning="Fallback: reviewer requested retry.")
    if review_decision == "HUMAN_REVIEW":
        return SupervisorOutput(next_agent="human_review", reasoning="Fallback: reviewer escalated.")
    return SupervisorOutput(next_agent="FINISH", reasoning="Fallback: all agents completed.")


class OpenRouterSupervisorAdapter:
    """
    Production Supervisor adapter using OpenRouter API.

    Uses a fast, cheap model (gpt-4o-mini by default) since the supervisor
    only needs to output a short JSON decision, not generate long text.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-4o-mini",
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 15.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._timeout = timeout

    async def supervise(
        self,
        *,
        message: str,
        customer_name: str,
        completed_agents: list[str],
        state_summary: dict,
    ) -> SupervisorOutput:
        user_msg = _build_user_message(message, customer_name, completed_agents, state_summary)

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._model,
                        "messages": [
                            {"role": "system", "content": _SYSTEM_PROMPT},
                            {"role": "user", "content": user_msg},
                        ],
                        "temperature": 0.0,  # deterministic — supervisor must be consistent
                        "max_tokens": 150,
                    },
                )
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"]

            parsed = _parse_supervisor_json(raw)
            if parsed is not None:
                logger.info(f"[supervisor_adapter] next={parsed.next_agent}")
                return parsed

            logger.warning(f"[supervisor_adapter] failed to parse LLM response: {raw!r}")

        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[supervisor_adapter] LLM call failed: {exc!r} — using fallback")

        # Safe degradation: deterministic fallback
        fallback = _fallback_decision(completed_agents, state_summary)
        logger.info(f"[supervisor_adapter] fallback next={fallback.next_agent}")
        return fallback
