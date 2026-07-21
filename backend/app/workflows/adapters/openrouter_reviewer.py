"""
adapters/openrouter_reviewer.py — Production Quality Reviewer adapter via OpenRouter.

Sends the resolution output along with full ticket/triage/routing context to the LLM
and asks it to review quality, safety, completeness and professionalism.

Returns a validated ReviewOutput with APPROVED / RETRY / HUMAN_REVIEW decision.
"""

from __future__ import annotations

import json

import httpx

from app.config import settings
from app.utils.logger import logger
from app.workflows.adapters.base import ReviewOutput
from app.workflows.errors import ReviewerModelError


_REVIEW_SYSTEM_PROMPT = """You are a Quality Reviewer for an AI customer-support system.

Your job is to evaluate a generated resolution and decide:
  APPROVED      — The response is complete, professional, safe and appropriate.
  RETRY         — The response has fixable issues; provide feedback notes.
  HUMAN_REVIEW  — The case is too complex or sensitive for automated handling.

Evaluate:
1. Alignment with the customer's original message.
2. Consistency with classification, priority and sentiment.
3. Absence of dangerous or unauthorised promises.
4. Completeness of the recommended_action.
5. Professional and appropriate tone.
6. Presence of potential sensitive data (PII, credentials).
7. Whether the case exceeds the scope of automated resolution.

Respond with ONLY a JSON object matching this schema:
{
  "decision": "APPROVED" | "RETRY" | "HUMAN_REVIEW",
  "confidence": <float 0.0-1.0>,
  "issues": [<string>, ...],
  "notes": "<feedback for retry or human reviewer>"
}
"""


class OpenRouterReviewerAdapter:
    """
    Production Quality Reviewer adapter.
    Calls OpenRouter API with the full resolution context and returns a ReviewOutput.
    """

    def __init__(self) -> None:
        self._api_key = settings.openrouter_api_key
        self._model = settings.openrouter_model
        self._base_url = "https://openrouter.ai/api/v1/chat/completions"

    async def review(
        self,
        *,
        ticket: dict,
        triage: dict,
        routing_decision: dict,
        resolution: dict,
    ) -> ReviewOutput:
        user_content = f"""
TICKET:
  customer_name: {ticket.get('customer_name')}
  email: {ticket.get('email')}
  message: {ticket.get('message')}

TRIAGE:
  classification: {triage.get('classification')}
  priority: {triage.get('priority')}
  sentiment: {triage.get('sentiment')}
  summary: {triage.get('summary')}
  confidence: {triage.get('confidence')}
  risk_flags: {triage.get('risk_flags')}

ROUTING:
  assigned_team: {routing_decision.get('assigned_team')}
  sla_hours: {routing_decision.get('sla_hours')}
  escalation_level: {routing_decision.get('escalation_level')}

RESOLUTION:
  suggested_response: {resolution.get('suggested_response')}
  recommended_action: {resolution.get('recommended_action')}
  tone: {resolution.get('tone')}
  confidence: {resolution.get('confidence')}

Review this resolution and respond with a JSON object only.
""".strip()

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    self._base_url,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._model,
                        "messages": [
                            {"role": "system", "content": _REVIEW_SYSTEM_PROMPT},
                            {"role": "user", "content": user_content},
                        ],
                        "temperature": 0.1,
                        "response_format": {"type": "json_object"},
                    },
                )
                resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ReviewerModelError(f"OpenRouter HTTP error in reviewer: {exc}") from exc

        try:
            raw_text = resp.json()["choices"][0]["message"]["content"]
            data = json.loads(raw_text)
            result = ReviewOutput(**data)
        except Exception as exc:
            raise ReviewerModelError(
                f"Failed to parse reviewer LLM response: {exc}"
            ) from exc

        logger.info(
            f"[openrouter_reviewer] decision={result.decision} "
            f"confidence={result.confidence:.2f} issues={len(result.issues)}"
        )
        return result
