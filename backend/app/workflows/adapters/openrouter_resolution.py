"""
adapters/openrouter_resolution.py — OpenRouter LLM adapter for the Resolution Agent.

Separate from the Triage adapter (Interface Segregation):
- Different prompt, different output schema, different error type.
- Shares only the httpx client pattern and settings.

Prompt design:
- Provides full triage context (classification, priority, sentiment, summary,
  confidence, risk_flags) and routing context (assigned_team, sla_hours,
  escalation_level, requires_human_review, internal_notes).
- Instructs the model to write a safe, professional, non-misleading response.
- Tone selection is driven by sentiment and priority.
- Raises ResolutionModelError on any failure — never HTTPException.
"""

from __future__ import annotations

import json

import httpx
from pydantic import ValidationError

from app.config import settings
from app.utils.logger import logger
from app.workflows.adapters.base import ResolutionOutput
from app.workflows.errors import ResolutionModelError

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

RESOLUTION_SYSTEM_PROMPT = """\
You are an expert AI customer support resolution specialist.

Your task: given a classified support ticket and its routing context, write:
1. A professional customer-facing response (suggested_response)
2. An internal recommended action for the support agent (recommended_action)

RESPONSE REQUIREMENTS:
- Be professional, clear, and concise.
- Be safe: do not make promises the system cannot keep.
- Be honest: do not mislead the customer about timelines or outcomes.
- Match tone to sentiment and priority:
  * negative/angry sentiment → empathetic tone; acknowledge frustration explicitly
  * critical/urgent priority → urgent tone; convey immediacy
  * enterprise/formal context → formal tone
  * informational/general → informational tone
- Do NOT include specific refund amounts, exact fix dates, or SLA numbers
  unless they are explicitly stated in the routing context.
- Do NOT address the customer by name in the response (privacy-safe).
- The suggested_response must be a complete, ready-to-send customer reply.
- The recommended_action must be a single, actionable internal instruction
  (e.g. "Escalate to finance team — verify duplicate charge on account.").

OUTPUT FORMAT — return ONLY valid JSON, no markdown, no explanations:
{
  "suggested_response": "<complete customer-facing reply, 10-2000 chars>",
  "recommended_action": "<internal action for support agent, 10-500 chars>",
  "tone": "<one of: empathetic, formal, urgent, informational>",
  "confidence": <float 0.0-1.0>
}
"""


def _build_user_message(
    ticket: dict,
    triage: dict,
    routing_decision: dict,
    context: list[str],
) -> str:
    """Build the user-turn message combining all available context."""
    lines = [
        "=== TICKET ===",
        f"Message: {ticket.get('message', '')}",
        "",
        "=== TRIAGE OUTPUT ===",
        f"Classification: {triage.get('classification', 'general')}",
        f"Priority: {triage.get('priority', 'medium')}",
        f"Sentiment: {triage.get('sentiment', 'neutral')}",
        f"Summary: {triage.get('summary', '')}",
        f"Confidence: {triage.get('confidence', 0.0):.2f}",
        f"Risk flags: {', '.join(triage.get('risk_flags', [])) or 'none'}",
        "",
        "=== ROUTING DECISION ===",
        f"Assigned team: {routing_decision.get('assigned_team', 'support_general')}",
        f"SLA hours: {routing_decision.get('sla_hours', 24)}",
        f"Escalation level: {routing_decision.get('escalation_level', 0)}",
        f"Requires human review: {routing_decision.get('requires_human_review', False)}",
        f"Internal notes: {routing_decision.get('internal_notes', '')}",
    ]
    if context:
        lines += ["", "=== CONTEXT ==="]
        lines += [f"- {c}" for c in context]
    return "\n".join(lines)


class OpenRouterResolutionAdapter:
    """
    Production Resolution Agent adapter.
    Calls OpenRouter with the resolution-specific prompt.
    Raises ResolutionModelError on any failure — never HTTPException.
    """

    async def generate_resolution(
        self,
        *,
        ticket: dict,
        triage: dict,
        routing_decision: dict,
        context: list[str],
    ) -> ResolutionOutput:
        user_content = _build_user_message(ticket, triage, routing_decision, context)

        logger.info(
            f"[OpenRouterResolutionAdapter] calling model={settings.openrouter_model} "
            f"cls={triage.get('classification')} pri={triage.get('priority')}"
        )

        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": settings.app_url,
            "X-Title": settings.app_name,
        }
        payload = {
            "model": settings.openrouter_model,
            "messages": [
                {"role": "system", "content": RESOLUTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.3,
            "max_tokens": 800,
            "response_format": {"type": "json_object"},
        }

        try:
            async with httpx.AsyncClient(timeout=settings.ai_request_timeout) as client:
                resp = await client.post(OPENROUTER_API_URL, headers=headers, json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ResolutionModelError(f"HTTP error calling OpenRouter: {exc}") from exc

        raw = resp.json()["choices"][0]["message"]["content"].strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ResolutionModelError(f"LLM returned invalid JSON: {raw[:300]}") from exc

        try:
            result = ResolutionOutput(**data)
        except (ValidationError, TypeError) as exc:
            raise ResolutionModelError(f"ResolutionOutput validation failed: {exc}") from exc

        logger.info(
            f"[OpenRouterResolutionAdapter] resolution complete "
            f"tone={result.tone} conf={result.confidence:.2f}"
        )
        return result
