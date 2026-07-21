"""
adapters/openrouter.py — OpenRouter LLM adapter for the Triage Agent.

Has its own prompt and structured output parsing.
Does NOT modify ai_service.py — this is a separate pipeline.
Shares only the httpx client pattern and settings.
"""

from __future__ import annotations

import json

import httpx
from pydantic import ValidationError

from app.config import settings
from app.models.ticket import TicketClassification, TicketPriority, TicketSentiment
from app.utils.logger import logger
from app.workflows.adapters.base import TriageOutput
from app.workflows.errors import TriageModelError

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Triage-specific prompt — covers all TicketClassification values,
# returns confidence and risk_flags in addition to the base fields.
TRIAGE_SYSTEM_PROMPT = """You are an expert AI triage analyst for a customer support operations platform.

Analyse the incoming support message and return ONLY a valid JSON object with exactly these fields:

{
  "classification": "<one of: billing, technical, account, shipping, general, refund, outage, enterprise_sales, cancellation, feature_request, bug_report, integration_problem>",
  "priority": "<one of: low, medium, high, critical>",
  "sentiment": "<one of: positive, neutral, negative, angry>",
  "summary": "<1-2 sentence summary of the customer issue>",
  "confidence": <float between 0.0 and 1.0 representing your certainty>,
  "risk_flags": [<optional list of strings: "security", "outage", "data_loss", "revenue_impact", "destructive_action">]
}

Classification guidelines:
- billing: payment issues, invoices, charges, subscriptions
- refund: explicit refund requests
- technical: bugs, errors, crashes, performance issues
- outage: service down, unavailable, P0 incidents
- account: login, password, profile, permissions, access
- shipping: delivery, tracking, returns, damaged goods
- cancellation: cancel subscription, churn risk
- enterprise_sales: enterprise deals, SLA negotiations, CTO-level escalations
- feature_request: requests for new functionality
- bug_report: specific bug reports with reproduction steps
- integration_problem: API, webhook, third-party integration issues
- general: anything else

Priority guidelines:
- critical: service completely down, data loss, security breach, revenue impact
- high: major functionality broken, billing error, urgent deadline
- medium: partial issue, workaround available
- low: cosmetic issue, general question, minor inconvenience

Confidence: how certain are you (0.0=uncertain, 1.0=completely certain)?

Risk flags: include "security" if there is any security concern, "outage" if service is down,
"data_loss" if data may be lost, "revenue_impact" if financial loss is occurring,
"destructive_action" if the customer requests deletion/destruction of data.

Return ONLY the JSON. No markdown. No explanations."""


class OpenRouterTriageAdapter:
    """
    Production LLM adapter. Calls OpenRouter with the triage-specific prompt.
    Raises TriageModelError on any failure — never HTTPException.
    """

    async def triage(self, customer_name: str, email: str, message: str) -> TriageOutput:
        user_content = f"Customer: {customer_name}\nEmail: {email}\nMessage: {message}"

        logger.info(
            f"[OpenRouterTriageAdapter] calling model={settings.openrouter_model} "
            f"msg_len={len(message)}"
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
                {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.1,
            "max_tokens": 500,
            "response_format": {"type": "json_object"},
        }

        try:
            async with httpx.AsyncClient(timeout=settings.ai_request_timeout) as client:
                resp = await client.post(OPENROUTER_API_URL, headers=headers, json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise TriageModelError(f"HTTP error calling OpenRouter: {exc}") from exc

        raw = resp.json()["choices"][0]["message"]["content"].strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TriageModelError(f"LLM returned invalid JSON: {raw[:300]}") from exc

        try:
            result = TriageOutput(**data)
        except (ValidationError, TypeError) as exc:
            raise TriageModelError(f"TriageOutput validation failed: {exc}") from exc

        logger.info(
            f"[OpenRouterTriageAdapter] triage complete "
            f"cls={result.classification} pri={result.priority} "
            f"conf={result.confidence:.2f}"
        )
        return result
