"""
services/ai_service.py — OpenRouter API integration.

Responsible for:
1. Building the structured prompt for the AI
2. Calling the OpenRouter API via httpx
3. Parsing and validating the JSON response
4. Returning a clean dict with all 6 AI-generated fields

This module has NO knowledge of the database or HTTP routing.
It only talks to the AI API and returns data.
"""

import json
import httpx
from app.config import settings
from app.utils.logger import logger


# The OpenRouter API endpoint (OpenAI-compatible)
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Expected fields in the AI JSON response
REQUIRED_AI_FIELDS = {
    "classification",
    "priority",
    "summary",
    "sentiment",
    "suggested_response",
    "recommended_action",
}

# System prompt — instructs the AI to return structured JSON
SYSTEM_PROMPT = """You are an expert customer support analyst working for a professional support operations team.

Your job is to analyze incoming customer support messages and return a structured analysis in JSON format.

You must return ONLY a valid JSON object — no markdown, no code blocks, no extra text — with exactly these fields:

{
  "classification": "<one of: billing, technical, account, shipping, general>",
  "priority": "<one of: low, medium, high, critical>",
  "summary": "<1-2 sentence summary of the customer issue>",
  "sentiment": "<one of: positive, neutral, negative, angry>",
  "suggested_response": "<professional, empathetic reply to send to the customer>",
  "recommended_action": "<internal action for the support team, e.g. ESCALATE_TO_BILLING, ASSIGN_TO_TECH_TEAM, etc.>"
}

Classification guidelines:
- billing: payment issues, invoices, charges, refunds, subscriptions
- technical: bugs, errors, crashes, performance, integrations
- account: login, password, profile, permissions, access
- shipping: delivery, tracking, returns, damaged goods
- general: anything that doesn't fit the above

Priority guidelines:
- critical: service is completely down, data loss, security breach
- high: major functionality broken, billing error, urgent deadline
- medium: partial issue, workaround available, moderate frustration
- low: cosmetic issue, general question, minor inconvenience

Return ONLY the JSON. Nothing else."""


async def process_ticket_with_ai(customer_message: str) -> dict:
    """
    Send the customer message to OpenRouter and return parsed AI analysis.

    Args:
        customer_message: The raw support message text.

    Returns:
        dict with keys: classification, priority, summary, sentiment,
                        suggested_response, recommended_action

    Raises:
        httpx.HTTPStatusError: If the API returns a non-2xx status.
        ValueError: If the AI response cannot be parsed as valid JSON
                    or is missing required fields.
    """
    logger.info(f"Calling OpenRouter API | model={settings.openrouter_model} | message_length={len(customer_message)}")

    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": settings.app_url,
        "X-Title": settings.app_name,
    }

    payload = {
        "model": settings.openrouter_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": customer_message},
        ],
        "temperature": 0.2,      # Low temperature for consistent structured output
        "max_tokens": 1000,
        "response_format": {"type": "json_object"},  # Enforce JSON output where supported
    }

    async with httpx.AsyncClient(timeout=settings.ai_request_timeout) as client:
        response = await client.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
        )

    # Raise immediately if API returned an error status
    response.raise_for_status()

    response_data = response.json()

    # Extract the AI message content
    raw_content = response_data["choices"][0]["message"]["content"].strip()
    logger.debug(f"Raw AI response: {raw_content[:200]}...")

    # Parse JSON from the AI response
    try:
        ai_result = json.loads(raw_content)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse AI response as JSON: {e} | raw={raw_content[:500]}")
        raise ValueError(f"AI returned invalid JSON. Raw response: {raw_content[:500]}")

    # Validate all required fields are present
    missing_fields = REQUIRED_AI_FIELDS - set(ai_result.keys())
    if missing_fields:
        logger.error(f"AI response missing fields: {missing_fields}")
        raise ValueError(f"AI response missing required fields: {missing_fields}")

    logger.info(
        f"AI processing complete | "
        f"classification={ai_result.get('classification')} | "
        f"priority={ai_result.get('priority')} | "
        f"sentiment={ai_result.get('sentiment')}"
    )

    return ai_result
