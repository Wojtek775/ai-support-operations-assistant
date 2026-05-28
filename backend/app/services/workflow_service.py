"""
services/workflow_service.py — Deterministic workflow routing engine.

This module contains NO database queries and makes NO AI API calls.
It applies fixed business rules to produce routing decisions based solely
on the AI-classified ticket fields: classification, priority, sentiment.

All internal_notes are hardcoded strings — produced by rule logic,
not by the language model. Same inputs always produce the same output.
"""

from dataclasses import dataclass


@dataclass
class WorkflowDecision:
    """
    The result of applying workflow routing rules to a classified ticket.

    All fields are populated deterministically — same inputs always
    produce the same outputs, with no external calls.
    """

    assigned_team: str
    sla_hours: int
    requires_human_review: bool
    escalation_level: int
    internal_notes: str


def route_ticket(
    classification: str,
    priority: str,
    sentiment: str,
) -> WorkflowDecision:
    """
    Apply deterministic routing rules to a classified ticket.

    Args:
        classification: AI-assigned ticket category (e.g. "billing", "outage").
        priority:       AI-assigned urgency level (e.g. "high", "urgent").
        sentiment:      AI-detected customer sentiment (e.g. "angry", "negative").

    Returns:
        WorkflowDecision with team assignment, SLA hours, escalation level,
        human-review flag, and deterministic internal notes.

    Primary routing rules (first match wins):
        1. outage + urgent/critical      → infrastructure_team,  sla=1h
        1b. technical + critical         → infrastructure_team,  sla=1h
        2. billing / refund              → finance_team,          sla=4h
        3. enterprise_sales              → account_executive,     sla=2h
        4. cancellation                  → retention_queue,       sla=2h
        5. technical (non-critical)      → technical_team,        sla=8h
        6. account                       → account_team,          sla=8h
        7. feature_request / bug_report  → product_team,          sla=72h
        default                          → support_general,        sla=24h

    Escalation overlays (applied on top of team assignment):
        8. angry + urgent/critical       → escalation_level=2, requires_human_review=True
        9. high/urgent/critical + negative/angry → requires_human_review=True, escalation_level>=1
    """
    c = classification.lower().strip()
    p = priority.lower().strip()
    s = sentiment.lower().strip()

    # --- Default values ---
    assigned_team = "support_general"
    sla_hours = 24
    requires_human_review = False
    escalation_level = 0
    notes_parts: list[str] = []

    # -----------------------------------------------------------------------
    # Primary team routing — first match wins
    # -----------------------------------------------------------------------

    # Rule 1: Urgent or critical outage → infrastructure team immediately
    if c == "outage" and p in ("urgent", "critical"):
        assigned_team = "infrastructure_team"
        sla_hours = 1
        notes_parts.append(
            "Urgent outage detected. Route to infrastructure team immediately."
        )

    # Rule 1b: Critical technical issue → infrastructure team
    elif c == "technical" and p == "critical":
        assigned_team = "infrastructure_team"
        sla_hours = 1
        notes_parts.append(
            "Critical technical issue. Route to infrastructure team immediately."
        )

    # Rule 2: Billing or refund → finance team
    elif c in ("billing", "refund"):
        assigned_team = "finance_team"
        sla_hours = 4
        notes_parts.append("Billing/refund issue. Route to finance team.")

    # Rule 3: Enterprise sales → account executive
    elif c == "enterprise_sales":
        assigned_team = "account_executive"
        sla_hours = 2
        notes_parts.append(
            "Enterprise sales enquiry. Route to account executive team."
        )

    # Rule 4: Cancellation threat → retention queue
    elif c == "cancellation":
        assigned_team = "retention_queue"
        sla_hours = 2
        notes_parts.append(
            "Cancellation risk detected. Route to retention queue."
        )

    # Rule 5: Technical (non-critical) → technical team
    elif c == "technical":
        assigned_team = "technical_team"
        sla_hours = 8
        notes_parts.append("Technical issue. Assigned to technical support team.")

    # Rule 6: Account access issues → account team
    elif c == "account":
        assigned_team = "account_team"
        sla_hours = 8
        notes_parts.append("Account issue. Assigned to account management team.")

    # Rule 7: Feature requests / bug reports → product team
    elif c in ("feature_request", "bug_report"):
        assigned_team = "product_team"
        sla_hours = 72
        notes_parts.append("Product feedback received. Assigned to product team.")

    # -----------------------------------------------------------------------
    # Escalation overlays — applied on top of team assignment
    # -----------------------------------------------------------------------

    # Rule 8: Angry customer with urgent/critical priority → escalate to level 2
    if s == "angry" and p in ("urgent", "critical"):
        escalation_level = 2
        requires_human_review = True
        notes_parts.append(
            "ESCALATED: Angry customer with urgent priority — immediate human review required."
        )

    # Rule 9: High+ priority with negative/angry sentiment → requires human review
    if p in ("high", "urgent", "critical") and s in ("negative", "angry"):
        requires_human_review = True
        if escalation_level == 0:
            escalation_level = 1

    return WorkflowDecision(
        assigned_team=assigned_team,
        sla_hours=sla_hours,
        requires_human_review=requires_human_review,
        escalation_level=escalation_level,
        internal_notes=" | ".join(notes_parts) if notes_parts else "Standard routing applied.",
    )
