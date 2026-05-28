# AI Support & Operations Assistant

> **Portfolio MVP** — AI-powered operational workflow system demonstrating intelligent ticket processing, classification, and response generation.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com)
[![OpenRouter](https://img.shields.io/badge/AI-OpenRouter-orange.svg)](https://openrouter.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 What This System Does

This is **not just a chatbot**. It is an AI-powered operational workflow engine that processes customer support messages through a structured pipeline:

1. **Receive** — Accept incoming support ticket via REST API
2. **Classify** — Categorize the issue (billing, technical, account, etc.)
3. **Detect Priority** — Assign urgency level (low / medium / high / critical)
4. **Summarize** — Generate a concise 1-2 sentence summary
5. **Detect Sentiment** — Identify emotional tone (positive / neutral / negative / angry)
6. **Suggest Response** — Draft a professional customer reply
7. **Recommend Action** — Propose an internal operational step
8. **Persist** — Store structured results in SQLite (PostgreSQL-ready)
9. **Expose** — Serve data via REST API for dashboard consumption

---

## 🏗️ Project Structure

```
ai-support-operations-assistant/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app entry point
│   │   ├── config.py            # Environment config (Pydantic Settings)
│   │   ├── models/
│   │   │   ├── ticket.py        # Pydantic request/response models
│   │   │   └── db_models.py     # SQLAlchemy ORM models
│   │   ├── services/
│   │   │   ├── ai_service.py    # OpenRouter API integration
│   │   │   └── ticket_service.py# Business logic: full pipeline
│   │   ├── routers/
│   │   │   ├── tickets.py       # POST /tickets, GET /tickets
│   │   │   └── health.py        # GET /health
│   │   ├── data/                # SQLite DB file lives here
│   │   └── utils/
│   │       └── logger.py        # Structured logging setup
│   ├── tests/
│   │   └── test_tickets.py      # Basic endpoint tests
│   ├── requirements.txt
│   └── .env.example
├── dashboard/
│   ├── app.py                   # Streamlit dashboard entry point
│   └── requirements.txt         # Dashboard-only dependencies
├── docs/
│   ├── TECHNICAL_ARCHITECTURE.md
│   └── ROADMAP.md
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone and set up environment

```bash
git clone https://github.com/your-username/ai-support-operations-assistant.git
cd ai-support-operations-assistant/backend

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env — add your OpenRouter API key
```

### 3. Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

### 4. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Submit a ticket
curl -X POST http://localhost:8000/tickets \
  -H "Content-Type: application/json" \
  -d '{"customer_name": "Jan Kowalski", "email": "jan@example.com", "message": "My invoice is wrong, I was charged twice this month."}'
```

Interactive docs: **http://localhost:8000/docs**

---

## 📡 API Endpoints

| Method | Endpoint         | Description                          |
|--------|-----------------|--------------------------------------|
| GET    | `/health`        | System health check                  |
| POST   | `/tickets`       | Submit a new support ticket          |
| GET    | `/tickets`       | List all processed tickets           |
| GET    | `/tickets/{id}`  | Get a single ticket by ID            |

---

## 🧠 AI Pipeline Output Example

```json
{
  "ticket_id": "tkt_20240115_001",
  "customer_name": "Jan Kowalski",
  "email": "jan@example.com",
  "original_message": "My invoice is wrong, I was charged twice this month.",
  "classification": "billing",
  "priority": "high",
  "summary": "Customer reports duplicate billing charge for the current month.",
  "sentiment": "negative",
  "suggested_response": "Dear Jan, we sincerely apologize for the billing error. Our finance team will review your account within 24 hours and issue a refund if a duplicate charge is confirmed.",
  "recommended_action": "ESCALATE_TO_BILLING — Investigate duplicate charge, verify payment records, initiate refund if confirmed.",
  "processed_at": "2024-01-15T10:30:00Z",
  "ai_model_used": "openai/gpt-4o-mini"
}
```

---

## 🔄 Workflow Orchestration

Beyond AI classification, the system applies **deterministic routing rules** to every ticket — no additional AI calls, no randomness. The workflow engine runs after the AI pipeline and produces five additional fields:

| Field | Description |
|---|---|
| `assigned_team` | Team responsible for handling the ticket |
| `sla_hours` | Response window in hours |
| `requires_human_review` | `true` when manual attention is mandatory |
| `escalation_level` | `0` = normal, `1` = elevated, `2` = urgent escalation |
| `internal_notes` | Hardcoded routing explanation (not AI-generated) |

### Routing Rules

| # | Condition | Result |
|---|---|---|
| 1 | `outage` + `urgent/critical` | `infrastructure_team`, SLA 1h |
| 1b | `technical` + `critical` | `infrastructure_team`, SLA 1h |
| 2 | `billing` or `refund` | `finance_team`, SLA 4h |
| 3 | `enterprise_sales` | `account_executive`, SLA 2h |
| 4 | `cancellation` | `retention_queue`, SLA 2h |
| 5 | `technical` (non-critical) | `technical_team`, SLA 8h |
| 6 | `account` | `account_team`, SLA 8h |
| — | default | `support_general`, SLA 24h |
| **+** | `angry` + `urgent/critical` | `escalation_level=2`, `requires_human_review=true` |
| **+** | `high+` priority + `negative/angry` | `requires_human_review=true`, `escalation_level≥1` |

### Internal Notes Examples

```
"Urgent outage detected. Route to infrastructure team immediately."
"Billing/refund issue. Route to finance team."
"Cancellation risk detected. Route to retention queue."
"ESCALATED: Angry customer with urgent priority — immediate human review required."
```

---

## 🖥️ Dashboard

A lightweight Streamlit dashboard that visualises backend metrics and processed tickets in real time.

### Run the backend first

```bash
cd ai-support-operations-assistant/backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Install dashboard dependencies

```bash
cd ai-support-operations-assistant/dashboard
pip install -r requirements.txt
```

### Run the dashboard

```bash
cd ai-support-operations-assistant/dashboard
streamlit run app.py
```

The dashboard opens at **http://localhost:8501** by default.

### Dashboard features

| Section | Description |
|---|---|
| **System Health** | Live backend status (🟢 Online / 🔴 Offline), version, URL |
| **AI Operations Overview** | 5 metric cards: total tickets, urgent, high-priority, negative sentiment, top category |
| **Recent Tickets** | Sortable table of up to 50 most recent processed tickets |
| **Sidebar** | Configurable backend URL, Refresh button, project info |

> If no tickets have been processed yet, the dashboard shows:
> *"No tickets processed yet. Submit a ticket via /docs or POST /tickets to populate the dashboard."*

### Dashboard screenshot

![Dashboard Screenshot](docs/dashboard_screenshot.png)

---

## 🧪 Running Tests

```bash
cd backend
pytest tests/ -v
```

---

## 🛠️ Tech Stack

| Layer       | Technology              |
|-------------|------------------------|
| Backend     | Python 3.11 + FastAPI  |
| AI Provider | OpenRouter API         |
| Database    | SQLite (→ PostgreSQL)  |
| ORM         | SQLAlchemy 2.x         |
| Validation  | Pydantic v2            |
| Testing     | pytest + httpx         |

---

---

## 🗃️ Synthetic Dataset

The project ships with a **100-ticket AI-ready evaluation dataset** designed to benchmark the pipeline against realistic, production-like support messages.

### Generate / Regenerate

```bash
cd backend
py -3.12 app/data/generate_dataset.py
```

Outputs:
- `backend/app/data/generated/tickets_dataset.json` — 100 tickets
- `backend/app/data/generated/dataset_summary.json` — distribution statistics

### Ticket Schema

```json
{
  "ticket_id": "SYN-0001",
  "customer_name": "Sarah Mitchell",
  "email": "sarah.mitchell@company.io",
  "message": "I was charged twice for my subscription...",
  "category": "billing",
  "priority": "high",
  "sentiment": "frustrated",
  "expected_action": "escalate_billing",
  "difficulty": "easy",
  "language": "en",
  "edge_case": null
}
```

### Dataset Composition

| Property | Values |
|---|---|
| **Categories** | billing, technical_issue, account_access, refund, feature_request, outage, enterprise_sales, bug_report, cancellation, integration_problem |
| **Priorities** | low, medium, high, urgent |
| **Sentiments** | positive, neutral, frustrated, angry, confused |
| **Difficulty** | easy (27), medium (46), hard (27) |
| **Regular tickets** | 80 (8 per category) |
| **Edge case tickets** | 20 — vague, angry, multi-issue, typo-heavy, cancellation threats, enterprise escalations |

### Edge Case Types

Specifically designed to stress-test AI classification accuracy:

| Type | Description |
|---|---|
| `vague_message` | "It's not working. Please help." |
| `angry_customer` | ALL CAPS, threats, emotional overload |
| `multiple_issues` | 3 different problems in one email |
| `missing_reference` | No invoice/order number provided |
| `typo_heavy` | "my acocunt is blokced" |
| `cancellation_threat` | Implicit churn signal inside a support request |
| `enterprise_escalation` | Formal SLA breach notification from CTO |
| `unclear_ownership` | Reference to previous conversation with no context |
| `emotional_low_info` | High emotion, zero actionable details |
| `urgent_ops` | PRODUCTION DOWN, P0, revenue impact |

### How This Dataset Will Be Used

1. **Phase 3** — Run the full AI pipeline against all 100 tickets
2. **Compare** `AI output` vs `expected_action / category / priority`
3. **Calculate** accuracy scores per category and per difficulty tier
4. **Identify** where the AI model underperforms (likely on edge cases)
5. **Future** — fine-tuning dataset or prompt optimization benchmark

> **Deterministic:** `random.seed(42)` — every regeneration produces identical files.
> Safe for CI/CD and reproducible experiments.

---

---

## 🔬 AI Evaluation System

The project includes a benchmarking pipeline to measure how accurately the AI model processes tickets against ground-truth labels.

### Run evaluation (requires OpenRouter API key)

```bash
cd backend
py -3.12 -m app.evals.run_evals
```

### Dry run (no API calls — schema check only)

```bash
cd backend
py -3.12 -m app.evals.run_evals --dry-run
```

### What it measures

| Dimension | AI field | Dataset field |
|---|---|---|
| Category | `classification` | `category` |
| Priority | `priority` | `priority` |
| Sentiment | `sentiment` | `sentiment` (normalized) |
| Action | `recommended_action` | `expected_action` |

### Sentiment normalization

The dataset uses richer labels (`frustrated`, `confused`) than the AI returns. Applied mapping:

```
frustrated → negative
confused   → neutral
```

### Output: `backend/app/evals/eval_results.json`

```json
{
  "total_tickets": 100,
  "overall_accuracy": 0.72,
  "category_accuracy": 0.85,
  "accuracy_by_difficulty": { "easy": 0.90, "medium": 0.74, "hard": 0.52 },
  "accuracy_by_edge_case": { "vague_message": 0.30, "angry_customer": 0.60 }
}
```

> **Note:** This eval measures **structured output alignment only**.
> Semantic quality evaluation (response quality, summary relevance) will be added in a future phase.

See full documentation: [`backend/app/evals/README.md`](backend/app/evals/README.md)

---

## 📌 Use Cases

- **LinkedIn content** — Demonstrating real AI-powered system design
- **GitHub portfolio** — Production-ready code structure
- **Job applications** — AI Integration Developer / AI Automation Engineer
- **Consulting** — Small business AI workflow automation starter

---

## 📄 Docs

- [Technical Architecture](docs/TECHNICAL_ARCHITECTURE.md)
- [Roadmap](docs/ROADMAP.md)

---

## 📝 License

MIT — free to use, modify, and distribute.
