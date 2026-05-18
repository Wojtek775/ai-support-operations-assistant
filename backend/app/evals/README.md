# AI Evaluation System

This module benchmarks the AI Support & Operations Assistant pipeline
against the synthetic dataset of 100 realistic support tickets.

---

## Purpose

Measure how accurately the AI model classifies incoming tickets across
four structured output dimensions:

| Dimension | AI field | Dataset field |
|---|---|---|
| Category classification | `classification` | `category` |
| Priority detection | `priority` | `priority` |
| Sentiment detection | `sentiment` | `sentiment` |
| Action recommendation | `recommended_action` | `expected_action` |

> **Note:** This eval runner measures **structured output alignment only**.
> It does not measure full semantic response quality yet.
> Semantic evaluation (suggested_response quality, summary relevance)
> will be added in a future phase using LLM-as-judge scoring.

---

## SCHEMA_MAPPING

### Field name differences between dataset and AI output

The dataset and AI pipeline use different field names for equivalent concepts:

```
dataset.category        → AI output: classification
dataset.expected_action → AI output: recommended_action
dataset.priority        → AI output: priority          (same)
dataset.sentiment       → AI output: sentiment         (same name, different values)
```

### Sentiment normalization

The synthetic dataset uses richer sentiment labels than the AI model produces.
The following normalization is applied before comparison:

```
frustrated  →  negative
confused    →  neutral
positive    →  positive    (no change)
neutral     →  neutral     (no change)
angry       →  angry       (no change)
```

Rationale: `frustrated` is the closest emotional equivalent to `negative` in
the AI's 4-value schema. `confused` maps to `neutral` (low arousal, low valence).

This normalization is intentional and documented. If the AI correctly labels
a frustrated customer as `negative`, that is counted as a correct prediction.

---

## Files

```
backend/app/evals/
├── run_evals.py       — evaluation pipeline (modular, async)
├── eval_results.json  — latest evaluation results (or placeholder zeros)
└── README.md          — this file

backend/tests/
└── test_evals.py      — unit tests for pure eval logic (no OpenRouter calls)
```

---

## Usage

### Run the full evaluation (requires OpenRouter API key in .env)

```bash
cd backend
py -3.12 -m app.evals.run_evals
```

### Dry run (schema check only, no AI calls)

```bash
cd backend
py -3.12 -m app.evals.run_evals --dry-run
```

### Run unit tests only (no API key needed)

```bash
cd backend
py -3.12 -m pytest tests/test_evals.py -v
```

---

## Output: eval_results.json

```json
{
  "meta": {
    "evaluated_at": "2024-01-15T10:30:00Z",
    "dataset_path": "...",
    "schema_version": "1.0.0",
    "note": "Structured output alignment only."
  },
  "total_tickets": 100,
  "overall_accuracy": 0.72,
  "category_accuracy": 0.85,
  "priority_accuracy": 0.78,
  "sentiment_accuracy": 0.81,
  "expected_action_accuracy": 0.69,
  "edge_case_accuracy": 0.45,
  "accuracy_by_category": { "billing": 0.91, ... },
  "accuracy_by_priority": { "urgent": 0.88, ... },
  "accuracy_by_difficulty": { "easy": 0.90, "medium": 0.74, "hard": 0.52 },
  "accuracy_by_edge_case": { "vague_message": 0.30, "angry_customer": 0.60, ... },
  "per_ticket_results": [ ... ]
}
```

---

## Interpreting Results

| Metric | What it tells you |
|---|---|
| `overall_accuracy` | % of tickets where ALL 4 fields are correct |
| `category_accuracy` | % of tickets correctly classified |
| `accuracy_by_difficulty` | Does model degrade on hard tickets? |
| `edge_case_accuracy` | How well does model handle messy real-world input? |
| `accuracy_by_edge_case` | Which edge case type is hardest for the model? |

---

## What Comes Next

After this eval is baseline-established:

1. **Prompt optimization** — improve prompts and re-run to compare
2. **Model comparison** — run the same eval against multiple OpenRouter models
3. **Semantic quality scoring** — LLM-as-judge for `suggested_response`
4. **Fine-tuning dataset** — curate tickets where model failed as training examples
