"""
evals/run_evals.py — AI Evaluation Runner

Measures classification quality of the AI pipeline against
the synthetic support ticket dataset.

WHAT IT MEASURES:
  For each of the 100 synthetic tickets, the runner:
  1. Sends only the ticket's `message` field to the AI pipeline
  2. Compares AI output against the ground-truth labels:
     - AI `classification`    vs dataset `category`
     - AI `priority`          vs dataset `priority`
     - AI `sentiment`         vs dataset `sentiment` (normalized)
     - AI `recommended_action` vs dataset `expected_action`
  3. Aggregates accuracy scores across multiple dimensions

FIELD MAPPING (dataset → AI output):
  dataset.category        → ai_output["classification"]
  dataset.expected_action → ai_output["recommended_action"]
  dataset.sentiment       → ai_output["sentiment"]  (after normalization)
  dataset.priority        → ai_output["priority"]

SENTIMENT NORMALIZATION:
  The dataset uses: frustrated, confused, positive, neutral, angry
  The AI returns:   positive, neutral, negative, angry
  Mapping applied:  frustrated → negative
                    confused   → neutral

NOTE:
  This eval runner measures structured output alignment only.
  It does not measure full semantic response quality yet.

USAGE:
  cd backend
  py -3.12 -m app.evals.run_evals

  Or to run a quick dry-run with mock predictions:
  py -3.12 -m app.evals.run_evals --dry-run
"""

import asyncio
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATASET_PATH = Path(__file__).parent.parent / "data" / "generated" / "tickets_dataset.json"
RESULTS_PATH = Path(__file__).parent / "eval_results.json"


# ---------------------------------------------------------------------------
# Sentiment normalization map
# Maps dataset sentiment values → AI pipeline output values
# ---------------------------------------------------------------------------

SENTIMENT_NORMALIZATION: dict[str, str] = {
    "frustrated": "negative",
    "confused": "neutral",
    "positive": "positive",
    "neutral": "neutral",
    "angry": "angry",
}


def normalize_sentiment(raw_sentiment: str) -> str:
    """
    Normalize a dataset sentiment value to the AI pipeline's output vocabulary.

    The dataset contains richer sentiment labels (frustrated, confused) that
    the AI model does not produce. This function maps them to the closest
    equivalent in the AI's output schema.

    Args:
        raw_sentiment: Sentiment value from the dataset (e.g. "frustrated").

    Returns:
        Normalized sentiment matching AI output vocabulary.
    """
    return SENTIMENT_NORMALIZATION.get(raw_sentiment.lower(), raw_sentiment.lower())


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_dataset(path: Path = DATASET_PATH) -> list[dict]:
    """
    Load the synthetic ticket dataset from JSON.

    Args:
        path: Path to tickets_dataset.json.

    Returns:
        List of ticket dicts. Each dict contains:
        ticket_id, customer_name, email, message, category, priority,
        sentiment, expected_action, difficulty, language, edge_case.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Run: py -3.12 app/data/generate_dataset.py"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Single-ticket evaluation
# ---------------------------------------------------------------------------

def evaluate_prediction(ticket: dict, ai_output: dict) -> dict:
    """
    Compare AI predictions against a single ticket's ground-truth labels.

    Field mapping:
      ticket["category"]        → ai_output["classification"]
      ticket["priority"]        → ai_output["priority"]
      ticket["sentiment"]       → ai_output["sentiment"]  (after normalization)
      ticket["expected_action"] → ai_output["recommended_action"]

    Args:
        ticket: A single dict from the dataset with ground-truth fields.
        ai_output: Dict returned by process_ticket_with_ai(), containing
                   keys: classification, priority, sentiment, recommended_action.

    Returns:
        Dict with per-field match booleans and metadata for aggregation.
    """
    expected_category = ticket["category"]
    expected_priority = ticket["priority"]
    expected_sentiment = normalize_sentiment(ticket["sentiment"])
    expected_action = ticket["expected_action"]

    predicted_category = (ai_output.get("classification") or "").lower().strip()
    predicted_priority = (ai_output.get("priority") or "").lower().strip()
    predicted_sentiment = (ai_output.get("sentiment") or "").lower().strip()
    predicted_action = (ai_output.get("recommended_action") or "").lower().strip()

    category_match = predicted_category == expected_category
    priority_match = predicted_priority == expected_priority
    sentiment_match = predicted_sentiment == expected_sentiment
    action_match = predicted_action == expected_action

    # Overall: all 4 fields must match for a ticket to be "fully correct"
    all_correct = all([category_match, priority_match, sentiment_match, action_match])

    return {
        "ticket_id": ticket["ticket_id"],
        "difficulty": ticket["difficulty"],
        "edge_case": ticket["edge_case"],
        # Ground truth
        "expected_category": expected_category,
        "expected_priority": expected_priority,
        "expected_sentiment": expected_sentiment,
        "expected_action": expected_action,
        # AI predictions
        "predicted_category": predicted_category,
        "predicted_priority": predicted_priority,
        "predicted_sentiment": predicted_sentiment,
        "predicted_action": predicted_action,
        # Match booleans
        "category_match": category_match,
        "priority_match": priority_match,
        "sentiment_match": sentiment_match,
        "action_match": action_match,
        "all_correct": all_correct,
    }


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------

def _safe_accuracy(correct: int, total: int) -> float:
    """Return accuracy as a float rounded to 4 decimal places, or 0.0 if total=0."""
    return round(correct / total, 4) if total > 0 else 0.0


def calculate_metrics(predictions: list[dict]) -> dict:
    """
    Aggregate per-ticket predictions into accuracy scores across all dimensions.

    Args:
        predictions: List of dicts returned by evaluate_prediction().

    Returns:
        Dict with overall accuracy, per-field accuracy, and breakdowns
        by category, priority, difficulty, and edge case type.
    """
    n = len(predictions)
    if n == 0:
        return {"total_tickets": 0, "overall_accuracy": 0.0}

    # --- Overall field accuracies ---
    cat_correct = sum(1 for p in predictions if p["category_match"])
    pri_correct = sum(1 for p in predictions if p["priority_match"])
    sen_correct = sum(1 for p in predictions if p["sentiment_match"])
    act_correct = sum(1 for p in predictions if p["action_match"])
    all_correct = sum(1 for p in predictions if p["all_correct"])

    # --- Breakdown by category ---
    by_cat: dict[str, dict] = defaultdict(lambda: {"total": 0, "correct": 0})
    for p in predictions:
        cat = p["expected_category"]
        by_cat[cat]["total"] += 1
        if p["category_match"]:
            by_cat[cat]["correct"] += 1
    accuracy_by_category = {
        k: _safe_accuracy(v["correct"], v["total"])
        for k, v in sorted(by_cat.items())
    }

    # --- Breakdown by priority ---
    by_pri: dict[str, dict] = defaultdict(lambda: {"total": 0, "correct": 0})
    for p in predictions:
        pri = p["expected_priority"]
        by_pri[pri]["total"] += 1
        if p["priority_match"]:
            by_pri[pri]["correct"] += 1
    accuracy_by_priority = {
        k: _safe_accuracy(v["correct"], v["total"])
        for k, v in sorted(by_pri.items())
    }

    # --- Breakdown by difficulty ---
    by_diff: dict[str, dict] = defaultdict(lambda: {"total": 0, "correct": 0})
    for p in predictions:
        diff = p["difficulty"]
        by_diff[diff]["total"] += 1
        if p["all_correct"]:
            by_diff[diff]["correct"] += 1
    accuracy_by_difficulty = {
        k: _safe_accuracy(v["correct"], v["total"])
        for k, v in sorted(by_diff.items())
    }

    # --- Breakdown by edge case type ---
    by_ec: dict[str, dict] = defaultdict(lambda: {"total": 0, "correct": 0})
    for p in predictions:
        ec = p["edge_case"]
        if ec is not None:
            by_ec[ec]["total"] += 1
            if p["all_correct"]:
                by_ec[ec]["correct"] += 1
    edge_case_total = sum(1 for p in predictions if p["edge_case"] is not None)
    edge_case_correct = sum(1 for p in predictions if p["edge_case"] is not None and p["all_correct"])
    accuracy_by_edge_case = {
        k: _safe_accuracy(v["correct"], v["total"])
        for k, v in sorted(by_ec.items())
    }

    return {
        "total_tickets": n,
        "overall_accuracy": _safe_accuracy(all_correct, n),
        "category_accuracy": _safe_accuracy(cat_correct, n),
        "priority_accuracy": _safe_accuracy(pri_correct, n),
        "sentiment_accuracy": _safe_accuracy(sen_correct, n),
        "expected_action_accuracy": _safe_accuracy(act_correct, n),
        "edge_case_accuracy": _safe_accuracy(edge_case_correct, edge_case_total),
        "accuracy_by_category": accuracy_by_category,
        "accuracy_by_priority": accuracy_by_priority,
        "accuracy_by_difficulty": accuracy_by_difficulty,
        "accuracy_by_edge_case": accuracy_by_edge_case,
    }


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

def save_results(metrics: dict, per_ticket: list[dict], path: Path = RESULTS_PATH) -> None:
    """
    Save evaluation results to a JSON file.

    Args:
        metrics: Output of calculate_metrics().
        per_ticket: List of per-ticket evaluation dicts.
        path: Output file path (default: evals/eval_results.json).
    """
    output = {
        "meta": {
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_path": str(DATASET_PATH),
            "schema_version": "1.0.0",
            "note": (
                "This eval measures structured output alignment only. "
                "Semantic response quality is not yet evaluated."
            ),
        },
        **metrics,
        "per_ticket_results": per_ticket,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

async def run_evaluation(
    ai_fn=None,
    dataset_path: Path = DATASET_PATH,
    results_path: Path = RESULTS_PATH,
    dry_run: bool = False,
) -> dict:
    """
    Full evaluation pipeline.

    Loads the dataset, calls the AI function for each ticket,
    evaluates predictions, calculates metrics, and saves results.

    Args:
        ai_fn: Async callable with signature async(message: str) -> dict.
               Must return keys: classification, priority, sentiment,
               recommended_action.
               If None, the real process_ticket_with_ai is used.
        dataset_path: Override for dataset file path.
        results_path: Override for output results file path.
        dry_run: If True, skip AI calls and use a no-op prediction
                 (all fields empty) — for schema testing only.

    Returns:
        The full metrics dict (same structure as eval_results.json).
    """
    if ai_fn is None and not dry_run:
        # Import here to avoid circular imports and to keep tests clean
        from app.services.ai_service import process_ticket_with_ai
        ai_fn = process_ticket_with_ai

    tickets = load_dataset(dataset_path)
    predictions = []
    total = len(tickets)

    print(f"[*] Starting evaluation on {total} tickets...")

    for i, ticket in enumerate(tickets, 1):
        if dry_run:
            ai_output = {
                "classification": "",
                "priority": "",
                "sentiment": "",
                "recommended_action": "",
            }
        else:
            try:
                ai_output = await ai_fn(ticket["message"])
            except Exception as e:
                print(f"  [WARN] AI call failed for {ticket['ticket_id']}: {e}")
                ai_output = {
                    "classification": "error",
                    "priority": "error",
                    "sentiment": "error",
                    "recommended_action": "error",
                }

        result = evaluate_prediction(ticket, ai_output)
        predictions.append(result)

        if i % 10 == 0:
            print(f"  Progress: {i}/{total}")

    metrics = calculate_metrics(predictions)
    save_results(metrics, predictions, results_path)

    # --- Terminal summary ---
    print("\n--- Evaluation Results ---")
    print(f"  Total tickets evaluated : {metrics['total_tickets']}")
    print(f"  Overall accuracy        : {metrics['overall_accuracy']:.1%}")
    print(f"  Category accuracy       : {metrics['category_accuracy']:.1%}")
    print(f"  Priority accuracy       : {metrics['priority_accuracy']:.1%}")
    print(f"  Sentiment accuracy      : {metrics['sentiment_accuracy']:.1%}")
    print(f"  Action accuracy         : {metrics['expected_action_accuracy']:.1%}")
    print(f"  Edge case accuracy      : {metrics['edge_case_accuracy']:.1%}")
    print(f"\n[OK] Results saved -> {results_path}")

    return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    if dry_run:
        print("[*] DRY RUN mode — no AI calls will be made.")
    asyncio.run(run_evaluation(dry_run=dry_run))
