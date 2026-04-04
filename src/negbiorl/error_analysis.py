"""Phase 1 diagnostic: error taxonomy and failure mode classification.

Classifies LLM errors across domains into a taxonomy:
- positivity_bias: model predicts "tested" for untested pairs
- false_alarm: model predicts "untested" for tested pairs
- calibration_failure: confidence doesn't correlate with correctness
- tier_sensitivity: performance varies by evidence tier
- reasoning_collapse: model gives correct answer but wrong reasoning (L3)
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from negbiorl.data_registry import (
    get_domain,
    get_gold_answer_field,
    get_l1_parser,
    load_export,
    load_predictions,
    parse_l4_unified,
)


# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------

ERROR_TYPES = [
    "positivity_bias",
    "false_alarm",
    "parse_failure",
    "wrong_class",
    "tier_sensitivity",
]


def classify_l1_errors(
    predictions: list[dict],
    export: list[dict],
    domain: str,
) -> list[dict[str, Any]]:
    """Classify L1 prediction errors using domain-specific parser."""
    gold_field = get_gold_answer_field(domain)
    class_field = get_domain(domain)["gold_class_field"]
    export_by_qid = {r["question_id"]: r for r in export}
    parser = get_l1_parser(domain)

    errors = []
    for pred in predictions:
        qid = pred["question_id"]
        rec = export_by_qid.get(qid)
        if rec is None:
            continue

        gold = rec.get(gold_field, "")
        raw_prediction = pred.get("prediction", "")
        parsed = parser(raw_prediction) if raw_prediction else None

        if parsed is None:
            errors.append({
                "question_id": qid,
                "error_type": "parse_failure",
                "gold": gold,
                "predicted": raw_prediction,
                "domain": domain,
                "tier": rec.get("confidence_tier") or rec.get("metadata", {}).get("confidence_tier") or "unknown",
                "difficulty": rec.get("difficulty"),
                "class": rec.get(class_field),
            })
        elif parsed.upper() != gold.strip().upper():
            errors.append({
                "question_id": qid,
                "error_type": "wrong_class",
                "gold": gold,
                "predicted": parsed,
                "domain": domain,
                "tier": rec.get("confidence_tier") or rec.get("metadata", {}).get("confidence_tier") or "unknown",
                "difficulty": rec.get("difficulty"),
                "class": rec.get(class_field),
            })

    return errors


def classify_l4_errors(
    predictions: list[dict],
    export: list[dict],
    domain: str,
) -> list[dict[str, Any]]:
    """Classify L4 prediction errors into positivity_bias or false_alarm.

    Uses domain-specific L4 parser to extract answer from raw prediction text.
    """
    gold_field = get_gold_answer_field(domain)
    export_by_qid = {r["question_id"]: r for r in export}

    errors = []
    for pred in predictions:
        qid = pred["question_id"]
        rec = export_by_qid.get(qid)
        if rec is None:
            continue

        gold = rec.get(gold_field, "")
        raw_prediction = pred.get("prediction", "")

        # Parse using domain-specific parser
        parsed_answer, _evidence = parse_l4_unified(raw_prediction, domain) if raw_prediction else (None, None)

        if parsed_answer is None:
            error_type = "parse_failure"
        elif parsed_answer.lower() != gold.strip().lower():
            if gold.strip().lower() == "untested" and parsed_answer.lower() == "tested":
                error_type = "positivity_bias"
            elif gold.strip().lower() == "tested" and parsed_answer.lower() == "untested":
                error_type = "false_alarm"
            else:
                error_type = "wrong_class"
        else:
            continue  # correct prediction

        errors.append({
            "question_id": qid,
            "error_type": error_type,
            "gold": gold,
            "predicted": parsed_answer or raw_prediction,
            "domain": domain,
            "temporal_group": rec.get("temporal_group"),
            "tier": rec.get("confidence_tier") or rec.get("metadata", {}).get("confidence_tier") or "unknown",
        })

    return errors


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def summarize_errors(errors: list[dict]) -> dict[str, Any]:
    """Summarize error taxonomy distribution."""
    type_counts = Counter(e["error_type"] for e in errors)
    total = len(errors)

    summary = {
        "total_errors": total,
        "error_distribution": {
            etype: {"count": count, "fraction": count / total if total > 0 else 0}
            for etype, count in type_counts.most_common()
        },
    }

    # Per-tier breakdown
    tier_errors: dict[str, Counter] = defaultdict(Counter)
    for e in errors:
        tier = e.get("tier") or "unknown"
        tier_errors[tier][e["error_type"]] += 1

    summary["per_tier"] = {
        tier: dict(counts) for tier, counts in sorted(tier_errors.items())
    }

    return summary


def run_error_analysis(
    domain: str,
    model: str,
    task: str,
    fewshot: str = "fs0",
    shot_config: str = "zero-shot",
) -> dict[str, Any]:
    """Run full error analysis for a domain/model/task combination.

    Returns error taxonomy with counts and distributions.
    """
    export = load_export(domain, task)
    predictions = load_predictions(domain, model, task, fewshot, shot_config=shot_config)

    if task == "l1":
        errors = classify_l1_errors(predictions, export, domain)
    elif task == "l4":
        errors = classify_l4_errors(predictions, export, domain)
    else:
        raise ValueError(f"Error analysis not implemented for task={task}")

    summary = summarize_errors(errors)
    summary["domain"] = domain
    summary["model"] = model
    summary["task"] = task
    summary["fewshot"] = fewshot
    summary["n_predictions"] = len(predictions)
    summary["error_rate"] = summary["total_errors"] / len(predictions) if predictions else 0

    return summary
