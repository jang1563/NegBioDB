"""Phase 4: Before/after evaluation pipeline.

Evaluates fine-tuned models against NegBioDB benchmarks and computes
improvement metrics (MCC delta, PBS delta, accuracy delta).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from negbiorl.data_registry import (
    get_domain,
    get_gold_answer_field,
    get_l1_parser,
    load_export,
    parse_l4_unified,
)
from negbiorl.pbs_metric import compute_pbs


def evaluate_l1(
    predictions: list[dict],
    domain: str,
) -> dict[str, float]:
    """Evaluate L1 MCQ predictions for a domain.

    Returns: accuracy, parse_rate, per_class_accuracy, mcc
    """
    from sklearn.metrics import matthews_corrcoef, accuracy_score

    gold_field = get_gold_answer_field(domain)
    export = load_export(domain, "l1")
    export_by_qid = {r["question_id"]: r for r in export}
    parser = get_l1_parser(domain)

    y_true, y_pred = [], []
    parse_failures = 0

    for pred in predictions:
        rec = export_by_qid.get(pred["question_id"])
        if rec is None:
            continue
        gold = rec.get(gold_field, "").strip()
        raw_prediction = pred.get("prediction", "")
        parsed = parser(raw_prediction) if raw_prediction else None
        if parsed is None:
            parse_failures += 1
            continue
        y_true.append(gold.upper())
        y_pred.append(parsed.upper())

    total = len(predictions)
    if not y_true:
        return {"accuracy": 0.0, "mcc": 0.0, "parse_rate": 0.0, "n": total}

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(set(y_true)) > 1 else 0.0,
        "parse_rate": 1.0 - parse_failures / total if total > 0 else 0.0,
        "n": total,
    }


def evaluate_l4(
    predictions: list[dict],
    domain: str,
) -> dict[str, float]:
    """Evaluate L4 tested/untested predictions.

    Returns: accuracy, mcc, pbs, parse_rate, contamination_gap
    """
    from sklearn.metrics import matthews_corrcoef, accuracy_score

    gold_field = get_gold_answer_field(domain)
    export = load_export(domain, "l4")
    export_by_qid = {r["question_id"]: r for r in export}

    y_true, y_pred = [], []
    temporal_results: dict[str, list[bool]] = {}
    parse_failures = 0

    for pred in predictions:
        rec = export_by_qid.get(pred["question_id"])
        if rec is None:
            continue
        gold = rec.get(gold_field, "").strip().lower()
        raw_prediction = pred.get("prediction", "")
        parsed_answer, _evidence = parse_l4_unified(raw_prediction, domain) if raw_prediction else (None, None)
        if parsed_answer is None:
            parse_failures += 1
            continue

        pred_clean = parsed_answer.lower()
        y_true.append(gold)
        y_pred.append(pred_clean)

        # Track temporal group for contamination analysis
        tg = rec.get("temporal_group")
        if tg:
            temporal_results.setdefault(tg, []).append(pred_clean == gold)

    total = len(predictions)
    if not y_true:
        return {"accuracy": 0.0, "mcc": 0.0, "pbs": float("nan"), "parse_rate": 0.0, "n": total}

    pbs = compute_pbs(y_pred, y_true)

    # Contamination gap
    contamination_gap = float("nan")
    groups = get_domain(domain).get("temporal_groups", [])
    if len(groups) == 2:
        early = temporal_results.get(groups[0], [])
        late = temporal_results.get(groups[1], [])
        if early and late:
            contamination_gap = float(np.mean(early)) - float(np.mean(late))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(set(y_true)) > 1 else 0.0,
        "pbs": pbs["pbs"],
        "parse_rate": 1.0 - parse_failures / total if total > 0 else 0.0,
        "contamination_gap": contamination_gap,
        "n": total,
    }


def evaluate_before_after(
    before_predictions: list[dict],
    after_predictions: list[dict],
    domain: str,
    task: str,
) -> dict[str, Any]:
    """Compute before/after improvement metrics."""
    if task == "l1":
        eval_fn = evaluate_l1
    elif task == "l4":
        eval_fn = evaluate_l4
    else:
        raise ValueError(f"Before/after eval not implemented for task={task}")

    before = eval_fn(before_predictions, domain)
    after = eval_fn(after_predictions, domain)

    # Compute deltas (skip count fields)
    deltas = {}
    for key in before:
        if key == "n":
            continue
        if isinstance(before[key], (int, float)) and isinstance(after[key], (int, float)):
            deltas[f"delta_{key}"] = after[key] - before[key]

    return {
        "before": before,
        "after": after,
        "deltas": deltas,
        "domain": domain,
        "task": task,
    }


# ---------------------------------------------------------------------------
# Main results table
# ---------------------------------------------------------------------------

def build_results_table(
    model_results: dict[str, dict[str, dict[str, Any]]],
) -> list[dict]:
    """Build the main results table from structured results.

    Args:
        model_results: {model_stage: {domain: {task: metrics}}}

    Returns:
        Flat list of rows for table rendering
    """
    rows = []
    for model_stage, domains in model_results.items():
        for domain, tasks in domains.items():
            for task, metrics in tasks.items():
                row = {
                    "model_stage": model_stage,
                    "domain": domain,
                    "task": task,
                }
                row.update(metrics)
                rows.append(row)
    return rows
