"""Phase 1 diagnostic: calibration analysis and item difficulty.

Measures how LLM confidence/consistency correlates with actual correctness
across tiers, domains, and difficulty levels.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from negbiorl.data_registry import (
    BENCHMARK_MODELS,
    get_gold_answer_field,
    get_l1_parser,
    load_export,
    load_predictions,
    parse_l4_unified,
)


def _check_correct(raw_prediction: str, gold: str, domain: str, task: str) -> bool:
    """Parse prediction using domain-specific parser and check correctness."""
    if not raw_prediction:
        return False
    if task == "l1":
        parser = get_l1_parser(domain)
        parsed = parser(raw_prediction)
        return parsed is not None and parsed.upper() == gold.strip().upper()
    elif task == "l4":
        parsed, _ = parse_l4_unified(raw_prediction, domain)
        return parsed is not None and parsed.lower() == gold.strip().lower()
    else:
        return (raw_prediction or "").strip().lower() == gold.strip().lower()


def compute_accuracy_by_tier(
    predictions: list[dict],
    export: list[dict],
    domain: str,
    task: str,
) -> dict[str, dict[str, float]]:
    """Compute accuracy stratified by confidence tier."""
    gold_field = get_gold_answer_field(domain)
    export_by_qid = {r["question_id"]: r for r in export}

    tier_results: dict[str, list[bool]] = defaultdict(list)
    for pred in predictions:
        qid = pred["question_id"]
        rec = export_by_qid.get(qid)
        if rec is None:
            continue
        gold = rec.get(gold_field, "")
        raw_prediction = pred.get("prediction", "")
        tier = rec.get("confidence_tier") or rec.get("metadata", {}).get("confidence_tier") or "unknown"

        correct = _check_correct(raw_prediction, gold, domain, task)
        tier_results[tier].append(correct)

    return {
        tier: {
            "accuracy": float(np.mean(results)),
            "n": len(results),
            "n_correct": sum(results),
        }
        for tier, results in sorted(tier_results.items())
    }


def compute_item_difficulty(
    domain: str,
    task: str,
    models: list[str] | None = None,
    fewshot: str = "fs0",
) -> dict[str, dict[str, Any]]:
    """Compute item difficulty = fraction of models that got each question right.

    Returns: {question_id: {"difficulty": float, "n_correct": int, "n_models": int}}
    """
    models = models or BENCHMARK_MODELS
    gold_field = get_gold_answer_field(domain)
    export = load_export(domain, task)
    export_by_qid = {r["question_id"]: r for r in export}

    # Collect predictions across models
    qid_correct: dict[str, list[bool]] = defaultdict(list)
    for model in models:
        try:
            preds = load_predictions(domain, model, task, fewshot)
        except FileNotFoundError:
            continue
        for pred in preds:
            qid = pred["question_id"]
            rec = export_by_qid.get(qid)
            if rec is None:
                continue
            gold = rec.get(gold_field, "")
            raw_prediction = pred.get("prediction", "")
            correct = _check_correct(raw_prediction, gold, domain, task)
            qid_correct[qid].append(correct)

    return {
        qid: {
            "difficulty": 1.0 - float(np.mean(results)),  # higher = harder
            "n_correct": sum(results),
            "n_models": len(results),
        }
        for qid, results in qid_correct.items()
    }


def compute_cross_model_correlation(
    domain: str,
    task: str,
    models: list[str] | None = None,
    fewshot: str = "fs0",
) -> dict[str, Any]:
    """Compute Pearson correlation of per-item accuracy across model pairs.

    High correlation = items are consistently easy/hard across models.
    Low correlation = model-specific failure modes.
    """
    models = models or BENCHMARK_MODELS
    gold_field = get_gold_answer_field(domain)
    export = load_export(domain, task)
    export_by_qid = {r["question_id"]: r for r in export}

    # Build model → {qid → correct} map
    model_qid_correct: dict[str, dict[str, int]] = {}
    for model in models:
        try:
            preds = load_predictions(domain, model, task, fewshot)
        except FileNotFoundError:
            continue
        qid_map = {}
        for pred in preds:
            qid = pred["question_id"]
            rec = export_by_qid.get(qid)
            if rec is None:
                continue
            gold = rec.get(gold_field, "")
            raw_prediction = pred.get("prediction", "")
            qid_map[qid] = 1 if _check_correct(raw_prediction, gold, domain, task) else 0
        model_qid_correct[model] = qid_map

    # Compute pairwise correlations
    model_names = list(model_qid_correct.keys())
    correlations = {}
    for i, m1 in enumerate(model_names):
        for m2 in model_names[i + 1:]:
            common_qids = sorted(set(model_qid_correct[m1]) & set(model_qid_correct[m2]))
            if len(common_qids) < 10:
                continue
            v1 = [model_qid_correct[m1][q] for q in common_qids]
            v2 = [model_qid_correct[m2][q] for q in common_qids]
            if np.std(v1) == 0 or np.std(v2) == 0:
                corr = float("nan")
            else:
                corr = float(np.corrcoef(v1, v2)[0, 1])
            correlations[f"{m1}_vs_{m2}"] = {
                "correlation": corr,
                "n_common": len(common_qids),
            }

    all_corrs = [v["correlation"] for v in correlations.values() if not np.isnan(v["correlation"])]
    return {
        "pairwise": correlations,
        "mean_correlation": float(np.mean(all_corrs)) if all_corrs else float("nan"),
        "n_pairs": len(correlations),
    }
