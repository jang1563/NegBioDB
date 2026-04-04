"""Publication Bias Score (PBS) — Novel metric for quantifying positivity bias.

PBS = P(predicts positive | true negative) - P(predicts positive | true positive)

Positive PBS = model is biased toward positive predictions even for negatives.
PBS near 0 = well-calibrated; PBS < 0 = rare (negative bias).

For L4 tasks: positive = "tested", negative = "untested"
For L1 tasks: depends on domain-specific class definitions
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_pbs(
    predictions: list[str],
    gold_answers: list[str],
    positive_label: str = "tested",
    negative_label: str = "untested",
) -> dict[str, float]:
    """Compute Publication Bias Score for a set of predictions.

    Args:
        predictions: Model predictions (after parsing)
        gold_answers: Ground truth labels
        positive_label: What counts as "positive" prediction
        negative_label: What counts as "negative" prediction

    Returns:
        dict with keys: pbs, p_pos_given_neg, p_pos_given_pos, n_true_pos, n_true_neg
    """
    true_pos_preds = []  # predictions where gold = positive
    true_neg_preds = []  # predictions where gold = negative

    for pred, gold in zip(predictions, gold_answers):
        if pred is None:
            continue
        pred_lower = pred.strip().lower()
        gold_lower = gold.strip().lower()

        if gold_lower == positive_label.lower():
            true_pos_preds.append(pred_lower == positive_label.lower())
        elif gold_lower == negative_label.lower():
            true_neg_preds.append(pred_lower == positive_label.lower())

    n_true_pos = len(true_pos_preds)
    n_true_neg = len(true_neg_preds)

    if n_true_pos == 0 or n_true_neg == 0:
        return {
            "pbs": float("nan"),
            "p_pos_given_neg": float("nan"),
            "p_pos_given_pos": float("nan"),
            "n_true_pos": n_true_pos,
            "n_true_neg": n_true_neg,
        }

    p_pos_given_pos = sum(true_pos_preds) / n_true_pos
    p_pos_given_neg = sum(true_neg_preds) / n_true_neg

    pbs = p_pos_given_neg - p_pos_given_pos

    return {
        "pbs": pbs,
        "p_pos_given_neg": p_pos_given_neg,
        "p_pos_given_pos": p_pos_given_pos,
        "n_true_pos": n_true_pos,
        "n_true_neg": n_true_neg,
    }


def compute_pbs_by_tier(
    predictions: list[str],
    gold_answers: list[str],
    tiers: list[str],
    positive_label: str = "tested",
    negative_label: str = "untested",
) -> dict[str, dict[str, float]]:
    """Compute PBS stratified by confidence tier."""
    tier_groups: dict[str, tuple[list, list]] = {}
    for pred, gold, tier in zip(predictions, gold_answers, tiers):
        if tier not in tier_groups:
            tier_groups[tier] = ([], [])
        tier_groups[tier][0].append(pred)
        tier_groups[tier][1].append(gold)

    results = {}
    for tier, (preds, golds) in sorted(tier_groups.items()):
        results[tier] = compute_pbs(preds, golds, positive_label, negative_label)

    return results


def compute_pbs_delta(
    pbs_before: dict[str, float],
    pbs_after: dict[str, float],
) -> dict[str, float]:
    """Compute change in PBS after training.

    Negative delta = bias reduced (improvement).
    """
    pbs_b = pbs_before["pbs"]
    pbs_a = pbs_after["pbs"]

    if np.isnan(pbs_b) or np.isnan(pbs_a):
        return {
            "delta_pbs": float("nan"),
            "pbs_before": pbs_b,
            "pbs_after": pbs_a,
            "bias_reduced": False,
            "abs_reduction": float("nan"),
        }

    return {
        "delta_pbs": pbs_a - pbs_b,
        "pbs_before": pbs_b,
        "pbs_after": pbs_a,
        "bias_reduced": abs(pbs_a) < abs(pbs_b),  # absolute bias decreased
        "abs_reduction": abs(pbs_b) - abs(pbs_a),  # positive = improvement
    }


def compute_multi_domain_pbs(
    domain_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate PBS across domains.

    Args:
        domain_results: {domain: {"predictions": [...], "gold_answers": [...], ...}}

    Returns:
        {"per_domain": {domain: pbs_dict}, "mean_pbs": float, "max_pbs": float}
    """
    per_domain = {}
    pbs_values = []

    for domain, data in domain_results.items():
        pbs = compute_pbs(
            data["predictions"],
            data["gold_answers"],
            data.get("positive_label", "tested"),
            data.get("negative_label", "untested"),
        )
        per_domain[domain] = pbs
        if not np.isnan(pbs["pbs"]):
            pbs_values.append(pbs["pbs"])

    return {
        "per_domain": per_domain,
        "mean_pbs": float(np.mean(pbs_values)) if pbs_values else float("nan"),
        "max_pbs": float(np.max(pbs_values)) if pbs_values else float("nan"),
        "n_domains": len(pbs_values),
    }
