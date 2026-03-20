"""Evaluation functions for LLM benchmark tasks L1–L4.

Pattern follows metrics.py: pure functions, NumPy-based, comprehensive.
"""

import json
import re
from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef


# ── L1: MCQ Classification ───────────────────────────────────────────────────


def parse_l1_answer(response: str) -> str | None:
    """Extract single letter answer (A/B/C/D) from LLM response."""
    response = response.strip()
    if not response:
        return None
    # Try exact single letter
    if response.upper() in ("A", "B", "C", "D"):
        return response.upper()
    # Try "Answer: X", "Answer is X", "(X)", "X." patterns
    for pattern in [
        r"(?:answer|choice)\s*(?:is|:)\s*\(?([ABCD])\)?",
        r"\(([ABCD])\)",
        r"^([ABCD])\.",
        r"^([ABCD])\)",
    ]:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    # Fallback: first letter if A-D
    first = response[0].upper()
    if first in ("A", "B", "C", "D"):
        return first
    # Last resort: any standalone A-D
    match = re.search(r"\b([ABCD])\b", response.upper())
    return match.group(1) if match else None


def evaluate_l1(
    predictions: list[str],
    gold_answers: list[str],
    gold_classes: list[str] | None = None,
) -> dict:
    """Evaluate L1 MCQ predictions.

    Returns: accuracy, weighted_f1, macro_f1, mcc, per_class_accuracy.
    """
    parsed = [parse_l1_answer(p) for p in predictions]
    valid_mask = [p is not None for p in parsed]
    valid_pred = [p for p, m in zip(parsed, valid_mask) if m]
    valid_gold = [g for g, m in zip(gold_answers, valid_mask) if m]

    if not valid_pred:
        return {
            "accuracy": 0.0,
            "weighted_f1": 0.0,
            "macro_f1": 0.0,
            "mcc": 0.0,
            "parse_rate": 0.0,
            "n_valid": 0,
            "n_total": len(predictions),
        }

    labels = sorted(set(valid_gold + valid_pred))
    result = {
        "accuracy": accuracy_score(valid_gold, valid_pred),
        "weighted_f1": f1_score(valid_gold, valid_pred, average="weighted", labels=labels),
        "macro_f1": f1_score(valid_gold, valid_pred, average="macro", labels=labels),
        "mcc": matthews_corrcoef(valid_gold, valid_pred),
        "parse_rate": sum(valid_mask) / len(predictions),
        "n_valid": sum(valid_mask),
        "n_total": len(predictions),
    }

    # Per-class accuracy
    if gold_classes:
        valid_classes = [c for c, m in zip(gold_classes, valid_mask) if m]
        class_correct = Counter()
        class_total = Counter()
        for pred, gold, cls in zip(valid_pred, valid_gold, valid_classes):
            class_total[cls] += 1
            if pred == gold:
                class_correct[cls] += 1
        result["per_class_accuracy"] = {
            cls: class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0.0
            for cls in sorted(class_total)
        }

    return result


# ── L2: Structured Extraction ─────────────────────────────────────────────────


def parse_l2_response(response: str) -> dict | None:
    """Parse JSON from LLM response for L2 extraction."""
    # Try to find JSON in response
    response = response.strip()
    # Remove markdown code fences
    response = re.sub(r"```json\s*", "", response)
    response = re.sub(r"```\s*$", "", response)

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to find JSON object in text
        match = re.search(r"\{[\s\S]*\}", response)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
    return None


def evaluate_l2(
    predictions: list[str],
    gold_records: list[dict],
) -> dict:
    """Evaluate L2 structured extraction.

    Returns: schema_compliance, field_f1, entity_f1, avg_field_f1.
    """
    n_valid_json = 0
    field_scores = []
    entity_scores = []

    for pred_str, gold in zip(predictions, gold_records):
        parsed = parse_l2_response(pred_str)
        if parsed is None:
            continue
        n_valid_json += 1

        # Schema compliance: check required fields
        gold_results = gold.get("negative_results", [])

        # Entity-level F1: match compound-target pairs
        pred_results = parsed.get("negative_results", [])
        if isinstance(pred_results, list) and isinstance(gold_results, list):
            gold_pairs = {
                (r.get("compound", "").lower(), r.get("target", "").lower())
                for r in gold_results
            }
            pred_pairs = {
                (r.get("compound", "").lower(), r.get("target", "").lower())
                for r in pred_results
            }
            if gold_pairs:
                tp = len(gold_pairs & pred_pairs)
                prec = tp / len(pred_pairs) if pred_pairs else 0.0
                rec = tp / len(gold_pairs)
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                entity_scores.append(f1)

        # Field-level F1 for top-level fields
        for field in ["total_inactive_count", "positive_results_mentioned"]:
            if field in gold:
                gold_val = gold[field]
                pred_val = parsed.get(field)
                if pred_val is not None:
                    if str(gold_val).lower() == str(pred_val).lower():
                        field_scores.append(1.0)
                    else:
                        field_scores.append(0.0)

    return {
        "schema_compliance": n_valid_json / len(predictions) if predictions else 0.0,
        "entity_f1": float(np.mean(entity_scores)) if entity_scores else 0.0,
        "field_accuracy": float(np.mean(field_scores)) if field_scores else 0.0,
        "n_valid_json": n_valid_json,
        "n_total": len(predictions),
    }


# ── L3: Reasoning (LLM-as-Judge) ─────────────────────────────────────────────

L3_JUDGE_PROMPT = """Rate the following scientific explanation of why a compound is inactive against a target.

Compound: {compound_name}
Target: {target_gene} ({target_uniprot})

Explanation to evaluate:
{response}

Rate on these 4 dimensions (1-5 each):
1. Accuracy: Are the scientific claims factually correct?
2. Reasoning: Is the logical chain from structure to inactivity sound?
3. Completeness: Are all relevant factors considered (binding, selectivity, SAR)?
4. Specificity: Does the explanation use specific molecular details, not generalities?

Respond in JSON: {{"accuracy": X, "reasoning": X, "completeness": X, "specificity": X}}"""


def parse_l3_judge_scores(response: str) -> dict | None:
    """Parse judge scores from response."""
    parsed = parse_l2_response(response)  # reuse JSON parser
    if parsed is None:
        return None
    dims = ["accuracy", "reasoning", "completeness", "specificity"]
    scores = {}
    for dim in dims:
        val = parsed.get(dim)
        if isinstance(val, (int, float)) and 1 <= val <= 5:
            scores[dim] = float(val)
    return scores if len(scores) == 4 else None


def evaluate_l3(judge_scores: list[dict]) -> dict:
    """Aggregate L3 judge scores.

    judge_scores: list of {"accuracy": X, "reasoning": X, ...} dicts.
    Returns mean ± std per dimension + overall.
    """
    dims = ["accuracy", "reasoning", "completeness", "specificity"]
    result = {}

    valid = [s for s in judge_scores if s is not None]
    if not valid:
        result = {dim: {"mean": 0.0, "std": 0.0} for dim in dims}
        result["overall"] = {"mean": 0.0, "std": 0.0}
        result["n_valid"] = 0
        result["n_total"] = len(judge_scores)
        return result

    for dim in dims:
        values = [s[dim] for s in valid if dim in s]
        result[dim] = {
            "mean": float(np.mean(values)) if values else 0.0,
            "std": float(np.std(values)) if values else 0.0,
        }

    all_scores = [
        np.mean([s[d] for d in dims]) for s in valid if all(d in s for d in dims)
    ]
    result["overall"] = {
        "mean": float(np.mean(all_scores)) if all_scores else 0.0,
        "std": float(np.std(all_scores)) if all_scores else 0.0,
    }
    result["n_valid"] = len(valid)
    result["n_total"] = len(judge_scores)

    return result


# ── L4: Tested vs Untested ────────────────────────────────────────────────────


def parse_l4_answer(response: str) -> tuple[str | None, str | None]:
    """Parse L4 response into (answer, evidence).

    Returns (tested/untested, evidence_text).
    """
    lines = response.strip().split("\n")
    if not lines:
        return None, None

    first = lines[0].strip().lower()
    answer = None
    if "untested" in first or "not tested" in first or "not been tested" in first:
        answer = "untested"
    elif "tested" in first:
        answer = "tested"

    evidence = "\n".join(lines[1:]).strip() if len(lines) > 1 else None
    return answer, evidence


def evaluate_l4(
    predictions: list[str],
    gold_answers: list[str],
    temporal_groups: list[str] | None = None,
) -> dict:
    """Evaluate L4 tested/untested predictions.

    Returns: accuracy, f1, mcc, evidence_citation_rate,
             temporal accuracy (pre_2023 vs post_2024).
    """
    parsed = [parse_l4_answer(p) for p in predictions]
    answers = [p[0] for p in parsed]
    evidences = [p[1] for p in parsed]

    valid_mask = [a is not None for a in answers]
    valid_pred = [a for a, m in zip(answers, valid_mask) if m]
    valid_gold = [g for g, m in zip(gold_answers, valid_mask) if m]

    if not valid_pred:
        return {
            "accuracy": 0.0,
            "f1": 0.0,
            "mcc": 0.0,
            "parse_rate": 0.0,
            "evidence_citation_rate": 0.0,
        }

    result = {
        "accuracy": accuracy_score(valid_gold, valid_pred),
        "f1": f1_score(
            valid_gold, valid_pred, average="binary",
            pos_label="tested", zero_division=0.0,
        ),
        "mcc": matthews_corrcoef(valid_gold, valid_pred),
        "parse_rate": sum(valid_mask) / len(predictions),
        "n_valid": sum(valid_mask),
        "n_total": len(predictions),
    }

    # Evidence citation rate (for correctly predicted "tested" pairs)
    tested_correct = [
        i
        for i, (a, g, m) in enumerate(zip(answers, gold_answers, valid_mask))
        if m and a == "tested" and g == "tested"
    ]
    # Evidence must be substantive: >50 chars or contain known DB/DOI keywords
    _EVIDENCE_KEYWORDS = {"chembl", "pubchem", "bindingdb", "doi", "pmid", "assay", "ic50", "ki ", "kd "}

    if tested_correct:
        with_evidence = sum(
            1
            for i in tested_correct
            if evidences[i] and (
                len(evidences[i]) > 50
                or any(kw in evidences[i].lower() for kw in _EVIDENCE_KEYWORDS)
            )
        )
        result["evidence_citation_rate"] = with_evidence / len(tested_correct)
    else:
        result["evidence_citation_rate"] = 0.0

    # Temporal accuracy breakdown
    if temporal_groups:
        valid_temporal = [t for t, m in zip(temporal_groups, valid_mask) if m]
        for group in ["pre_2023", "post_2024"]:
            group_pred = [
                p for p, t in zip(valid_pred, valid_temporal) if t == group
            ]
            group_gold = [
                g for g, t in zip(valid_gold, valid_temporal) if t == group
            ]
            if group_pred:
                result[f"accuracy_{group}"] = accuracy_score(group_gold, group_pred)

        # Contamination flag
        pre = result.get("accuracy_pre_2023")
        post = result.get("accuracy_post_2024")
        if pre is not None and post is not None:
            gap = pre - post
            result["contamination_gap"] = round(gap, 4)
            result["contamination_flag"] = gap > 0.15

    return result


# ── Dispatch ──────────────────────────────────────────────────────────────────


def compute_all_llm_metrics(
    task: str,
    predictions: list[str],
    gold: list[dict],
) -> dict:
    """Compute all metrics for a given task.

    Args:
        task: 'l1', 'l2', 'l3', 'l4'
        predictions: list of raw LLM response strings
        gold: list of gold-standard records (from JSONL)

    Returns: dict of metrics
    """
    if task == "l1":
        gold_answers = [g["correct_answer"] for g in gold]
        gold_classes = [g.get("class") for g in gold]
        return evaluate_l1(predictions, gold_answers, gold_classes)

    elif task == "l2":
        return evaluate_l2(predictions, gold)

    elif task == "l3":
        # L3 expects judge scores, not raw predictions
        judge_scores = [parse_l3_judge_scores(p) for p in predictions]
        return evaluate_l3(judge_scores)

    elif task == "l4":
        gold_answers = [g["correct_answer"] for g in gold]
        temporal = [g.get("temporal_group") for g in gold]
        return evaluate_l4(predictions, gold_answers, temporal)

    else:
        raise ValueError(f"Unknown task: {task}")
