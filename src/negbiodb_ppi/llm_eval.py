"""Evaluation functions for PPI LLM benchmark tasks PPI-L1 through PPI-L4.

Mirrors src/negbiodb_ct/llm_eval.py structure from CT domain.
Key differences from CT:
  - PPI-L1: 4-way (A-D) like DTI, not 5-way (A-E)
  - PPI-L2: entity_f1 on protein pair matching, not field_f1_micro
  - PPI-L3: 4-dimension judge (biological_plausibility, structural_reasoning,
            mechanistic_completeness, specificity)
  - PPI-L4: temporal groups pre_2015/post_2020 (IntAct publication years)
  - PPI uses "gold_answer" field name consistently
"""

from __future__ import annotations

import json
import re
from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef


# ── PPI-L1: MCQ Classification (4-way A-D) ──────────────────────────────

_PPI_L1_LETTERS = {"A", "B", "C", "D"}


def parse_ppi_l1_answer(response: str) -> str | None:
    """Extract single letter answer (A/B/C/D) from LLM response."""
    response = response.strip()
    if not response:
        return None
    # Try exact single letter
    if response.upper() in _PPI_L1_LETTERS:
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
    if first in _PPI_L1_LETTERS:
        return first
    # Last resort: any standalone A-D
    match = re.search(r"\b([ABCD])\b", response.upper())
    return match.group(1) if match else None


def evaluate_ppi_l1(
    predictions: list[str],
    gold_answers: list[str],
    gold_classes: list[str] | None = None,
    difficulties: list[str] | None = None,
) -> dict:
    """Evaluate PPI-L1 MCQ predictions.

    Returns: accuracy, weighted_f1, macro_f1, mcc, parse_rate,
             per_class_accuracy (if gold_classes), per_difficulty_accuracy (if difficulties).
    """
    parsed = [parse_ppi_l1_answer(p) for p in predictions]
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
    result: dict = {
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
        class_correct: Counter = Counter()
        class_total: Counter = Counter()
        for pred, gold, cls in zip(valid_pred, valid_gold, valid_classes):
            class_total[cls] += 1
            if pred == gold:
                class_correct[cls] += 1
        result["per_class_accuracy"] = {
            cls: class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0.0
            for cls in sorted(class_total)
        }

    # Per-difficulty accuracy
    if difficulties:
        valid_diffs = [d for d, m in zip(difficulties, valid_mask) if m]
        diff_correct: Counter = Counter()
        diff_total: Counter = Counter()
        for pred, gold, diff in zip(valid_pred, valid_gold, valid_diffs):
            diff_total[diff] += 1
            if pred == gold:
                diff_correct[diff] += 1
        result["per_difficulty_accuracy"] = {
            d: diff_correct[d] / diff_total[d] if diff_total[d] > 0 else 0.0
            for d in sorted(diff_total)
        }

    return result


# ── PPI-L2: Structured Extraction ───────────────────────────────────────

PPI_L2_REQUIRED_FIELDS = [
    "non_interacting_pairs",
    "total_negative_count",
    "positive_interactions_mentioned",
]


def parse_ppi_l2_response(response: str) -> dict | None:
    """Parse JSON from LLM response for PPI-L2 extraction."""
    response = response.strip()
    # Remove markdown code fences
    response = re.sub(r"```json\s*", "", response)
    response = re.sub(r"```\s*$", "", response)

    try:
        result = json.loads(response)
        if isinstance(result, list) and len(result) > 0:
            result = result[0]
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", response)
        if match:
            try:
                result = json.loads(match.group())
                return result if isinstance(result, dict) else None
            except json.JSONDecodeError:
                return None
    return None


def _normalize_protein(name: str) -> str:
    """Normalize protein identifier for matching."""
    return name.strip().upper().replace("-", "").replace("_", "")


def _extract_pair_set(pairs_list: list[dict]) -> set[tuple[str, str]]:
    """Extract normalized protein pair set from list of pair dicts."""
    result = set()
    for pair in pairs_list:
        p1 = _normalize_protein(str(pair.get("protein_1", "")))
        p2 = _normalize_protein(str(pair.get("protein_2", "")))
        if p1 and p2:
            result.add((min(p1, p2), max(p1, p2)))
    return result


def evaluate_ppi_l2(
    predictions: list[str],
    gold_records: list[dict],
) -> dict:
    """Evaluate PPI-L2 non-interaction extraction.

    Returns: schema_compliance, entity_f1, field_accuracy, parse_rate.
    """
    n_total = len(predictions)
    n_valid_json = 0
    n_schema_compliant = 0

    # Entity-level F1: protein pair matching
    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Field accuracy for total_negative_count and positive_interactions_mentioned
    count_correct = 0
    count_total = 0
    positive_correct = 0
    positive_total = 0

    for pred_str, gold in zip(predictions, gold_records):
        parsed = parse_ppi_l2_response(pred_str)
        if parsed is None:
            gold_pairs = gold.get("gold_extraction", {}).get("non_interacting_pairs", [])
            total_fn += len(gold_pairs)
            continue
        n_valid_json += 1

        # Schema compliance
        has_all = all(f in parsed for f in PPI_L2_REQUIRED_FIELDS)
        if has_all:
            n_schema_compliant += 1

        # Entity F1: protein pair matching
        gold_ext = gold.get("gold_extraction", gold)
        gold_pairs = _extract_pair_set(gold_ext.get("non_interacting_pairs", []))
        pred_pairs = _extract_pair_set(parsed.get("non_interacting_pairs", []))

        tp = len(gold_pairs & pred_pairs)
        total_tp += tp
        total_fp += len(pred_pairs - gold_pairs)
        total_fn += len(gold_pairs - pred_pairs)

        # Count accuracy
        gold_count = gold_ext.get("total_negative_count")
        pred_count = parsed.get("total_negative_count")
        if gold_count is not None:
            count_total += 1
            if pred_count is not None and int(pred_count) == int(gold_count):
                count_correct += 1

        # Positive mentions accuracy
        gold_pos = gold_ext.get("positive_interactions_mentioned")
        pred_pos = parsed.get("positive_interactions_mentioned")
        if gold_pos is not None:
            positive_total += 1
            if pred_pos is not None and bool(pred_pos) == bool(gold_pos):
                positive_correct += 1

    # Compute entity F1
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    entity_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    result: dict = {
        "schema_compliance": n_schema_compliant / n_total if n_total else 0.0,
        "entity_f1": entity_f1,
        "entity_precision": precision,
        "entity_recall": recall,
        "count_accuracy": count_correct / count_total if count_total else 0.0,
        "positive_mention_accuracy": positive_correct / positive_total if positive_total else 0.0,
        "parse_rate": n_valid_json / n_total if n_total else 0.0,
        "n_valid_json": n_valid_json,
        "n_schema_compliant": n_schema_compliant,
        "n_total": n_total,
    }

    return result


# ── PPI-L3: Reasoning (LLM-as-Judge) ───────────────────────────────────

PPI_L3_JUDGE_PROMPT = (
    "You are evaluating a scientific explanation for why two proteins do NOT "
    "physically interact.\n\n"
    "PROTEIN PAIR CONTEXT:\n{context_text}\n\n"
    "EXPERIMENTAL EVIDENCE: {detection_method} assay confirmed no physical interaction.\n\n"
    "RESPONSE TO EVALUATE:\n{response_text}\n\n"
    "Score the response on 4 dimensions (1-5 each):\n"
    "1. biological_plausibility: Are biological reasons (function, pathway, localization) "
    "scientifically sound?\n"
    "2. structural_reasoning: Does the explanation address binding interfaces, "
    "domains, or steric factors?\n"
    "3. mechanistic_completeness: Are multiple relevant factors considered "
    "(expression, tissue, regulation)?\n"
    "4. specificity: Are claims specific to these proteins or generic?\n\n"
    'Return ONLY a JSON object: {{"biological_plausibility": N, '
    '"structural_reasoning": N, "mechanistic_completeness": N, "specificity": N}}'
)

_PPI_L3_DIMS = [
    "biological_plausibility",
    "structural_reasoning",
    "mechanistic_completeness",
    "specificity",
]


def parse_ppi_l3_judge_scores(response: str) -> dict | None:
    """Parse judge scores from response."""
    parsed = parse_ppi_l2_response(response)  # reuse JSON parser
    if parsed is None:
        return None
    scores = {}
    for dim in _PPI_L3_DIMS:
        val = parsed.get(dim)
        if isinstance(val, (int, float)) and 1 <= val <= 5:
            scores[dim] = float(val)
    return scores if len(scores) == 4 else None


def evaluate_ppi_l3(judge_scores: list[dict | None]) -> dict:
    """Aggregate PPI-L3 judge scores.

    judge_scores: list of dicts with dimension scores, or None.
    Returns mean +/- std per dimension + overall.
    """
    result: dict = {}

    valid = [s for s in judge_scores if s is not None]
    if not valid:
        result = {dim: {"mean": 0.0, "std": 0.0} for dim in _PPI_L3_DIMS}
        result["overall"] = {"mean": 0.0, "std": 0.0}
        result["n_valid"] = 0
        result["n_total"] = len(judge_scores)
        return result

    for dim in _PPI_L3_DIMS:
        values = [s[dim] for s in valid if dim in s]
        result[dim] = {
            "mean": float(np.mean(values)) if values else 0.0,
            "std": float(np.std(values)) if values else 0.0,
        }

    all_scores = [
        np.mean([s[d] for d in _PPI_L3_DIMS])
        for s in valid
        if all(d in s for d in _PPI_L3_DIMS)
    ]
    result["overall"] = {
        "mean": float(np.mean(all_scores)) if all_scores else 0.0,
        "std": float(np.std(all_scores)) if all_scores else 0.0,
    }
    result["n_valid"] = len(valid)
    result["n_total"] = len(judge_scores)

    return result


# ── PPI-L4: Tested vs Untested ──────────────────────────────────────────

PPI_EVIDENCE_KEYWORDS = {
    "uniprot", "intact", "huri", "biogrid", "string", "co-ip",
    "two-hybrid", "yeast two-hybrid", "pulldown", "affinity",
    "mass spectrometry", "co-fractionation", "interaction",
    "co-immunoprecipitation", "spr", "surface plasmon",
}


def parse_ppi_l4_answer(response: str) -> tuple[str | None, str | None]:
    """Parse PPI-L4 response into (answer, evidence).

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


def evaluate_ppi_l4(
    predictions: list[str],
    gold_answers: list[str],
    temporal_groups: list[str] | None = None,
) -> dict:
    """Evaluate PPI-L4 tested/untested predictions.

    Returns: accuracy, f1, mcc, evidence_citation_rate,
             temporal accuracy (pre_2015 vs post_2020).
    """
    parsed = [parse_ppi_l4_answer(p) for p in predictions]
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

    result: dict = {
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

    # Evidence citation rate
    tested_correct = [
        i
        for i, (a, g, m) in enumerate(zip(answers, gold_answers, valid_mask))
        if m and a == "tested" and g == "tested"
    ]
    if tested_correct:
        with_evidence = sum(
            1
            for i in tested_correct
            if evidences[i] and (
                len(evidences[i]) > 50
                or any(kw in evidences[i].lower() for kw in PPI_EVIDENCE_KEYWORDS)
            )
        )
        result["evidence_citation_rate"] = with_evidence / len(tested_correct)
    else:
        result["evidence_citation_rate"] = 0.0

    # Temporal accuracy breakdown: PPI uses pre_2015/post_2020
    if temporal_groups:
        valid_temporal = [t for t, m in zip(temporal_groups, valid_mask) if m]
        for group in ["pre_2015", "post_2020"]:
            group_pred = [
                p for p, t in zip(valid_pred, valid_temporal) if t == group
            ]
            group_gold = [
                g for g, t in zip(valid_gold, valid_temporal) if t == group
            ]
            if group_pred:
                result[f"accuracy_{group}"] = accuracy_score(group_gold, group_pred)

        # Contamination flag
        pre = result.get("accuracy_pre_2015")
        post = result.get("accuracy_post_2020")
        if pre is not None and post is not None:
            gap = pre - post
            result["contamination_gap"] = gap
            result["contamination_flag"] = gap > 0.15

    return result


# ── Dispatch ───────────────────────────────────────────────────────────


def compute_all_ppi_llm_metrics(
    task: str,
    predictions: list[str],
    gold: list[dict],
) -> dict:
    """Compute all metrics for a given PPI task.

    Args:
        task: 'ppi-l1', 'ppi-l2', 'ppi-l3', 'ppi-l4'
        predictions: list of raw LLM response strings
        gold: list of gold-standard records (from JSONL)

    Returns: dict of metrics
    """
    if task == "ppi-l1":
        gold_answers = [g["gold_answer"] for g in gold]
        gold_classes = [g.get("gold_category") for g in gold]
        difficulties = [g.get("difficulty") for g in gold]
        return evaluate_ppi_l1(predictions, gold_answers, gold_classes, difficulties)

    elif task == "ppi-l2":
        return evaluate_ppi_l2(predictions, gold)

    elif task == "ppi-l3":
        judge_scores = [parse_ppi_l3_judge_scores(p) for p in predictions]
        return evaluate_ppi_l3(judge_scores)

    elif task == "ppi-l4":
        gold_answers = [g["gold_answer"] for g in gold]
        temporal = [g.get("temporal_group") for g in gold]
        return evaluate_ppi_l4(predictions, gold_answers, temporal)

    else:
        raise ValueError(f"Unknown task: {task}. Choose from ppi-l1, ppi-l2, ppi-l3, ppi-l4")
