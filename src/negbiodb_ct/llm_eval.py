"""Evaluation functions for CT LLM benchmark tasks CT-L1 through CT-L4.

Mirrors src/negbiodb/llm_eval.py structure from DTI domain.
Key differences from DTI:
  - CT-L1: 5-way (A-E) not 4-way (A-D)
  - CT-L2: Phase 1 uses failure_category as sole gold; 7 JSON fields
  - CT-L3: 4-dimension judge (same dims, different context)
  - CT-L4: temporal groups pre_2020/post_2023 (not DTI pre_2023/post_2024)
  - CT uses "gold_answer" field name (not DTI's "correct_answer")
"""

from __future__ import annotations

import json
import re
from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef


# ── CT-L1: MCQ Classification ──────────────────────────────────────────────

_CT_L1_LETTERS = {"A", "B", "C", "D", "E"}


def parse_ct_l1_answer(response: str) -> str | None:
    """Extract single letter answer (A/B/C/D/E) from LLM response."""
    response = response.strip()
    if not response:
        return None
    # Try exact single letter
    if response.upper() in _CT_L1_LETTERS:
        return response.upper()
    # Try "Answer: X", "Answer is X", "(X)", "X." patterns
    for pattern in [
        r"(?:answer|choice)\s*(?:is|:)\s*\(?([ABCDE])\)?",
        r"\(([ABCDE])\)",
        r"^([ABCDE])\.",
        r"^([ABCDE])\)",
    ]:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    # Fallback: first letter if A-E
    first = response[0].upper()
    if first in _CT_L1_LETTERS:
        return first
    # Last resort: any standalone A-E
    match = re.search(r"\b([ABCDE])\b", response.upper())
    return match.group(1) if match else None


def evaluate_ct_l1(
    predictions: list[str],
    gold_answers: list[str],
    gold_classes: list[str] | None = None,
    difficulties: list[str] | None = None,
) -> dict:
    """Evaluate CT-L1 MCQ predictions.

    Returns: accuracy, weighted_f1, macro_f1, mcc, parse_rate,
             per_class_accuracy (if gold_classes), per_difficulty_accuracy (if difficulties).
    """
    parsed = [parse_ct_l1_answer(p) for p in predictions]
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


# ── CT-L2: Structured Extraction ───────────────────────────────────────────

CT_L2_REQUIRED_FIELDS = [
    "failure_category",
    "failure_subcategory",
    "affected_system",
    "severity_indicator",
    "quantitative_evidence",
    "decision_maker",
    "patient_impact",
]
# Note: CT_L2_CATEGORY_VALUES, CT_L2_SEVERITY_VALUES, CT_L2_DECISION_VALUES were
# removed — unused in Phase 1 evaluation. Phase 2 (multi-field accuracy) is deferred.


def parse_ct_l2_response(response: str) -> dict | None:
    """Parse JSON from LLM response for CT-L2 extraction."""
    response = response.strip()
    # Remove markdown code fences
    response = re.sub(r"```json\s*", "", response)
    response = re.sub(r"```\s*$", "", response)

    try:
        result = json.loads(response)
        if isinstance(result, list) and len(result) > 0:
            result = result[0]  # take first element if array
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        # Try to find JSON object in text
        match = re.search(r"\{[\s\S]*\}", response)
        if match:
            try:
                result = json.loads(match.group())
                return result if isinstance(result, dict) else None
            except json.JSONDecodeError:
                return None
    return None


def evaluate_ct_l2(
    predictions: list[str],
    gold_records: list[dict],
    phase: int = 1,
) -> dict:
    """Evaluate CT-L2 structured extraction.

    Phase 1 (automated): schema_compliance, category_accuracy, field_f1_micro, parse_rate.
    Phase 2 (deferred): severity_accuracy, decision_maker_accuracy, subcategory_f1.

    NOTE: In Phase 1, only failure_category has gold annotations.  field_f1_micro
    is approximate — it compares tokens across all 7 fields, but 6 of 7 gold fields
    will be empty strings.  The primary Phase 1 metric is category_accuracy.
    """
    n_total = len(predictions)
    n_valid_json = 0
    n_schema_compliant = 0
    category_correct = 0
    category_total = 0

    # For field_f1_micro: collect all token-level predictions vs gold
    all_pred_tokens: list[str] = []
    all_gold_tokens: list[str] = []

    for pred_str, gold in zip(predictions, gold_records):
        parsed = parse_ct_l2_response(pred_str)
        if parsed is None:
            continue
        n_valid_json += 1

        # Schema compliance: all 7 fields present
        has_all = all(f in parsed for f in CT_L2_REQUIRED_FIELDS)
        if has_all:
            n_schema_compliant += 1

        # Category accuracy (Phase 1 gold)
        gold_cat = gold.get("gold_answer") or gold.get("gold_category")
        if gold_cat:
            category_total += 1
            pred_cat = parsed.get("failure_category", "")
            if isinstance(pred_cat, str) and pred_cat.lower() == gold_cat.lower():
                category_correct += 1

        # field_f1_micro: token-level F1 across string fields
        gold_extraction = gold.get("gold_extraction", gold)
        for field in CT_L2_REQUIRED_FIELDS:
            gold_val = str(gold_extraction.get(field, "") or "").lower().split()
            pred_val = str(parsed.get(field, "") or "").lower().split()
            all_gold_tokens.extend(gold_val)
            all_pred_tokens.extend(pred_val)

    # Compute field F1 micro
    if all_gold_tokens and all_pred_tokens:
        gold_counter = Counter(all_gold_tokens)
        pred_counter = Counter(all_pred_tokens)
        tp = sum((gold_counter & pred_counter).values())
        prec = tp / sum(pred_counter.values()) if pred_counter else 0.0
        rec = tp / sum(gold_counter.values()) if gold_counter else 0.0
        field_f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    else:
        field_f1 = 0.0

    result: dict = {
        "schema_compliance": n_schema_compliant / n_total if n_total else 0.0,
        "category_accuracy": category_correct / category_total if category_total else 0.0,
        "field_f1_micro": field_f1,
        "parse_rate": n_valid_json / n_total if n_total else 0.0,
        "n_valid_json": n_valid_json,
        "n_schema_compliant": n_schema_compliant,
        "n_total": n_total,
    }

    return result


# ── CT-L3: Reasoning (LLM-as-Judge) ───────────────────────────────────────

CT_L3_JUDGE_PROMPT = (
    "You are evaluating a scientific explanation for a clinical trial failure.\n\n"
    "TRIAL CONTEXT:\n{context_text}\n\n"
    "GROUND TRUTH CATEGORY: {failure_category}\n\n"
    "RESPONSE TO EVALUATE:\n{response_text}\n\n"
    "Score the response on 4 dimensions (1-5 each):\n"
    "1. accuracy: Are factual claims about the drug, target, and disease correct?\n"
    "2. reasoning: Is the causal explanation logically coherent?\n"
    "3. completeness: Does it address mechanism, evidence, clinical factors, and context?\n"
    "4. specificity: Does it reference specific trial data (p-values, endpoints) "
    "rather than making generic statements?\n\n"
    'Return ONLY a JSON object: {{"accuracy": N, "reasoning": N, "completeness": N, "specificity": N}}'
)

_L3_DIMS = ["accuracy", "reasoning", "completeness", "specificity"]


def parse_ct_l3_judge_scores(response: str) -> dict | None:
    """Parse judge scores from response."""
    parsed = parse_ct_l2_response(response)  # reuse JSON parser
    if parsed is None:
        return None
    scores = {}
    for dim in _L3_DIMS:
        val = parsed.get(dim)
        if isinstance(val, (int, float)) and 1 <= val <= 5:
            scores[dim] = float(val)
    return scores if len(scores) == 4 else None


def evaluate_ct_l3(judge_scores: list[dict | None]) -> dict:
    """Aggregate CT-L3 judge scores.

    judge_scores: list of {"accuracy": X, "reasoning": X, ...} dicts or None.
    Returns mean ± std per dimension + overall.
    """
    result: dict = {}

    valid = [s for s in judge_scores if s is not None]
    if not valid:
        result = {dim: {"mean": 0.0, "std": 0.0} for dim in _L3_DIMS}
        result["overall"] = {"mean": 0.0, "std": 0.0}
        result["n_valid"] = 0
        result["n_total"] = len(judge_scores)
        return result

    for dim in _L3_DIMS:
        values = [s[dim] for s in valid if dim in s]
        result[dim] = {
            "mean": float(np.mean(values)) if values else 0.0,
            "std": float(np.std(values)) if values else 0.0,
        }

    all_scores = [
        np.mean([s[d] for d in _L3_DIMS]) for s in valid if all(d in s for d in _L3_DIMS)
    ]
    result["overall"] = {
        "mean": float(np.mean(all_scores)) if all_scores else 0.0,
        "std": float(np.std(all_scores)) if all_scores else 0.0,
    }
    result["n_valid"] = len(valid)
    result["n_total"] = len(judge_scores)

    return result


# ── CT-L4: Tested vs Untested ──────────────────────────────────────────────

CT_EVIDENCE_KEYWORDS = {
    "clinicaltrials", "nct", "pubmed", "doi", "pmid", "p-value",
    "hazard", "aact", "eudract", "fda", "endpoint",
}


def parse_ct_l4_answer(response: str) -> tuple[str | None, str | None]:
    """Parse CT-L4 response into (answer, evidence).

    Returns (tested/untested, evidence_text).
    """
    lines = response.strip().split("\n")
    if not lines:
        return None, None

    first = lines[0].strip().lower()
    answer = None
    _untested_phrases = (
        "untested", "not tested", "not been tested", "never been tested",
        "never tested", "hasn't been tested", "has not been tested",
        "no testing", "no evidence of testing",
    )
    if any(p in first for p in _untested_phrases):
        answer = "untested"
    elif "tested" in first:
        answer = "tested"

    evidence = "\n".join(lines[1:]).strip() if len(lines) > 1 else None
    return answer, evidence


def evaluate_ct_l4(
    predictions: list[str],
    gold_answers: list[str],
    temporal_groups: list[str] | None = None,
) -> dict:
    """Evaluate CT-L4 tested/untested predictions.

    Returns: accuracy, f1, mcc, evidence_citation_rate,
             temporal accuracy (pre_2020 vs post_2023).
    """
    parsed = [parse_ct_l4_answer(p) for p in predictions]
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
                and any(kw in evidences[i].lower() for kw in CT_EVIDENCE_KEYWORDS)
            )
        )
        result["evidence_citation_rate"] = with_evidence / len(tested_correct)
    else:
        result["evidence_citation_rate"] = 0.0

    # Temporal accuracy breakdown: CT uses pre_2020/post_2023
    if temporal_groups:
        valid_temporal = [t for t, m in zip(temporal_groups, valid_mask) if m]
        for group in ["pre_2020", "post_2023"]:
            group_pred = [
                p for p, t in zip(valid_pred, valid_temporal) if t == group
            ]
            group_gold = [
                g for g, t in zip(valid_gold, valid_temporal) if t == group
            ]
            if group_pred:
                result[f"accuracy_{group}"] = accuracy_score(group_gold, group_pred)

        # Contamination flag
        pre = result.get("accuracy_pre_2020")
        post = result.get("accuracy_post_2023")
        if pre is not None and post is not None:
            gap = pre - post
            result["contamination_gap"] = gap
            result["contamination_flag"] = gap > 0.15

    return result


# ── Dispatch ───────────────────────────────────────────────────────────────


def compute_all_ct_llm_metrics(
    task: str,
    predictions: list[str],
    gold: list[dict],
) -> dict:
    """Compute all metrics for a given CT task.

    Args:
        task: 'ct-l1', 'ct-l2', 'ct-l3', 'ct-l4'
        predictions: list of raw LLM response strings
        gold: list of gold-standard records (from JSONL)

    Returns: dict of metrics
    """
    if task == "ct-l1":
        gold_answers = [g["gold_answer"] for g in gold]
        gold_classes = [g.get("gold_category") for g in gold]
        difficulties = [g.get("difficulty") for g in gold]
        return evaluate_ct_l1(predictions, gold_answers, gold_classes, difficulties)

    elif task == "ct-l2":
        return evaluate_ct_l2(predictions, gold)

    elif task == "ct-l3":
        # L3 expects judge scores, not raw predictions
        judge_scores = [parse_ct_l3_judge_scores(p) for p in predictions]
        return evaluate_ct_l3(judge_scores)

    elif task == "ct-l4":
        gold_answers = [g["gold_answer"] for g in gold]
        temporal = [g.get("temporal_group") for g in gold]
        return evaluate_ct_l4(predictions, gold_answers, temporal)

    else:
        raise ValueError(f"Unknown task: {task}. Choose from ct-l1, ct-l2, ct-l3, ct-l4")
