"""Evaluation functions for the Cell Painting LLM benchmark."""

from __future__ import annotations

import json
import re
from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

_CP_L1_LETTERS = {"A", "B", "C", "D"}
CP_L2_REQUIRED_FIELDS = [
    "compound_identifier",
    "dose",
    "dose_unit",
    "cell_line",
    "batch_id",
    "dmso_distance_summary",
    "reproducibility_summary",
    "qc_summary",
    "outcome_label",
]
CP_EVIDENCE_KEYWORDS = {
    "tested in cell painting", "dmso distance", "dmso centroid",
    "replicate reproducibility", "viability ratio",
    "assay-valid observation", "perturbation result",
    "morphological profile", "phenotypic screen",
}
CP_L3_JUDGE_PROMPT = (
    "Score the explanation on a 1-5 scale for the following dimensions: "
    "evidence_grounding, assay_reasoning, specificity, non_speculation. "
    "Return JSON only."
)
_CP_L3_DIMS = [
    "evidence_grounding",
    "assay_reasoning",
    "specificity",
    "non_speculation",
]


def parse_cp_l1_answer(response: str) -> str | None:
    response = response.strip()
    if not response:
        return None
    if response.upper() in _CP_L1_LETTERS:
        return response.upper()
    for pattern in [
        r"(?:answer|choice|classification)\s*(?:is|:)\s*\(?([ABCD])\)?",
        r"\(([ABCD])\)",
        r"^([ABCD])\.",
        r"^([ABCD])\)",
    ]:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    first = response[0].upper()
    if first in _CP_L1_LETTERS:
        return first
    match = re.search(r"\b([ABCD])\b", response.upper())
    return match.group(1) if match else None


def evaluate_cp_l1(
    predictions: list[str],
    gold_answers: list[str],
    gold_classes: list[str] | None = None,
    difficulties: list[str] | None = None,
) -> dict:
    parsed = [parse_cp_l1_answer(p) for p in predictions]
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

    if gold_classes:
        valid_classes = [c for c, m in zip(gold_classes, valid_mask) if m]
        totals = Counter(valid_classes)
        correct = Counter(
            cls for pred, gold, cls in zip(valid_pred, valid_gold, valid_classes)
            if pred == gold
        )
        result["per_class_accuracy"] = {
            cls: correct[cls] / totals[cls] if totals[cls] else 0.0
            for cls in sorted(totals)
        }
    if difficulties:
        valid_diffs = [d for d, m in zip(difficulties, valid_mask) if m]
        totals = Counter(valid_diffs)
        correct = Counter(
            diff for pred, gold, diff in zip(valid_pred, valid_gold, valid_diffs)
            if pred == gold
        )
        result["per_difficulty_accuracy"] = {
            diff: correct[diff] / totals[diff] if totals[diff] else 0.0
            for diff in sorted(totals)
        }
    return result


def parse_cp_l2_response(response: str) -> dict | None:
    response = response.strip()
    response = re.sub(r"```json\s*", "", response)
    response = re.sub(r"```\s*$", "", response)
    try:
        result = json.loads(response)
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", response)
        if match:
            try:
                parsed = json.loads(match.group())
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                return None
    return None


def evaluate_cp_l2(predictions: list[str], gold_records: list[dict]) -> dict:
    n_total = len(predictions)
    n_valid = 0
    n_schema = 0
    exact_fields = 0
    exact_total = 0

    for pred, gold in zip(predictions, gold_records):
        parsed = parse_cp_l2_response(pred)
        if parsed is None:
            continue
        n_valid += 1
        if all(field in parsed for field in CP_L2_REQUIRED_FIELDS):
            n_schema += 1

        metadata = gold.get("metadata", {})
        gold_extraction = gold.get("gold_extraction") or metadata.get("gold_extraction") or gold
        for field in CP_L2_REQUIRED_FIELDS:
            exact_total += 1
            if str(parsed.get(field, "")).strip().lower() == str(gold_extraction.get(field, "")).strip().lower():
                exact_fields += 1

    return {
        "parse_rate": n_valid / n_total if n_total else 0.0,
        "schema_compliance": n_schema / n_total if n_total else 0.0,
        "field_accuracy": exact_fields / exact_total if exact_total else 0.0,
        "n_total": n_total,
        "n_valid": n_valid,
    }


def parse_cp_l3_judge_scores(response: str) -> dict | None:
    parsed = parse_cp_l2_response(response)
    if parsed is None:
        return None
    scores = {}
    for dim in _CP_L3_DIMS:
        value = parsed.get(dim)
        if isinstance(value, (int, float)) and 1 <= value <= 5:
            scores[dim] = float(value)
    return scores if len(scores) == len(_CP_L3_DIMS) else None


def evaluate_cp_l3(judge_scores: list[dict | None]) -> dict:
    valid = [score for score in judge_scores if score is not None]
    if not valid:
        return {
            "parse_rate": 0.0,
            "n_total": len(judge_scores),
            "n_valid": 0,
        }

    result = {
        "parse_rate": len(valid) / len(judge_scores) if judge_scores else 0.0,
        "n_total": len(judge_scores),
        "n_valid": len(valid),
    }
    for dim in _CP_L3_DIMS:
        vals = [float(score[dim]) for score in valid]
        result[dim] = float(np.mean(vals))
        result[f"{dim}_std"] = float(np.std(vals))
    overall = [float(np.mean([score[dim] for dim in _CP_L3_DIMS])) for score in valid]
    result["overall_mean"] = float(np.mean(overall))
    result["overall_std"] = float(np.std(overall))
    return result


def parse_cp_l4_answer(response: str) -> tuple[str | None, str | None]:
    lines = response.strip().split("\n")
    if not lines:
        return None, None
    first = lines[0].strip().lower()
    answer = None
    if "untested" in first or "not tested" in first:
        answer = "untested"
    elif "tested" in first:
        answer = "tested"
    evidence = "\n".join(lines[1:]).strip() if len(lines) > 1 else None
    return answer, evidence


def evaluate_cp_l4(predictions: list[str], gold_answers: list[str]) -> dict:
    parsed = [parse_cp_l4_answer(p) for p in predictions]
    answers = [pair[0] for pair in parsed]
    evidences = [pair[1] for pair in parsed]
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
            "n_valid": 0,
            "n_total": len(predictions),
        }

    tested_correct = [
        i for i, (pred, gold, valid) in enumerate(zip(answers, gold_answers, valid_mask))
        if valid and pred == "tested" and gold == "tested"
    ]
    with_evidence = 0
    for i in tested_correct:
        evidence = evidences[i]
        if evidence and any(keyword in evidence.lower() for keyword in CP_EVIDENCE_KEYWORDS):
            with_evidence += 1

    return {
        "accuracy": accuracy_score(valid_gold, valid_pred),
        "f1": f1_score(valid_gold, valid_pred, average="binary", pos_label="tested", zero_division=0.0),
        "mcc": matthews_corrcoef(valid_gold, valid_pred),
        "parse_rate": sum(valid_mask) / len(predictions) if predictions else 0.0,
        "evidence_citation_rate": with_evidence / len(tested_correct) if tested_correct else 0.0,
        "n_valid": sum(valid_mask),
        "n_total": len(predictions),
    }


def compute_all_cp_llm_metrics(task: str, predictions: list[str], gold: list[dict]) -> dict:
    task = task.lower()
    if task == "cp-l1":
        gold_answers = [g["gold_answer"] for g in gold]
        gold_classes = [g.get("gold_category") for g in gold]
        difficulties = [g.get("difficulty") for g in gold]
        return evaluate_cp_l1(predictions, gold_answers, gold_classes, difficulties)
    if task == "cp-l2":
        return evaluate_cp_l2(predictions, gold)
    if task == "cp-l3":
        return evaluate_cp_l3([parse_cp_l3_judge_scores(p) for p in predictions])
    if task == "cp-l4":
        return evaluate_cp_l4(predictions, [g["gold_answer"] for g in gold])
    raise ValueError(f"Unknown task: {task}")
