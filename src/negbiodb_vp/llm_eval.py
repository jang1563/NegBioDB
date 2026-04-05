"""Evaluation functions for VP LLM benchmark tasks VP-L1 through VP-L4.

Mirrors src/negbiodb_ppi/llm_eval.py structure from PPI domain.
Key differences from PPI:
  - VP-L1: 4-way (A-D) pathogenicity classification
  - VP-L2: variant interpretation extraction with criteria_f1 (novel metric)
  - VP-L3: 4-dimension judge (population_reasoning, computational_evidence,
            functional_reasoning, gene_disease_specificity)
  - VP-L4: temporal groups pre_2020/post_2023 (ClinVar submission years)
"""

from __future__ import annotations

import json
import re
from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef


# ── VP-L1: MCQ Classification (4-way A-D) ───────────────────────────────

_VP_L1_LETTERS = {"A", "B", "C", "D"}


def _s(val: object) -> str:
    """Safely coerce a JSON field value to str; returns '' for None/float/NaN."""
    if isinstance(val, str):
        return val
    return ""


def parse_vp_l1_answer(response: str) -> str | None:
    """Extract single letter answer (A/B/C/D) from LLM response."""
    response = response.strip()
    if not response:
        return None
    # Try exact single letter
    if response.upper() in _VP_L1_LETTERS:
        return response.upper()
    # Try "Answer: X", "Answer is X", "(X)", "X." patterns
    for pattern in [
        r"(?:answer|choice|classification)\s*(?:is|:)\s*\(?([ABCD])\)?",
        r"\(([ABCD])\)",
        r"^([ABCD])\.",
        r"^([ABCD])\)",
    ]:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    # Fallback: first letter if A-D
    first = response[0].upper()
    if first in _VP_L1_LETTERS:
        return first
    # Last resort: any standalone A-D
    match = re.search(r"\b([ABCD])\b", response.upper())
    return match.group(1) if match else None


def evaluate_vp_l1(
    predictions: list[str],
    gold_answers: list[str],
    gold_classes: list[str] | None = None,
    difficulties: list[str] | None = None,
) -> dict:
    """Evaluate VP-L1 MCQ predictions.

    Returns: accuracy, weighted_f1, macro_f1, mcc, parse_rate,
             per_class_accuracy (if gold_classes), per_difficulty_accuracy.
    """
    parsed = [parse_vp_l1_answer(p) for p in predictions]
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


# ── VP-L2: Structured Extraction ────────────────────────────────────────

VP_L2_REQUIRED_FIELDS = [
    "variants",
    "total_variants_discussed",
    "classification_method",
]


def parse_vp_l2_response(response: str) -> dict | None:
    """Parse JSON from LLM response for VP-L2 extraction."""
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


def _normalize_classification(cls: str) -> str:
    """Normalize pathogenicity classification string."""
    cls = cls.strip().lower().replace(" ", "_").replace("/", "_")
    # Map common variants
    mapping = {
        "pathogenic": "pathogenic",
        "likely_pathogenic": "likely_pathogenic",
        "uncertain_significance": "uncertain_significance",
        "vus": "uncertain_significance",
        "likely_benign": "likely_benign",
        "benign": "benign",
        "benign_likely_benign": "benign",
    }
    return mapping.get(cls, cls)


def _normalize_criteria(criteria: list) -> set[str]:
    """Normalize a list of ACMG criteria codes to uppercase set."""
    result = set()
    for c in criteria:
        c = str(c).strip().upper()
        # Match standard ACMG codes
        m = re.match(r"(PVS1|PS[1-4]|PM[1-6]|PP[1-5]|BA1|BS[1-4]|BP[1-7])", c)
        if m:
            result.add(m.group(1))
    return result


def evaluate_vp_l2(
    predictions: list[str],
    gold_records: list[dict],
) -> dict:
    """Evaluate VP-L2 variant interpretation extraction.

    Returns: schema_compliance, field_f1, classification_accuracy,
    criteria_precision, criteria_recall, criteria_f1, parse_rate.
    """
    n_total = len(predictions)
    n_valid_json = 0
    n_schema_compliant = 0

    # Field-level matching (gene, hgvs, classification)
    field_tp = 0
    field_fp = 0
    field_fn = 0

    # Classification accuracy (for matched variants)
    cls_correct = 0
    cls_total = 0

    # ACMG criteria matching
    criteria_tp = 0
    criteria_fp = 0
    criteria_fn = 0

    # Count and method accuracy
    count_correct = 0
    count_total = 0
    method_correct = 0
    method_total = 0

    for pred_str, gold in zip(predictions, gold_records):
        parsed = parse_vp_l2_response(pred_str)
        if parsed is None:
            gold_vars = gold.get("gold_extraction", {}).get("variants", [])
            field_fn += len(gold_vars)
            for gv in gold_vars:
                criteria_fn += len(_normalize_criteria(gv.get("acmg_criteria_met", [])))
            continue
        n_valid_json += 1

        # Schema compliance
        has_all = all(f in parsed for f in VP_L2_REQUIRED_FIELDS)
        if has_all:
            n_schema_compliant += 1

        gold_ext = gold.get("gold_extraction", gold)
        gold_variants = gold_ext.get("variants", [])
        pred_variants = parsed.get("variants", [])

        # Build gold variant map by (gene, hgvs) for matching
        gold_map: dict[tuple[str, str], dict] = {}
        for gv in gold_variants:
            gene = _s(gv.get("gene")).strip().upper()
            hgvs = _s(gv.get("hgvs")).strip()
            if gene and hgvs:
                gold_map[(gene, hgvs)] = gv

        # Match predicted variants to gold
        matched_pred = set()
        matched_gold = set()
        for pv in pred_variants:
            pgene = _s(pv.get("gene")).strip().upper()
            phgvs = _s(pv.get("hgvs")).strip()
            if not pgene or not phgvs:
                continue
            key = (pgene, phgvs)
            if key in gold_map and key not in matched_gold:
                matched_gold.add(key)
                matched_pred.add(id(pv))
                field_tp += 1

                gv = gold_map[key]

                # Classification accuracy
                gold_cls = _normalize_classification(gv.get("classification", ""))
                pred_cls = _normalize_classification(pv.get("classification", ""))
                if gold_cls:
                    cls_total += 1
                    if pred_cls == gold_cls:
                        cls_correct += 1

                # ACMG criteria matching
                gold_crit = _normalize_criteria(gv.get("acmg_criteria_met", []))
                pred_crit = _normalize_criteria(pv.get("acmg_criteria_met", []))
                criteria_tp += len(gold_crit & pred_crit)
                criteria_fp += len(pred_crit - gold_crit)
                criteria_fn += len(gold_crit - pred_crit)
            else:
                field_fp += 1
                # Count FP criteria
                pred_crit = _normalize_criteria(pv.get("acmg_criteria_met", []))
                criteria_fp += len(pred_crit)

        # Unmatched gold variants
        for key, gv in gold_map.items():
            if key not in matched_gold:
                field_fn += 1
                criteria_fn += len(_normalize_criteria(gv.get("acmg_criteria_met", [])))

        # Count accuracy
        gold_count = gold_ext.get("total_variants_discussed")
        pred_count = parsed.get("total_variants_discussed")
        if gold_count is not None:
            count_total += 1
            if pred_count is not None and int(pred_count) == int(gold_count):
                count_correct += 1

        # Method accuracy
        gold_method = (gold_ext.get("classification_method") or "").strip().lower()
        pred_method = (parsed.get("classification_method") or "").strip().lower()
        if gold_method:
            method_total += 1
            if pred_method and (pred_method in gold_method or gold_method in pred_method):
                method_correct += 1

    # Compute field F1
    field_prec = field_tp / (field_tp + field_fp) if (field_tp + field_fp) > 0 else 0.0
    field_rec = field_tp / (field_tp + field_fn) if (field_tp + field_fn) > 0 else 0.0
    field_f1 = (
        2 * field_prec * field_rec / (field_prec + field_rec)
        if (field_prec + field_rec) > 0
        else 0.0
    )

    # Compute criteria F1
    crit_prec = criteria_tp / (criteria_tp + criteria_fp) if (criteria_tp + criteria_fp) > 0 else 0.0
    crit_rec = criteria_tp / (criteria_tp + criteria_fn) if (criteria_tp + criteria_fn) > 0 else 0.0
    crit_f1 = (
        2 * crit_prec * crit_rec / (crit_prec + crit_rec)
        if (crit_prec + crit_rec) > 0
        else 0.0
    )

    result: dict = {
        "schema_compliance": n_schema_compliant / n_total if n_total else 0.0,
        "field_f1": field_f1,
        "field_precision": field_prec,
        "field_recall": field_rec,
        "classification_accuracy": cls_correct / cls_total if cls_total else 0.0,
        "criteria_f1": crit_f1,
        "criteria_precision": crit_prec,
        "criteria_recall": crit_rec,
        "count_accuracy": count_correct / count_total if count_total else 0.0,
        "method_accuracy": method_correct / method_total if method_total else 0.0,
        "parse_rate": n_valid_json / n_total if n_total else 0.0,
        "n_valid_json": n_valid_json,
        "n_schema_compliant": n_schema_compliant,
        "n_total": n_total,
    }

    return result


# ── VP-L3: Reasoning (LLM-as-Judge) ────────────────────────────────────

VP_L3_JUDGE_PROMPT = (
    "You are evaluating a scientific explanation for why a DNA variant is classified "
    "as benign or likely benign.\n\n"
    "VARIANT CONTEXT:\n{context_text}\n\n"
    "RESPONSE TO EVALUATE:\n{response_text}\n\n"
    "Score the response on 4 dimensions (1-5 each):\n"
    "1. population_reasoning: Correct interpretation of allele frequency evidence "
    "(BA1/BS1 thresholds, population-specific patterns, gnomAD data)\n"
    "2. computational_evidence: Appropriate use of in silico predictors "
    "(CADD, REVEL, AlphaMissense, conservation scores) with nuanced interpretation\n"
    "3. functional_reasoning: Biological mechanism explanation quality "
    "(protein function, domain impact, biochemical consequence of AA change)\n"
    "4. gene_disease_specificity: Understanding of gene-disease relationship, "
    "inheritance pattern, and why variant doesn't disrupt disease mechanism\n\n"
    'Return ONLY a JSON object: {{"population_reasoning": N, '
    '"computational_evidence": N, "functional_reasoning": N, '
    '"gene_disease_specificity": N}}'
)

_VP_L3_DIMS = [
    "population_reasoning",
    "computational_evidence",
    "functional_reasoning",
    "gene_disease_specificity",
]


def parse_vp_l3_judge_scores(response: str) -> dict | None:
    """Parse judge scores from response."""
    parsed = parse_vp_l2_response(response)  # reuse JSON parser
    if parsed is None:
        return None
    scores = {}
    for dim in _VP_L3_DIMS:
        val = parsed.get(dim)
        if isinstance(val, (int, float)) and 1 <= val <= 5:
            scores[dim] = float(val)
    return scores if len(scores) == 4 else None


def evaluate_vp_l3(judge_scores: list[dict | None]) -> dict:
    """Aggregate VP-L3 judge scores.

    judge_scores: list of dicts with dimension scores, or None.
    Returns mean +/- std per dimension + overall.
    """
    result: dict = {}

    valid = [s for s in judge_scores if s is not None]
    if not valid:
        result = {dim: {"mean": 0.0, "std": 0.0} for dim in _VP_L3_DIMS}
        result["overall"] = {"mean": 0.0, "std": 0.0}
        result["n_valid"] = 0
        result["n_total"] = len(judge_scores)
        return result

    for dim in _VP_L3_DIMS:
        values = [s[dim] for s in valid if dim in s]
        result[dim] = {
            "mean": float(np.mean(values)) if values else 0.0,
            "std": float(np.std(values)) if values else 0.0,
        }

    all_scores = [
        np.mean([s[d] for d in _VP_L3_DIMS])
        for s in valid
        if all(d in s for d in _VP_L3_DIMS)
    ]
    result["overall"] = {
        "mean": float(np.mean(all_scores)) if all_scores else 0.0,
        "std": float(np.std(all_scores)) if all_scores else 0.0,
    }
    result["n_valid"] = len(valid)
    result["n_total"] = len(judge_scores)

    return result


# ── VP-L4: Tested vs Untested ───────────────────────────────────────────

VP_EVIDENCE_KEYWORDS = {
    "clinvar", "gnomad", "acmg", "benign", "pathogenic", "variant",
    "allele frequency", "population", "submitter", "clinical testing",
    "genetic testing", "exome", "genome sequencing", "panel testing",
    "classification", "clingen", "expert panel", "review status",
    "likely benign", "likely pathogenic", "uncertain significance",
}


def parse_vp_l4_answer(response: str) -> tuple[str | None, str | None]:
    """Parse VP-L4 response into (answer, evidence).

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
        "no testing", "no evidence of testing", "not been assessed",
        "never been assessed", "not assessed",
    )
    if any(p in first for p in _untested_phrases):
        answer = "untested"
    elif "tested" in first or "assessed" in first:
        answer = "tested"

    evidence = "\n".join(lines[1:]).strip() if len(lines) > 1 else None
    return answer, evidence


def evaluate_vp_l4(
    predictions: list[str],
    gold_answers: list[str],
    temporal_groups: list[str] | None = None,
) -> dict:
    """Evaluate VP-L4 tested/untested predictions.

    Returns: accuracy, f1, mcc, evidence_citation_rate,
             temporal accuracy (pre_2020 vs post_2023).
    """
    parsed = [parse_vp_l4_answer(p) for p in predictions]
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
                and any(kw in evidences[i].lower() for kw in VP_EVIDENCE_KEYWORDS)
            )
        )
        result["evidence_citation_rate"] = with_evidence / len(tested_correct)
    else:
        result["evidence_citation_rate"] = 0.0

    # Temporal accuracy breakdown: VP uses pre_2020/post_2023
    if temporal_groups:
        valid_temporal = [t for t, m in zip(temporal_groups, valid_mask) if m]
        for group in ["pre_2020", "post_2023", "untested_trick", "untested_rare"]:
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
            result["contamination_flag"] = gap > 0.20  # VP uses 0.20 threshold

    return result


# ── Dispatch ────────────────────────────────────────────────────────────


def compute_all_vp_llm_metrics(
    task: str,
    predictions: list[str],
    gold: list[dict],
) -> dict:
    """Compute all metrics for a given VP task.

    Args:
        task: 'vp-l1', 'vp-l2', 'vp-l3', 'vp-l4'
        predictions: list of raw LLM response strings
        gold: list of gold-standard records (from JSONL)

    Returns: dict of metrics
    """
    if task == "vp-l1":
        gold_answers = [g["gold_answer"] for g in gold]
        gold_classes = [g.get("gold_category") for g in gold]
        difficulties = [g.get("difficulty") for g in gold]
        return evaluate_vp_l1(predictions, gold_answers, gold_classes, difficulties)

    elif task == "vp-l2":
        return evaluate_vp_l2(predictions, gold)

    elif task == "vp-l3":
        judge_scores = [parse_vp_l3_judge_scores(p) for p in predictions]
        return evaluate_vp_l3(judge_scores)

    elif task == "vp-l4":
        gold_answers = [g["gold_answer"] for g in gold]
        temporal = [g.get("temporal_group") for g in gold]
        return evaluate_vp_l4(predictions, gold_answers, temporal)

    else:
        raise ValueError(f"Unknown task: {task}. Choose from vp-l1, vp-l2, vp-l3, vp-l4")
