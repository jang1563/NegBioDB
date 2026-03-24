"""LLM response evaluation for GE benchmark tasks GE-L1 through GE-L4.

Mirrors src/negbiodb_ppi/llm_eval.py structure:
  - parse_*_answer/response: Extract structured data from raw LLM output
  - evaluate_*: Compute task-specific metrics
  - compute_all_ge_llm_metrics: Dispatch to task-specific evaluator
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter

import numpy as np

logger = logging.getLogger(__name__)

# Valid labels for each task
_L1_VALID = {"A", "B", "C", "D"}
_L4_VALID = {"tested", "untested"}


# ── GE-L1: 4-way classification ──────────────────────────────────────────


def parse_ge_l1_answer(raw: str) -> str | None:
    """Extract single-letter answer (A-D) from LLM response.

    Tries:
      1. Single letter on first line
      2. Pattern like "Answer: B" or "(B)"
      3. First A-D word token in full text
    """
    if not raw:
        return None
    if raw.startswith("ERROR:"):
        return None

    raw = raw.strip()
    first_line = raw.split("\n")[0].strip()

    # Single letter
    if first_line.upper() in _L1_VALID:
        return first_line.upper()

    # "Answer: B" or "(B)" or "B)"
    m = re.search(r"(?:answer[:\s]*|[\(\[])\s*([A-D])\s*[)\]]?", first_line, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # First A-D as a whole word/token (not embedded in words)
    m = re.search(r"\b([A-D])\b", raw, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return None


def evaluate_ge_l1(
    predictions: list[str],
    gold_labels: list[str],
) -> dict:
    """Compute GE-L1 metrics: accuracy, weighted_f1, macro_f1, MCC."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        matthews_corrcoef,
    )

    parsed = [parse_ge_l1_answer(p) for p in predictions]
    valid_mask = [p is not None for p in parsed]
    n_valid = sum(valid_mask)
    n_total = len(predictions)

    if n_valid == 0:
        return {
            "accuracy": 0.0, "weighted_f1": 0.0, "macro_f1": 0.0,
            "mcc": 0.0, "valid_rate": 0.0, "n_valid": 0, "n_total": n_total,
        }

    y_pred = [p for p, v in zip(parsed, valid_mask) if v]
    y_true = [g for g, v in zip(gold_labels, valid_mask) if v]

    labels = sorted(_L1_VALID)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "valid_rate": n_valid / n_total,
        "n_valid": n_valid,
        "n_total": n_total,
    }


# ── GE-L2: Information extraction ────────────────────────────────────────


def parse_ge_l2_response(raw: str) -> dict | None:
    """Parse JSON response from GE-L2 extraction task.

    Tries:
      1. Direct JSON parse
      2. Extract JSON from markdown code block
      3. Regex for JSON object
    """
    if not raw:
        return None

    raw = raw.strip()

    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try markdown code block
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try extracting JSON object
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _get_gene_list(extraction: dict) -> list[dict]:
    """Extract gene list from either 'genes' or legacy 'essentiality_findings' key."""
    return extraction.get("genes") or extraction.get("essentiality_findings") or []


def evaluate_ge_l2(
    predictions: list[str],
    gold_extractions: list[dict],
) -> dict:
    """Compute GE-L2 metrics: parse_rate, schema_compliance, field_f1,
    essentiality_accuracy.

    Normalised gold schema uses 'genes' key (legacy 'essentiality_findings'
    is also accepted).
    """
    n_total = len(predictions)
    parsed = [parse_ge_l2_response(p) for p in predictions]
    n_parsed = sum(1 for p in parsed if p is not None)

    required_fields = {"genes", "total_genes_mentioned", "screen_type"}
    n_compliant = 0
    field_scores: list[float] = []
    essentiality_correct = 0
    essentiality_total = 0

    for pred, gold in zip(parsed, gold_extractions):
        if pred is None:
            field_scores.append(0.0)
            continue

        # Schema compliance — normalise pred key (accept either)
        pred_norm_keys = set(pred.keys())
        if "essentiality_findings" in pred_norm_keys and "genes" not in pred_norm_keys:
            pred_norm_keys = (pred_norm_keys - {"essentiality_findings"}) | {"genes"}
        if required_fields.issubset(pred_norm_keys):
            n_compliant += 1

        # Field-level F1 against normalised gold keys
        gold_norm = {}
        if gold:
            gold_norm = dict(gold)
            if "essentiality_findings" in gold_norm and "genes" not in gold_norm:
                gold_norm["genes"] = gold_norm.pop("essentiality_findings")
            if "total_gene_count" in gold_norm and "total_genes_mentioned" not in gold_norm:
                gold_norm["total_genes_mentioned"] = gold_norm.pop("total_gene_count")
        gold_fields = set(gold_norm.keys())
        pred_fields = pred_norm_keys
        if not gold_fields and not pred_fields:
            field_scores.append(1.0)
        elif not gold_fields or not pred_fields:
            field_scores.append(0.0)
        else:
            tp = len(gold_fields & pred_fields)
            precision = tp / max(len(pred_fields), 1)
            recall = tp / max(len(gold_fields), 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-10)
            field_scores.append(f1)

        # Essentiality accuracy — compare per-gene essentiality_status
        pred_genes = _get_gene_list(pred)
        gold_genes = _get_gene_list(gold_norm)
        if gold_genes:
            # Build gold lookup: gene_name → essentiality_status
            gold_lookup = {
                g.get("gene_name", "").upper(): g.get("essentiality_status", "")
                for g in gold_genes
                if isinstance(g, dict)
            }
            for pg in pred_genes:
                if not isinstance(pg, dict):
                    continue
                gname = pg.get("gene_name", "").upper()
                pred_status = pg.get("essentiality_status", "").lower().strip()
                if gname in gold_lookup:
                    essentiality_total += 1
                    gold_status = gold_lookup[gname].lower().strip()
                    if pred_status == gold_status:
                        essentiality_correct += 1

    return {
        "parse_rate": n_parsed / max(n_total, 1),
        "schema_compliance": n_compliant / max(n_total, 1),
        "field_f1": float(np.mean(field_scores)) if field_scores else 0.0,
        "essentiality_accuracy": (
            essentiality_correct / essentiality_total
            if essentiality_total > 0
            else 0.0
        ),
        "essentiality_n": essentiality_total,
        "n_parsed": n_parsed,
        "n_total": n_total,
    }


# ── GE-L3: Reasoning evaluation ──────────────────────────────────────────

GE_L3_JUDGE_PROMPT = (
    "You are evaluating a scientific explanation for why a gene is "
    "NON-ESSENTIAL in a specific cancer cell line.\n\n"
    "GENE-CELL LINE CONTEXT:\n{context_text}\n\n"
    "RESPONSE TO EVALUATE:\n{response_text}\n\n"
    "Score the response on 4 dimensions (1-5 each):\n"
    "1. biological_plausibility: Are biological reasons (gene function, "
    "pathway role, tissue context) scientifically sound?\n"
    "2. pathway_reasoning: Does the explanation address pathway redundancy, "
    "compensatory mechanisms, or lineage-specific dispensability?\n"
    "3. context_specificity: Are claims specific to this gene in this cell "
    "line/lineage or generic?\n"
    "4. mechanistic_depth: Are multiple relevant factors considered "
    "(expression, copy number, mutation status, tissue of origin)?\n\n"
    'Return ONLY a JSON object: {{"biological_plausibility": N, '
    '"pathway_reasoning": N, "context_specificity": N, "mechanistic_depth": N}}'
)


def parse_ge_l3_judge_scores(raw: str) -> dict[str, float] | None:
    """Parse judge model scores for GE-L3 reasoning evaluation.

    Expected format (from judge prompt):
        biological_plausibility: 4
        pathway_reasoning: 3
        context_specificity: 5
        mechanistic_depth: 4
    """
    if not raw:
        return None

    dimensions = [
        "biological_plausibility", "pathway_reasoning",
        "context_specificity", "mechanistic_depth",
    ]
    scores = {}

    for dim in dimensions:
        pattern = rf"{dim}\s*[:=]\s*(\d(?:\.\d)?)"
        m = re.search(pattern, raw, re.IGNORECASE)
        if m:
            scores[dim] = float(m.group(1))

    # Also try JSON format (with or without markdown code fence)
    if not scores:
        json_str = raw
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if m:
            json_str = m.group(1).strip()
        try:
            data = json.loads(json_str)
            if isinstance(data, dict):
                for dim in dimensions:
                    if dim in data:
                        scores[dim] = float(data[dim])
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    return scores if scores else None


def evaluate_ge_l3(
    judge_outputs: list[str],
) -> dict:
    """Compute GE-L3 metrics from judge model outputs."""
    dimensions = [
        "biological_plausibility", "pathway_reasoning",
        "context_specificity", "mechanistic_depth",
    ]
    all_scores: dict[str, list[float]] = {d: [] for d in dimensions}
    n_parsed = 0

    for output in judge_outputs:
        scores = parse_ge_l3_judge_scores(output)
        if scores is None:
            continue
        n_parsed += 1
        for dim in dimensions:
            if dim in scores:
                all_scores[dim].append(scores[dim])

    result: dict = {"n_parsed": n_parsed, "n_total": len(judge_outputs)}
    for dim in dimensions:
        vals = all_scores[dim]
        result[f"{dim}_mean"] = float(np.mean(vals)) if vals else 0.0
        result[f"{dim}_std"] = float(np.std(vals)) if vals else 0.0

    # Overall mean
    all_vals = [v for vals in all_scores.values() for v in vals]
    result["overall_mean"] = float(np.mean(all_vals)) if all_vals else 0.0
    result["overall_std"] = float(np.std(all_vals)) if all_vals else 0.0

    return result


# ── GE-L4: Tested/Untested discrimination ────────────────────────────────


def parse_ge_l4_answer(raw: str) -> str | None:
    """Extract 'tested' or 'untested' from GE-L4 response."""
    if not raw:
        return None
    if raw.startswith("ERROR:"):
        return None

    raw = raw.strip().lower()
    first_line = raw.split("\n")[0].strip()

    if first_line in _L4_VALID:
        return first_line

    # Search for keywords
    if "untested" in first_line:
        return "untested"
    if "tested" in first_line:
        return "tested"

    # Try full text
    if "untested" in raw:
        return "untested"
    if "tested" in raw:
        return "tested"

    return None


def evaluate_ge_l4(
    predictions: list[str],
    gold_labels: list[str],
) -> dict:
    """Compute GE-L4 metrics: accuracy, MCC, temporal contamination gap."""
    from sklearn.metrics import accuracy_score, matthews_corrcoef

    parsed = [parse_ge_l4_answer(p) for p in predictions]
    valid_mask = [p is not None for p in parsed]
    n_valid = sum(valid_mask)
    n_total = len(predictions)

    if n_valid == 0:
        return {
            "accuracy": 0.0, "mcc": 0.0, "valid_rate": 0.0,
            "n_valid": 0, "n_total": n_total,
        }

    y_pred = [p for p, v in zip(parsed, valid_mask) if v]
    y_true = [g for g, v in zip(gold_labels, valid_mask) if v]

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "valid_rate": n_valid / n_total,
        "n_valid": n_valid,
        "n_total": n_total,
        "prediction_distribution": dict(Counter(y_pred)),
        "gold_distribution": dict(Counter(y_true)),
    }


# ── Dispatch ──────────────────────────────────────────────────────────────


def compute_all_ge_llm_metrics(
    task: str,
    predictions: list[str],
    gold_data: list,
) -> dict:
    """Dispatch to task-specific evaluator.

    Args:
        task: 'ge-l1', 'ge-l2', 'ge-l3', or 'ge-l4'
        predictions: Raw LLM responses
        gold_data: Gold labels/extractions/judge outputs — may be either
            plain values (str/dict) or full record dicts with a 'gold_answer'
            or 'gold_extraction' key.

    Returns:
        Metrics dict
    """
    # Normalise gold_data: if records are dicts with dataset keys, extract
    # the relevant field for each task.
    def _extract(records, field):
        if records and isinstance(records[0], dict) and field in records[0]:
            return [r.get(field) for r in records]
        return records

    if task == "ge-l1":
        return evaluate_ge_l1(predictions, _extract(gold_data, "gold_answer"))
    elif task == "ge-l2":
        return evaluate_ge_l2(predictions, _extract(gold_data, "gold_extraction"))
    elif task == "ge-l3":
        return evaluate_ge_l3(predictions)
    elif task == "ge-l4":
        return evaluate_ge_l4(predictions, _extract(gold_data, "gold_answer"))
    else:
        raise ValueError(f"Unknown task: {task}")
