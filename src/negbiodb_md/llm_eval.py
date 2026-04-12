"""Evaluation metrics for MD LLM benchmark tasks.

Mirrors src/negbiodb_dc/llm_eval.py structure.

Metrics:
  L1: MCC, accuracy (4-way MCQ, A/B/C/D)
  L2: field_f1 (per-field exact match), schema_compliance
  L3: rubric mean (0-5), per-axis scores (4 axes)
  L4: MCC, accuracy (binary: real=1 vs synthetic=0)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ── L1 evaluation ─────────────────────────────────────────────────────────────

def parse_l1_response(response: str) -> str | None:
    """Extract single letter (A-D) from LLM response."""
    if not response:
        return None
    response = response.strip()
    if len(response) == 1 and response.upper() in "ABCD":
        return response.upper()
    match = re.search(r"\b([A-D])\b", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def eval_l1(records: list[dict], responses: list[str]) -> dict:
    """Compute L1 metrics: MCC and accuracy."""
    preds, golds = [], []
    for rec, resp in zip(records, responses):
        gold = rec.get("gold_answer", "").upper()
        pred = parse_l1_response(resp)
        if gold and pred:
            preds.append(pred)
            golds.append(gold)

    if not preds:
        return {"mcc": None, "accuracy": None, "n": 0}

    # Map letters to integers for sklearn
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    y_true = [label_map[g] for g in golds]
    y_pred = [label_map.get(p, -1) for p in preds]

    try:
        from sklearn.metrics import matthews_corrcoef, accuracy_score
        mcc = matthews_corrcoef(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
    except Exception:
        correct = sum(1 for g, p in zip(golds, preds) if g == p)
        acc = correct / len(preds)
        mcc = None

    return {"mcc": mcc, "accuracy": acc, "n": len(preds)}


# ── L2 evaluation ─────────────────────────────────────────────────────────────

L2_FIELDS = ["metabolite", "disease", "fold_change", "platform", "biofluid", "outcome"]
PLATFORM_ALIASES = {
    "lc-ms": "lc_ms", "lcms": "lc_ms", "lc/ms": "lc_ms",
    "gc-ms": "gc_ms", "gcms": "gc_ms", "gc/ms": "gc_ms",
    "nmr": "nmr",
}


def parse_l2_response(response: str) -> dict | None:
    """Parse JSON from L2 LLM response."""
    if not response:
        return None
    # Strip markdown code fences
    response = re.sub(r"```(?:json)?\s*", "", response).strip().rstrip("```")
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to extract JSON object
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def _field_match(pred_val, gold_val) -> bool:
    """Flexible field matching: numeric fields within 10%, string fields case-insensitive."""
    if gold_val is None:
        return pred_val is None
    if pred_val is None:
        return False

    if isinstance(gold_val, (int, float)):
        try:
            return abs(float(pred_val) - float(gold_val)) / (abs(float(gold_val)) + 1e-9) < 0.1
        except (ValueError, TypeError):
            return False

    # String: case-insensitive, normalize aliases
    pred_str = str(pred_val).lower().strip()
    gold_str = str(gold_val).lower().strip()
    pred_str = PLATFORM_ALIASES.get(pred_str, pred_str)
    gold_str = PLATFORM_ALIASES.get(gold_str, gold_str)
    return pred_str == gold_str


def eval_l2(records: list[dict], responses: list[str]) -> dict:
    """Compute L2 metrics: per-field accuracy and schema_compliance."""
    field_correct: dict[str, int] = {f: 0 for f in L2_FIELDS}
    field_total: dict[str, int] = {f: 0 for f in L2_FIELDS}
    schema_compliant = 0
    n = 0

    for rec, resp in zip(records, responses):
        parsed = parse_l2_response(resp)
        gold = rec.get("gold_fields", {})

        if parsed is not None and isinstance(parsed, dict):
            schema_compliant += 1

        for field in L2_FIELDS:
            gold_val = gold.get(field)
            if gold_val is None:
                continue
            field_total[field] += 1
            pred_val = (parsed or {}).get(field)
            if _field_match(pred_val, gold_val):
                field_correct[field] += 1
        n += 1

    per_field = {
        f: field_correct[f] / field_total[f] if field_total[f] else None
        for f in L2_FIELDS
    }
    all_vals = [v for v in per_field.values() if v is not None]
    field_f1 = float(np.mean(all_vals)) if all_vals else None

    return {
        "field_f1": field_f1,
        "per_field_accuracy": per_field,
        "schema_compliance": schema_compliant / n if n else None,
        "n": n,
    }


# ── L3 evaluation ─────────────────────────────────────────────────────────────

L3_AXES = [
    "metabolite_biology",
    "disease_mechanism",
    "study_context",
    "alternative_hypothesis",
]


def eval_l3_with_judge(
    records: list[dict],
    responses: list[str],
    judge_scores: list[dict],
) -> dict:
    """Compute L3 metrics from LLM-as-judge scores.

    Args:
        records:      L3 records from md_l3.jsonl
        responses:    Model responses (one per record)
        judge_scores: Judge output dicts with per-axis scores (0-5)

    Returns:
        Dict with overall_mean, per_axis means, n
    """
    axis_scores: dict[str, list[float]] = {ax: [] for ax in L3_AXES}

    for score_dict in judge_scores:
        if not isinstance(score_dict, dict):
            continue
        for ax in L3_AXES:
            val = score_dict.get(ax)
            if val is not None:
                try:
                    axis_scores[ax].append(float(val))
                except (ValueError, TypeError):
                    pass

    per_axis = {
        ax: float(np.mean(scores)) if scores else None
        for ax, scores in axis_scores.items()
    }
    all_vals = [v for v in per_axis.values() if v is not None]
    overall = float(np.mean(all_vals)) if all_vals else None

    return {
        "overall_mean": overall,
        "per_axis": per_axis,
        "n": len(records),
    }


# ── L4 evaluation ─────────────────────────────────────────────────────────────

def parse_l4_response(response: str) -> int | None:
    """Parse L4 response into 1 (real) or 0 (synthetic)."""
    if not response:
        return None
    response = response.strip()
    if re.search(r"\bA\b", response, re.IGNORECASE):
        return 1  # A = Real
    if re.search(r"\bB\b", response, re.IGNORECASE):
        return 0  # B = Synthetic
    # Keywords
    if any(w in response.lower() for w in ("real", "actual", "study", "measured")):
        return 1
    if any(w in response.lower() for w in ("synthetic", "random", "generated", "never")):
        return 0
    return None


def eval_l4(records: list[dict], responses: list[str]) -> dict:
    """Compute L4 metrics: MCC and accuracy (binary real vs synthetic)."""
    preds, golds = [], []
    for rec, resp in zip(records, responses):
        gold = rec.get("label")
        pred = parse_l4_response(resp)
        if gold is not None and pred is not None:
            golds.append(int(gold))
            preds.append(int(pred))

    if not preds:
        return {"mcc": None, "accuracy": None, "n": 0}

    try:
        from sklearn.metrics import matthews_corrcoef, accuracy_score
        mcc = matthews_corrcoef(golds, preds)
        acc = accuracy_score(golds, preds)
    except Exception:
        correct = sum(1 for g, p in zip(golds, preds) if g == p)
        acc = correct / len(preds)
        mcc = None

    return {"mcc": mcc, "accuracy": acc, "n": len(preds)}


# ── Results I/O ───────────────────────────────────────────────────────────────

def save_results(results: dict, out_path: str | Path) -> None:
    """Save evaluation results to JSON."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved results: %s", out_path)


def load_jsonl(path: str | Path) -> list[dict]:
    """Load all records from a JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
