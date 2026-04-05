"""LLM response evaluation for DC benchmark tasks DC-L1 through DC-L4.

Mirrors src/negbiodb_vp/llm_eval.py structure:
  - parse_*_answer/response: Extract structured data from raw LLM output
  - evaluate_*: Compute task-specific metrics
  - compute_all_dc_llm_metrics: Dispatch to task-specific evaluator

Key DC-specific metrics:
  - DC-L2: mechanism_f1 (novel) — mechanism_of_interaction + affected_pathways matching
  - DC-L2: interaction_accuracy — interaction_type classification
  - DC-L3: 4 judge dimensions (mechanistic_reasoning, pathway_analysis,
            pharmacological_context, therapeutic_relevance)
  - DC-L4: per-group accuracy (classic_combos, recent_combos, untested_plausible,
            untested_rare) + contamination gap
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

# L2 required JSON fields
_L2_REQUIRED_FIELDS = {
    "drug_a", "drug_b", "shared_targets", "interaction_type",
    "mechanism_of_interaction", "affected_pathways",
}

# L2 valid interaction types
_VALID_INTERACTION_TYPES = {"antagonistic", "synergistic", "additive"}

# L2 valid mechanism categories
_VALID_MECHANISMS = {
    "competitive_binding", "pathway_crosstalk", "metabolic_interference",
    "pharmacokinetic_interaction", "target_redundancy", "feedback_activation",
    "opposing_pathways", "unknown",
}

# L3 judge dimensions
_L3_DIMS = [
    "mechanistic_reasoning",
    "pathway_analysis",
    "pharmacological_context",
    "therapeutic_relevance",
]

# L4 temporal groups
_L4_GROUPS = ["classic_combos", "recent_combos", "untested_plausible", "untested_rare"]

# Evidence keywords for L4
_L4_EVIDENCE_KEYWORDS = {
    "drugcomb", "almanac", "nci-almanac", "dream", "astrazeneca", "sanger",
    "synergy", "antagonism", "zip", "bliss", "loewe", "hsa",
    "combination screen", "drug combination", "combo screen",
    "clinical trial", "combination therapy", "phase i", "phase ii",
}


# ── DC-L1: 4-way synergy classification ─────────────────────────────────


def parse_dc_l1_answer(raw: str) -> str | None:
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

    # First A-D as whole word token
    m = re.search(r"\b([A-D])\b", raw, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return None


def evaluate_dc_l1(
    predictions: list[str],
    gold_labels: list[str],
) -> dict:
    """Compute DC-L1 metrics: accuracy, weighted_f1, macro_f1, MCC."""
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

    parsed = [parse_dc_l1_answer(p) for p in predictions]
    valid_mask = [p is not None for p in parsed]
    n_valid = sum(valid_mask)
    n_total = len(predictions)

    if n_valid == 0:
        return {
            "accuracy": 0.0, "weighted_f1": 0.0, "macro_f1": 0.0,
            "mcc": 0.0, "parse_rate": 0.0, "n_valid": 0, "n_total": n_total,
        }

    y_pred = [p for p, v in zip(parsed, valid_mask) if v]
    y_true = [g for g, v in zip(gold_labels, valid_mask) if v]

    labels = sorted(_L1_VALID)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "parse_rate": n_valid / n_total,
        "n_valid": n_valid,
        "n_total": n_total,
    }


# ── DC-L2: Mechanism extraction ─────────────────────────────────────────


def parse_dc_l2_response(raw: str) -> dict | None:
    """Parse JSON response from DC-L2 extraction task.

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


def _normalize_targets(targets: list | str | None) -> set[str]:
    """Normalize target lists to a set of uppercase gene symbols."""
    if targets is None:
        return set()
    if isinstance(targets, str):
        # comma-separated or semicolon-separated
        targets = re.split(r"[;,]\s*", targets)
    return {t.strip().upper() for t in targets if t and t.strip()}


def _normalize_pathways(pathways: list | str | None) -> set[str]:
    """Normalize pathway lists to lowercase set."""
    if pathways is None:
        return set()
    if isinstance(pathways, str):
        pathways = re.split(r"[;,]\s*", pathways)
    return {p.strip().lower() for p in pathways if p and p.strip()}


def _normalize_mechanism(mech: str | None) -> str:
    """Normalize mechanism string to lowercase underscore form."""
    if not mech:
        return ""
    return mech.strip().lower().replace(" ", "_").replace("-", "_")


def evaluate_dc_l2(
    predictions: list[str],
    gold_extractions: list[dict],
) -> dict:
    """Compute DC-L2 metrics: parse_rate, schema_compliance, field_f1,
    interaction_accuracy, mechanism_f1.

    mechanism_f1 = arithmetic mean of mechanism_accuracy and pathway_f1.
    """
    n_total = len(predictions)
    parsed = [parse_dc_l2_response(p) for p in predictions]
    n_parsed = sum(1 for p in parsed if p is not None)

    n_compliant = 0
    field_scores: list[float] = []
    interaction_correct = 0
    interaction_total = 0
    mechanism_correct = 0
    mechanism_total = 0

    # Pathway-level F1 accumulators
    pathway_tp = 0
    pathway_fp = 0
    pathway_fn = 0

    # Target-level F1 accumulators (shared_targets)
    target_tp = 0
    target_fp = 0
    target_fn = 0

    for pred, gold in zip(parsed, gold_extractions):
        if pred is None:
            field_scores.append(0.0)
            gold_ext = gold.get("gold_extraction", gold) if isinstance(gold, dict) else {}
            gold_paths = _normalize_pathways(gold_ext.get("affected_pathways"))
            pathway_fn += len(gold_paths)
            gold_tgts = _normalize_targets(gold_ext.get("shared_targets"))
            target_fn += len(gold_tgts)
            continue

        gold_ext = gold.get("gold_extraction", gold) if isinstance(gold, dict) else {}

        # Schema compliance
        pred_keys = set(pred.keys())
        if _L2_REQUIRED_FIELDS.issubset(pred_keys):
            n_compliant += 1

        # Field-level F1 (presence of top-level keys)
        gold_keys = set(gold_ext.keys()) if gold_ext else set()
        if not gold_keys and not pred_keys:
            field_scores.append(1.0)
        elif not gold_keys or not pred_keys:
            field_scores.append(0.0)
        else:
            tp = len(gold_keys & pred_keys)
            prec = tp / max(len(pred_keys), 1)
            rec = tp / max(len(gold_keys), 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-10)
            field_scores.append(f1)

        # Interaction type accuracy
        gold_itype = (gold_ext.get("interaction_type") or "").strip().lower()
        pred_itype = (pred.get("interaction_type") or "").strip().lower()
        if gold_itype:
            interaction_total += 1
            if pred_itype == gold_itype:
                interaction_correct += 1

        # Mechanism of interaction accuracy
        gold_mech = _normalize_mechanism(gold_ext.get("mechanism_of_interaction"))
        pred_mech = _normalize_mechanism(pred.get("mechanism_of_interaction"))
        if gold_mech:
            mechanism_total += 1
            if pred_mech == gold_mech:
                mechanism_correct += 1

        # Affected pathways F1
        gold_paths = _normalize_pathways(gold_ext.get("affected_pathways"))
        pred_paths = _normalize_pathways(pred.get("affected_pathways"))
        pathway_tp += len(gold_paths & pred_paths)
        pathway_fp += len(pred_paths - gold_paths)
        pathway_fn += len(gold_paths - pred_paths)

        # Shared targets F1
        gold_tgts = _normalize_targets(gold_ext.get("shared_targets"))
        pred_tgts = _normalize_targets(pred.get("shared_targets"))
        target_tp += len(gold_tgts & pred_tgts)
        target_fp += len(pred_tgts - gold_tgts)
        target_fn += len(gold_tgts - pred_tgts)

    # Compute aggregated F1 scores
    def _f1(tp, fp, fn):
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    pathway_f1 = _f1(pathway_tp, pathway_fp, pathway_fn)
    target_f1 = _f1(target_tp, target_fp, target_fn)

    # mechanism_f1 = average of mechanism_accuracy and pathway_f1
    mech_acc = mechanism_correct / mechanism_total if mechanism_total > 0 else 0.0
    mechanism_f1 = (mech_acc + pathway_f1) / 2.0

    return {
        "parse_rate": n_parsed / max(n_total, 1),
        "schema_compliance": n_compliant / max(n_total, 1),
        "field_f1": float(np.mean(field_scores)) if field_scores else 0.0,
        "interaction_accuracy": (
            interaction_correct / interaction_total if interaction_total > 0 else 0.0
        ),
        "mechanism_accuracy": mech_acc,
        "pathway_f1": pathway_f1,
        "mechanism_f1": mechanism_f1,
        "target_f1": target_f1,
        "n_parsed": n_parsed,
        "n_compliant": n_compliant,
        "n_total": n_total,
    }


# ── DC-L3: Antagonism reasoning evaluation ──────────────────────────────

DC_L3_JUDGE_PROMPT = (
    "You are evaluating a scientific explanation for why a drug combination "
    "is ANTAGONISTIC — the drugs interfere with each other's effects.\n\n"
    "DRUG COMBINATION CONTEXT:\n{context_text}\n\n"
    "RESPONSE TO EVALUATE:\n{response_text}\n\n"
    "Score the response on 4 dimensions (1-5 each):\n"
    "1. mechanistic_reasoning: Does the explanation correctly identify how "
    "the drugs' mechanisms of action conflict or interfere?\n"
    "2. pathway_analysis: Are pathway-level interactions (competition, "
    "feedback loops, crosstalk) properly addressed?\n"
    "3. pharmacological_context: Are PK/PD interactions, metabolic "
    "interference, or dosing considerations discussed?\n"
    "4. therapeutic_relevance: Does the response address clinical "
    "implications and suggest alternatives?\n\n"
    'Return ONLY a JSON object: {{"mechanistic_reasoning": N, '
    '"pathway_analysis": N, "pharmacological_context": N, '
    '"therapeutic_relevance": N}}'
)


def parse_dc_l3_judge_scores(raw: str) -> dict[str, float] | None:
    """Parse judge model scores for DC-L3 reasoning evaluation.

    Accepts both key-value and JSON formats.
    """
    if not raw:
        return None

    scores = {}

    # Try JSON format first (with or without markdown code fence)
    json_str = raw
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if m:
        json_str = m.group(1).strip()
    try:
        data = json.loads(json_str)
        if isinstance(data, dict):
            for dim in _L3_DIMS:
                if dim in data:
                    val = float(data[dim])
                    if 1 <= val <= 5:
                        scores[dim] = val
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Fallback: key-value pattern
    if not scores:
        for dim in _L3_DIMS:
            pattern = rf"{dim}\s*[:=]\s*(\d(?:\.\d)?)"
            m_dim = re.search(pattern, raw, re.IGNORECASE)
            if m_dim:
                val = float(m_dim.group(1))
                if 1 <= val <= 5:
                    scores[dim] = val

    return scores if scores else None


def evaluate_dc_l3(
    judge_outputs: list[str],
) -> dict:
    """Compute DC-L3 metrics from judge model outputs.

    Returns per-dimension mean/std + overall mean/std.
    """
    all_scores: dict[str, list[float]] = {d: [] for d in _L3_DIMS}
    n_parsed = 0

    for output in judge_outputs:
        scores = parse_dc_l3_judge_scores(output)
        if scores is None:
            continue
        n_parsed += 1
        for dim in _L3_DIMS:
            if dim in scores:
                all_scores[dim].append(scores[dim])

    result: dict = {"n_parsed": n_parsed, "n_total": len(judge_outputs)}
    for dim in _L3_DIMS:
        vals = all_scores[dim]
        result[f"{dim}_mean"] = float(np.mean(vals)) if vals else 0.0
        result[f"{dim}_std"] = float(np.std(vals)) if vals else 0.0

    # Overall mean across all dimensions
    all_vals = [v for vals in all_scores.values() for v in vals]
    result["overall_mean"] = float(np.mean(all_vals)) if all_vals else 0.0
    result["overall_std"] = float(np.std(all_vals)) if all_vals else 0.0

    return result


# ── DC-L4: Tested/Untested discrimination ───────────────────────────────


def parse_dc_l4_answer(raw: str) -> tuple[str | None, str | None]:
    """Extract 'tested'/'untested' and evidence from DC-L4 response.

    Returns (answer, evidence_text).
    """
    if not raw:
        return None, None
    if raw.startswith("ERROR:"):
        return None, None

    lines = raw.strip().split("\n")
    first = lines[0].strip().lower()

    answer = None
    _untested_phrases = (
        "untested", "not tested", "not been tested", "never been tested",
        "never tested", "hasn't been tested", "has not been tested",
    )
    if any(p in first for p in _untested_phrases):
        answer = "untested"
    elif "tested" in first:
        answer = "tested"

    evidence = "\n".join(lines[1:]).strip() if len(lines) > 1 else None
    return answer, evidence


def evaluate_dc_l4(
    predictions: list[str],
    gold_labels: list[str],
    temporal_groups: list[str] | None = None,
) -> dict:
    """Compute DC-L4 metrics: accuracy, MCC, per-group accuracy, contamination gap.

    Contamination gap: accuracy(classic_combos) - accuracy(recent_combos).
    If gap > 0.20, flag as contamination risk.
    """
    from sklearn.metrics import accuracy_score, matthews_corrcoef

    parsed = [parse_dc_l4_answer(p) for p in predictions]
    answers = [p[0] for p in parsed]
    evidences = [p[1] for p in parsed]

    valid_mask = [a is not None for a in answers]
    n_valid = sum(valid_mask)
    n_total = len(predictions)

    if n_valid == 0:
        return {
            "accuracy": 0.0, "mcc": 0.0, "valid_rate": 0.0,
            "evidence_citation_rate": 0.0,
            "n_valid": 0, "n_total": n_total,
        }

    y_pred = [a for a, v in zip(answers, valid_mask) if v]
    y_true = [g for g, v in zip(gold_labels, valid_mask) if v]

    result: dict = {
        "accuracy": accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "valid_rate": n_valid / n_total,
        "n_valid": n_valid,
        "n_total": n_total,
        "prediction_distribution": dict(Counter(y_pred)),
        "gold_distribution": dict(Counter(y_true)),
    }

    # Evidence citation rate for true positives
    tested_correct = [
        i for i, (a, g, m) in enumerate(zip(answers, gold_labels, valid_mask))
        if m and a == "tested" and g == "tested"
    ]
    if tested_correct:
        with_evidence = sum(
            1 for i in tested_correct
            if evidences[i] and len(evidences[i]) > 30
            and any(kw in evidences[i].lower() for kw in _L4_EVIDENCE_KEYWORDS)
        )
        result["evidence_citation_rate"] = with_evidence / len(tested_correct)
    else:
        result["evidence_citation_rate"] = 0.0

    # Per-group accuracy breakdown
    if temporal_groups:
        valid_groups = [t for t, m in zip(temporal_groups, valid_mask) if m]
        for group in _L4_GROUPS:
            group_pred = [p for p, g in zip(y_pred, valid_groups) if g == group]
            group_gold = [g for g, tg in zip(y_true, valid_groups) if tg == group]
            if group_pred:
                result[f"accuracy_{group}"] = accuracy_score(group_gold, group_pred)

        # Contamination gap: classic (well-known) vs recent
        classic_acc = result.get("accuracy_classic_combos")
        recent_acc = result.get("accuracy_recent_combos")
        if classic_acc is not None and recent_acc is not None:
            gap = classic_acc - recent_acc
            result["contamination_gap"] = gap
            result["contamination_flag"] = gap > 0.20

    return result


# ── Dispatch ────────────────────────────────────────────────────────────


def compute_all_dc_llm_metrics(
    task: str,
    predictions: list[str],
    gold_data: list,
) -> dict:
    """Dispatch to task-specific evaluator.

    Args:
        task: 'dc-l1', 'dc-l2', 'dc-l3', or 'dc-l4'
        predictions: Raw LLM responses
        gold_data: Gold labels/extractions/judge outputs — may be either
            plain values (str/dict) or full record dicts with a 'gold_answer'
            or 'gold_extraction' key.

    Returns:
        Metrics dict
    """
    def _extract(records, field):
        if records and isinstance(records[0], dict) and field in records[0]:
            return [r.get(field) for r in records]
        return records

    if task == "dc-l1":
        return evaluate_dc_l1(predictions, _extract(gold_data, "gold_answer"))
    elif task == "dc-l2":
        return evaluate_dc_l2(predictions, gold_data)
    elif task == "dc-l3":
        return evaluate_dc_l3(predictions)
    elif task == "dc-l4":
        gold_labels = _extract(gold_data, "gold_answer")
        temporal = None
        if gold_data and isinstance(gold_data[0], dict) and "temporal_group" in gold_data[0]:
            temporal = [r.get("temporal_group") for r in gold_data]
        return evaluate_dc_l4(predictions, gold_labels, temporal)
    else:
        raise ValueError(f"Unknown task: {task}")
