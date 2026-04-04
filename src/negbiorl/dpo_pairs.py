"""DPO pair construction for the DPO-vs-GRPO ablation study.

Uses L3 judge scores and L4 predictions to construct (chosen, rejected) pairs.
These pairs test whether DPO matches GRPO performance, supporting the
"GRPO is secretly DPO" (arXiv:2510.00977) hypothesis in scientific domains.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from negbiorl.data_registry import (
    BENCHMARK_MODELS,
    TRAIN_DOMAINS,
    get_domain,
    get_gold_answer_field,
    get_prefixed_task,
    get_results_dir,
    load_jsonl,
    parse_l4_unified,
)
from negbiorl.sft_data import _format_prompt

# Map short model names to actual file-path substrings
_MODEL_PATH_PATTERNS = {
    "haiku": "claude-haiku",
    "gemini": "gemini-2-5-flash_",  # trailing underscore excludes flash-lite
    "gpt": "gpt-4o",
    "qwen": "qwen2-5",
    "llama": "llama-3",
}


# ---------------------------------------------------------------------------
# L3 pairs: chosen = highest judge score, rejected = lowest
# ---------------------------------------------------------------------------

def _collect_l3_responses(domain: str) -> dict[str, list[dict]]:
    """Collect L3 predictions + judge scores across models for a domain.

    Returns: {question_id: [{model, prediction, scores, overall_score}, ...]}
    """
    results_dir = get_results_dir(domain)
    prefix = get_prefixed_task(domain, "l3")  # e.g. "ct-l3" or "l3"
    by_qid: dict[str, list[dict]] = defaultdict(list)

    for model in BENCHMARK_MODELS:
        pat = _MODEL_PATH_PATTERNS.get(model, model)
        # Actual paths: {prefix}_{full-model-name}_{shot}_fs{seed}/predictions.jsonl
        # Judge paths:  {prefix}_{full-model-name}_{shot}_fs{seed}_judged/judge_scores.jsonl
        pred_matches = sorted(results_dir.glob(f"{prefix}_{pat}*/predictions.jsonl"))
        judge_matches = sorted(results_dir.glob(f"{prefix}_{pat}*_judged/judge_scores.jsonl"))

        if not pred_matches or not judge_matches:
            continue

        # Match pred/judge from the same run config by pairing on run directory name
        # Judge dir = "{run_dir}_judged" → strip suffix to find matching pred dir
        pred_by_dir = {p.parent.name: p for p in pred_matches}
        paired = None
        for j in judge_matches:
            run_dir = j.parent.name.removesuffix("_judged")
            if run_dir in pred_by_dir:
                # Prefer zero-shot_fs0
                if "zero-shot_fs0" in run_dir:
                    paired = (pred_by_dir[run_dir], j)
                    break
                elif paired is None:
                    paired = (pred_by_dir[run_dir], j)
        if paired is None:
            continue
        pred_file, judge_file = paired

        preds = {p["question_id"]: p for p in load_jsonl(pred_file)}
        judges = {j["question_id"]: j for j in load_jsonl(judge_file)}

        for qid, pred in preds.items():
            if qid not in judges:
                continue
            scores = judges[qid].get("scores", {})
            if not scores:
                continue
            overall = sum(scores.values()) / len(scores) if scores else 0.0
            by_qid[qid].append({
                "model": model,
                "prediction": pred.get("prediction", ""),
                "scores": scores,
                "overall_score": overall,
            })

    return dict(by_qid)


def build_l3_dpo_pairs(
    domains: list[str] | None = None,
    min_score_gap: float = 0.5,
) -> list[dict[str, Any]]:
    """Build DPO pairs from L3 judge scores.

    For each question, chosen = model with highest judge score,
    rejected = model with lowest judge score. Only pairs with
    sufficient score gap are included.

    Returns list of:
        {"prompt": [msgs], "chosen": str, "rejected": str,
         "domain": str, "task": "l3", "score_gap": float}
    """
    domains = domains or TRAIN_DOMAINS
    pairs = []

    for domain in domains:
        by_qid = _collect_l3_responses(domain)
        # We need the L3 export to reconstruct prompts
        reg = get_domain(domain)
        export_path = Path(__file__).resolve().parents[2] / reg["exports_dir"] / reg["l3_file"]
        if not export_path.exists():
            continue
        exports = {r["question_id"]: r for r in load_jsonl(export_path)}

        for qid, responses in by_qid.items():
            if len(responses) < 2:
                continue
            if qid not in exports:
                continue

            # Sort by score
            responses.sort(key=lambda r: r["overall_score"])
            worst = responses[0]
            best = responses[-1]
            gap = best["overall_score"] - worst["overall_score"]

            if gap < min_score_gap:
                continue

            system_prompt, user_prompt = _format_prompt(domain, "l3", exports[qid])
            pairs.append({
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "chosen": best["prediction"],
                "rejected": worst["prediction"],
                "domain": domain,
                "task": "l3",
                "score_gap": gap,
            })

    return pairs


# ---------------------------------------------------------------------------
# L4 pairs: correct = chosen, incorrect = rejected
# ---------------------------------------------------------------------------

def _collect_l4_responses(domain: str) -> dict[str, list[dict]]:
    """Collect L4 predictions across models for a domain.

    Returns: {question_id: [{model, prediction, gold_answer, correct}, ...]}
    """
    results_dir = get_results_dir(domain)
    prefix = get_prefixed_task(domain, "l4")
    gold_field = get_gold_answer_field(domain)
    by_qid: dict[str, list[dict]] = defaultdict(list)

    # Load export for gold labels
    reg = get_domain(domain)
    export_path = Path(__file__).resolve().parents[2] / reg["exports_dir"] / reg["l4_file"]
    if not export_path.exists():
        return {}
    exports = {r["question_id"]: r for r in load_jsonl(export_path)}

    for model in BENCHMARK_MODELS:
        pat = _MODEL_PATH_PATTERNS.get(model, model)
        pred_matches = sorted(results_dir.glob(f"{prefix}_{pat}*/predictions.jsonl"))
        if not pred_matches:
            continue

        # Use zero-shot fs0 if available
        pred_file = pred_matches[0]
        for p in pred_matches:
            if "zero-shot_fs0" in str(p):
                pred_file = p
                break

        for pred in load_jsonl(pred_file):
            qid = pred["question_id"]
            rec = exports.get(qid)
            if rec is None:
                continue
            gold = rec.get(gold_field, "")
            raw_prediction = pred.get("prediction", "")
            parsed, _ = parse_l4_unified(raw_prediction, domain) if raw_prediction else (None, None)
            correct = parsed is not None and parsed.lower() == gold.strip().lower()
            by_qid[qid].append({
                "model": model,
                "prediction": raw_prediction,
                "gold_answer": gold,
                "correct": correct,
            })

    return dict(by_qid)


def build_l4_dpo_pairs(
    domains: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Build DPO pairs from L4 predictions (correct vs incorrect).

    For each question, we need at least one correct and one incorrect response.

    Returns list of:
        {"prompt": [msgs], "chosen": str, "rejected": str,
         "domain": str, "task": "l4"}
    """
    domains = domains or TRAIN_DOMAINS
    pairs = []

    for domain in domains:
        by_qid = _collect_l4_responses(domain)
        reg = get_domain(domain)
        export_path = Path(__file__).resolve().parents[2] / reg["exports_dir"] / reg["l4_file"]
        if not export_path.exists():
            continue
        exports = {r["question_id"]: r for r in load_jsonl(export_path)}

        for qid, responses in by_qid.items():
            if qid not in exports:
                continue
            correct = [r for r in responses if r["correct"]]
            incorrect = [r for r in responses if not r["correct"]]
            if not correct or not incorrect:
                continue

            system_prompt, user_prompt = _format_prompt(domain, "l4", exports[qid])
            pairs.append({
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "chosen": correct[0]["prediction"],
                "rejected": incorrect[0]["prediction"],
                "domain": domain,
                "task": "l4",
            })

    return pairs


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------

_ERROR_PATTERNS = ("ERROR:", "HTTP Error", "Rate limit", "timed out", "Connection refused")


def _is_error_response(text: str) -> bool:
    """Check if a response is an HTTP/API error rather than a real model output."""
    return any(pat in text for pat in _ERROR_PATTERNS)


def build_all_dpo_pairs(
    domains: list[str] | None = None,
    min_l3_score_gap: float = 0.5,
) -> list[dict[str, Any]]:
    """Build all DPO pairs (L3 + L4) for the ablation study.

    Filters out pairs where either response is an API/HTTP error.
    """
    l3_pairs = build_l3_dpo_pairs(domains, min_l3_score_gap)
    l4_pairs = build_l4_dpo_pairs(domains)
    all_pairs = l3_pairs + l4_pairs
    return [p for p in all_pairs
            if not _is_error_response(p["chosen"]) and not _is_error_response(p["rejected"])]


def save_dpo_pairs(pairs: list[dict], path: Path) -> int:
    """Save DPO pairs as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return len(pairs)
