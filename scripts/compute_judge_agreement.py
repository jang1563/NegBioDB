#!/usr/bin/env python3
"""Compute inter-rater agreement between primary and second judge for L3.

Reads:
  - Primary judge scores: results/llm/l3_{run}_judged/judge_scores.jsonl
  - Second judge scores: results/llm/judge_validation/{judge}_scores.jsonl
  - (Optional) Human scores: results/llm/judge_validation/human_scores.jsonl

Computes:
  - Cohen's quadratic weighted kappa per dimension
  - Pearson correlation per dimension
  - Overall agreement summary

Usage:
    python scripts/compute_judge_agreement.py
    python scripts/compute_judge_agreement.py --include-human

Output:
    results/llm/judge_validation/agreement_report.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "llm"
VALIDATION_DIR = RESULTS_DIR / "judge_validation"
DIMS = ["accuracy", "reasoning", "completeness", "specificity"]


def load_primary_scores(results_dir: Path) -> dict[tuple[str, str], dict]:
    """Load primary judge scores indexed by (run_name, question_id)."""
    scores = {}
    for judged_dir in sorted(results_dir.iterdir()):
        if not judged_dir.is_dir() or not judged_dir.name.endswith("_judged"):
            continue
        if not judged_dir.name.startswith("l3_"):
            continue
        run_name = judged_dir.name.replace("_judged", "")
        scores_path = judged_dir / "judge_scores.jsonl"
        if not scores_path.exists():
            continue
        with open(scores_path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("scores") is not None:
                    scores[(run_name, rec["question_id"])] = rec["scores"]
    return scores


def load_second_scores(scores_path: Path) -> dict[tuple[str, str], dict]:
    """Load second judge scores indexed by (source_run, question_id)."""
    scores = {}
    if not scores_path.exists():
        return scores
    with open(scores_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("scores") is not None:
                key = (rec["source_run"], rec["question_id"])
                scores[key] = rec["scores"]
    return scores


def load_human_scores(scores_path: Path) -> dict[tuple[str, str], dict]:
    """Load human scores indexed by (source_run, question_id)."""
    return load_second_scores(scores_path)  # Same format


def compute_agreement(
    scores_a: dict[tuple[str, str], dict],
    scores_b: dict[tuple[str, str], dict],
    label_a: str,
    label_b: str,
) -> dict:
    """Compute agreement metrics between two sets of scores."""
    # Find common items
    common_keys = set(scores_a.keys()) & set(scores_b.keys())
    if not common_keys:
        return {
            "n_common": 0,
            "error": "No common items found",
            "label_a": label_a,
            "label_b": label_b,
        }

    result = {
        "n_common": len(common_keys),
        "label_a": label_a,
        "label_b": label_b,
        "dimensions": {},
    }

    overall_a = []
    overall_b = []

    for dim in DIMS:
        vals_a = []
        vals_b = []
        for key in sorted(common_keys):
            a_val = scores_a[key].get(dim)
            b_val = scores_b[key].get(dim)
            if a_val is not None and b_val is not None:
                vals_a.append(a_val)
                vals_b.append(b_val)

        if len(vals_a) < 2:
            result["dimensions"][dim] = {
                "n": len(vals_a),
                "error": "Insufficient data",
            }
            continue

        arr_a = np.array(vals_a)
        arr_b = np.array(vals_b)

        # Cohen's quadratic weighted kappa
        # Round to integers for kappa (scores are 1-5)
        int_a = np.clip(np.round(arr_a).astype(int), 1, 5)
        int_b = np.clip(np.round(arr_b).astype(int), 1, 5)
        try:
            kappa = cohen_kappa_score(int_a, int_b, weights="quadratic")
        except ValueError:
            kappa = None

        # Pearson correlation
        if np.std(arr_a) > 0 and np.std(arr_b) > 0:
            pearson = float(np.corrcoef(arr_a, arr_b)[0, 1])
        else:
            pearson = None

        # Mean absolute difference
        mad = float(np.mean(np.abs(arr_a - arr_b)))

        result["dimensions"][dim] = {
            "n": len(vals_a),
            "kappa": round(kappa, 4) if kappa is not None else None,
            "pearson": round(pearson, 4) if pearson is not None else None,
            "mean_abs_diff": round(mad, 4),
            "mean_a": round(float(np.mean(arr_a)), 3),
            "mean_b": round(float(np.mean(arr_b)), 3),
        }

        overall_a.extend(vals_a)
        overall_b.extend(vals_b)

    # Overall (all dimensions pooled)
    if len(overall_a) >= 2:
        arr_a = np.array(overall_a)
        arr_b = np.array(overall_b)
        int_a = np.clip(np.round(arr_a).astype(int), 1, 5)
        int_b = np.clip(np.round(arr_b).astype(int), 1, 5)
        try:
            kappa = cohen_kappa_score(int_a, int_b, weights="quadratic")
        except ValueError:
            kappa = None
        if np.std(arr_a) > 0 and np.std(arr_b) > 0:
            pearson = float(np.corrcoef(arr_a, arr_b)[0, 1])
        else:
            pearson = None
        result["overall"] = {
            "n": len(overall_a),
            "kappa": round(kappa, 4) if kappa is not None else None,
            "pearson": round(pearson, 4) if pearson is not None else None,
            "mean_abs_diff": round(float(np.mean(np.abs(arr_a - arr_b))), 4),
        }

    return result


def main():
    parser = argparse.ArgumentParser(description="Compute L3 judge agreement")
    parser.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR,
    )
    parser.add_argument(
        "--second-judge-file", type=str, default=None,
        help="Second judge scores file name (default: auto-detect)",
    )
    parser.add_argument(
        "--include-human", action="store_true",
        help="Include human scores in agreement analysis",
    )
    args = parser.parse_args()

    val_dir = args.results_dir / "judge_validation"
    if not val_dir.exists():
        print("No judge_validation directory found.")
        return

    # Load primary (Gemini Flash-Lite) scores
    print("Loading primary judge scores...")
    primary = load_primary_scores(args.results_dir)
    print(f"  {len(primary)} primary scores loaded")

    # Find second judge scores
    if args.second_judge_file:
        second_path = val_dir / args.second_judge_file
    else:
        # Auto-detect: find *_scores.jsonl (not human_scores.jsonl)
        candidates = [
            p for p in val_dir.glob("*_scores.jsonl")
            if p.name != "human_scores.jsonl"
        ]
        if not candidates:
            print("No second judge scores found.")
            return
        second_path = candidates[0]

    print(f"Loading second judge scores from: {second_path.name}")
    second = load_second_scores(second_path)
    print(f"  {len(second)} second judge scores loaded")

    # Compute primary vs second agreement
    report = {}
    judge_name = second_path.stem.replace("_scores", "")
    print(f"\n=== Primary (Flash-Lite) vs Second ({judge_name}) ===")
    agreement = compute_agreement(primary, second, "flash-lite", judge_name)
    report["primary_vs_second"] = agreement

    print(f"  Common items: {agreement['n_common']}")
    if "dimensions" in agreement:
        for dim in DIMS:
            d = agreement["dimensions"].get(dim, {})
            kappa = d.get("kappa")
            pearson = d.get("pearson")
            kappa_str = f"{kappa:.3f}" if kappa is not None else "N/A"
            pearson_str = f"{pearson:.3f}" if pearson is not None else "N/A"
            print(f"  {dim:15s}  kappa={kappa_str}  r={pearson_str}  "
                  f"MAD={d.get('mean_abs_diff', 'N/A')}")
    if "overall" in agreement:
        o = agreement["overall"]
        print(f"  {'OVERALL':15s}  kappa={o['kappa']:.3f}  r={o['pearson']:.3f}  "
              f"MAD={o['mean_abs_diff']:.3f}")

    # Human scores (optional)
    if args.include_human:
        human_path = val_dir / "human_scores.jsonl"
        if human_path.exists():
            print(f"\nLoading human scores...")
            human = load_human_scores(human_path)
            print(f"  {len(human)} human scores loaded")

            # Primary vs human
            print("\n=== Primary (Flash-Lite) vs Human ===")
            ph_agreement = compute_agreement(primary, human, "flash-lite", "human")
            report["primary_vs_human"] = ph_agreement
            if "dimensions" in ph_agreement:
                for dim in DIMS:
                    d = ph_agreement["dimensions"].get(dim, {})
                    kappa = d.get("kappa")
                    kappa_str = f"{kappa:.3f}" if kappa is not None else "N/A"
                    print(f"  {dim:15s}  kappa={kappa_str}")

            # Second vs human
            print(f"\n=== Second ({judge_name}) vs Human ===")
            sh_agreement = compute_agreement(second, human, judge_name, "human")
            report["second_vs_human"] = sh_agreement
            if "dimensions" in sh_agreement:
                for dim in DIMS:
                    d = sh_agreement["dimensions"].get(dim, {})
                    kappa = d.get("kappa")
                    kappa_str = f"{kappa:.3f}" if kappa is not None else "N/A"
                    print(f"  {dim:15s}  kappa={kappa_str}")
        else:
            print(f"\n  Human scores not found at {human_path}")

    # Save report
    report_path = val_dir / "agreement_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {report_path}")

    # Interpretation
    if "overall" in agreement and agreement["overall"].get("kappa") is not None:
        kappa = agreement["overall"]["kappa"]
        if kappa >= 0.8:
            interp = "almost perfect"
        elif kappa >= 0.6:
            interp = "substantial"
        elif kappa >= 0.4:
            interp = "moderate"
        elif kappa >= 0.2:
            interp = "fair"
        else:
            interp = "slight/poor"
        print(f"\nOverall kappa = {kappa:.3f} ({interp} agreement)")
        if kappa < 0.6:
            print("  Note: kappa < 0.6 — consider adding caveat in paper")


if __name__ == "__main__":
    main()
