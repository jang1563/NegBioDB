#!/usr/bin/env python
"""Phase 4: Evaluate fine-tuned model and compare with baseline.

Runs vLLM inference on held-out test split, evaluates with domain-specific
metrics, and computes before/after improvement.

Usage:
    python scripts_rl/07_eval_before_after.py \
        --before-dir results/negbiorl/phase4_eval/baseline/ \
        --after-dir results/negbiorl/phase4_eval/grpo/ \
        --domains dti ct ppi ge vp
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from negbiorl.data_registry import ALL_DOMAINS, PROJECT_ROOT, load_jsonl
from negbiorl.eval_pipeline import evaluate_before_after, evaluate_l1, evaluate_l4
from negbiorl.pbs_metric import compute_pbs_delta


def main():
    parser = argparse.ArgumentParser(description="Before/after evaluation")
    parser.add_argument("--before-dir", type=Path, required=True, help="Dir with baseline predictions")
    parser.add_argument("--after-dir", type=Path, required=True, help="Dir with fine-tuned predictions")
    parser.add_argument("--domains", nargs="+", default=ALL_DOMAINS)
    parser.add_argument("--tasks", nargs="+", default=["l1", "l4"])
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "negbiorl" / "phase4_eval" / "before_after.json",
    )
    args = parser.parse_args()

    results = {}
    for domain in args.domains:
        results[domain] = {}
        for task in args.tasks:
            before_path = args.before_dir / f"{domain}_{task}_predictions.jsonl"
            after_path = args.after_dir / f"{domain}_{task}_predictions.jsonl"

            if not before_path.exists() or not after_path.exists():
                print(f"  SKIP {domain}/{task}: missing predictions")
                continue

            before_preds = load_jsonl(before_path)
            after_preds = load_jsonl(after_path)

            result = evaluate_before_after(before_preds, after_preds, domain, task)
            results[domain][task] = result

            # Print summary
            deltas = result["deltas"]
            key = "mcc" if task == "l4" else "accuracy"
            delta_key = f"delta_{key}"
            before_val = result["before"].get(key, 0)
            after_val = result["after"].get(key, 0)
            delta_val = deltas.get(delta_key, 0)
            sign = "+" if delta_val >= 0 else ""
            print(f"  {domain}/{task}: {key} {before_val:.3f} → {after_val:.3f} ({sign}{delta_val:.3f})")

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
