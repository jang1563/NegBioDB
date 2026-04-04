#!/usr/bin/env python
"""Phase 4: Build cross-domain transfer matrix (5×5).

Evaluates each single-domain model on all 5 domains.

Usage:
    python scripts_rl/08_cross_domain_transfer.py \
        --results-dir results/negbiorl/phase4_eval/single_domain/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from negbiorl.data_registry import ALL_DOMAINS, PROJECT_ROOT, load_jsonl
from negbiorl.cross_domain import (
    build_transfer_matrix,
    compute_transfer_gain,
    format_transfer_matrix_latex,
)
from negbiorl.eval_pipeline import evaluate_l4


def main():
    parser = argparse.ArgumentParser(description="Cross-domain transfer matrix")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--baseline-file", type=Path, default=None, help="Baseline MCC values")
    parser.add_argument("--task", default="l4")
    parser.add_argument("--metric", default="mcc")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "negbiorl" / "phase4_eval",
    )
    args = parser.parse_args()

    # Expect structure: results-dir/{train_domain}/{eval_domain}_l4_predictions.jsonl
    train_sources = [d for d in args.results_dir.iterdir() if d.is_dir()]
    results = {}

    for src_dir in sorted(train_sources):
        src_name = src_dir.name
        results[src_name] = {}
        for eval_domain in ALL_DOMAINS:
            pred_path = src_dir / f"{eval_domain}_{args.task}_predictions.jsonl"
            if not pred_path.exists():
                continue
            preds = load_jsonl(pred_path)
            metrics = evaluate_l4(preds, eval_domain)
            results[src_name][eval_domain] = metrics
            print(f"  {src_name} → {eval_domain}: MCC={metrics.get('mcc', 'N/A'):.3f}")

    # Build transfer matrix
    matrix = build_transfer_matrix(results, metric=args.metric)

    # Compute gains if baseline provided
    gains = None
    if args.baseline_file and args.baseline_file.exists():
        with open(args.baseline_file) as f:
            baseline = json.load(f)
        gains = compute_transfer_gain(matrix, baseline)

    # Generate LaTeX
    latex = format_transfer_matrix_latex(matrix, gains)

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "transfer_matrix.json", "w") as f:
        json.dump(matrix, f, indent=2, default=str)
    with open(args.output_dir / "transfer_matrix.tex", "w") as f:
        f.write(latex)
    print(f"\nSaved to {args.output_dir}")
    print("\nLaTeX table:")
    print(latex)


if __name__ == "__main__":
    main()
