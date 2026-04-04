#!/usr/bin/env python
"""Phase 1: Run error analysis across domains and models.

Usage:
    python scripts_rl/01_analyze_errors.py --domains dti ct ppi ge
    python scripts_rl/01_analyze_errors.py --domains dti --models qwen llama --tasks l1 l4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from negbiorl.data_registry import BENCHMARK_MODELS, TRAIN_DOMAINS, PROJECT_ROOT
from negbiorl.error_analysis import run_error_analysis


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Error taxonomy analysis")
    parser.add_argument("--domains", nargs="+", default=TRAIN_DOMAINS)
    parser.add_argument("--models", nargs="+", default=BENCHMARK_MODELS)
    parser.add_argument("--tasks", nargs="+", default=["l1", "l4"])
    parser.add_argument("--fewshot", default="fs0")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "negbiorl" / "phase1_diagnostics",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for domain in args.domains:
        for model in args.models:
            for task in args.tasks:
                try:
                    result = run_error_analysis(domain, model, task, args.fewshot)
                    all_results.append(result)
                    print(
                        f"  {domain}/{model}/{task}: "
                        f"{result['total_errors']} errors / {result['n_predictions']} predictions "
                        f"({result['error_rate']:.1%})"
                    )
                except FileNotFoundError as e:
                    print(f"  {domain}/{model}/{task}: SKIP ({e})")

    # Save combined results
    out_path = args.output_dir / "error_taxonomy.jsonl"
    with open(out_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"\nSaved {len(all_results)} analyses to {out_path}")


if __name__ == "__main__":
    main()
