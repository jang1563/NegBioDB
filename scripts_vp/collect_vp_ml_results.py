#!/usr/bin/env python3
"""Collect and aggregate VP ML baseline results.

Reads all results.json from results/vp_baselines/ subdirectories.

Usage:
    python scripts_vp/collect_vp_ml_results.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ML_DIR = PROJECT_ROOT / "results" / "vp_baselines"


def main():
    parser = argparse.ArgumentParser(description="Collect VP ML results.")
    parser.add_argument("--results-dir", type=Path, default=ML_DIR)
    args = parser.parse_args()

    if not args.results_dir.exists():
        print("No results directory found.")
        return

    results = []
    for run_dir in sorted(args.results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        results_file = run_dir / "results.json"
        if not results_file.exists():
            continue
        with open(results_file) as f:
            data = json.load(f)
        data["run_name"] = run_dir.name
        results.append(data)

    if not results:
        print("No results found.")
        return

    print(f"Found {len(results)} runs\n")

    # Aggregate by model+dataset+split (across seeds)
    groups: dict[tuple, list] = defaultdict(list)
    for r in results:
        key = (r.get("model", "?"), r.get("dataset", "?"), r.get("split", "?"))
        groups[key].append(r)

    # Print table
    print(f"{'Model':<12} {'Dataset':<16} {'Split':<18} {'Seeds':>5} {'AUROC':>10} {'MCC':>10} {'F1':>10}")
    print("-" * 83)

    for (model, dataset, split), runs in sorted(groups.items()):
        aurocs = [r["auroc"] for r in runs if r.get("auroc") is not None]
        mccs = [r["mcc"] for r in runs if r.get("mcc") is not None]
        f1s = [r["f1"] for r in runs if r.get("f1") is not None]

        auroc_str = f"{np.mean(aurocs):.3f}" if aurocs else "--"
        mcc_str = f"{np.mean(mccs):.3f}" if mccs else "--"
        f1_str = f"{np.mean(f1s):.3f}" if f1s else "--"

        if len(aurocs) > 1:
            auroc_str += f"+/-{np.std(aurocs, ddof=1):.3f}"
        if len(mccs) > 1:
            mcc_str += f"+/-{np.std(mccs, ddof=1):.3f}"

        print(f"{model:<12} {dataset:<16} {split:<18} {len(runs):>5} {auroc_str:>10} {mcc_str:>10} {f1_str:>10}")

    # Save CSV
    csv_path = args.results_dir / "vp_ml_summary.csv"
    with open(csv_path, "w") as f:
        cols = ["run_name", "model", "dataset", "split", "seed", "auroc", "auprc", "mcc", "f1", "accuracy", "n_train", "n_val", "n_test"]
        f.write(",".join(cols) + "\n")
        for r in results:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"\nSaved CSV: {csv_path}")


if __name__ == "__main__":
    main()
