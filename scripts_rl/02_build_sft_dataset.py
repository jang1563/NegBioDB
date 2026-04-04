#!/usr/bin/env python
"""Phase 2: Build SFT dataset from NegBioDB exports.

Usage:
    python scripts_rl/02_build_sft_dataset.py
    python scripts_rl/02_build_sft_dataset.py --domains dti ct --max-per 500
"""

from __future__ import annotations

import argparse
from pathlib import Path

from negbiorl.data_registry import TRAIN_DOMAINS, PROJECT_ROOT
from negbiorl.sft_data import build_sft_dataset, save_dataset


def main():
    parser = argparse.ArgumentParser(description="Build SFT training dataset")
    parser.add_argument("--domains", nargs="+", default=TRAIN_DOMAINS)
    parser.add_argument("--tasks", nargs="+", default=["l1", "l4"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-per", type=int, default=None, help="Max records per domain×task")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "negbiorl" / "phase2_training_data",
    )
    args = parser.parse_args()

    print(f"Building SFT dataset: domains={args.domains}, tasks={args.tasks}, split={args.split}")
    records = build_sft_dataset(
        domains=args.domains,
        tasks=args.tasks,
        split=args.split,
        max_per_domain_task=args.max_per,
    )

    out_path = args.output_dir / "sft_dataset.jsonl"
    count = save_dataset(records, out_path)
    print(f"Saved {count} SFT records to {out_path}")

    # Print domain/task breakdown
    from collections import Counter
    dist = Counter((r["domain"], r["task"]) for r in records)
    for (d, t), c in sorted(dist.items()):
        print(f"  {d}/{t}: {c}")


if __name__ == "__main__":
    main()
